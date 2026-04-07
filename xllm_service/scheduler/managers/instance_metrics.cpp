/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm-service/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "instance_metrics.h"

#include <glog/logging.h>

#include <algorithm>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "common/global_gflags.h"
#include "scheduler/managers/instance_topology.h"

namespace {
constexpr const char* kEtcdLoadMetricsPrefix = "XLLM:LOADMETRICS:";
}  // namespace

namespace xllm_service {

InstanceMetricsImpl::InstanceMetricsImpl(
    const Options& options,
    const std::shared_ptr<EtcdClient>& etcd_client,
    bool is_master_service)
    : options_(options),
      etcd_client_(etcd_client),
      is_master_service_(is_master_service) {
  if (!is_master_service_) {
    auto handle_load_metrics =
        std::bind(&InstanceMetricsImpl::update_load_metrics,
                  this,
                  std::placeholders::_1,
                  std::placeholders::_2);
    etcd_client_->add_watch(kEtcdLoadMetricsPrefix, handle_load_metrics);
  }
}

InstanceMetricsImpl::~InstanceMetricsImpl() { shutdown(); }

void InstanceMetricsImpl::set_topology(InstanceTopologyImpl* topology) {
  topology_impl_ = topology;
}

void InstanceMetricsImpl::load_initial_load_metrics_from_etcd() {
  std::unordered_map<std::string, LoadMetrics> loaded_metrics;
  etcd_client_->get_prefix(kEtcdLoadMetricsPrefix, &loaded_metrics);
  std::unique_lock<std::shared_mutex> lock(metrics_mutex_);
  load_metrics_ = std::move(loaded_metrics);
}

void InstanceMetricsImpl::shutdown() {
  exited_.store(true, std::memory_order_release);
}

void InstanceMetricsImpl::fill_load_balance_infos_no_lock(
    const std::vector<std::string>& prefill_instances,
    const std::vector<std::string>& decode_instances,
    LoadBalanceInfos* infos) {
  infos->prefill_load_metrics.clear();
  infos->decode_load_metrics.clear();
  infos->prefill_max_waiting_requests_num = 0;
  infos->decode_max_waiting_requests_num = 0;

  for (const auto& name : prefill_instances) {
    auto it = load_metrics_.find(name);
    if (it != load_metrics_.end()) {
      infos->prefill_load_metrics.insert_or_assign(name, it->second);
    }
  }
  for (const auto& name : decode_instances) {
    auto it = load_metrics_.find(name);
    if (it != load_metrics_.end()) {
      infos->decode_load_metrics.insert_or_assign(name, it->second);
    }
  }

  uint64_t prefill_max = 0;
  for (const auto& p : infos->prefill_load_metrics) {
    prefill_max = std::max(prefill_max, p.second.waiting_requests_num);
  }
  infos->prefill_max_waiting_requests_num = prefill_max;

  uint64_t decode_max = 0;
  for (const auto& p : infos->decode_load_metrics) {
    decode_max = std::max(decode_max, p.second.waiting_requests_num);
  }
  infos->decode_max_waiting_requests_num = decode_max;

  infos->request_metrics.clear();
  for (const auto& p : request_metrics_) {
    infos->request_metrics.insert_or_assign(p.first, p.second);
  }
}

void InstanceMetricsImpl::fill_load_balance_infos(
    const std::vector<std::string>& prefill_instances,
    const std::vector<std::string>& decode_instances,
    LoadBalanceInfos* infos) {
  std::shared_lock<std::shared_mutex> metric_lock(metrics_mutex_);
  infos->instance_infos.clear();
  fill_load_balance_infos_no_lock(prefill_instances, decode_instances, infos);
}

void InstanceMetricsImpl::record_load_metrics_update(
    const std::string& instance_name,
    const proto::LoadMetrics& load_metrics) {
  std::unique_lock<std::shared_mutex> lock(metrics_mutex_);

  updated_metrics_.insert_or_assign(
      instance_name,
      LoadMetrics(load_metrics.waiting_requests_num(),
                  load_metrics.gpu_cache_usage_perc()));
}

bool InstanceMetricsImpl::upload_load_metrics() {
  std::unordered_map<std::string, LoadMetrics> upload_snapshot;
  std::unordered_set<std::string> remove_snapshot;
  {
    std::unique_lock<std::shared_mutex> lk(metrics_mutex_);
    for (auto& iter : updated_metrics_) {
      load_metrics_.insert_or_assign(iter.first, iter.second);
    }
    for (auto& iter : removed_instance_) {
      load_metrics_.erase(iter);
    }
    upload_snapshot = updated_metrics_;
    remove_snapshot = removed_instance_;
    updated_metrics_.clear();
    removed_instance_.clear();
  }
  bool status = etcd_client_->set(kEtcdLoadMetricsPrefix, upload_snapshot);
  status = status && etcd_client_->rm(kEtcdLoadMetricsPrefix, remove_snapshot);
  return status;
}

void InstanceMetricsImpl::update_latency_metrics(
    const std::string& instance_name,
    const proto::LatencyMetrics& latency_metrics) {
  std::unique_lock<std::shared_mutex> lock(metrics_mutex_);

  latency_metrics_.insert_or_assign(
      instance_name,
      LatencyMetrics(latency_metrics.recent_max_ttft(),
                     latency_metrics.recent_max_tbt()));
}

void InstanceMetricsImpl::update_request_metrics(
    std::shared_ptr<Request> request,
    RequestAction action) {
  if (options_.load_balance_policy() != "SLO_AWARE") {
    return;
  }
  if (topology_impl_ == nullptr) {
    LOG(ERROR) << "update_request_metrics: topology not set";
    return;
  }

  std::string flip_decode_name;
  std::string flip_prefill_name;
  {
    std::unique_lock<std::shared_mutex> lock(metrics_mutex_);

    auto prefill_it = request_metrics_.find(request->routing.prefill_name);
    if (prefill_it == request_metrics_.end()) {
      LOG(ERROR) << "Failed to find instance request metrics, instance name : "
                 << request->routing.prefill_name;
      return;
    }

    auto decode_it = request_metrics_.find(request->routing.decode_name);
    if (decode_it == request_metrics_.end()) {
      LOG(ERROR) << "Failed to find instance request metrics, instance name : "
                 << request->routing.decode_name;
      return;
    }

    int64_t estimated_tpot = 0;
    int64_t num_prompt_tokens = request->token_ids.size();
    int64_t num_generated_tokens = request->num_generated_tokens;
    switch (action) {
      case RequestAction::SCHEDULE: {
        prefill_it->second.prefill_request_num += 1;
        prefill_it->second.prefill_token_num += num_prompt_tokens;

        decode_it->second.decode_request_num += 1;
        decode_it->second.decode_token_num += num_prompt_tokens;

        const std::string prefill_name = request->routing.prefill_name;
        const std::string decode_name = request->routing.decode_name;
        const int32_t ttft_token_len =
            static_cast<int32_t>(request->token_ids.size());
        const int32_t tpot_total_len =
            static_cast<int32_t>(decode_it->second.decode_token_num);
        const int32_t tpot_batch =
            static_cast<int32_t>(decode_it->second.decode_request_num);

        auto prefill_tp_it = time_predictors_.find(prefill_name);
        auto decode_tp_it = time_predictors_.find(decode_name);
        if (prefill_tp_it == time_predictors_.end() ||
            decode_tp_it == time_predictors_.end()) {
          LOG(FATAL) << "Find TimePredictor failed, prefill: " << prefill_name
                     << " decode: " << decode_name;
        }
        const int64_t estimated_ttft = static_cast<int64_t>(
            prefill_tp_it->second.predict_ttft(ttft_token_len));
        estimated_tpot = static_cast<int64_t>(
            decode_tp_it->second.predict_tpot(tpot_total_len, tpot_batch));

        prefill_it->second.estimated_prefill_time += estimated_ttft;
        request->estimated_ttft = estimated_ttft;
        break;
      }
      case RequestAction::FINISH_PREFILL:
        prefill_it->second.prefill_request_num -= 1;
        prefill_it->second.prefill_token_num -= num_prompt_tokens;
        prefill_it->second.estimated_prefill_time -= request->estimated_ttft;

        decode_it->second.decode_token_num += 1;
        break;
      case RequestAction::GENERATE:
        decode_it->second.decode_token_num += 1;
        break;
      case RequestAction::FINISH_DECODE:
        decode_it->second.decode_request_num -= 1;
        decode_it->second.decode_token_num -=
            (num_prompt_tokens + num_generated_tokens);

        break;
      case RequestAction::CANCEL:
        prefill_it->second.prefill_request_num -= 1;
        prefill_it->second.prefill_token_num -= num_prompt_tokens;
        prefill_it->second.estimated_prefill_time -= request->estimated_ttft;

        decode_it->second.decode_request_num -= 1;
        decode_it->second.decode_token_num -=
            (num_prompt_tokens + num_generated_tokens);

        break;
      default:
        LOG(ERROR) << "Unknown RequestAction: " << static_cast<int32_t>(action);
        break;
    }

    if (decode_it->second.decode_request_num == 0) {
      flip_decode_name = request->routing.decode_name;
    }
    if (estimated_tpot > FLAGS_target_tpot) {
      flip_prefill_name = request->routing.prefill_name;
    }
  }

  if (!flip_decode_name.empty()) {
    topology_impl_->flip_decode_to_prefill(flip_decode_name);
  }

  if (!flip_prefill_name.empty()) {
    topology_impl_->flip_prefill_to_decode(flip_prefill_name);
  }
}

double InstanceMetricsImpl::predict_tpot(const std::string& instance_name,
                                         int32_t total_length,
                                         int32_t batch_size) {
  std::shared_lock<std::shared_mutex> lock(metrics_mutex_);
  auto it = time_predictors_.find(instance_name);
  if (it == time_predictors_.end()) {
    LOG(FATAL) << "Find TimePredictor failed, instance name : "
               << instance_name;
  }
  return it->second.predict_tpot(total_length, batch_size);
}

void InstanceMetricsImpl::set_as_master() {
  is_master_service_.store(true, std::memory_order_release);
  etcd_client_->remove_watch(kEtcdLoadMetricsPrefix);
}

void InstanceMetricsImpl::add_instance_metrics(const std::string& name,
                                               const InstanceMetaInfo& info) {
  std::unique_lock<std::shared_mutex> lock(metrics_mutex_);

  time_predictors_.insert_or_assign(
      name, TimePredictor(info.ttft_profiling_data, info.tpot_profiling_data));

  request_metrics_.insert_or_assign(name, RequestMetrics());
}

void InstanceMetricsImpl::remove_instance_metrics(const std::string& name) {
  std::unique_lock<std::shared_mutex> lock(metrics_mutex_);
  time_predictors_.erase(name);
  request_metrics_.erase(name);
  latency_metrics_.erase(name);
  updated_metrics_.erase(name);
  removed_instance_.insert(name);
  load_metrics_.erase(name);
}

void InstanceMetricsImpl::update_load_metrics(const etcd::Response& response,
                                              const uint64_t& prefix_len) {
  if (response.events().empty() || exited_.load(std::memory_order_acquire)) {
    return;
  }

  load_metrics_threadpool_.schedule([this,
                                     response = std::move(response),
                                     prefix_len = std::move(prefix_len)] {
    if (exited_.load(std::memory_order_acquire)) return;
    std::unordered_map<std::string, LoadMetrics> put_map;
    std::vector<std::string> delete_list;

    for (const auto& event : response.events()) {
      std::string instance_name = event.kv().key().substr(prefix_len);

      if (event.event_type() == etcd::Event::EventType::PUT) {
        LoadMetrics load_metrics;
        auto json_str = event.kv().as_string();
        if (!load_metrics.parse_from_json(json_str)) {
          LOG(ERROR) << "pase json:" << json_str << " error!";
          continue;
        }

        put_map.insert(std::make_pair(instance_name, std::move(load_metrics)));

      } else if (event.event_type() == etcd::Event::EventType::DELETE_) {
        delete_list.push_back(instance_name);
      }
    }

    {
      std::unique_lock<std::shared_mutex> lock(metrics_mutex_);
      for (auto& iter : put_map) {
        load_metrics_.insert_or_assign(iter.first, std::move(iter.second));
      }

      for (auto& iter : delete_list) {
        load_metrics_.erase(iter);
      }
    }
  });
}

}  // namespace xllm_service
