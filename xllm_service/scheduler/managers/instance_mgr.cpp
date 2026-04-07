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

#include "instance_mgr.h"

#include <glog/logging.h>

#include "instance_kvcache.h"

#include <limits>
#include <mutex>
#include <string>
#include <vector>

#include "common/global_gflags.h"
#include "common/types.h"

namespace {
bool is_instance_schedulable(const xllm_service::InstanceMetaInfo& info) {
  return info.runtime_state != xllm_service::InstanceRuntimeState::SUSPECT;
}

}  // namespace

namespace xllm_service {

InstanceMgr::InstanceMgr(const Options& options,
                         const std::shared_ptr<EtcdClient>& etcd_client,
                         const bool is_master_service,
                         OnInstanceDeregisteredCallback on_instance_deregistered)
    : etcd_client_(etcd_client),
      metrics_impl_(std::make_unique<InstanceMetricsImpl>(
          options,
          etcd_client,
          is_master_service)),
      topology_impl_(std::make_unique<InstanceTopologyImpl>(
          options,
          etcd_client,
          std::move(on_instance_deregistered),
          [this](const std::string& n, const InstanceMetaInfo& i) {
            metrics_impl_->add_instance_metrics(n, i);
          },
          [this](const std::string& n) {
            metrics_impl_->remove_instance_metrics(n);
          })),
      kvcache_(std::make_unique<InstanceKVCache>(
          options,
          etcd_client,
          is_master_service)) {
  metrics_impl_->set_topology(topology_impl_.get());
  init();
}

void InstanceMgr::init() {
  topology_impl_->init_from_etcd_register_all();
  metrics_impl_->load_initial_load_metrics_from_etcd();
}

InstanceMgr::~InstanceMgr() {
  metrics_impl_->shutdown();
}

InstanceMetaInfo InstanceMgr::get_instance_info(
    const std::string& instance_name) {
  return topology_impl_->get_instance_info(instance_name);
}

std::vector<std::string> InstanceMgr::get_schedulable_prefill_instances() {
  std::shared_lock<std::shared_mutex> lock(topology_impl_->cluster_mutex_);
  const auto& idx = topology_impl_->prefill_index_;
  const auto& inst = topology_impl_->instances_;
  const bool suspect_empty = topology_impl_->suspect_instances_.empty();
  std::vector<std::string> out;
  out.reserve(idx.size());
  for (const auto& name : idx) {
    if (suspect_empty) {
      out.push_back(name);
      continue;
    }
    auto it = inst.find(name);
    if (it != inst.end() && is_instance_schedulable(it->second)) {
      out.push_back(name);
    }
  }
  return out;
}

std::vector<std::string> InstanceMgr::get_schedulable_decode_instances() {
  std::shared_lock<std::shared_mutex> lock(topology_impl_->cluster_mutex_);
  const auto& idx = topology_impl_->decode_index_;
  const auto& inst = topology_impl_->instances_;
  const bool suspect_empty = topology_impl_->suspect_instances_.empty();
  std::vector<std::string> out;
  out.reserve(idx.size());
  for (const auto& name : idx) {
    if (suspect_empty) {
      out.push_back(name);
      continue;
    }
    auto it = inst.find(name);
    if (it != inst.end() && is_instance_schedulable(it->second)) {
      out.push_back(name);
    }
  }
  return out;
}

std::vector<std::string> InstanceMgr::get_static_decode_list(
    const std::string& instance_name) {
  return topology_impl_->get_static_decode_list(instance_name);
}

std::vector<std::string> InstanceMgr::get_static_prefill_list(
    const std::string& instance_name) {
  return topology_impl_->get_static_prefill_list(instance_name);
}

void InstanceMgr::get_load_metrics(LoadBalanceInfos* infos) {
  TopologySnapshot topo = topology_impl_->snapshot();
  metrics_impl_->get_load_metrics(infos, topo);
}

void InstanceMgr::kvcache_match(const Slice<int32_t>& token_ids,
                                OverlapScores* overlap_scores) {
  kvcache_->match(token_ids, overlap_scores);
}

bool InstanceMgr::upload_master_state_to_etcd() {
  const bool kvcache_ok = kvcache_->upload_kvcache();
  const bool load_ok = metrics_impl_->upload_load_metrics();
  return kvcache_ok && load_ok;
}

void InstanceMgr::set_as_master() {
  metrics_impl_->set_as_master();
  kvcache_->set_as_master();
}

std::shared_ptr<brpc::Channel> InstanceMgr::get_channel(
    const std::string& instance_name) {
  return topology_impl_->get_channel(instance_name);
}

bool InstanceMgr::bind_request_instance_incarnations(
    const std::shared_ptr<Request>& request) {
  return topology_impl_->bind_request_instance_incarnations(request);
}

bool InstanceMgr::on_instance_heartbeat(const proto::HeartbeatRequest& req) {
  if (!topology_impl_->record_instance_heartbeat(req.name(),
                                                req.incarnation_id())) {
    return false;
  }
  kvcache_->record_updated_kvcaches(req.name(), req.cache_event());
  metrics_impl_->record_load_metrics_update(req.name(), req.load_metrics());
  metrics_impl_->update_latency_metrics(req.name(), req.latency_metrics());
  return true;
}

void InstanceMgr::update_request_metrics(std::shared_ptr<Request> request,
                                         RequestAction action) {
  metrics_impl_->update_request_metrics(request, action);
}

bool InstanceMgr::select_instance_pair_on_slo(
    std::shared_ptr<Request> request) {
  std::string flip_prefill_target;
  {
    std::scoped_lock<std::shared_mutex, std::shared_mutex> lock(
        topology_impl_->cluster_mutex_, metrics_impl_->metrics_mutex_);

    auto& instances_ = topology_impl_->instances_;
    auto& prefill_index_ = topology_impl_->prefill_index_;
    auto& decode_index_ = topology_impl_->decode_index_;
    auto& suspect_instances_ = topology_impl_->suspect_instances_;
    auto& request_metrics_ = metrics_impl_->request_metrics_;

    const bool has_unschedulable_instances = !suspect_instances_.empty();

    std::string min_prefill_instance;
    int64_t min_prefill_time = std::numeric_limits<int64_t>::max();
    int64_t total_prefill_time = 0;
    size_t schedulable_prefill_count = 0;
    for (const auto& prefill_instance : prefill_index_) {
      if (has_unschedulable_instances) {
        auto it = instances_.find(prefill_instance);
        if (it == instances_.end() || !is_instance_schedulable(it->second)) {
          continue;
        }
      }

      int64_t prefill_time =
          request_metrics_[prefill_instance].estimated_prefill_time;
      total_prefill_time += prefill_time;
      if (prefill_time < min_prefill_time) {
        min_prefill_instance = prefill_instance;
        min_prefill_time = prefill_time;
      }
      ++schedulable_prefill_count;
    }

    if (schedulable_prefill_count == 0) {
      LOG(ERROR) << "No prefill or default instance found!";
      return false;
    }
    int64_t avg_prefill_time = total_prefill_time / schedulable_prefill_count;

    std::string min_decode_instance;
    int64_t min_estimated_tpot = std::numeric_limits<int64_t>::max();
    std::string target_decode_instance;
    size_t schedulable_decode_count = 0;
    for (const auto& decode_instance : decode_index_) {
      if (has_unschedulable_instances) {
        auto it = instances_.find(decode_instance);
        if (it == instances_.end() || !is_instance_schedulable(it->second)) {
          continue;
        }
      }

      int64_t token_num = request_metrics_[decode_instance].decode_token_num;
      int64_t request_num =
          request_metrics_[decode_instance].decode_request_num;
      auto& time_predictor =
          metrics_impl_->time_predictor_unlocked(decode_instance);
      int64_t estimated_tpot = static_cast<int64_t>(time_predictor.predict_tpot(
          static_cast<int32_t>(token_num + request->token_ids.size()),
          static_cast<int32_t>(request_num + 1)));
      if (estimated_tpot <= FLAGS_target_tpot &&
          target_decode_instance.empty()) {
        target_decode_instance = decode_instance;
      }

      if (estimated_tpot < min_estimated_tpot) {
        min_decode_instance = decode_instance;
        min_estimated_tpot = estimated_tpot;
      }
      ++schedulable_decode_count;
    }

    if (schedulable_decode_count == 0) {
      LOG(ERROR) << "No decode instance found!";
      return false;
    }

    if (!target_decode_instance.empty()) {
      request->routing.decode_name = target_decode_instance;
    } else {
      request->routing.decode_name = min_decode_instance;
    }

    float tpot_threshold =
        (schedulable_decode_count - 1.0f) / schedulable_decode_count;
    if (min_prefill_time > FLAGS_target_ttft &&
        target_decode_instance != min_decode_instance &&
        min_estimated_tpot < FLAGS_target_tpot * tpot_threshold &&
        request_metrics_[min_decode_instance].estimated_prefill_time <
            min_prefill_time) {
      request->routing.prefill_name = min_decode_instance;
      auto& time_predictor =
          metrics_impl_->time_predictor_unlocked(min_decode_instance);
      request->estimated_ttft =
          static_cast<int64_t>(time_predictor.predict_ttft(
              static_cast<int32_t>(request->token_ids.size())));
      request_metrics_[min_decode_instance].estimated_prefill_time +=
          request->estimated_ttft;
    } else {
      request->routing.prefill_name = min_prefill_instance;
      auto& time_predictor =
          metrics_impl_->time_predictor_unlocked(min_prefill_instance);
      request->estimated_ttft =
          static_cast<int64_t>(time_predictor.predict_ttft(
              static_cast<int32_t>(request->token_ids.size())));
      request_metrics_[min_prefill_instance].estimated_prefill_time +=
          request->estimated_ttft;
    }

    float ttft_threshold =
        (schedulable_prefill_count - 1.0f) / schedulable_prefill_count;
    if (target_decode_instance.empty() &&
        (avg_prefill_time < FLAGS_target_ttft * ttft_threshold ||
         schedulable_decode_count < schedulable_prefill_count)) {
      flip_prefill_target = request->routing.prefill_name;
    }
  }

  if (!flip_prefill_target.empty()) {
    topology_impl_->flip_prefill_to_decode(flip_prefill_target);
  }

  return true;
}

}  // namespace xllm_service
