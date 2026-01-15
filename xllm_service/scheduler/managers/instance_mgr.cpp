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

#include <absl/strings/str_join.h>
#include <glog/logging.h>

#include <chrono>
#include <iostream>
#include <nlohmann/json.hpp>

#include "common/global_gflags.h"
#include "common/types.h"
#include "common/utils.h"

namespace {
using xllm_service::InstanceType;
std::unordered_map<InstanceType, std::string> ETCD_KEYS_PREFIX_MAP = {
    {InstanceType::DEFAULT, "XLLM:DEFAULT:"},
    {InstanceType::PREFILL, "XLLM:PREFILL:"},
    {InstanceType::DECODE, "XLLM:DECODE:"},
    {InstanceType::MIX, "XLLM:MIX:"},
};

std::string ETCD_ALL_KEYS_PREFIX = "XLLM:";
std::string ETCD_LOADMETRICS_PREFIX = "XLLM:LOADMETRICS:";
}  // namespace

namespace xllm_service {

InstanceMgr::InstanceMgr(const Options& options,
                         const std::shared_ptr<EtcdClient>& etcd_client,
                         const bool is_master_service)
    : options_(options),
      is_master_service_(is_master_service),
      etcd_client_(etcd_client) {
  auto handle_instance_metainfo =
      std::bind(&InstanceMgr::update_instance_metainfo,
                this,
                std::placeholders::_1,
                std::placeholders::_2);
  for (auto& it : ETCD_KEYS_PREFIX_MAP) {
    etcd_client_->add_watch(it.second, handle_instance_metainfo);
  }
  if (!is_master_service_) {
    auto handle_load_metrics = std::bind(&InstanceMgr::update_load_metrics,
                                         this,
                                         std::placeholders::_1,
                                         std::placeholders::_2);
    etcd_client_->add_watch(ETCD_LOADMETRICS_PREFIX, handle_load_metrics);
  }

  init();
}

void InstanceMgr::init() {
  {
    std::unique_lock<std::shared_mutex> lock(inst_mutex_);
    for (auto& it : ETCD_KEYS_PREFIX_MAP) {
      etcd_client_->get_prefix(it.second, &instances_);
    }
    // create ttft predictor and request metrics for each instance
    {
      std::lock_guard<std::mutex> time_predictor_lock(time_predictor_mutex_);
      std::lock_guard<std::mutex> request_metrics_lock(request_metrics_mutex_);
      for (auto& pair : instances_) {
        time_predictors_.insert_or_assign(
            pair.first,
            TimePredictor(pair.second.ttft_profiling_data,
                          pair.second.tpot_profiling_data));
        request_metrics_.insert_or_assign(pair.first, RequestMetrics());
      }
    }
    LOG(INFO) << "Load instance info from etcd:" << instances_.size();
    std::vector<std::string> channel_creat_fail_insts;
    prefill_index_.reserve(instances_.size());
    decode_index_.reserve(instances_.size());

    for (auto& ist : instances_) {
      if (!create_channel(ist.first)) {
        channel_creat_fail_insts.emplace_back(ist.first);
      } else {
        switch (ist.second.type) {
          case InstanceType::DEFAULT:
          case InstanceType::PREFILL:
            ist.second.instance_index = prefill_index_.size();
            prefill_index_.emplace_back(ist.first);
            LOG(INFO) << "Register a new prefill instance, instance name : "
                      << ist.first;
            break;
          case InstanceType::DECODE:
            ist.second.instance_index = decode_index_.size();
            decode_index_.emplace_back(ist.first);
            LOG(INFO) << "Register a new decode instance, instance name : "
                      << ist.first;
            break;
          case InstanceType::MIX:
            // In the initial state, we set the first MIX type instance as a
            // decode instance, while all subsequent instances are set as
            // prefill instances.
            if (decode_index_.size() > 0) {
              ist.second.instance_index = prefill_index_.size();
              ist.second.current_type = InstanceType::PREFILL;
              prefill_index_.emplace_back(ist.first);
              LOG(INFO) << "Register a new prefill instance, instance name : "
                        << ist.first;
            } else {
              ist.second.instance_index = decode_index_.size();
              ist.second.current_type = InstanceType::DECODE;
              decode_index_.emplace_back(ist.first);
              LOG(INFO) << "Register a new decode instance, instance name : "
                        << ist.first;
            }
            break;
          default:
            LOG(WARNING) << "Unknown InstanceType: " << int(ist.second.type);
            channel_creat_fail_insts.emplace_back(ist.first);
            break;
        }
      }
    }
    for (auto& name : channel_creat_fail_insts) {
      instances_.erase(name);
      {
        std::lock_guard<std::mutex> time_predictor_lock(time_predictor_mutex_);
        std::lock_guard<std::mutex> request_metrics_lock(
            request_metrics_mutex_);
        time_predictors_.erase(name);
        request_metrics_.erase(name);
      }
    }
  }
  {
    std::unique_lock<std::shared_mutex> lock(load_metric_mutex_);
    etcd_client_->get_prefix(ETCD_LOADMETRICS_PREFIX, &load_metrics_);
  }

  for (int i = 0; i < prefill_index_.size(); i++) {
    LOG(INFO) << i << " : " << prefill_index_[i];
  }
}

InstanceMgr::~InstanceMgr() { exited_ = true; }

InstanceMetaInfo InstanceMgr::get_instance_info(
    const std::string& instance_name) {
  std::shared_lock<std::shared_mutex> lock(inst_mutex_);
  if (instances_.find(instance_name) == instances_.end()) {
    LOG(ERROR) << "Get instance info failed, instance is not registered, "
                  "instance_name: "
               << instance_name;
    return InstanceMetaInfo();
  }
  return instances_[instance_name];
}

bool InstanceMgr::get_next_instance_pair(Routing* routing) {
  std::unique_lock<std::shared_mutex> lock(inst_mutex_);
  if (prefill_index_.empty()) {
    LOG(ERROR) << "No prefill or default instance found!";
    return false;
  }
  next_prefill_index_ = next_prefill_index_ % prefill_index_.size();
  routing->prefill_name = prefill_index_[next_prefill_index_];
  next_prefill_index_++;
  if (decode_index_.empty()) {
    return true;
  }
  next_decode_index_ = next_decode_index_ % decode_index_.size();
  routing->decode_name = decode_index_[next_decode_index_];
  next_decode_index_++;
  return true;
}

// TODO: refactor later, currently return all decode instances
std::vector<std::string> InstanceMgr::get_static_decode_list(
    const std::string& instance_name) {
  std::vector<std::string> decode_list;
  std::shared_lock<std::shared_mutex> lock(inst_mutex_);
  for (auto& inst : instances_) {
    if (inst.second.type == InstanceType::DECODE) {
      decode_list.emplace_back(inst.second.name);
    }
  }

  return decode_list;
}

// TODO: refactor later, currently return all prefill instances
std::vector<std::string> InstanceMgr::get_static_prefill_list(
    const std::string& instance_name) {
  std::vector<std::string> prefill_list;
  std::shared_lock<std::shared_mutex> lock(inst_mutex_);
  for (auto& inst : instances_) {
    if (inst.second.type == InstanceType::PREFILL ||
        inst.second.type == InstanceType::DEFAULT) {
      prefill_list.emplace_back(inst.second.name);
    }
  }

  return prefill_list;
}

void InstanceMgr::get_load_metrics(LoadBalanceInfos* infos) {
  std::shared_lock<std::shared_mutex> inst_lock(inst_mutex_);
  std::shared_lock<std::shared_mutex> metric_lock(load_metric_mutex_);

  for (auto name : infos->overlap_scores.instances) {
    auto it = load_metrics_.find(name);
    if (it == load_metrics_.end()) {
      continue;
    }
    auto instance_it = instances_.find(name);
    if (instance_it == instances_.end()) {
      continue;
    }

    if (instance_it->second.type == InstanceType::DECODE) {
      infos->decode_load_metrics.insert(std::make_pair(name, it->second));
      infos->decode_max_waiting_requests_num =
          std::max(infos->decode_max_waiting_requests_num,
                   it->second.waiting_requests_num);
    } else {
      infos->prefill_load_metrics.insert(std::make_pair(name, it->second));
      infos->prefill_max_waiting_requests_num =
          std::max(infos->prefill_max_waiting_requests_num,
                   it->second.waiting_requests_num);
    }
  }

  std::string least_loaded_prefill_instance;
  float least_loaded_prefill_gpu_cache_usage_perc = 1;
  std::string least_loaded_decode_instance;
  float least_loaded_decode_gpu_cache_usage_perc = 1;

  if (infos->prefill_load_metrics.size() == 0 ||
      infos->decode_load_metrics.size() == 0) {
    for (const auto& metric : load_metrics_) {
      auto instance_it = instances_.find(metric.first);
      if (instance_it != instances_.end()) {
        if (instance_it->second.type != InstanceType::DECODE) {
          if (metric.second.gpu_cache_usage_perc <
              least_loaded_prefill_gpu_cache_usage_perc) {
            least_loaded_prefill_gpu_cache_usage_perc =
                metric.second.gpu_cache_usage_perc;
            least_loaded_prefill_instance = metric.first;
          }
        } else {
          if (metric.second.gpu_cache_usage_perc <
              least_loaded_decode_gpu_cache_usage_perc) {
            least_loaded_decode_gpu_cache_usage_perc =
                metric.second.gpu_cache_usage_perc;
            least_loaded_decode_instance = metric.first;
          }
        }
      }
    }
  }

  if (infos->prefill_load_metrics.size() == 0 &&
      !least_loaded_prefill_instance.empty()) {
    infos->prefill_load_metrics.insert(
        std::make_pair(least_loaded_prefill_instance,
                       load_metrics_[least_loaded_prefill_instance]));
  }

  if (infos->decode_load_metrics.size() == 0 &&
      !least_loaded_decode_instance.empty()) {
    infos->decode_load_metrics.insert(
        std::make_pair(least_loaded_decode_instance,
                       load_metrics_[least_loaded_decode_instance]));
  }
}

void InstanceMgr::record_load_metrics_update(
    const std::string& instance_name,
    const proto::LoadMetrics& load_metrics) {
  std::lock_guard<std::mutex> lock(update_mutex_);

  updated_metrics_.insert_or_assign(
      instance_name,
      LoadMetrics(load_metrics.waiting_requests_num(),
                  load_metrics.gpu_cache_usage_perc()));
}

bool InstanceMgr::upload_load_metrics() {
  std::lock_guard<std::mutex> lock(update_mutex_);
  bool status = etcd_client_->set(ETCD_LOADMETRICS_PREFIX, updated_metrics_);
  status =
      status && etcd_client_->rm(ETCD_LOADMETRICS_PREFIX, removed_instance_);
  {
    std::unique_lock<std::shared_mutex> lock(inst_mutex_);
    for (auto& iter : updated_metrics_) {
      load_metrics_.insert_or_assign(iter.first, std::move(iter.second));
    }
    for (auto& iter : removed_instance_) {
      load_metrics_.erase(iter);
    }
  }
  updated_metrics_.clear();
  removed_instance_.clear();

  return status;
}

void InstanceMgr::set_as_master() {
  is_master_service_ = true;
  etcd_client_->remove_watch(ETCD_LOADMETRICS_PREFIX);
}

std::shared_ptr<brpc::Channel> InstanceMgr::get_channel(
    const std::string& instance_name) {
  std::shared_lock<std::shared_mutex> lock(inst_mutex_);
  auto iter = cached_channels_.find(instance_name);
  if (iter == cached_channels_.end()) {
    return nullptr;
  }
  return iter->second;
}

bool InstanceMgr::create_channel(const std::string& instance_name) {
  if (cached_channels_.find(instance_name) == cached_channels_.end()) {
    auto channel = std::make_shared<brpc::Channel>();
    brpc::ChannelOptions options;
    // Add to params
    options.protocol = "http";
    options.timeout_ms = options_.timeout_ms(); /*milliseconds*/
    options.max_retry = 3;
    options.connect_timeout_ms = options_.connect_timeout_ms();
    std::string load_balancer = "";
    if (channel->Init(instance_name.c_str(), load_balancer.c_str(), &options) !=
        0) {
      LOG(ERROR) << "Fail to initialize channel for " << instance_name;
      return false;
    }
    cached_channels_[instance_name] = std::move(channel);
  }

  return true;
}

void InstanceMgr::update_instance_metainfo(const etcd::Response& response,
                                           const uint64_t& prefix_len) {
  if (response.events().empty() || exited_) {
    return;
  }

  threadpool_.schedule([this,
                        response = std::move(response),
                        prefix_len = std::move(prefix_len)] {
    if (exited_) return;
    std::unordered_map<std::string, InstanceMetaInfo> put_map;
    std::vector<std::string> delete_list;

    for (const auto& event : response.events()) {
      std::string instance_name = event.kv().key().substr(prefix_len);

      if (event.event_type() == etcd::Event::EventType::PUT) {
        InstanceMetaInfo metainfo;
        auto json_str = event.kv().as_string();
        if (!metainfo.parse_from_json(json_str)) {
          LOG(ERROR) << "pase json:" << json_str << " error!";
          continue;
        }

        put_map.insert(std::make_pair(instance_name, std::move(metainfo)));

      } else if (event.event_type() == etcd::Event::EventType::DELETE_) {
        delete_list.push_back(instance_name);
      }
    }

    {
      std::unique_lock<std::shared_mutex> lock(inst_mutex_);
      for (auto& iter : put_map) {
        if (instances_.find(iter.first) != instances_.end()) {
          LOG(ERROR) << "Instance is already registered, instance_name: "
                     << iter.first;
          continue;
        }

        if (!create_channel(iter.first)) {
          LOG(ERROR) << "create channel fail: " << iter.first;
          continue;
        }

        {
          std::lock_guard<std::mutex> time_predictor_lock(
              time_predictor_mutex_);
          std::lock_guard<std::mutex> request_metrics_lock(
              request_metrics_mutex_);
          // create ttft predictor for instance
          time_predictors_.emplace(
              iter.first,
              TimePredictor(iter.second.ttft_profiling_data,
                            iter.second.tpot_profiling_data));

          // create request metrics for instance
          request_metrics_.emplace(iter.first, RequestMetrics());
        }

        switch (iter.second.type) {
          case InstanceType::DEFAULT:
          case InstanceType::PREFILL:
            iter.second.instance_index = prefill_index_.size();
            prefill_index_.emplace_back(iter.first);
            LOG(INFO) << "Register a new prefill instance, instance name : "
                      << iter.first;
            break;
          case InstanceType::DECODE:
            iter.second.instance_index = decode_index_.size();
            decode_index_.emplace_back(iter.first);
            LOG(INFO) << "Register a new decode instance, instance name : "
                      << iter.first;
            break;
          case InstanceType::MIX:
            // In the initial state, we set the first MIX type instance as a
            // decode instance, while all subsequent instances are set as
            // prefill instances.
            if (decode_index_.size() > 0) {
              iter.second.instance_index = prefill_index_.size();
              iter.second.current_type = InstanceType::PREFILL;
              prefill_index_.emplace_back(iter.first);
              LOG(INFO) << "Register a new prefill instance, instance name : "
                        << iter.first;
            } else {
              iter.second.instance_index = decode_index_.size();
              iter.second.current_type = InstanceType::DECODE;
              decode_index_.emplace_back(iter.first);
              LOG(INFO) << "Register a new decode instance, instance name : "
                        << iter.first;
            }
            break;
          default:
            LOG(WARNING) << "Unknown InstanceType: " << int(iter.second.type);
            break;
        }

        instances_.insert(std::make_pair(iter.first, std::move(iter.second)));
      }

      for (auto& iter : delete_list) {
        LOG(INFO) << "delete instance: " << iter;
        if (instances_.find(iter) == instances_.end()) {
          LOG(ERROR) << "Instance is already deleted, instance_name: " << iter;
          continue;
        }
        // TODO: notify cache manager to clear expire cache
        uint64_t index = instances_[iter].instance_index;

        switch (instances_[iter].type) {
          case InstanceType::DEFAULT:
          case InstanceType::PREFILL:
            if (index == -1 || index >= prefill_index_.size()) {
              break;
            }
            std::swap(prefill_index_[index], prefill_index_.back());
            instances_[prefill_index_[index]].instance_index = index;
            prefill_index_.pop_back();
            break;
          case InstanceType::DECODE:
            if (index == -1 || index >= decode_index_.size()) {
              break;
            }
            std::swap(decode_index_[index], decode_index_.back());
            instances_[decode_index_[index]].instance_index = index;
            decode_index_.pop_back();
            break;
          case InstanceType::MIX:
            if (index == -1) {
              break;
            }
            if (instances_[iter].current_type == InstanceType::PREFILL) {
              if (index >= prefill_index_.size()) {
                break;
              }
              std::swap(prefill_index_[index], prefill_index_.back());
              instances_[prefill_index_[index]].instance_index = index;
              prefill_index_.pop_back();
            } else {
              if (index >= decode_index_.size()) {
                break;
              }
              std::swap(decode_index_[index], decode_index_.back());
              instances_[decode_index_[index]].instance_index = index;
              decode_index_.pop_back();
            }
            break;
          default:
            LOG(WARNING) << "Unknown InstanceType: "
                         << int(instances_[iter].type);
            break;
        }

        instances_.erase(iter);
        cached_channels_.erase(iter);
        {
          std::lock_guard<std::mutex> time_predictor_lock(
              time_predictor_mutex_);
          std::lock_guard<std::mutex> request_metrics_lock(
              request_metrics_mutex_);
          time_predictors_.erase(iter);
          request_metrics_.erase(iter);
        }
        {
          std::lock_guard<std::mutex> lock(update_mutex_);
          updated_metrics_.erase(iter);
          removed_instance_.insert(iter);
        }
      }
    }
  });
}

void InstanceMgr::update_load_metrics(const etcd::Response& response,
                                      const uint64_t& prefix_len) {
  if (response.events().empty() || exited_) {
    return;
  }
  threadpool_.schedule([this,
                        response = std::move(response),
                        prefix_len = std::move(prefix_len)] {
    if (exited_) return;
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
      std::unique_lock<std::shared_mutex> lock(load_metric_mutex_);
      for (auto& iter : put_map) {
        load_metrics_.insert_or_assign(iter.first, std::move(iter.second));
      }

      for (auto& iter : delete_list) {
        load_metrics_.erase(iter);
      }
    }
  });
}

void InstanceMgr::update_latency_metrics(
    const std::string& instance_name,
    const proto::LatencyMetrics& latency_metrics) {
  std::lock_guard<std::mutex> lock(latency_metrics_mutex_);

  latency_metrics_.insert_or_assign(
      instance_name,
      LatencyMetrics(latency_metrics.recent_max_ttft(),
                     latency_metrics.recent_max_tbt()));
}

void InstanceMgr::update_request_metrics(std::shared_ptr<Request> request,
                                         RequestAction action) {
  std::lock_guard<std::mutex> lock(request_metrics_mutex_);

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

  int64_t num_prompt_tokens = request->token_ids.size();
  int64_t num_generated_tokens = request->num_generated_tokens;
  switch (action) {
    case RequestAction::SCHEDULE:
      // update the request metrics for prefill and decode instances when
      // request is scheduled
      prefill_it->second.prefill_request_num += 1;
      prefill_it->second.prefill_token_num += num_prompt_tokens;

      decode_it->second.decode_request_num += 1;
      decode_it->second.decode_token_num += num_prompt_tokens;
      break;
    case RequestAction::FINISH_PREFILL:
      // update the request metrics for prefill and decode instance when request
      // finishes the prefill phase
      prefill_it->second.prefill_request_num -= 1;
      prefill_it->second.prefill_token_num -= num_prompt_tokens;
      prefill_it->second.estimated_prefill_time -= request->estimated_ttft;

      decode_it->second.decode_token_num += 1;
      break;
    case RequestAction::GENERATE:
      // update the request metrics for decode instance when request generate a
      // token
      decode_it->second.decode_token_num += 1;
      break;
    case RequestAction::FINISH_DECODE:
      // update the request metrics for decode instance when request finishes
      // the decode phase
      decode_it->second.decode_request_num -= 1;
      decode_it->second.decode_token_num -=
          (num_prompt_tokens + num_generated_tokens);
      break;
    case RequestAction::CANCEL:
      // update the request metrics for prefill and decode instances when
      // request is cancelled
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

  if (options_.load_balance_policy() == "SLO_AWARE" &&
      decode_it->second.decode_request_num == 0) {
    std::unique_lock<std::shared_mutex> instance_lock(inst_mutex_);
    flip_decode_to_prefill(request->routing.decode_name);
  }
}

bool InstanceMgr::select_instance_pair_on_slo(
    std::shared_ptr<Request> request) {
  std::unique_lock<std::shared_mutex> lock(inst_mutex_);
  std::lock_guard<std::mutex> request_metrics_lock(request_metrics_mutex_);

  if (prefill_index_.empty()) {
    LOG(ERROR) << "No prefill or default instance found!";
    return false;
  }

  // get min prefill time instance from request metrics
  auto min_prefill_instance = prefill_index_[0];
  int64_t min_prefill_time = std::numeric_limits<int64_t>::max();
  int64_t total_prefill_time = 0;
  for (auto& prefill_instance : prefill_index_) {
    int64_t prefill_time =
        request_metrics_[prefill_instance].estimated_prefill_time;
    total_prefill_time += prefill_time;
    if (prefill_time < min_prefill_time) {
      min_prefill_instance = prefill_instance;
      min_prefill_time = prefill_time;
    }
  }
  int64_t avg_prefill_time = total_prefill_time / prefill_index_.size();

  if (decode_index_.empty()) {
    LOG(ERROR) << "No decode instance found!";
    return false;
  }

  // select decode instance
  auto min_decode_instance = decode_index_[0];
  int64_t min_estimated_tpot = std::numeric_limits<int64_t>::max();
  std::string target_decode_instance;
  for (auto& decode_instance : decode_index_) {
    int64_t token_num = request_metrics_[decode_instance].decode_token_num;
    int64_t request_num = request_metrics_[decode_instance].decode_request_num;
    auto& time_predictor = get_time_predictor(decode_instance);
    // calculate the estimated tpot
    int64_t estimated_tpot = time_predictor.predict_tpot(
        token_num + request->token_ids.size(), request_num + 1);
    // If the estimated tpot meets the requirements, the request will be
    // directly dispatched to that instance.
    if (estimated_tpot <= FLAGS_target_tpot && target_decode_instance.empty()) {
      target_decode_instance = decode_instance;
    }

    // Record the instance with the minimum estimated tpot.
    if (estimated_tpot < min_estimated_tpot) {
      min_decode_instance = decode_instance;
      min_estimated_tpot = estimated_tpot;
    }
  }

  if (!target_decode_instance.empty()) {
    request->routing.decode_name = target_decode_instance;
  } else {
    request->routing.decode_name = min_decode_instance;
  }

  // select prefill instance
  float tpot_threshold = (decode_index_.size() - 1.0f) / decode_index_.size();
  // When the prefill instances are already overloaded and there are other
  // instances with lower loads in the decode group, we will dispatch the
  // prefill requests to those instances to alleviate the pressure on the
  // prefill instances.
  if (min_prefill_time > FLAGS_target_ttft &&
      target_decode_instance != min_decode_instance &&
      min_estimated_tpot < FLAGS_target_tpot * tpot_threshold &&
      request_metrics_[min_decode_instance].estimated_prefill_time <
          min_prefill_time) {
    request->routing.prefill_name = min_decode_instance;
    // update estimated ttft
    auto& time_predictor = get_time_predictor(min_decode_instance);
    request->estimated_ttft =
        time_predictor.predict_ttft(request->token_ids.size());
    request_metrics_[min_decode_instance].estimated_prefill_time +=
        request->estimated_ttft;
  } else {
    request->routing.prefill_name = min_prefill_instance;
    // update estimated ttft
    auto& time_predictor = get_time_predictor(min_prefill_instance);
    request->estimated_ttft =
        time_predictor.predict_ttft(request->token_ids.size());
    request_metrics_[min_prefill_instance].estimated_prefill_time +=
        request->estimated_ttft;
  }

  // If there are no decode instances that meet the requirements, switch a
  // prefill instance to decode if the number of instances allows. Since the
  // current disaggregated PD mode does not support prefill and decode using the
  // same instance, we only switch the instance here, without dispatching the
  // decode request to this instance.
  float ttft_threshold = (prefill_index_.size() - 1.0f) / prefill_index_.size();
  if (target_decode_instance.empty() &&
      (avg_prefill_time < FLAGS_target_ttft * ttft_threshold ||
       decode_index_.size() < prefill_index_.size())) {
    flip_prefill_to_decode(request->routing.prefill_name);
  }

  return true;
}

void InstanceMgr::flip_prefill_to_decode(std::string& instance_name) {
  if (prefill_index_.size() <= 1) {
    // Ensure there is at least one prefill instance.
    return;
  }

  if (instances_.find(instance_name) == instances_.end()) {
    LOG(ERROR) << "Can't find instance, instance_name: " << instance_name;
    return;
  }

  // delete instance name from prefill_index_
  uint64_t index = instances_[instance_name].instance_index;
  std::swap(prefill_index_[index], prefill_index_.back());
  instances_[prefill_index_[index]].instance_index = index;
  prefill_index_.pop_back();

  // insert instance name to decode_index_
  instances_[instance_name].instance_index = decode_index_.size();
  instances_[instance_name].current_type = InstanceType::DECODE;
  decode_index_.emplace_back(instance_name);

  LOG(INFO) << "Flip prefill to decode, instance name : " << instance_name;
}

void InstanceMgr::flip_decode_to_prefill(std::string& instance_name) {
  if (decode_index_.size() <= 1) {
    // Ensure there is at least one decode instance.
    return;
  }

  if (instances_.find(instance_name) == instances_.end()) {
    LOG(ERROR) << "Can't find instance, instance_name: " << instance_name;
    return;
  }

  // delete instance name from decode_index_
  uint64_t index = instances_[instance_name].instance_index;
  std::swap(decode_index_[index], decode_index_.back());
  instances_[decode_index_[index]].instance_index = index;
  decode_index_.pop_back();

  // insert instance name to prefill_index
  instances_[instance_name].instance_index = prefill_index_.size();
  instances_[instance_name].current_type = InstanceType::PREFILL;
  prefill_index_.emplace_back(instance_name);

  LOG(INFO) << "Flip decode to prefill, instance name : " << instance_name;
}

TimePredictor& InstanceMgr::get_time_predictor(
    const std::string& instance_name) {
  std::lock_guard<std::mutex> lock(time_predictor_mutex_);

  auto it = time_predictors_.find(instance_name);
  if (it == time_predictors_.end()) {
    LOG(FATAL) << "Find TimePredictor failed, instance name : "
               << instance_name;
  }
  return it->second;
}

}  // namespace xllm_service
