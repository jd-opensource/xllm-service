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
#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <brpc/controller.h>
#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/global_gflags.h"
#include "common/types.h"
#include "common/utils.h"
#include "common/xllm/output.h"
#include "common/xllm/status.h"
#include "disagg_pd.pb.h"
#include "scheduler/scheduler.h"

namespace {
using xllm_service::InstanceRuntimeState;
using xllm_service::InstanceType;
std::unordered_map<InstanceType, std::string> ETCD_KEYS_PREFIX_MAP = {
    {InstanceType::DEFAULT, "XLLM:DEFAULT:"},
    {InstanceType::PREFILL, "XLLM:PREFILL:"},
    {InstanceType::DECODE, "XLLM:DECODE:"},
    {InstanceType::MIX, "XLLM:MIX:"},
};

std::string ETCD_ALL_KEYS_PREFIX = "XLLM:";
std::string ETCD_LOADMETRICS_PREFIX = "XLLM:LOADMETRICS:";

constexpr char kHealthPath[] = "/health";
constexpr int64_t kDeleteProbeRetryBackoffMs = 100;

uint64_t current_time_ms() {
  return static_cast<uint64_t>(
      absl::ToInt64Milliseconds(absl::Now() - absl::UnixEpoch()));
}

bool is_instance_schedulable(const xllm_service::InstanceMetaInfo& info) {
  // LEASE_LOST instances can still be reused while heartbeats continue.
  return info.runtime_state != InstanceRuntimeState::SUSPECT;
}

bool select_next_schedulable_instance(
    const std::unordered_map<std::string, xllm_service::InstanceMetaInfo>&
        instances,
    const std::vector<std::string>& index,
    uint64_t* next_index,
    std::string* instance_name) {
  if (index.empty()) {
    return false;
  }

  const uint64_t start_index = *next_index % index.size();
  for (uint64_t offset = 0; offset < index.size(); ++offset) {
    const uint64_t index_pos = (start_index + offset) % index.size();
    auto it = instances.find(index[index_pos]);
    if (it == instances.end() || !is_instance_schedulable(it->second)) {
      continue;
    }
    *instance_name = index[index_pos];
    *next_index = index_pos + 1;
    return true;
  }
  return false;
}

size_t count_schedulable_instances(
    const std::unordered_map<std::string, xllm_service::InstanceMetaInfo>&
        instances,
    const std::vector<std::string>& index) {
  size_t count = 0;
  for (const auto& name : index) {
    auto it = instances.find(name);
    if (it == instances.end() || !is_instance_schedulable(it->second)) {
      continue;
    }
    ++count;
  }
  return count;
}

InstanceType get_cleanup_type(const xllm_service::InstanceMetaInfo& info) {
  if (info.type == InstanceType::DEFAULT) {
    return InstanceType::PREFILL;
  }
  if (info.type == InstanceType::MIX) {
    return info.current_type;
  }
  return info.type;
}
}  // namespace

namespace xllm_service {

InstanceMgr::InstanceMgr(const Options& options,
                         const std::shared_ptr<EtcdClient>& etcd_client,
                         const bool is_master_service,
                         Scheduler* scheduler)
    : options_(options),
      is_master_service_(is_master_service),
      etcd_client_(etcd_client),
      scheduler_(scheduler) {
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

  state_reconcile_thread_ = std::make_unique<std::thread>(
      &InstanceMgr::reconcile_instance_states, this);
}

void InstanceMgr::init() {
  std::unordered_map<std::string, InstanceMetaInfo> loaded_instances;
  {
    std::unique_lock<std::shared_mutex> lock(inst_mutex_);
    for (auto& it : ETCD_KEYS_PREFIX_MAP) {
      etcd_client_->get_prefix(it.second, &loaded_instances);
    }
    LOG(INFO) << "Load instance info from etcd:" << loaded_instances.size();

    prefill_index_.reserve(loaded_instances.size());
    decode_index_.reserve(loaded_instances.size());

    for (auto& pair : loaded_instances) {
      if (!register_instance(pair.first, pair.second)) {
        LOG(ERROR) << "Fail to register instance: " << pair.first;
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

InstanceMgr::~InstanceMgr() {
  exited_ = true;
  if (state_reconcile_thread_ && state_reconcile_thread_->joinable()) {
    state_reconcile_thread_->join();
  }
}

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

  routing->decode_name.clear();
  if (suspect_instances_.empty()) {
    // Fast path for the common case: no suspect instances, keep plain RR.
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

  if (!select_next_schedulable_instance(instances_,
                                        prefill_index_,
                                        &next_prefill_index_,
                                        &routing->prefill_name)) {
    LOG(ERROR) << "No schedulable prefill or default instance found!";
    return false;
  }

  if (!decode_index_.empty()) {
    select_next_schedulable_instance(
        instances_, decode_index_, &next_decode_index_, &routing->decode_name);
  }
  return true;
}

// TODO: refactor later, currently return all decode instances
std::vector<std::string> InstanceMgr::get_static_decode_list(
    const std::string& instance_name) {
  std::vector<std::string> decode_list;
  std::shared_lock<std::shared_mutex> lock(inst_mutex_);
  for (auto& inst : instances_) {
    if (inst.second.type == InstanceType::DECODE &&
        is_instance_schedulable(inst.second)) {
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
    if ((inst.second.type == InstanceType::PREFILL ||
         inst.second.type == InstanceType::DEFAULT) &&
        is_instance_schedulable(inst.second)) {
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
    if (instance_it == instances_.end() ||
        !is_instance_schedulable(instance_it->second)) {
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
      if (instance_it == instances_.end() ||
          !is_instance_schedulable(instance_it->second)) {
        continue;
      }
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

bool InstanceMgr::bind_request_instance_incarnations(
    const std::shared_ptr<Request>& request) {
  std::shared_lock<std::shared_mutex> lock(inst_mutex_);

  // Bind the selected routing to a concrete incarnation before dispatch.
  request->prefill_incarnation_id.clear();
  request->decode_incarnation_id.clear();

  if (!request->routing.prefill_name.empty()) {
    auto prefill_it = instances_.find(request->routing.prefill_name);
    if (prefill_it == instances_.end()) {
      LOG(ERROR) << "Prefill instance is not registered when binding request: "
                 << request->routing.prefill_name;
      return false;
    }
    if (!is_instance_schedulable(prefill_it->second)) {
      LOG(ERROR) << "Prefill instance is not schedulable when binding request: "
                 << request->routing.prefill_name << ", state: "
                 << runtime_state_name(prefill_it->second.runtime_state);
      return false;
    }
    request->prefill_incarnation_id = prefill_it->second.incarnation_id;
  }

  if (!request->routing.decode_name.empty()) {
    auto decode_it = instances_.find(request->routing.decode_name);
    if (decode_it == instances_.end()) {
      LOG(ERROR) << "Decode instance is not registered when binding request: "
                 << request->routing.decode_name;
      return false;
    }
    if (!is_instance_schedulable(decode_it->second)) {
      LOG(ERROR) << "Decode instance is not schedulable when binding request: "
                 << request->routing.decode_name << ", state: "
                 << runtime_state_name(decode_it->second.runtime_state);
      return false;
    }
    request->decode_incarnation_id = decode_it->second.incarnation_id;
  }

  return true;
}

bool InstanceMgr::record_instance_heartbeat(const std::string& instance_name,
                                            const std::string& incarnation_id) {
  std::unique_lock<std::shared_mutex> lock(inst_mutex_);
  auto it = instances_.find(instance_name);
  if (it == instances_.end()) {
    LOG(WARNING) << "Ignore heartbeat from unknown instance: " << instance_name;
    return false;
  }

  if (it->second.incarnation_id != incarnation_id) {
    LOG(WARNING) << "Ignore stale heartbeat from instance: " << instance_name
                 << ", current incarnation_id: " << it->second.incarnation_id
                 << ", heartbeat incarnation_id: " << incarnation_id;
    return false;
  }

  it->second.latest_timestamp = current_time_ms();
  if (it->second.runtime_state == InstanceRuntimeState::SUSPECT) {
    // A recovered suspect instance first goes back to LEASE_LOST and must
    // keep heartbeating before becoming fully active again via registration.
    clear_suspect_instance(instance_name, incarnation_id);
    it->second.runtime_state = InstanceRuntimeState::LEASE_LOST;
    LOG(WARNING) << "Heartbeat recovered for suspect instance, move to "
                 << "lease lost state: " << instance_name
                 << ", incarnation_id: " << incarnation_id;
  }
  return true;
}

bool InstanceMgr::create_channel(const std::string& instance_name) {
  if (cached_channels_.find(instance_name) == cached_channels_.end()) {
    auto channel = std::make_shared<brpc::Channel>();
    brpc::ChannelOptions options;
    // Add to params
    // options.protocol = "http";
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

bool InstanceMgr::probe_instance_health(const std::string& instance_name) {
  const int attempts = std::max(1, options_.instance_delete_probe_attempts());
  const int timeout_ms =
      std::max(1, options_.instance_delete_probe_timeout_ms());
  const std::string url = "http://" + instance_name + kHealthPath;

  for (int attempt = 1; attempt <= attempts; ++attempt) {
    brpc::Channel channel;
    brpc::ChannelOptions options;
    options.protocol = "http";
    options.timeout_ms = timeout_ms;
    options.connect_timeout_ms = timeout_ms;
    options.max_retry = 0;
    if (channel.Init(url.c_str(), "", &options) != 0) {
      LOG(WARNING) << "Failed to initialize health probe channel, instance: "
                   << instance_name << ", attempt: " << attempt << "/"
                   << attempts;
    } else {
      brpc::Controller cntl;
      cntl.http_request().uri() = url;
      cntl.http_request().set_method(brpc::HTTP_METHOD_GET);
      channel.CallMethod(nullptr, &cntl, nullptr, nullptr, nullptr);
      if (!cntl.Failed() && cntl.http_response().status_code() == 200) {
        return true;
      }
      LOG(WARNING) << "Health probe failed, instance: " << instance_name
                   << ", attempt: " << attempt << "/" << attempts << ", error: "
                   << (cntl.Failed() ? cntl.ErrorText()
                                     : std::to_string(
                                           cntl.http_response().status_code()));
    }

    if (attempt < attempts) {
      std::this_thread::sleep_for(
          std::chrono::milliseconds(kDeleteProbeRetryBackoffMs));
    }
  }

  return false;
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
    for (const auto& event : response.events()) {
      const std::string instance_name = get_event_key_suffix(event, prefix_len);
      if (instance_name.empty()) {
        continue;
      }

      if (event.event_type() == etcd::Event::EventType::PUT) {
        InstanceMetaInfo metainfo;
        auto json_str = get_event_value(event);
        if (!metainfo.parse_from_json(json_str)) {
          LOG(ERROR) << "Parse instance json failed: " << json_str;
          continue;
        }
        std::unique_lock<std::shared_mutex> lock(inst_mutex_);
        auto existing_it = instances_.find(instance_name);
        if (existing_it == instances_.end()) {
          if (!register_instance(instance_name, metainfo)) {
            LOG(ERROR) << "Fail to register instance: " << instance_name;
          }
          continue;
        }

        if (existing_it->second.incarnation_id == metainfo.incarnation_id) {
          const auto previous_state = existing_it->second.runtime_state;
          refresh_instance_registration(instance_name, metainfo);
          clear_suspect_instance(instance_name, metainfo.incarnation_id);
          if (previous_state != InstanceRuntimeState::ACTIVE) {
            LOG(INFO) << "Instance registration restored, back to active: "
                      << instance_name
                      << ", incarnation_id: " << metainfo.incarnation_id
                      << ", previous_state: "
                      << runtime_state_name(previous_state);
          }
          continue;
        }

        const std::string old_incarnation_id =
            existing_it->second.incarnation_id;
        LOG(WARNING) << "Detected instance replacement, instance_name: "
                     << instance_name
                     << ", old incarnation_id: " << old_incarnation_id
                     << ", new incarnation_id: " << metainfo.incarnation_id;
        deregister_instance(instance_name, old_incarnation_id);
        if (!register_instance(instance_name, metainfo)) {
          LOG(ERROR) << "Fail to register replacement instance: "
                     << instance_name;
        }
        continue;
      }

      if (event.event_type() != etcd::Event::EventType::DELETE_) {
        continue;
      }

      InstanceMetaInfo deleted_info;
      std::string deleted_incarnation_id;
      const auto deleted_value = get_event_value(event);
      if (!deleted_value.empty() &&
          deleted_info.parse_from_json(deleted_value)) {
        deleted_incarnation_id = deleted_info.incarnation_id;
      }
      std::string tracked_incarnation_id;
      {
        std::unique_lock<std::shared_mutex> lock(inst_mutex_);
        auto existing_it = instances_.find(instance_name);
        if (existing_it == instances_.end()) {
          continue;
        }

        if (!deleted_incarnation_id.empty() &&
            existing_it->second.incarnation_id != deleted_incarnation_id) {
          LOG(INFO) << "Ignore stale delete for replaced instance: "
                    << instance_name
                    << ", deleted incarnation_id: " << deleted_incarnation_id
                    << ", current incarnation_id: "
                    << existing_it->second.incarnation_id;
          continue;
        }
        tracked_incarnation_id = existing_it->second.incarnation_id;
      }

      // Keep delete handling event-driven: use this event, one health probe,
      // and later heartbeats or PUT events to drive recovery.
      const bool probe_success = probe_instance_health(instance_name);

      std::unique_lock<std::shared_mutex> lock(inst_mutex_);
      auto existing_it = instances_.find(instance_name);
      if (existing_it == instances_.end() ||
          existing_it->second.incarnation_id != tracked_incarnation_id) {
        continue;
      }

      if (probe_success) {
        // Keep the instance in service temporarily and wait for heartbeats.
        clear_suspect_instance(instance_name, tracked_incarnation_id);
        existing_it->second.runtime_state = InstanceRuntimeState::LEASE_LOST;
        existing_it->second.latest_timestamp = current_time_ms();
        LOG(WARNING) << "Instance lease deleted, probe succeeded, enter "
                     << "lease lost state: " << instance_name
                     << ", incarnation_id: " << tracked_incarnation_id;
        continue;
      }

      mark_instance_suspect(instance_name, tracked_incarnation_id);
      LOG(WARNING) << "Instance lease deleted, probe failed, enter suspect "
                   << "state: " << instance_name
                   << ", incarnation_id: " << tracked_incarnation_id;
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

void InstanceMgr::reconcile_instance_states() {
  const auto suspect_interval_ms =
      std::max<int64_t>(1, options_.detect_disconnected_instance_interval()) *
      1000;
  const auto heartbeat_timeout_ms =
      std::max<int64_t>(1, options_.lease_lost_heartbeat_timeout_ms());
  while (!exited_) {
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::unique_lock<std::shared_mutex> lock(inst_mutex_);
    if (exited_) {
      return;
    }

    const uint64_t now_ms = current_time_ms();
    for (auto& [instance_name, info] : instances_) {
      // LEASE_LOST is a grace period after etcd delete but before hard
      // eviction.
      if (info.runtime_state != InstanceRuntimeState::LEASE_LOST) {
        continue;
      }
      if (now_ms - info.latest_timestamp < heartbeat_timeout_ms) {
        continue;
      }
      mark_instance_suspect(instance_name, info.incarnation_id);
      LOG(WARNING) << "Lease lost instance heartbeat timed out, enter suspect "
                   << "state: " << instance_name
                   << ", incarnation_id: " << info.incarnation_id;
    }

    for (auto it = suspect_instances_.begin();
         it != suspect_instances_.end();) {
      const std::string instance_name = it->first;
      const std::string incarnation_id = it->second.incarnation_id;
      const uint64_t enter_ts_ms = it->second.enter_ts_ms;
      ++it;

      if (now_ms - enter_ts_ms < suspect_interval_ms) {
        continue;
      }

      auto inst_it = instances_.find(instance_name);
      if (inst_it == instances_.end() ||
          inst_it->second.incarnation_id != incarnation_id) {
        suspect_instances_.erase(instance_name);
        continue;
      }

      LOG(WARNING) << "Suspect window expired, deregister instance: "
                   << instance_name << ", incarnation_id: " << incarnation_id;
      deregister_instance(instance_name, incarnation_id);
    }
  }
}

void InstanceMgr::refresh_instance_registration(const std::string& name,
                                                const InstanceMetaInfo& info) {
  auto it = instances_.find(name);
  if (it == instances_.end()) {
    return;
  }

  // Preserve local scheduling/index state across etcd refreshes.
  const auto instance_index = it->second.instance_index;
  const auto current_type = it->second.current_type;

  it->second = info;
  it->second.instance_index = instance_index;
  it->second.current_type = current_type;
  it->second.latest_timestamp = current_time_ms();
  it->second.runtime_state = InstanceRuntimeState::ACTIVE;
}

void InstanceMgr::mark_instance_suspect(const std::string& name,
                                        const std::string& incarnation_id) {
  SuspectInstanceInfo info;
  info.incarnation_id = incarnation_id;
  info.enter_ts_ms = current_time_ms();
  suspect_instances_[name] = std::move(info);
  auto it = instances_.find(name);
  if (it != instances_.end() && it->second.incarnation_id == incarnation_id) {
    it->second.runtime_state = InstanceRuntimeState::SUSPECT;
  }
}

void InstanceMgr::clear_suspect_instance(const std::string& name,
                                         const std::string& incarnation_id) {
  auto it = suspect_instances_.find(name);
  if (it == suspect_instances_.end()) {
    return;
  }
  if (!incarnation_id.empty() && it->second.incarnation_id != incarnation_id) {
    return;
  }
  suspect_instances_.erase(it);
}

void InstanceMgr::update_request_metrics(std::shared_ptr<Request> request,
                                         RequestAction action) {
  // skip request metrics update if policy is not SLO_AWARE
  if (options_.load_balance_policy() != "SLO_AWARE") {
    return;
  }

  bool need_flip_decode_to_prefill = false;
  {
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
        // update the request metrics for prefill and decode instance when
        // request finishes the prefill phase
        prefill_it->second.prefill_request_num -= 1;
        prefill_it->second.prefill_token_num -= num_prompt_tokens;
        prefill_it->second.estimated_prefill_time -= request->estimated_ttft;

        decode_it->second.decode_token_num += 1;
        break;
      case RequestAction::GENERATE:
        // update the request metrics for decode instance when request generate
        // a token
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

    // Defer index mutation until after releasing request_metrics_mutex_.
    need_flip_decode_to_prefill = decode_it->second.decode_request_num == 0;
  }

  if (need_flip_decode_to_prefill) {
    std::unique_lock<std::shared_mutex> instance_lock(inst_mutex_);
    flip_decode_to_prefill(request->routing.decode_name);
  }
}

bool InstanceMgr::select_instance_pair_on_slo(
    std::shared_ptr<Request> request) {
  std::unique_lock<std::shared_mutex> lock(inst_mutex_);
  std::lock_guard<std::mutex> request_metrics_lock(request_metrics_mutex_);
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
    int64_t request_num = request_metrics_[decode_instance].decode_request_num;
    auto& time_predictor = get_time_predictor(decode_instance);
    int64_t estimated_tpot = time_predictor.predict_tpot(
        token_num + request->token_ids.size(), request_num + 1);
    if (estimated_tpot <= FLAGS_target_tpot && target_decode_instance.empty()) {
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

  // select prefill instance
  float tpot_threshold =
      (schedulable_decode_count - 1.0f) / schedulable_decode_count;
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
  float ttft_threshold =
      (schedulable_prefill_count - 1.0f) / schedulable_prefill_count;
  if (target_decode_instance.empty() &&
      (avg_prefill_time < FLAGS_target_ttft * ttft_threshold ||
       schedulable_decode_count < schedulable_prefill_count)) {
    flip_prefill_to_decode(request->routing.prefill_name);
  }

  return true;
}

void InstanceMgr::flip_prefill_to_decode(std::string& instance_name) {
  if (count_schedulable_instances(instances_, prefill_index_) <= 1) {
    // Ensure there is at least one prefill instance.
    return;
  }

  if (instances_.find(instance_name) == instances_.end()) {
    LOG(ERROR) << "Can't find instance, instance_name: " << instance_name;
    return;
  }

  // delete instance name from prefill_index_
  remove_instance_from_index(instance_name, instances_[instance_name]);

  // insert instance name to decode_index_
  instances_[instance_name].current_type = InstanceType::DECODE;
  add_instance_to_index(instance_name, instances_[instance_name]);

  LOG(INFO) << "Flip prefill to decode, instance name : " << instance_name;
}

void InstanceMgr::flip_decode_to_prefill(std::string& instance_name) {
  if (count_schedulable_instances(instances_, decode_index_) <= 1) {
    // Ensure there is at least one decode instance.
    return;
  }

  if (instances_.find(instance_name) == instances_.end()) {
    LOG(ERROR) << "Can't find instance, instance_name: " << instance_name;
    return;
  }

  // delete instance name from decode_index_
  remove_instance_from_index(instance_name, instances_[instance_name]);

  // insert instance name to prefill_index
  instances_[instance_name].current_type = InstanceType::PREFILL;
  add_instance_to_index(instance_name, instances_[instance_name]);

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

bool InstanceMgr::call_link_instance(const std::string& target_rpc_addr,
                                     const InstanceMetaInfo& peer_info) {
  brpc::Channel channel;
  brpc::ChannelOptions options;
  options.protocol = "http";
  options.timeout_ms = options_.timeout_ms();
  options.max_retry = 3;
  if (channel.Init(target_rpc_addr.c_str(), "", &options) != 0) {
    LOG(ERROR) << "Fail to initialize channel for LinkInstance to "
               << target_rpc_addr;
    return false;
  }
  xllm::proto::DisaggPDService_Stub stub(&channel);
  brpc::Controller cntl;
  xllm::proto::InstanceClusterInfo req;
  req.set_instance_name(peer_info.name);
  for (auto& cluster_id : peer_info.cluster_ids) {
    req.add_cluster_ids(cluster_id);
  }
  for (auto& addr : peer_info.addrs) {
    req.add_addrs(addr);
  }
  for (auto& ip : peer_info.device_ips) {
    req.add_device_ips(ip);
  }
  for (auto& port : peer_info.ports) {
    req.add_ports(port);
  }
  req.set_dp_size(peer_info.dp_size);
  xllm::proto::Status res;
  stub.LinkInstance(&cntl, &req, &res, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "LinkInstance failed, target: " << target_rpc_addr
               << ", peer: " << peer_info.name
               << ", error: " << cntl.ErrorText();
    return false;
  }
  return res.ok();
}

bool InstanceMgr::call_unlink_instance(const std::string& target_rpc_addr,
                                       const InstanceMetaInfo& peer_info) {
  brpc::Channel channel;
  brpc::ChannelOptions options;
  options.protocol = "http";
  options.timeout_ms = options_.timeout_ms();
  options.max_retry = 3;
  if (channel.Init(target_rpc_addr.c_str(), "", &options) != 0) {
    LOG(ERROR) << "Fail to initialize channel for UnlinkInstance to "
               << target_rpc_addr;
    return false;
  }
  xllm::proto::DisaggPDService_Stub stub(&channel);
  brpc::Controller cntl;
  xllm::proto::InstanceClusterInfo req;
  req.set_instance_name(peer_info.name);
  for (auto& cluster_id : peer_info.cluster_ids) {
    req.add_cluster_ids(cluster_id);
  }
  for (auto& addr : peer_info.addrs) {
    req.add_addrs(addr);
  }
  for (auto& ip : peer_info.device_ips) {
    req.add_device_ips(ip);
  }
  for (auto& port : peer_info.ports) {
    req.add_ports(port);
  }
  req.set_dp_size(peer_info.dp_size);
  xllm::proto::Status res;
  stub.UnlinkInstance(&cntl, &req, &res, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "UnlinkInstance failed, target: " << target_rpc_addr
               << ", peer: " << peer_info.name
               << ", error: " << cntl.ErrorText();
    return false;
  }
  return res.ok();
}

bool InstanceMgr::register_instance(const std::string& name,
                                    InstanceMetaInfo& info) {
  info.runtime_state = InstanceRuntimeState::ACTIVE;
  info.latest_timestamp = current_time_ms();
  if (!create_channel(name)) {
    LOG(ERROR) << "create channel fail: " << name;
    return false;
  }

  add_instance_resources(name, info);

  if (!link_instance_internal(name, info)) {
    remove_instance_resources(name);
    return false;
  }

  add_instance_to_index(name, info);
  instances_.insert(std::make_pair(name, info));
  return true;
}

void InstanceMgr::deregister_instance(
    const std::string& name,
    const std::string& expected_incarnation_id) {
  auto it = instances_.find(name);
  if (it == instances_.end()) {
    LOG(ERROR) << "Instance is not registered, instance_name: " << name;
    return;
  }

  if (!expected_incarnation_id.empty() &&
      it->second.incarnation_id != expected_incarnation_id) {
    LOG(INFO) << "Skip deregistering stale incarnation, instance_name: " << name
              << ", current incarnation_id: " << it->second.incarnation_id
              << ", expected incarnation_id: " << expected_incarnation_id;
    return;
  }

  const auto& info = it->second;
  clear_suspect_instance(name, info.incarnation_id);
  unlink_instance_internal(name, info);
  remove_instance_from_index(name, info);

  scheduler_->clear_requests_on_failed_instance(
      name, info.incarnation_id, get_cleanup_type(info));

  remove_instance_resources(name);
  instances_.erase(it);
  LOG(INFO) << "delete instance: " << name;
}

void InstanceMgr::add_instance_resources(const std::string& name,
                                         const InstanceMetaInfo& info) {
  std::lock_guard<std::mutex> time_predictor_lock(time_predictor_mutex_);
  std::lock_guard<std::mutex> request_metrics_lock(request_metrics_mutex_);

  time_predictors_.insert_or_assign(
      name, TimePredictor(info.ttft_profiling_data, info.tpot_profiling_data));

  request_metrics_.insert_or_assign(name, RequestMetrics());
}

void InstanceMgr::remove_instance_resources(const std::string& name) {
  cached_channels_.erase(name);
  {
    std::lock_guard<std::mutex> time_predictor_lock(time_predictor_mutex_);
    std::lock_guard<std::mutex> request_metrics_lock(request_metrics_mutex_);
    time_predictors_.erase(name);
    request_metrics_.erase(name);
  }
  {
    std::lock_guard<std::mutex> lock(latency_metrics_mutex_);
    latency_metrics_.erase(name);
  }
  {
    std::lock_guard<std::mutex> lock(update_mutex_);
    updated_metrics_.erase(name);
    removed_instance_.insert(name);
  }
  {
    std::unique_lock<std::shared_mutex> lock(load_metric_mutex_);
    load_metrics_.erase(name);
  }
}

bool InstanceMgr::link_instance_internal(const std::string& name,
                                         InstanceMetaInfo& info) {
  bool link_ok = true;
  int32_t linked_p_count = 0;
  int32_t linked_d_count = 0;
  std::vector<std::string> linked_mix_names;

  switch (info.type) {
    case InstanceType::DEFAULT:
      break;
    case InstanceType::PREFILL: {
      for (auto& d_name : decode_index_) {
        if (!call_link_instance(instances_[d_name].rpc_address, info)) {
          link_ok = false;
          break;
        }
        linked_d_count++;
      }
      break;
    }
    case InstanceType::DECODE: {
      for (auto& p_name : prefill_index_) {
        if (!call_link_instance(info.rpc_address, instances_[p_name])) {
          link_ok = false;
          break;
        }
        linked_p_count++;
      }
      break;
    }
    case InstanceType::MIX: {
      for (const auto& [peer_name, peer_info] : instances_) {
        if (peer_name == name) {
          continue;
        }
        if (!call_link_instance(info.rpc_address, peer_info)) {
          link_ok = false;
          break;
        }
        linked_mix_names.emplace_back(peer_name);
      }
      break;
    }
    default:
      LOG(WARNING) << "Unknown InstanceType: " << int(info.type);
      return false;
  }

  if (!link_ok) {
    LOG(ERROR) << "Fail to link instance during registration, instance: "
               << name;
    // Rollback
    if (info.type == InstanceType::PREFILL) {
      for (int i = 0; i < linked_d_count; i++) {
        call_unlink_instance(instances_[decode_index_[i]].rpc_address, info);
      }
    } else if (info.type == InstanceType::DECODE) {
      for (int i = 0; i < linked_p_count; i++) {
        call_unlink_instance(info.rpc_address, instances_[prefill_index_[i]]);
      }
    } else if (info.type == InstanceType::MIX) {
      for (const auto& linked_name : linked_mix_names) {
        auto it = instances_.find(linked_name);
        if (it == instances_.end()) {
          continue;
        }
        call_unlink_instance(info.rpc_address, it->second);
      }
    }
    return false;
  }

  return true;
}

void InstanceMgr::unlink_instance_internal(const std::string& name,
                                           const InstanceMetaInfo& info) {
  if (info.type == InstanceType::PREFILL) {
    for (auto& d_name : decode_index_) {
      call_unlink_instance(instances_[d_name].rpc_address, info);
    }
  } else if (info.type == InstanceType::DECODE) {
    for (auto& p_name : prefill_index_) {
      call_unlink_instance(instances_[p_name].rpc_address, info);
    }
  } else if (info.type == InstanceType::MIX) {
    for (const auto& [peer_name, peer_info] : instances_) {
      if (peer_name == name) {
        continue;
      }
      call_unlink_instance(peer_info.rpc_address, info);
    }
  }
}

void InstanceMgr::add_instance_to_index(const std::string& name,
                                        InstanceMetaInfo& info) {
  switch (info.type) {
    case InstanceType::DEFAULT:
      info.instance_index = prefill_index_.size();
      prefill_index_.emplace_back(name);
      LOG(INFO) << "Register a new default instance, instance name : " << name;
      break;
    case InstanceType::PREFILL:
      info.instance_index = prefill_index_.size();
      prefill_index_.emplace_back(name);
      LOG(INFO) << "Register a new prefill instance, instance name : " << name;
      break;
    case InstanceType::DECODE:
      info.instance_index = decode_index_.size();
      decode_index_.emplace_back(name);
      LOG(INFO) << "Register a new decode instance, instance name : " << name;
      break;
    case InstanceType::MIX:
      if (decode_index_.size() > 0) {
        info.instance_index = prefill_index_.size();
        info.current_type = InstanceType::PREFILL;
        prefill_index_.emplace_back(name);
        LOG(INFO) << "Register a new prefill instance, instance name : "
                  << name;
      } else {
        info.instance_index = decode_index_.size();
        info.current_type = InstanceType::DECODE;
        decode_index_.emplace_back(name);
        LOG(INFO) << "Register a new decode instance, instance name : " << name;
      }
      break;
    default:
      break;
  }
}

void InstanceMgr::remove_instance_from_index(const std::string& name,
                                             const InstanceMetaInfo& info) {
  uint64_t index = info.instance_index;
  if (index == -1) return;

  auto remove_from_vec = [&](std::vector<std::string>& vec) {
    if (index >= vec.size()) return;
    std::swap(vec[index], vec.back());
    instances_[vec[index]].instance_index = index;
    vec.pop_back();
  };

  switch (info.type) {
    case InstanceType::DEFAULT:
    case InstanceType::PREFILL:
      remove_from_vec(prefill_index_);
      break;
    case InstanceType::DECODE:
      remove_from_vec(decode_index_);
      break;
    case InstanceType::MIX:
      if (info.current_type == InstanceType::PREFILL) {
        remove_from_vec(prefill_index_);
      } else {
        remove_from_vec(decode_index_);
      }
      break;
    default:
      break;
  }
}

}  // namespace xllm_service
