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

#include "disagg_pd_policy.h"

#include <glog/logging.h>

#include "common/utils.h"

namespace xllm_service {

namespace {

void debug_print(const std::string& action,
                 const std::string& name,
                 const std::string& type,
                 int idx) {
  if (utils::enable_debug_log()) {
    LOG(INFO) << "DisaggPdPolicy " << action << " instance, name = " << name
              << ", type = " << type << ", idx = " << idx;
  }
}

}  // namespace

DisaggPdPolicy::DisaggPdPolicy() {}

DisaggPdPolicy::~DisaggPdPolicy() {}

void DisaggPdPolicy::insert_instance(const std::string& name,
                                     InstanceMetaInfo* info) {
  std::lock_guard<std::mutex> guard(mutex_);
  InstanceType type = info->type;
  if (type == InstanceType::DEFAULT || type == InstanceType::PREFILL) {
    auto it = prefill_instance_to_index_.find(name);
    if (it != prefill_instance_to_index_.end()) {
      LOG(ERROR) << "Insert instance is already existed, name: " << name
                 << ", type: " << static_cast<int32_t>(type);
      return;
    }
    prefill_instance_.emplace_back(info);
    prefill_instance_to_index_[name] = prefill_instance_.size() - 1;
    debug_print(
        "insert", name, "prefill or default", prefill_instance_to_index_[name]);
  } else {
    auto it = decode_instance_to_index_.find(name);
    if (it != decode_instance_to_index_.end()) {
      LOG(ERROR) << "Insert instance is already existed, name: " << name
                 << ", type: " << static_cast<int32_t>(type);
      return;
    }
    decode_instance_.emplace_back(info);
    decode_instance_to_index_[name] = decode_instance_.size() - 1;
    debug_print("insert", name, "decode", decode_instance_to_index_[name]);
  }
}

void DisaggPdPolicy::update_instance(const std::string& name,
                                     InstanceMetaInfo* info) {
  std::lock_guard<std::mutex> guard(mutex_);
  InstanceType type = info->type;
  if (type == InstanceType::DEFAULT || type == InstanceType::PREFILL) {
    auto it = prefill_instance_to_index_.find(name);
    if (it == prefill_instance_to_index_.end()) {
      LOG(ERROR) << "Update instance is not existed, name: " << name
                 << ", type: " << static_cast<int32_t>(type);
      return;
    }
    prefill_instance_[it->second] = info;
    debug_print(
        "update", name, "prefill or default", prefill_instance_to_index_[name]);
  } else {
    auto it = decode_instance_to_index_.find(name);
    if (it == decode_instance_to_index_.end()) {
      LOG(ERROR) << "Update instance is not existed, name: " << name
                 << ", type: " << static_cast<int32_t>(type);
      return;
    }
    decode_instance_[it->second] = info;
    debug_print("update", name, "decode", decode_instance_to_index_[name]);
  }
}

void DisaggPdPolicy::remove_instance(const std::string& name,
                                     InstanceType type) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (type == InstanceType::DEFAULT || type == InstanceType::PREFILL) {
    auto it = prefill_instance_to_index_.find(name);
    if (it == prefill_instance_to_index_.end()) {
      LOG(ERROR) << "Remove instance not found, name: " << name
                 << ", type: " << static_cast<int32_t>(type);
      return;
    }
    auto idx = it->second;
    // Label the instance be deleted
    prefill_instance_[idx] = nullptr;
    prefill_instance_to_index_.erase(name);
    debug_print("remove", name, "prefill or default", idx);
  } else {
    auto it = decode_instance_to_index_.find(name);
    if (it == decode_instance_to_index_.end()) {
      LOG(ERROR) << "Remove instance not found, name: " << name
                 << ", type: " << static_cast<int32_t>(type);
      return;
    }
    auto idx = it->second;
    // Label the instance be deleted
    decode_instance_[idx] = nullptr;
    decode_instance_to_index_.erase(name);
    debug_print("remove", name, "decode", idx);
  }
}

RoundRobinDisaggPdPolicy::RoundRobinDisaggPdPolicy() {
  LOG(INFO) << "Enable RoundRobin disaggregated pd policy.";
}

RoundRobinDisaggPdPolicy::~RoundRobinDisaggPdPolicy() {}

InstancesPair RoundRobinDisaggPdPolicy::select_instances_pair(
    bool only_prefill) {
  std::lock_guard<std::mutex> guard(mutex_);
  // return the first available prefill instance
  if (only_prefill) {
    InstancesPair inst_pair;
    for (const auto& inst : prefill_instance_) {
      if (inst != nullptr) {
        inst_pair.prefill_instance_http_addr = inst->name;
        break;
      }
    }
    return inst_pair;
  }

  int prefill_count = prefill_instance_.size();
  int decode_count = decode_instance_.size();
  InstancesPair inst_pair;
  // select prefill instance
  if (prefill_count > 0) {
    auto start_idx = next_prefill_idx_;
    bool inst_not_existed = false;
    while (prefill_instance_[next_prefill_idx_] == nullptr) {
      ++next_prefill_idx_;
      next_prefill_idx_ %= prefill_count;
      if (next_prefill_idx_ == start_idx) {
        inst_not_existed = true;
        break;
      }
    }
    if (!inst_not_existed) {
      inst_pair.prefill_instance_http_addr =
          prefill_instance_[next_prefill_idx_]->name;
    }

    ++next_prefill_idx_;
    next_prefill_idx_ %= prefill_count;
  }

  // select decode instance
  if (decode_count > 0) {
    auto start_idx = next_decode_idx_;
    bool inst_not_existed = false;
    while (decode_instance_[next_decode_idx_] == nullptr) {
      ++next_decode_idx_;
      next_decode_idx_ %= decode_count;
      if (next_decode_idx_ == start_idx) {
        inst_not_existed = true;
        break;
      }
    }
    if (!inst_not_existed) {
      inst_pair.decode_instance_http_addr =
          decode_instance_[next_decode_idx_]->name;
    }

    ++next_decode_idx_;
    next_decode_idx_ %= decode_count;
  }

  return inst_pair;
}

std::unordered_map<std::string, InstanceType>
RoundRobinDisaggPdPolicy::reallocate_instances_type(/*params here*/) {
  // TODO: implement this function
  return {};
}

std::unordered_map<std::string, std::vector<std::string>>
RoundRobinDisaggPdPolicy::allocate_pd_pairs(/*params here*/) {
  // TODO: implement this function
  return {};
}

}  // namespace xllm_service
