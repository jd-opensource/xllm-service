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

#pragma once

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/types.h"

namespace xllm_service {

class DisaggPdPolicy {
 public:
  DisaggPdPolicy();
  virtual ~DisaggPdPolicy();

  // re-allocate instance types: prefill or decode
  virtual std::unordered_map<std::string, InstanceType>
      reallocate_instances_type(/*params here*/) = 0;

  // Allocate prefill and decode pairs, return prefill -> [decode instances]
  // Allow multiple decode instances for each prefill instance and
  // multiple prefill instances for each decode instance
  virtual std::unordered_map<std::string, std::vector<std::string>>
      allocate_pd_pairs(/*params here*/) = 0;

  // select instances(prefill/decode/default etc.) to handle request
  // according the disagg pd policy.
  virtual InstancesPair select_instances_pair(bool only_prefill = false) = 0;

  void insert_instance(const std::string& name, InstanceMetaInfo* info);
  void update_instance(const std::string& name, InstanceMetaInfo* info);
  void remove_instance(const std::string& name, InstanceType type);

 protected:
  std::vector<InstanceMetaInfo*> prefill_instance_;
  std::vector<InstanceMetaInfo*> decode_instance_;
  // map the instance name to vector index
  std::unordered_map<std::string, int> prefill_instance_to_index_;
  std::unordered_map<std::string, int> decode_instance_to_index_;

  std::mutex mutex_;
};

class RoundRobinDisaggPdPolicy : public DisaggPdPolicy {
 public:
  RoundRobinDisaggPdPolicy();
  ~RoundRobinDisaggPdPolicy();

  virtual std::unordered_map<std::string, InstanceType>
      reallocate_instances_type(/*params here*/) override;
  virtual std::unordered_map<std::string, std::vector<std::string>>
      allocate_pd_pairs(/*params here*/) override;
  virtual InstancesPair select_instances_pair(
      bool only_prefill = false) override;

 private:
  int next_prefill_idx_ = 0;
  int next_decode_idx_ = 0;
};

}  // namespace xllm_service
