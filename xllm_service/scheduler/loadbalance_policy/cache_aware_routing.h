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

#include "common/macros.h"
#include "loadbalance_policy.h"

namespace xllm_service {

class CacheAwareRouting final : public LoadBalancePolicy {
 public:
  CacheAwareRouting(std::shared_ptr<InstanceMgr> instance_mgr,
                    std::shared_ptr<GlobalKVCacheMgr> global_kvcache_mgr)
      : LoadBalancePolicy(instance_mgr, global_kvcache_mgr) {};

  virtual ~CacheAwareRouting() = default;

  bool select_instances_pair(ScheduleResult* res) override;

 protected:
  DISALLOW_COPY_AND_ASSIGN(CacheAwareRouting);

  void cost_function(
      const std::unordered_map<std::string, uint32_t>& overlap_scores,
      const uint32_t& max_block_num,
      const std::unordered_map<std::string, LoadMetrics>& load_metrics,
      const int64_t& max_waiting_requests_num,
      std::string* best_choice);
};

}  // namespace xllm_service
