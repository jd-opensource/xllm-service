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

class RoundRobin final : public LoadBalancePolicy {
 public:
  RoundRobin(std::shared_ptr<InstanceMgr> instance_mgr,
             std::shared_ptr<GlobalKVCacheMgr> global_kvcache_mgr)
      : LoadBalancePolicy(instance_mgr, global_kvcache_mgr) {};

  virtual ~RoundRobin() = default;

  bool select_instances_pair(ScheduleResult* res) override;

 protected:
  DISALLOW_COPY_AND_ASSIGN(RoundRobin);
};

}  // namespace xllm_service
