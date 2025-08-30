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

#include "../managers/global_kvcache_mgr.h"
#include "../managers/instance_mgr.h"
#include "common/types.h"

namespace xllm_service {

class LoadBalancePolicy {
 public:
  LoadBalancePolicy(std::shared_ptr<InstanceMgr> instance_mgr,
                    std::shared_ptr<GlobalKVCacheMgr> global_kvcache_mgr)
      : instance_mgr_(instance_mgr), global_kvcache_mgr_(global_kvcache_mgr) {}

  virtual ~LoadBalancePolicy() = default;

  virtual bool select_instances_pair(ScheduleResult* res) = 0;

 protected:
  std::shared_ptr<InstanceMgr> instance_mgr_;

  std::shared_ptr<GlobalKVCacheMgr> global_kvcache_mgr_;
};

}  // namespace xllm_service
