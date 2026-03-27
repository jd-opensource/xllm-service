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

#include <cstdint>
#include <memory>

#include "common/macros.h"
#include "common/types.h"
#include "loadbalance_policy.h"

namespace xllm_service {

class RoundRobin final : public LoadBalancePolicy {
 public:
  RoundRobin(std::shared_ptr<InstanceMgr> instance_mgr)
      : LoadBalancePolicy(instance_mgr) {};

  virtual ~RoundRobin() = default;

  bool select_instances_pair(std::shared_ptr<Request> request) override;

 private:
  DISALLOW_COPY_AND_ASSIGN(RoundRobin);

  uint64_t next_prefill_index_ = 0;
  uint64_t next_decode_index_ = 0;
};

// Shared round-robin selection over get_schedulable_* lists (used by RoundRobin
// and SloAwarePolicy when token_ids is empty).
bool SelectRoutingRoundRobin(const std::shared_ptr<InstanceMgr>& instance_mgr,
                             uint64_t* next_prefill_index,
                             uint64_t* next_decode_index,
                             Routing* routing);

}  // namespace xllm_service
