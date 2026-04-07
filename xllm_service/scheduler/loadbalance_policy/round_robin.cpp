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

#include "round_robin.h"

#include <glog/logging.h>

#include <vector>

namespace xllm_service {

bool SelectRoutingRoundRobin(const std::shared_ptr<InstanceMgr>& instance_mgr,
                             uint64_t* next_prefill_index,
                             uint64_t* next_decode_index,
                             Routing* routing) {
  const std::vector<std::string> prefill =
      instance_mgr->get_schedulable_prefill_instances();
  const std::vector<std::string> decode =
      instance_mgr->get_schedulable_decode_instances();

  if (prefill.empty()) {
    LOG(ERROR) << "No prefill or default instance found!";
    return false;
  }

  if (decode.empty()) {
    LOG(ERROR) << "No decode or default instance found!";
    return false;
  }

  *next_prefill_index = *next_prefill_index % prefill.size();
  routing->prefill_name = prefill[*next_prefill_index];
  (*next_prefill_index)++;
  *next_decode_index = *next_decode_index % decode.size();
  routing->decode_name = decode[*next_decode_index];
  (*next_decode_index)++;
  return true;
}

bool RoundRobin::select_instances_pair(std::shared_ptr<Request> request) {
  return SelectRoutingRoundRobin(instance_mgr_,
                                 &next_prefill_index_,
                                 &next_decode_index_,
                                 &request->routing);
}

}  // namespace xllm_service
