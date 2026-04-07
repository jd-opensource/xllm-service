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

#include "slo_aware_policy.h"

#include <glog/logging.h>

#include <limits>

#include "common/global_gflags.h"

namespace xllm_service {

SloAwarePolicy::SloAwarePolicy(const Options& options,
                               std::shared_ptr<InstanceMgr> instance_mgr)
    : options_(options), LoadBalancePolicy(instance_mgr) {}

bool SloAwarePolicy::load_balance(const std::shared_ptr<const Request>& request,
                                  const LoadBalanceCandidates* candidates,
                                  LoadBalanceResult* result) {
  if (request->token_ids.empty()) {
    return false;
  }

  const auto& rm = candidates->load_balance_infos.request_metrics;

  std::string min_prefill_instance;
  int64_t min_prefill_time = std::numeric_limits<int64_t>::max();
  for (const auto& prefill_instance : candidates->prefill_candidates) {
    int64_t prefill_time = 0;
    auto it = rm.find(prefill_instance);
    if (it != rm.end()) {
      prefill_time = it->second.estimated_prefill_time;
    }
    if (prefill_time < min_prefill_time) {
      min_prefill_instance = prefill_instance;
      min_prefill_time = prefill_time;
    }
  }
  if (min_prefill_instance.empty()) {
    LOG(ERROR) << "No prefill candidate for SLO load balance";
    return false;
  }

  std::string min_decode_instance;
  int64_t min_estimated_tpot = std::numeric_limits<int64_t>::max();
  std::string target_decode_instance;
  const size_t schedulable_decode_count = candidates->decode_candidates.size();

  for (const auto& decode_instance : candidates->decode_candidates) {
    int64_t token_num = 0;
    int64_t request_num = 0;
    auto it = rm.find(decode_instance);
    if (it != rm.end()) {
      token_num = it->second.decode_token_num;
      request_num = it->second.decode_request_num;
    }

    int64_t estimated_tpot = static_cast<int64_t>(instance_mgr_->predict_tpot(
        decode_instance,
        static_cast<int32_t>(token_num + request->token_ids.size()),
        static_cast<int32_t>(request_num + 1)));

    if (estimated_tpot <= FLAGS_target_tpot && target_decode_instance.empty()) {
      target_decode_instance = decode_instance;
    }

    if (estimated_tpot < min_estimated_tpot) {
      min_decode_instance = decode_instance;
      min_estimated_tpot = estimated_tpot;
    }
  }

  if (min_decode_instance.empty()) {
    LOG(ERROR) << "No decode candidate for SLO load balance";
    return false;
  }

  if (!target_decode_instance.empty()) {
    result->decode_name = target_decode_instance;
  } else {
    result->decode_name = min_decode_instance;
  }

  const float tpot_threshold =
      schedulable_decode_count > 0
          ? static_cast<float>(schedulable_decode_count - 1) /
                static_cast<float>(schedulable_decode_count)
          : 0.0f;

  int64_t min_decode_prefill_est = 0;
  auto md_it = rm.find(min_decode_instance);
  if (md_it != rm.end()) {
    min_decode_prefill_est = md_it->second.estimated_prefill_time;
  }

  if (min_prefill_time > FLAGS_target_ttft &&
      target_decode_instance != min_decode_instance &&
      min_estimated_tpot < FLAGS_target_tpot * tpot_threshold &&
      min_decode_prefill_est < min_prefill_time) {
    result->prefill_name = min_decode_instance;
  } else {
    result->prefill_name = min_prefill_instance;
  }

  return true;
}

}  // namespace xllm_service
