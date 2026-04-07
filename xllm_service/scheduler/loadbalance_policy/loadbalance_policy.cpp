/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "loadbalance_policy.h"

#include <glog/logging.h>

namespace xllm_service {

bool LoadBalancePolicy::select_instances_pair(
    std::shared_ptr<Request> request) {
  constexpr int kMaxAttempts = 2;
  for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
    // 1. prepare load balance candidates, filter out non-schedulable
    // instances，and take a snapshot of the "candidate list + instance_infos +
    // load_balance_infos" as a single generation within a lock-holding
    // sequence, subsequently, we should strive to perform load balancing based
    // on this snapshot information
    LoadBalanceCandidates candidates;
    if (!load_balance_pre_process(request, &candidates)) {
      return false;
    }

    // 2. load balance, select the best instances pair, each loadbalance policy
    // covers the implementation of this method, if failed, use round robin
    // policy instead.
    LoadBalanceResult result;
    if (!load_balance(request, &candidates, &result)) {
      LOG(ERROR) << "Failed to load balance!, use round robin policy instead";
      // the original round robin policy will not reach this point.
      pick_round_robin_candidates(candidates, &result);
    }

    // 3. post process, update the request with the selected instances pair.
    if (!load_balance_post_process(request, &candidates, &result)) {
      return false;
    }

    // 4. validate the selected instances pair, the snapshot obtained in the
    // first step may have undergone changes，if failed, retry.
    if (instance_mgr_->validate_scheduled_routing(*request)) {
      return true;
    }

    if (attempt + 1 < kMaxAttempts) {
      LOG(WARNING)
          << "select_instances_pair: validate_scheduled_routing failed, "
             "retrying once";
    }
  }

  LOG(ERROR)
      << "select_instances_pair: validate_scheduled_routing failed after "
      << kMaxAttempts << " attempt(s)";
  return false;
}

bool LoadBalancePolicy::load_balance(
    const std::shared_ptr<const Request>& request,
    const LoadBalanceCandidates* candidates,
    LoadBalanceResult* result) {
  return true;
}

bool LoadBalancePolicy::should_instance_schedulable(
    const std::shared_ptr<const Request>& request,
    const InstanceMetaInfo& info) const {
  (void)request;
  return info.runtime_state != InstanceRuntimeState::SUSPECT;
}

bool LoadBalancePolicy::load_balance_pre_process(
    const std::shared_ptr<const Request>& request,
    LoadBalanceCandidates* candidates) {
  if (!instance_mgr_->prepare_load_balance_candidates(
          [this, request](const InstanceMetaInfo& info) {
            return should_instance_schedulable(request, info);
          },
          candidates)) {
    LOG(ERROR) << "No schedulable instances found!";
    return false;
  }
  return true;
}

void LoadBalancePolicy::pick_round_robin_candidates(
    const LoadBalanceCandidates& candidates,
    LoadBalanceResult* result) {
  if (candidates.prefill_candidates.empty() ||
      candidates.decode_candidates.empty()) {
    return;
  }
  const uint64_t prefill_idx =
      next_prefill_index_ % candidates.prefill_candidates.size();
  const uint64_t decode_idx =
      next_decode_index_ % candidates.decode_candidates.size();
  result->prefill_name = candidates.prefill_candidates[prefill_idx];
  result->decode_name = candidates.decode_candidates[decode_idx];
  next_prefill_index_++;
  next_decode_index_++;
}

bool LoadBalancePolicy::load_balance_post_process(
    std::shared_ptr<Request> request,
    const LoadBalanceCandidates* candidates,
    LoadBalanceResult* result) {
  if (result->prefill_name.empty() || result->decode_name.empty()) {
    return false;
  }

  request->routing.prefill_name = result->prefill_name;
  request->routing.decode_name = result->decode_name;

  auto pre_it =
      candidates->load_balance_infos.instance_infos.find(result->prefill_name);
  if (pre_it != candidates->load_balance_infos.instance_infos.end()) {
    request->prefill_incarnation_id = pre_it->second.incarnation_id;
  }

  auto dec_it =
      candidates->load_balance_infos.instance_infos.find(result->decode_name);
  if (dec_it != candidates->load_balance_infos.instance_infos.end()) {
    request->decode_incarnation_id = dec_it->second.incarnation_id;
  }

  return true;
}

}  // namespace xllm_service