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

#include "cache_aware_routing.h"

namespace xllm_service {

constexpr float MIN_SCORE = -2.0;

bool CacheAwareRouting::load_balance(
    const std::shared_ptr<const Request>& request,
    const LoadBalanceCandidates* candidates,
    LoadBalanceResult* result) {
  OverlapScores overlap_scores;
  instance_mgr_->kvcache_match(request->token_ids, &overlap_scores);
  if (overlap_scores.instances.empty()) {
    return false;
  }

  // find preifll
  const auto& infos = candidates->load_balance_infos;
  cost_function(overlap_scores.hbm_instance_score,
                overlap_scores.max_block_num,
                infos.prefill_load_metrics,
                infos.prefill_max_waiting_requests_num,
                &result->prefill_name);

  // find decode
  cost_function(overlap_scores.hbm_instance_score,
                overlap_scores.max_block_num,
                infos.decode_load_metrics,
                infos.decode_max_waiting_requests_num,
                &result->decode_name);

  return true;
}

void CacheAwareRouting::cost_function(
    const std::unordered_map<std::string, uint32_t>& overlap_scores,
    const uint32_t& max_block_num,
    const std::unordered_map<std::string, LoadMetrics>& load_metrics,
    const int64_t& max_waiting_requests_num,
    std::string* best_choice) {
  float best_score = MIN_SCORE;
  for (const auto& it : load_metrics) {
    const auto matched_blocks_it = overlap_scores.find(it.first);
    uint32_t matched_blocks = 0;
    if (matched_blocks_it != overlap_scores.end()) {
      matched_blocks = matched_blocks_it->second;
    }

    auto score =
        (max_block_num == 0 ? 0 : matched_blocks / max_block_num) -
        it.second.gpu_cache_usage_perc -
        (max_waiting_requests_num == 0
             ? 0
             : it.second.waiting_requests_num / max_waiting_requests_num);

    if (score > best_score) {
      best_score = score;
      *best_choice = it.first;
    }
  }
}

}  // namespace xllm_service
