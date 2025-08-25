#pragma once

#include "loadbalance_policy.h"

namespace xllm_service {

constexpr float MIN_SCORE = -2.0;

void LoadBalancePolicy::select_instances_pair(const LoadBalanceInfos& infos,
                                              Routing* routing) {
  // find preifll
  cost_function(infos.overlap_scores.hbm_instance_score,
                infos.overlap_scores.max_block_num,
                infos.prefill_load_metrics,
                infos.prefill_max_waiting_requests_num,
                &routing->prefill_name);

  // find decode
  if (infos.decode_load_metrics.size()) {
    cost_function(infos.overlap_scores.hbm_instance_score,
                  infos.overlap_scores.max_block_num,
                  infos.decode_load_metrics,
                  infos.decode_max_waiting_requests_num,
                  &routing->decode_name);
  }
}

void LoadBalancePolicy::cost_function(
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
