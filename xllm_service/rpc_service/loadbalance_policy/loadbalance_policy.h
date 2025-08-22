#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "common/macros.h"
#include "common/types.h"

namespace xllm_service {

class LoadBalancePolicy {
 public:
  LoadBalancePolicy() = default;

  virtual ~LoadBalancePolicy() = default;

  virtual void select_instances_pair(const LoadBalanceInfos& infos,
                                     Routing* routing);

 protected:
  DISALLOW_COPY_AND_ASSIGN(LoadBalancePolicy);

  virtual void cost_function(
      const std::unordered_map<std::string, uint32_t>& overlap_scores,
      const uint32_t& max_block_num,
      const std::unordered_map<std::string, LoadMetrics>& load_metrics,
      const int64_t& max_waiting_requests_num,
      std::string* best_choice);
};

}  // namespace xllm_service
