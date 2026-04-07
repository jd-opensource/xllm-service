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
#include <string>
#include <vector>

#include "common/macros.h"
#include "common/options.h"
#include "common/slice.h"
#include "common/types.h"
#include "request/request.h"
#include "scheduler/etcd_client/etcd_client.h"
#include "scheduler/managers/instance_metrics.h"
#include "scheduler/managers/instance_topology.h"
#include "xllm_rpc_service.pb.h"

namespace xllm_service {
class InstanceKVCache;

class InstanceMgr final {
 public:
  explicit InstanceMgr(const Options& options,
                       const std::shared_ptr<EtcdClient>& etcd_client,
                       const bool is_master_service,
                       OnInstanceDeregisteredCallback on_instance_deregistered);

  ~InstanceMgr();

  InstanceMetaInfo get_instance_info(const std::string& instance_name);
  std::shared_ptr<brpc::Channel> get_channel(const std::string& instance_name);

  // Prefill / decode names in topology index order for load balancing.
  // When no instance is in SUSPECT, returns full index lists (same as legacy
  // fast-path RR). When any suspect exists, only instances that are
  // schedulable (non-SUSPECT) are included.
  std::vector<std::string> get_schedulable_prefill_instances();
  std::vector<std::string> get_schedulable_decode_instances();

  std::vector<std::string> get_static_decode_list(
      const std::string& instance_name);

  std::vector<std::string> get_static_prefill_list(
      const std::string& instance_name);

  // Single entry for worker heartbeat: topology liveness, KV cache events,
  // load metrics, and latency metrics. Returns false when heartbeat is
  // rejected (unknown instance or incarnation mismatch).
  bool on_instance_heartbeat(const proto::HeartbeatRequest& req);

  // Master-only: flush aggregated KV-cache locations and load metrics to etcd.
  // Returns true only if both uploads succeed. Order matches legacy behavior
  // (KV cache first, then load metrics).
  bool upload_master_state_to_etcd();

  void get_load_metrics(LoadBalanceInfos* infos);

  bool bind_request_instance_incarnations(
      const std::shared_ptr<Request>& request);

  void kvcache_match(const Slice<int32_t>& token_ids,
                     OverlapScores* overlap_scores);

  void update_request_metrics(std::shared_ptr<Request> request,
                              RequestAction action);

  bool select_instance_pair_on_slo(std::shared_ptr<Request> request);

  // Master promotion: stop follower watches on metrics/KV-cache paths.
  void set_as_master();

 private:
  DISALLOW_COPY_AND_ASSIGN(InstanceMgr);

  void init();

  std::shared_ptr<EtcdClient> etcd_client_;

  std::unique_ptr<InstanceMetricsImpl> metrics_impl_;
  std::unique_ptr<InstanceTopologyImpl> topology_impl_;
  std::unique_ptr<InstanceKVCache> kvcache_;
};

}  // namespace xllm_service
