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
#include <functional>
#include <memory>
#include <string>
#include <utility>
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

// Single facade for the managers layer: instance topology (etcd discovery,
// brpc channels, prefill/decode indices, link state), per-instance metrics
// (load, request counters, latency, TimePredictor), and master-aggregated
// KV-cache block locations. Scheduler and load-balance code should use this
// class instead of touching InstanceTopologyImpl / InstanceMetricsImpl /
// InstanceKVCache directly.
class InstanceMgr final {
 public:
  // Constructs sub-managers, wires topology<->metrics callbacks (register and
  // tear down per-instance metric maps with instance membership), and loads
  // initial topology from etcd plus cached load metrics when applicable.
  explicit InstanceMgr(const Options& options,
                       const std::shared_ptr<EtcdClient>& etcd_client,
                       const bool is_master_service,
                       OnInstanceDeregisteredCallback on_instance_deregistered);

  // Stops metrics background work (etcd watches, etc.).
  ~InstanceMgr();

  // Returns the latest InstanceMetaInfo for a registered instance name
  // (empty fields if unknown). Thread-safe read through topology.
  InstanceMetaInfo get_instance_info(const std::string& instance_name);

  // Cached brpc::Channel for RPC to the given instance, or nullptr if missing.
  std::shared_ptr<brpc::Channel> get_channel(const std::string& instance_name);

  // Decode-side instance names in static topology order (see topology indices).
  // prepare_load_balance_candidates。
  std::vector<std::string> get_static_decode_list();

  // Prefill-side instance names in static topology order.
  // prepare_load_balance_candidates。
  std::vector<std::string> get_static_prefill_list();

  // Single entry for worker heartbeat: topology liveness, KV cache events,
  // load metrics, and latency metrics. Returns false when heartbeat is
  // rejected (unknown instance or incarnation mismatch).
  bool on_instance_heartbeat(const proto::HeartbeatRequest& req);

  // Master-only: flush aggregated KV-cache locations and load metrics to etcd.
  // Returns true only if both uploads succeed. Order matches legacy behavior
  // (KV cache first, then load metrics).
  bool upload_master_state_to_etcd();

  // Computes KV overlap scores for the given prompt token ids against the
  // aggregated cache map (used by cache-aware routing).
  void kvcache_match(const Slice<int32_t>& token_ids,
                     OverlapScores* overlap_scores);

  // Updates per-instance request counters and SLO-related estimates when
  // load_balance_policy is SLO_AWARE (schedule / prefill finish / generate /
  // decode finish / cancel). May trigger optional PD flip hints via topology.
  void update_request_metrics(std::shared_ptr<Request> request,
                              RequestAction action);

  // Single snapshot for load balancing: same as get_static_* lists + per-name
  // meta + fill_load_balance_infos, under one pair of shared locks (topology
  // then metrics). Prefer this over separate calls for a consistent view.
  bool prepare_load_balance_candidates(
      const std::function<bool(const InstanceMetaInfo&)>& is_schedulable,
      LoadBalanceCandidates* candidates);

  // After scheduling: checks non-empty routing names and incarnation ids,
  // instances exist, runtime_state is ACTIVE, brpc channels exist, and
  // incarnation ids match current topology (detects replacement / stale ids).
  // On failure, logs one line and returns false.
  bool validate_scheduled_routing(const Request& request);

  // Per-instance TPOT estimate from the metrics TimePredictor (holds
  // metrics_mutex_ internally). total_length / batch_size match TimePredictor.
  double predict_tpot(const std::string& instance_name,
                      int32_t total_length,
                      int32_t batch_size);

  // Master promotion: stop follower watches on metrics/KV-cache paths.
  void set_as_master();

 private:
  DISALLOW_COPY_AND_ASSIGN(InstanceMgr);

  // Loads topology from etcd and initial load metrics snapshot (constructor
  // body).
  void init();

  std::shared_ptr<EtcdClient> etcd_client_;
  std::unique_ptr<InstanceMetricsImpl> metrics_impl_;
  std::unique_ptr<InstanceTopologyImpl> topology_impl_;
  std::unique_ptr<InstanceKVCache> kvcache_;
};

}  // namespace xllm_service
