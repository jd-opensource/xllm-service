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

#include <brpc/channel.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/macros.h"
#include "common/options.h"
#include "common/threadpool.h"
#include "common/types.h"
#include "scheduler/etcd_client/etcd_client.h"

namespace xllm_service {

using AddInstanceMetricsCallback =
    std::function<void(const std::string&, const InstanceMetaInfo&)>;
using RemoveInstanceMetricsCallback = std::function<void(const std::string&)>;
// Invoked when an instance is removed from topology (e.g. etcd deregister).
// Typically forwards to Scheduler::clear_requests_on_failed_instance.
using OnInstanceDeregisteredCallback = std::function<void(
    const std::string& instance_name,
    const std::string& incarnation_id,
    InstanceType cleanup_type)>;

// Cluster instance registry, indices, runtime state, and brpc channels.
class InstanceTopology {
 public:
  virtual ~InstanceTopology() = default;

  virtual InstanceMetaInfo get_instance_info(
      const std::string& instance_name) = 0;

  virtual std::vector<std::string> get_static_decode_list() = 0;

  virtual std::vector<std::string> get_static_prefill_list() = 0;

  virtual std::shared_ptr<brpc::Channel> get_channel(
      const std::string& instance_name) = 0;

  virtual bool record_instance_heartbeat(const std::string& instance_name,
                                         const std::string& incarnation_id) = 0;

  virtual void flip_prefill_to_decode(const std::string& instance_name) = 0;
  virtual void flip_decode_to_prefill(const std::string& instance_name) = 0;
};

// Default implementation extracted from InstanceMgr (etcd discovery, channels,
// link/unlink, suspect / lease-lost state machine). Metrics maps live in
// InstanceMgr; register/deregister invoke callbacks to add/remove them.
class InstanceTopologyImpl final : public InstanceTopology {
 public:
  InstanceTopologyImpl(const Options& options,
                       const std::shared_ptr<EtcdClient>& etcd_client,
                       OnInstanceDeregisteredCallback on_instance_deregistered,
                       AddInstanceMetricsCallback add_instance_metrics,
                       RemoveInstanceMetricsCallback remove_instance_metrics_maps);

  ~InstanceTopologyImpl() override;

  // Load instance keys from etcd and register each (mirrors InstanceMgr::init
  // instance loop). Starts the reconcile thread at the end.
  void init_from_etcd_register_all();

  InstanceMetaInfo get_instance_info(const std::string& instance_name) override;
  std::vector<std::string> get_static_decode_list() override;
  std::vector<std::string> get_static_prefill_list() override;
  std::shared_ptr<brpc::Channel> get_channel(
      const std::string& instance_name) override;
  bool record_instance_heartbeat(const std::string& instance_name,
                                 const std::string& incarnation_id) override;
  void flip_prefill_to_decode(const std::string& instance_name) override;
  void flip_decode_to_prefill(const std::string& instance_name) override;

 private:
  friend class InstanceMgr;
  DISALLOW_COPY_AND_ASSIGN(InstanceTopologyImpl);

  // cluster_mutex_ must already be held (shared or unique). Matches
  // get_static_prefill_list / get_static_decode_list iteration and filters.
  void collect_load_balance_lists_locked(
      std::vector<std::string>* prefill_out,
      std::vector<std::string>* decode_out,
      std::unordered_map<std::string, InstanceMetaInfo>* instance_infos_out,
      const std::function<bool(const InstanceMetaInfo&)>& is_schedulable)
      const;

  bool register_instance(const std::string& name, InstanceMetaInfo& info);
  void deregister_instance(const std::string& name,
                           const std::string& expected_incarnation_id = "");

  bool init_brpc_channel(const std::string& target_uri,
                         std::shared_ptr<brpc::Channel>* out_channel);
  bool probe_instance_health(const std::string& instance_name);
  void reconcile_instance_states();
  void refresh_instance_registration(const std::string& name,
                                     const InstanceMetaInfo& info);
  void mark_instance_suspect(const std::string& name,
                             const std::string& incarnation_id);
  void clear_suspect_instance(const std::string& name,
                              const std::string& incarnation_id = "");
  void update_instance_metainfo(const etcd::Response& response,
                                const uint64_t& prefix_len);

  bool gather_link_operations(
      const InstanceMetaInfo& info,
      std::vector<std::pair<std::string, InstanceMetaInfo>>* out_ops);
  bool run_link_operations(
      const std::vector<std::pair<std::string, InstanceMetaInfo>>& ops);
  void gather_unlink_operations(
      const std::string& name,
      const InstanceMetaInfo& info,
      std::vector<std::pair<std::string, InstanceMetaInfo>>* out_ops);
  void add_instance_to_index(const std::string& name, InstanceMetaInfo& info);
  void remove_instance_from_index(const std::string& name,
                                  const InstanceMetaInfo& info);
  bool call_link_instance(const std::string& target_rpc_addr,
                          const InstanceMetaInfo& peer_info);
  bool call_unlink_instance(const std::string& target_rpc_addr,
                            const InstanceMetaInfo& peer_info);

  // Caller must hold cluster_mutex_ exclusively.
  void remove_channel_and_metrics_maps(const std::string& name);

  Options options_;
  bool exited_ = false;
  std::shared_ptr<EtcdClient> etcd_client_;
  OnInstanceDeregisteredCallback on_instance_deregistered_cb_;
  AddInstanceMetricsCallback add_instance_metrics_cb_;
  RemoveInstanceMetricsCallback remove_instance_metrics_maps_cb_;

  mutable std::shared_mutex cluster_mutex_;
  std::unordered_map<std::string, InstanceMetaInfo> instances_;
  struct SuspectInstanceInfo {
    std::string incarnation_id;
    uint64_t enter_ts_ms = 0;
  };
  std::unordered_map<std::string, SuspectInstanceInfo> suspect_instances_;
  std::vector<std::string> prefill_index_;
  std::vector<std::string> decode_index_;
  std::unordered_map<std::string, std::shared_ptr<brpc::Channel>>
      cached_channels_;

  ThreadPool threadpool_;
  std::unique_ptr<std::thread> state_reconcile_thread_;
};

}  // namespace xllm_service
