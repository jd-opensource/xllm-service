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

#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "common/macros.h"
#include "common/types.h"
#include "etcd_client.h"
#include "xllm_rpc_service.pb.h"

namespace xllm_service {

class InstanceMgr final {
 public:
  explicit InstanceMgr(const std::shared_ptr<EtcdClient>& etcd_client,
                       const HttpServiceConfig& config,
                       const bool is_master_service);

  ~InstanceMgr();

  InstanceMetaInfo get_instance_info(const std::string& instance_name);

  std::vector<std::string> get_static_decode_list(
      const std::string& instance_name);

  void get_load_metrics(LoadBalanceInfos* infos);

  std::shared_ptr<brpc::Channel> get_channel(const std::string& instance_name);

  void record_load_metrics_update(const std::string& instance_name,
                                  const proto::LoadMetrics& load_metrics);
  bool upload_load_metrics();

  void set_as_master();

 private:
  DISALLOW_COPY_AND_ASSIGN(InstanceMgr);

  void init();

  bool create_channel(const std::string& target_uri);
  // use etcd as ServiceDiscovery
  void handle_instance_metainfo_watch(const etcd::Response& response,
                                      const uint64_t& prefix_len);

  void handle_load_metrics_watch(const etcd::Response& response,
                                 const uint64_t prefix_len);

 private:
  bool exited_ = false;
  bool use_etcd_ = false;
  std::atomic_bool is_master_service_ = false;

  RpcServiceConfig config_;
  HttpServiceConfig http_config_;

  std::shared_ptr<EtcdClient> etcd_client_;

  std::shared_mutex inst_mutex_;
  std::unordered_map<std::string, InstanceMetaInfo> instances_;

  std::shared_mutex load_metric_mutex_;
  std::unordered_map<std::string, LoadMetrics> load_metrics_;
  std::unordered_map<std::string, std::shared_ptr<brpc::Channel>>
      cached_channels_;
  std::unordered_set<std::string> zombie_nodes_;

  std::mutex update_mutex_;
  std::unordered_map<std::string, LoadMetrics> updated_metrics_;
  std::unordered_set<std::string> removed_instance_;
};

}  // namespace xllm_service
