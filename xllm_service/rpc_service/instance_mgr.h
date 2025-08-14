#pragma once

#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "common/types.h"
#include "disagg_pd_policy.h"
#include "etcd_client.h"

namespace xllm_service {

class InstanceMgr {
 public:
  explicit InstanceMgr(const RpcServiceConfig& config);
  ~InstanceMgr();
  ErrorCode heartbeat(const std::string& instance_name);

  ErrorCode register_instance(const std::string& instance_name);
  ErrorCode register_instance(const std::string& instance_name,
                              const InstanceMetaInfo& metainfo);
  ErrorCode update_instance_metainfo(const std::string& instance_name,
                                     const InstanceMetaInfo& metainfo);
  InstanceMetaInfo get_instance_info(const std::string& instance_name);

  // select instances(prefill/decode/default etc.) to handle request
  // according the disagg pd policy (or some other policies.).
  InstancesPair select_instances_pair(bool only_prefill = false);

  std::vector<std::string> get_static_decode_list(
      const std::string& instance_name);

 private:
  void internal_init();
  // save instance metainfo to etcd
  void save_persistence_metainfo(const InstanceMetaInfo& metainfo);
  // delete instance metainfo from etcd
  void delete_persistence_metainfo(
      const std::vector<std::string>& instance_names);
  void detect_disconnected_instances();
  void update_instance_timestamp(const std::string& inst_name);

 private:
  RpcServiceConfig config_;
  bool exited_ = false;
  std::mutex inst_mutex_;
  std::unordered_map<std::string, InstanceMetaInfo> instances_;
  std::unique_ptr<std::thread> heartbeat_thread_;

  std::unique_ptr<DisaggPdPolicy> disagg_pd_policy_;

  bool use_etcd_ = false;
  std::unique_ptr<EtcdClient> etcd_client_;
};

}  // namespace xllm_service
