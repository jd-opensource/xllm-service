#pragma once

#include <mutex>
#include <nlohmann/json.hpp>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "chat_template/jinja_chat_template.h"
#include "common/hash_util.h"
#include "common/macros.h"
#include "common/types.h"
#include "etcd_client/etcd_client.h"
#include "loadbalance_policy/loadbalance_policy.h"
#include "managers/global_kvcache_mgr.h"
#include "managers/instance_mgr.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/tokenizer_args.h"
#include "xllm_rpc_service.pb.h"

namespace xllm_service {

class Scheduler {
 public:
  explicit Scheduler(const RpcServiceConfig& rpc_config,
                     const ModelConfig& model_config,
                     const HttpServiceConfig& http_config);

  ~Scheduler();

  bool schedule(const ChatMessages& messages, ScheduleResult* res);

  bool schedule(const std::string& prompt, ScheduleResult* res);

  std::shared_ptr<brpc::Channel> get_channel(const std::string& target_name);

  InstanceMetaInfo get_instance_info(const std::string& instance_name);

  std::vector<std::string> get_static_decode_list(
      const std::string& instance_name);

  void handle_instance_heartbeat(const proto::HeartbeatRequest* req);

  void exited() { exited_ = true; }

 private:
  DISALLOW_COPY_AND_ASSIGN(Scheduler);

  void update_master_service_heartbeat();

  void handle_master_service_watch(const etcd::Response& response,
                                   const uint64_t& prefix_len);

  Tokenizer* get_tls_tokenizer();

 private:
  bool exited_ = false;

  bool is_master_service_ = false;

  TokenizerArgs tokenizer_args_;

  RpcServiceConfig rpc_config_;

  ModelConfig model_config_;

  HttpServiceConfig http_config_;

  // chat template instance
  std::unique_ptr<JinjaChatTemplate> chat_template_;

  std::shared_ptr<EtcdClient> etcd_client_;

  std::unique_ptr<Tokenizer> tokenizer_;

  std::shared_ptr<InstanceMgr> instance_mgr_;

  std::shared_ptr<GlobalKVCacheMgr> global_kvcache_mgr_;

  std::unique_ptr<LoadBalancePolicy> lb_policy_;

  std::unique_ptr<std::thread> heartbeat_thread_;
};

}  // namespace xllm_service
