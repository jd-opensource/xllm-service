#include "scheduler.h"

#include <algorithm>
#include <array>
#include <nlohmann/json.hpp>

#include "chat_template/jinja_chat_template.h"
#include "common.pb.h"
#include "common/hash_util.h"
#include "loadbalance_policy/cache_aware_routing.h"
#include "loadbalance_policy/round_robin.h"
#include "tokenizer/tokenizer_factory.h"

static constexpr int kHeartbeatInterval = 3;  // in seconds
static std::string ETCD_MASTER_SERVICE_KEY = "XLLM:SERVICE:MASTER";

namespace xllm_service {

Scheduler::Scheduler(const RpcServiceConfig& rpc_config,
                     const ModelConfig& model_config,
                     const HttpServiceConfig& http_config)
    : rpc_config_(rpc_config),
      model_config_(model_config),
      http_config_(http_config) {
  tokenizer_ = TokenizerFactory::create_tokenizer(model_config_.tokenizer_path,
                                                  &tokenizer_args_);
  chat_template_ = std::make_unique<JinjaChatTemplate>(tokenizer_args_);

  etcd_client_ = std::make_shared<EtcdClient>(rpc_config_.etcd_addr);

  if (!etcd_client_->get(ETCD_MASTER_SERVICE_KEY, nullptr)) {
    is_master_service_ = etcd_client_->set(
        ETCD_MASTER_SERVICE_KEY, rpc_config_.service_name, kHeartbeatInterval);
    LOG(INFO) << "Set current service as master!";
  }

  instance_mgr_ = std::make_shared<InstanceMgr>(
      etcd_client_, http_config_, is_master_service_);

  global_kvcache_mgr_ = std::make_shared<GlobalKVCacheMgr>(
      etcd_client_, model_config_, is_master_service_);

  if (rpc_config_.load_balance_policy == "CAR") {
    lb_policy_ =
        std::make_unique<CacheAwareRouting>(instance_mgr_, global_kvcache_mgr_);
  } else {
    lb_policy_ =
        std::make_unique<RoundRobin>(instance_mgr_, global_kvcache_mgr_);
  }

  if (is_master_service_) {
    heartbeat_thread_ = std::make_unique<std::thread>(
        &Scheduler::update_master_service_heartbeat, this);
  } else {
    auto handle_master = std::bind(&Scheduler::handle_master_service_watch,
                                   this,
                                   std::placeholders::_1,
                                   std::placeholders::_2);
    etcd_client_->add_watch(ETCD_MASTER_SERVICE_KEY, handle_master);
  }
}

Scheduler::~Scheduler() { etcd_client_->stop_watch(); }

bool Scheduler::schedule(const ChatMessages& messages, ScheduleResult* res) {
  if (chat_template_ == nullptr) {
    LOG(ERROR) << "Chat template has not configured for model type: "
               << model_config_.model_type;
    return false;
  }

  auto prompt = chat_template_->apply(messages);
  if (!prompt.has_value()) {
    LOG(ERROR) << "Failed to construct prompt from messages";
    return false;
  }

  return schedule(prompt.value(), res);
}

bool Scheduler::schedule(const std::string& prompt, ScheduleResult* res) {
  if (prompt.size() != 0) {
    if (!get_tls_tokenizer()->encode(prompt, &res->token_ids)) {
      LOG(ERROR) << "Encode prompt faill: " << prompt;
      return false;
    }
  }
  auto ret = lb_policy_->select_instances_pair(res);
  DLOG(INFO) << res->routing.debug_string();

  return ret;
}

std::shared_ptr<brpc::Channel> Scheduler::get_channel(
    const std::string& target_name) {
  return instance_mgr_->get_channel(target_name);
}

void Scheduler::update_master_service_heartbeat() {
  while (!exited_) {
    std::this_thread::sleep_for(std::chrono::seconds(kHeartbeatInterval));

    global_kvcache_mgr_->upload_kvcache();

    instance_mgr_->upload_load_metrics();
  }
}

void Scheduler::handle_instance_heartbeat(const proto::HeartbeatRequest* req) {
  if (exited_) {
    return;
  }
  global_kvcache_mgr_->record_updated_kvcaches(req->name(), req->cache_event());
  instance_mgr_->record_load_metrics_update(req->name(), req->load_metrics());
}

void Scheduler::handle_master_service_watch(const etcd::Response& response,
                                            const uint64_t& prefix_len) {
  if (exited_ || response.events().empty()) {
    return;
  }

  if (etcd_client_->set(ETCD_MASTER_SERVICE_KEY,
                        rpc_config_.service_name,
                        kHeartbeatInterval)) {
    is_master_service_ = true;

    heartbeat_thread_ = std::make_unique<std::thread>(
        &Scheduler::update_master_service_heartbeat, this);

    global_kvcache_mgr_->set_as_master();
    instance_mgr_->set_as_master();
  }
}

InstanceMetaInfo Scheduler::get_instance_info(
    const std::string& instance_name) {
  return instance_mgr_->get_instance_info(instance_name);
}

std::vector<std::string> Scheduler::get_static_decode_list(
    const std::string& instance_name) {
  return instance_mgr_->get_static_decode_list(instance_name);
}

Tokenizer* Scheduler::get_tls_tokenizer() {
  thread_local std::unique_ptr<Tokenizer> tls_tokenizer(tokenizer_->clone());
  return tls_tokenizer.get();
}

}  // namespace xllm_service