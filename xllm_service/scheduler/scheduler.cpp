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

#include "scheduler/scheduler.h"

#include "common/xllm/status.h"
#include "loadbalance_policy/cache_aware_routing.h"
#include "loadbalance_policy/round_robin.h"
#include "tokenizer/tokenizer_factory.h"

static constexpr int kHeartbeatInterval = 3;  // in seconds
static std::string ETCD_MASTER_SERVICE_KEY = "XLLM:SERVICE:MASTER";

namespace xllm_service {

Scheduler::Scheduler(const Options& options) : options_(options) {
  tokenizer_ = TokenizerFactory::create_tokenizer(options_.tokenizer_path(),
                                                  &tokenizer_args_);
  chat_template_ = std::make_unique<JinjaChatTemplate>(tokenizer_args_);

  etcd_client_ = std::make_shared<EtcdClient>(options_.etcd_addr());
  if (!etcd_client_->get(ETCD_MASTER_SERVICE_KEY, nullptr)) {
    is_master_service_ = etcd_client_->set(
        ETCD_MASTER_SERVICE_KEY, options_.service_name(), kHeartbeatInterval);
    LOG(INFO) << "Set current service as master!";
  }

  instance_mgr_ =
      std::make_unique<InstanceMgr>(options, etcd_client_, is_master_service_);

  global_kvcache_mgr_ = std::make_shared<GlobalKVCacheMgr>(
      options, etcd_client_, is_master_service_);

  if (options.load_balance_policy() == "CAR") {
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
    LOG(ERROR) << "Chat template has not configured.";
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
                        options_.service_name(),
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

bool Scheduler::record_new_request(std::shared_ptr<ChatCallData> call_data,
                                   const std::string& service_request_id,
                                   bool stream,
                                   const std::string& model,
                                   bool include_usage) {
  {
    std::lock_guard<std::mutex> guard(callback_mutex_);
    if (callbacks_.find(service_request_id) != callbacks_.end()) {
      LOG(ERROR) << "The request ID already exists. Requests with the same ID "
                    "are not allowed. "
                 << service_request_id;
      return false;
    }
    callbacks_[service_request_id] =
        [this,
         call_data,
         model,
         stream,
         include_usage,
         first_message_sent = std::unordered_set<size_t>(),
         service_request_id,
         created_time = absl::ToUnixSeconds(absl::Now())](
            const llm::RequestOutput& req_output) mutable -> bool {
      if (req_output.status.has_value()) {
        const auto& status = req_output.status.value();
        if (!status.ok()) {
          return call_data->finish_with_error(status.message());
        }
      }

      if (stream) {
        return response_handler_.send_delta_to_client(call_data,
                                                      &first_message_sent,
                                                      include_usage,
                                                      service_request_id,
                                                      created_time,
                                                      model,
                                                      req_output);
      }

      return response_handler_.send_result_to_client(
          call_data, service_request_id, created_time, model, req_output);
    };
  }

  {
    // allocate thread for the request
    std::lock_guard<std::mutex> guard(thread_map_mutex_);
    remote_requests_output_thread_map_[service_request_id] = next_thread_idx;
    next_thread_idx = (++next_thread_idx) % kOutputTheadNum_;
  }

  return true;
}

bool Scheduler::record_new_request(
    std::shared_ptr<CompletionCallData> call_data,
    const std::string& service_request_id,
    bool stream,
    const std::string& model,
    bool include_usage) {
  {
    std::lock_guard<std::mutex> guard(callback_mutex_);
    if (callbacks_.find(service_request_id) != callbacks_.end()) {
      LOG(ERROR) << "The request ID already exists. Requests with the same ID "
                    "are not allowed. "
                 << service_request_id;
      return false;
    }
    callbacks_[service_request_id] =
        [this,
         call_data,
         model,
         stream,
         include_usage,
         service_request_id,
         created_time = absl::ToUnixSeconds(absl::Now())](
            const llm::RequestOutput& req_output) mutable -> bool {
      if (req_output.status.has_value()) {
        const auto& status = req_output.status.value();
        if (!status.ok()) {
          return call_data->finish_with_error(status.message());
        }
      }

      if (stream) {
        return response_handler_.send_delta_to_client(call_data,
                                                      include_usage,
                                                      service_request_id,
                                                      created_time,
                                                      model,
                                                      req_output);
      }

      return response_handler_.send_result_to_client(
          call_data, service_request_id, created_time, model, req_output);
    };
  }

  {
    // allocate thread for the request
    std::lock_guard<std::mutex> guard(thread_map_mutex_);
    remote_requests_output_thread_map_[service_request_id] = next_thread_idx;
    next_thread_idx = (++next_thread_idx) % kOutputTheadNum_;
  }

  return true;
}

void Scheduler::finish_request(const std::string& service_request_id) {
  {
    std::lock_guard<std::mutex> guard(callback_mutex_);
    callbacks_.erase(service_request_id);
  }

  {
    std::lock_guard<std::mutex> guard(thread_map_mutex_);
    remote_requests_output_thread_map_.erase(service_request_id);
  }
}

bool Scheduler::handle_generation(const llm::RequestOutput& request_output) {
  const std::string& service_request_id = request_output.service_request_id;
  OutputCallback cb;
  {
    std::lock_guard<std::mutex> guard(callback_mutex_);
    auto it = callbacks_.find(service_request_id);
    if (it == callbacks_.end()) {
      LOG(ERROR) << "Can not found the callback for the received request "
                    "output, request id is: "
                 << service_request_id;
      return false;
    }
    cb = it->second;
  }

  size_t req_thread_idx = -1;
  {
    std::lock_guard<std::mutex> guard(thread_map_mutex_);
    auto it = remote_requests_output_thread_map_.find(service_request_id);
    if (it == remote_requests_output_thread_map_.end()) {
      LOG(ERROR) << "Can not found the thread for the received request output, "
                    "request id is: "
                 << service_request_id;
      return false;
    }
    req_thread_idx = it->second;
  }

  output_threadpools_[req_thread_idx].schedule(
      [this,
       service_request_id,
       cb,
       request_output = std::move(request_output)]() mutable {
        if (!cb(request_output) || request_output.finished) {
          finish_request(service_request_id);
        }
      });

  return true;
}

}  // namespace xllm_service