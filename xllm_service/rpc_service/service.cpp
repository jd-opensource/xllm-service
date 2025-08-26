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

#include "rpc_service/service.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <brpc/closure_guard.h>

#include "common/types.h"
#include "common/utils.h"
#include "common/xllm/status.h"

namespace xllm_service {

namespace {
grpc::StatusCode to_grpc_status_code(llm::StatusCode code) {
  switch (code) {
    case llm::StatusCode::OK:
      return grpc::StatusCode::OK;
    case llm::StatusCode::CANCELLED:
      return grpc::StatusCode::CANCELLED;
    case llm::StatusCode::UNKNOWN:
      return grpc::StatusCode::UNKNOWN;
    case llm::StatusCode::INVALID_ARGUMENT:
      return grpc::StatusCode::INVALID_ARGUMENT;
    case llm::StatusCode::DEADLINE_EXCEEDED:
      return grpc::StatusCode::DEADLINE_EXCEEDED;
    case llm::StatusCode::RESOURCE_EXHAUSTED:
      return grpc::StatusCode::RESOURCE_EXHAUSTED;
    case llm::StatusCode::UNAUTHENTICATED:
      return grpc::StatusCode::UNAUTHENTICATED;
    case llm::StatusCode::UNAVAILABLE:
      return grpc::StatusCode::UNAVAILABLE;
    case llm::StatusCode::UNIMPLEMENTED:
      return grpc::StatusCode::UNIMPLEMENTED;
    default:
      LOG(WARNING) << "Unknown status code: " << static_cast<uint8_t>(code);
  }
  return grpc::StatusCode::UNKNOWN;
}
}  // namespace

XllmRpcServiceImpl::XllmRpcServiceImpl(const RpcServiceConfig& rpc_config,
                                       const ModelConfig& model_config,
                                       const HttpServiceConfig& http_config) {
  enable_decode_response_to_service_ =
      utils::get_bool_env("ENABLE_DECODE_RESPONSE_TO_SERVICE", false);

  scheduler_ =
      std::make_unique<Scheduler>(rpc_config, model_config, http_config);
}

XllmRpcServiceImpl::~XllmRpcServiceImpl() { scheduler_->exited(); }

void XllmRpcServiceImpl::heartbeat(const proto::HeartbeatRequest* req) {
  scheduler_->handle_instance_heartbeat(req);
}

InstanceMetaInfo XllmRpcServiceImpl::get_instance_info(
    const std::string& instance_name) {
  return scheduler_->get_instance_info(instance_name);
}

std::vector<std::string> XllmRpcServiceImpl::get_static_decode_list(
    const std::string& instance_name) {
  return scheduler_->get_static_decode_list(instance_name);
}

bool XllmRpcServiceImpl::schedule(const std::string& prompt,
                                  ScheduleResult* res) {
  return scheduler_->schedule(prompt, res);
}

bool XllmRpcServiceImpl::schedule(const ChatMessages& messages,
                                  ScheduleResult* res) {
  return scheduler_->schedule(messages, res);
}

std::shared_ptr<brpc::Channel> XllmRpcServiceImpl::get_channel(
    const std::string& target_name) {
  return scheduler_->get_channel(target_name);
}

bool XllmRpcServiceImpl::handle_generation(
    const llm::RequestOutput& request_output) {
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

void XllmRpcServiceImpl::finish_request(const std::string& service_request_id) {
  {
    std::lock_guard<std::mutex> guard(callback_mutex_);
    callbacks_.erase(service_request_id);
  }

  {
    std::lock_guard<std::mutex> guard(thread_map_mutex_);
    remote_requests_output_thread_map_.erase(service_request_id);
  }
}

bool XllmRpcServiceImpl::record_new_request(
    std::shared_ptr<ChatCallData> call_data,
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
          return call_data->finish_with_error(
              to_grpc_status_code(status.code()), status.message());
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

bool XllmRpcServiceImpl::record_new_request(
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
          return call_data->finish_with_error(
              to_grpc_status_code(status.code()), status.message());
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

ServiceConfig XllmRpcServiceImpl::get_config() {
  return ServiceConfig(enable_decode_response_to_service_);
}

XllmRpcService::XllmRpcService(std::shared_ptr<XllmRpcServiceImpl> service)
    : xllm_service_(service) {}

XllmRpcService::~XllmRpcService() {}

void XllmRpcService::Hello(google::protobuf::RpcController* cntl_base,
                           const proto::Empty* req,
                           proto::Status* resp,
                           google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  resp->set_ok(true);
}

void XllmRpcService::GetInstanceInfo(google::protobuf::RpcController* cntl_base,
                                     const proto::InstanceID* req,
                                     proto::InstanceMetaInfo* resp,
                                     google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  InstanceMetaInfo metainfo = xllm_service_->get_instance_info(req->name());
  resp->set_name(metainfo.name);
  resp->set_rpc_address(metainfo.rpc_address);
  if (metainfo.type == InstanceType::PREFILL) {
    resp->set_type(proto::InstanceType::PREFILL);
  } else if (metainfo.type == InstanceType::DECODE) {
    resp->set_type(proto::InstanceType::DECODE);
  } else {
    resp->set_type(proto::InstanceType::DEFAULT);
  }
  for (auto& cluster_id : metainfo.cluster_ids) {
    *(resp->mutable_cluster_ids()->Add()) = cluster_id;
  }
  for (auto& addr : metainfo.addrs) {
    *(resp->mutable_addrs()->Add()) = addr;
  }
  for (auto& k_cache_id : metainfo.k_cache_ids) {
    *(resp->mutable_k_cache_ids()->Add()) = k_cache_id;
  }
  for (auto& v_cache_id : metainfo.v_cache_ids) {
    *(resp->mutable_v_cache_ids()->Add()) = v_cache_id;
  }
  resp->set_dp_size(metainfo.dp_size);
}

void XllmRpcService::Heartbeat(google::protobuf::RpcController* cntl_base,
                               const proto::HeartbeatRequest* req,
                               proto::Status* resp,
                               google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  xllm_service_->heartbeat(req);
  resp->set_ok(true);
}

void XllmRpcService::GetStaticDecodeList(
    google::protobuf::RpcController* cntl_base,
    const proto::InstanceID* req,
    proto::InstanceIDs* resp,
    google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  std::vector<std::string> decode_list =
      xllm_service_->get_static_decode_list(req->name());
  for (auto& d : decode_list) {
    *(resp->mutable_names()->Add()) = std::move(d);
  }
}

void XllmRpcService::Generations(google::protobuf::RpcController* cntl_base,
                                 const proto::DisaggStreamGenerations* req,
                                 proto::StatusSet* resp,
                                 google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);

  // TODO: use threadpool here
  //
  for (auto& request : req->gens()) {
    // convert proto request to `RequestOutput`
    llm::RequestOutput request_output;
    request_output.request_id = request.req_id();
    request_output.service_request_id = request.service_req_id();
    if (request.has_gen_status()) {
      request_output.status = llm::Status(
          static_cast<llm::StatusCode>(request.gen_status().status_code()),
          request.gen_status().status_msg());
    }
    if (request.has_usage()) {
      llm::Usage u;
      u.num_prompt_tokens = request.usage().num_prompt_tokens();
      u.num_generated_tokens = request.usage().num_generated_tokens();
      u.num_total_tokens = request.usage().num_total_tokens();
      request_output.usage = std::move(u);
    }
    request_output.finished = request.finished();
    for (auto& output : request.outputs()) {
      llm::SequenceOutput sequence_output;
      sequence_output.index = output.index();
      sequence_output.text = output.text();
      sequence_output.token_ids = std::vector<int32_t>(
          output.token_ids().begin(), output.token_ids().end());
      if (!output.finish_reason().empty()) {
        sequence_output.finish_reason = output.finish_reason();
      }
      if (output.logprobs().size() > 0) {
        std::vector<llm::LogProb> logprobs;
        for (auto& logprob : output.logprobs()) {
          llm::LogProb lp;
          lp.token = logprob.log_prob_data().token();
          lp.token_id = logprob.log_prob_data().token_id();
          lp.logprob = logprob.log_prob_data().logprob();
          lp.finished_token = logprob.log_prob_data().finished_token();
          if (logprob.top_logprobs().size() > 0) {
            std::vector<llm::LogProbData> top_logprobs;
            for (auto& top_logprob : logprob.top_logprobs()) {
              llm::LogProbData lpd;
              lpd.token = top_logprob.token();
              lpd.token_id = top_logprob.token_id();
              lpd.logprob = top_logprob.logprob();
              lpd.finished_token = top_logprob.finished_token();
              top_logprobs.emplace_back(std::move(lpd));
            }
            lp.top_logprobs = std::move(top_logprobs);
          }
          logprobs.emplace_back(std::move(lp));
        }
        sequence_output.logprobs = std::move(logprobs);
      }
      request_output.outputs.emplace_back(std::move(sequence_output));
    }
    resp->mutable_all_status()->Add()->set_ok(
        xllm_service_->handle_generation(request_output));
  }
}

void XllmRpcService::GetConfig(google::protobuf::RpcController* cntl_base,
                               const proto::Empty* req,
                               proto::ServiceConfig* resp,
                               google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  auto config = xllm_service_->get_config();
  resp->set_enable_decode_response_to_service(
      config.enable_decode_response_to_service);
}

}  // namespace xllm_service
