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

#include <mutex>
#include <unordered_map>

#include "chat.pb.h"
#include "common/call_data.h"
#include "common/threadpool.h"
#include "common/xllm/output.h"
#include "common/xllm/status.h"
#include "completion.pb.h"
#include "instance_mgr.h"
#include "response_handler.h"
#include "scheduler.h"
#include "xllm_rpc_service.pb.h"

namespace xllm_service {

using OutputCallback = std::function<bool(llm::RequestOutput output)>;

struct ServiceConfig {
  ServiceConfig(bool decode_to_service)
      : enable_decode_response_to_service(decode_to_service) {}

  bool enable_decode_response_to_service = false;
};

class XllmRpcServiceImpl final {
 public:
  XllmRpcServiceImpl(const RpcServiceConfig& rpc_config,
                     const ModelConfig& model_config,
                     const HttpServiceConfig& http_config);
  ~XllmRpcServiceImpl();

  void heartbeat(const proto::HeartbeatRequest* req);

  InstanceMetaInfo get_instance_info(const std::string& instance_name);

  ServiceConfig get_config();

  std::vector<std::string> get_static_decode_list(
      const std::string& prefill_name);

 public:
  // handle generations from prefill/decode instance
  bool handle_generation(const llm::RequestOutput& request_output);

  // register new requests from http service
  // keep http callback util request finished.
  // `handle_generation` will handle response with these callbacks.
  bool record_new_request(std::shared_ptr<ChatCallData> call_data,
                          const std::string& service_request_id,
                          bool stream,
                          const std::string& model,
                          bool include_usage);
  bool record_new_request(std::shared_ptr<CompletionCallData> call_data,
                          const std::string& service_request_id,
                          bool stream,
                          const std::string& model,
                          bool include_usage);
  void finish_request(const std::string& service_request_id);

  bool schedule(const std::string& prompt, SchduleResult* res);

  bool schedule(const ChatMessages& messages, SchduleResult* res);

  std::shared_ptr<brpc::Channel> get_channel(const std::string& target_name);

 private:
  // `request` -> `callback` map
  std::unordered_map<std::string, OutputCallback> callbacks_;
  std::mutex callback_mutex_;

  // use threadpool to handle all RequestOuputs queue
  static constexpr size_t kOutputTheadNum_ = 128;  // magic num
  ThreadPool output_threadpools_[kOutputTheadNum_];
  // A request will be handled in the same thread to guarantee the token's
  // order.
  std::unordered_map<std::string, size_t> remote_requests_output_thread_map_;
  size_t next_thread_idx = 0;
  std::mutex thread_map_mutex_;

  // In disagg pd mode, we support receive generated token from
  // prefill or from decode directly.
  // 1.
  // [service] ---req---> [prefill] ---req---> [decode]
  // [service] <---first resp--- [prefill] ---first resp---> [decode]
  // [service] <---resp--- [prefill] <---resp--- [decode]
  //
  // 2.
  // [service] ---req---> [prefill] ---req---> [decode]
  // [service] <---first resp-- [prefill] --first resp---> [decode]
  // [service] <---resp-- [decode]
  //
  bool enable_decode_response_to_service_ = false;

  // used when receive token from decode instance.
  ResponseHandler response_handler_;

  // instance discovery by register to etcd
  std::unique_ptr<Scheduler> scheduler_;
};

// parse proto data and call XllmRpcService
class XllmRpcService : public proto::XllmRpcService {
 public:
  explicit XllmRpcService(std::shared_ptr<XllmRpcServiceImpl> service);
  virtual ~XllmRpcService();

  virtual void Hello(google::protobuf::RpcController* cntl_base,
                     const proto::Empty* req,
                     proto::Status* resp,
                     google::protobuf::Closure* done) override;

  virtual void Heartbeat(google::protobuf::RpcController* cntl_base,
                         const proto::HeartbeatRequest* req,
                         proto::Status* resp,
                         google::protobuf::Closure* done) override;

  virtual void GetInstanceInfo(google::protobuf::RpcController* cntl_base,
                               const proto::InstanceID* req,
                               proto::InstanceMetaInfo* resp,
                               google::protobuf::Closure* done) override;

  virtual void GetStaticDecodeList(google::protobuf::RpcController* cntl_base,
                                   const proto::InstanceID* req,
                                   proto::InstanceIDs* resp,
                                   google::protobuf::Closure* done) override;

  // xllm service receive response from decode instance directly in disagg pd
  // mode. This can eliminate the cost brought by forwarding through prefill.
  virtual void Generations(google::protobuf::RpcController* cntl_base,
                           const proto::DisaggStreamGenerations* req,
                           proto::StatusSet* resp,
                           google::protobuf::Closure* done) override;

  virtual void GetConfig(google::protobuf::RpcController* cntl_base,
                         const proto::Empty* req,
                         proto::ServiceConfig* resp,
                         google::protobuf::Closure* done) override;

 private:
  std::shared_ptr<XllmRpcServiceImpl> xllm_service_;
};

}  // namespace xllm_service
