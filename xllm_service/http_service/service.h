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

#include <iostream>
#include <mutex>

#include "chat.pb.h"
#include "common/call_data.h"
#include "common/threadpool.h"
#include "common/types.h"
#include "completion.pb.h"
#include "request_tracer.h"
#include "xllm_http_service.pb.h"

namespace xllm_service {

using CompletionCallData = StreamCallData<llm::proto::CompletionRequest,
                                          llm::proto::CompletionResponse>;

using ChatCallData =
    StreamCallData<llm::proto::ChatRequest, llm::proto::ChatResponse>;

class XllmRpcServiceImpl;

class XllmHttpServiceImpl : public proto::XllmHttpService {
 public:
  XllmHttpServiceImpl(const HttpServiceConfig& config);
  XllmHttpServiceImpl(std::shared_ptr<XllmRpcServiceImpl> rpc_service,
                      const HttpServiceConfig& config);
  ~XllmHttpServiceImpl();

  void Hello(::google::protobuf::RpcController* controller,
             const proto::HttpHelloRequest* request,
             proto::HttpHelloResponse* response,
             ::google::protobuf::Closure* done) override;

  void Completions(::google::protobuf::RpcController* controller,
                   const proto::HttpRequest* request,
                   proto::HttpResponse* response,
                   ::google::protobuf::Closure* done) override;

  void ChatCompletions(::google::protobuf::RpcController* controller,
                       const proto::HttpRequest* request,
                       proto::HttpResponse* response,
                       ::google::protobuf::Closure* done) override;

  void Embeddings(::google::protobuf::RpcController* controller,
                  const proto::HttpRequest* request,
                  proto::HttpResponse* response,
                  ::google::protobuf::Closure* done) override;

  void Models(::google::protobuf::RpcController* controller,
              const proto::HttpRequest* request,
              proto::HttpResponse* response,
              ::google::protobuf::Closure* done) override;

  void Metrics(::google::protobuf::RpcController* controller,
               const proto::HttpRequest* request,
               proto::HttpResponse* response,
               ::google::protobuf::Closure* done) override;

 private:
  bool create_channel(const std::string& target_uri);
  // only prefill is true means only prefill instance is returned
  std::string get_redirect_uri(bool only_prefill = false);
  void post_serving(const std::string& serving_method,
                    ::google::protobuf::RpcController* controller,
                    const proto::HttpRequest* request,
                    proto::HttpResponse* response,
                    ::google::protobuf::Closure* done);

  template <typename T>
  void handle(std::shared_ptr<T> call_data,
              const std::string& req_attachment,
              const std::string& service_request_id,
              bool stream,
              const std::string& model,
              bool include_usage,
              const std::string& target_uri,
              const std::string& method);

  void handle_v1_chat_completions(std::shared_ptr<ChatCallData> call_data,
                                  const std::string& req_attachment,
                                  const std::string& service_request_id,
                                  bool stream,
                                  const std::string& model,
                                  bool include_usage,
                                  const std::string& target_uri);

  void handle_v1_completions(std::shared_ptr<CompletionCallData> call_data,
                             const std::string& req_attachment,
                             const std::string& service_request_id,
                             bool stream,
                             const std::string& model,
                             bool include_usage,
                             const std::string& target_uri);

  void get_serving(const std::string& serving_method,
                   ::google::protobuf::RpcController* controller,
                   const proto::HttpRequest* request,
                   proto::HttpResponse* response,
                   ::google::protobuf::Closure* done);

 private:
  bool initialized_ = false;
  HttpServiceConfig config_;

  std::shared_ptr<XllmRpcServiceImpl> rpc_service_;

  std::unique_ptr<RequestTracer> request_tracer_;
  // uri -> channel
  // e.g. 127.0.0.1:9999/v1/completions -> channel1
  //      127.0.0.1:9999/v1/chat/completions -> channel2
  // NOTE: different methods to one instance has different channels
  std::unordered_map<std::string, brpc::Channel*> cached_channels_;
  std::unique_ptr<ThreadPool> thread_pool_;
  std::mutex channel_mutex_;

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
};

}  // namespace xllm_service
