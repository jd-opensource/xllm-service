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

#include "http_service/service.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <brpc/controller.h>
#include <brpc/progressive_reader.h>
#include <glog/logging.h>

#include <functional>
#include <nlohmann/json.hpp>

#include "chat.pb.h"
#include "common/call_data.h"
#include "common/closure_guard.h"
#include "common/utils.h"
#include "common/xllm/uuid.h"
#include "completion.pb.h"
#include "scheduler/scheduler.h"

namespace xllm_service {

namespace {
thread_local llm::ShortUUID short_uuid;
std::string generate_service_request_id(const std::string& method) {
  std::stringstream ss;
  ss << method << "-";
  ss << std::this_thread::get_id();
  ss << "-";
  ss << short_uuid.random();
  return ss.str();
}
}  // namespace

XllmHttpServiceImpl::XllmHttpServiceImpl(const Options& options,
                                         Scheduler* scheduler)
    : options_(options), scheduler_(scheduler) {
  enable_decode_response_to_service_ =
      utils::get_bool_env("ENABLE_DECODE_RESPONSE_TO_SERVICE", false);
  initialized_ = true;
  thread_pool_ = std::make_unique<ThreadPool>(options_.num_threads());
  request_tracer_ =
      std::make_unique<RequestTracer>(options_.enable_request_trace());
}

XllmHttpServiceImpl::~XllmHttpServiceImpl() {}

void XllmHttpServiceImpl::Hello(::google::protobuf::RpcController* controller,
                                const proto::HttpHelloRequest* request,
                                proto::HttpHelloResponse* response,
                                ::google::protobuf::Closure* done) {
  assert(initialized_);
  brpc::ClosureGuard done_guard(done);
  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    return;
  }

  LOG(INFO) << "Get request: " << request->ping();

  response->set_pong(request->ping());
}

namespace {
template <typename T>
void handle_non_stream_response(brpc::Controller* cntl,
                                std::shared_ptr<T> call_data) {
  std::unique_ptr<brpc::Controller> cntl_guard(cntl);
  if (cntl->Failed()) {
    LOG(WARNING) << "Fail to send stream generation, " << cntl->ErrorText();
    return;
  }
  call_data->write_and_finish(cntl->response_attachment().to_string());
}

template <typename T>
void handle_first_response(brpc::Controller* cntl,
                           std::shared_ptr<T> call_data,
                           bool stream) {
  std::unique_ptr<brpc::Controller> cntl_guard(cntl);
  if (cntl->Failed()) {
    LOG(WARNING) << "Fail to send stream generation, " << cntl->ErrorText();
    return;
  }
  if (stream) {
    // write first token from prefill
    call_data->write(cntl->response_attachment().to_string());
  }
  // non-stream, all generated tokens will be sent from decode via rpc service.
}

template <typename T>
class CustomProgressiveReader : public brpc::ProgressiveReader {
 public:
  explicit CustomProgressiveReader(brpc::Controller* redirect_cntl,
                                   std::shared_ptr<T> call_data)
      : redirect_cntl_(redirect_cntl), call_data_(call_data) {}

  virtual ~CustomProgressiveReader() { delete redirect_cntl_; }

  // Called when one part was read.
  // Error returned is treated as *permanent* and the socket where the
  // data was read will be closed.
  // A temporary error may be handled by blocking this function, which
  // may block the HTTP parsing on the socket.
  virtual butil::Status OnReadOnePart(const void* data, size_t length) {
    call_data_->write(std::string((char*)data, length));
    return butil::Status::OK();
  }

  // Called when there's nothing to read anymore. The `status' is a hint for
  // why this method is called.
  // - status.ok(): the message is complete and successfully consumed.
  // - otherwise: socket was broken or OnReadOnePart() failed.
  // This method will be called once and only once. No other methods will
  // be called after. User can release the memory of this object inside.
  virtual void OnEndOfMessage(const butil::Status& status) { delete this; }

 private:
  brpc::Controller* redirect_cntl_ = nullptr;
  std::shared_ptr<T> call_data_;
};
}  // namespace

template <typename T>
void XllmHttpServiceImpl::handle(std::shared_ptr<T> call_data,
                                 const std::string& req_attachment,
                                 const std::string& service_request_id,
                                 bool stream,
                                 const std::string& model,
                                 bool include_usage,
                                 const std::string& target_uri,
                                 const std::string& method) {
  // record request when enable_decode_response_to_service.
  if (enable_decode_response_to_service_) {
    bool success = scheduler_->record_new_request(
        call_data, service_request_id, stream, model, include_usage);
    if (!success) {
      LOG(ERROR) << "rpc service add new request error: " << service_request_id;
      call_data->finish_with_error("Internal runtime error.");
      scheduler_->finish_request(service_request_id);
      return;
    }
  }

  // async redistribute the request and wait the response
  // TODO: optimize the thread pool to async mode.
  brpc::Channel* channel_ptr = scheduler_->get_channel(target_uri).get();

  // send request to prefill instance.
  thread_pool_->schedule([this,
                          service_request_id,
                          stream,
                          req_attachment = std::move(req_attachment),
                          call_data,
                          channel_ptr,
                          target_uri = target_uri + method]() {
    brpc::Controller* redirect_cntl = new brpc::Controller();
    redirect_cntl->http_request().uri() = target_uri.c_str();
    redirect_cntl->http_request().set_method(brpc::HTTP_METHOD_POST);

    // redirect the input request content
    redirect_cntl->request_attachment().append(req_attachment);

    // 1. tokens will be received via rpc channel.
    //
    if (enable_decode_response_to_service_) {
      google::protobuf::Closure* done = brpc::NewCallback(
          &handle_first_response<T>, redirect_cntl, call_data, stream);
      channel_ptr->CallMethod(NULL, redirect_cntl, NULL, NULL, done);
      if (redirect_cntl->Failed()) {
        LOG(ERROR) << "Redirect to instance error: "
                   << redirect_cntl->ErrorText();
        call_data->finish_with_error(redirect_cntl->ErrorText());
        scheduler_->finish_request(service_request_id);
        delete done;
        delete redirect_cntl;
        return;
      }
      return;
    }

    // 2. tokens will be received via http channel.
    //
    if (stream) {
      // receive tokens in progressive mode.
      redirect_cntl->response_will_be_read_progressively();

      // Because `done'(last parameter) is NULL, this function waits until
      // the response comes back or error occurs(including timeout).
      channel_ptr->CallMethod(NULL, redirect_cntl, NULL, NULL, NULL);
      if (redirect_cntl->Failed()) {
        LOG(ERROR) << "Redirect to instance error: "
                   << redirect_cntl->ErrorText();
        call_data->finish_with_error(redirect_cntl->ErrorText());
        delete redirect_cntl;
        return;
      }
      auto reader = new CustomProgressiveReader<T>(redirect_cntl, call_data);
      // redirect_cntl and reader will be deleted in CustomProgressiveReader.
      redirect_cntl->ReadProgressiveAttachmentBy(reader);
    } else {
      google::protobuf::Closure* done = brpc::NewCallback(
          &handle_non_stream_response<T>, redirect_cntl, call_data);
      channel_ptr->CallMethod(NULL, redirect_cntl, NULL, NULL, done);
      if (redirect_cntl->Failed()) {
        LOG(ERROR) << "Redirect to instance error: "
                   << redirect_cntl->ErrorText();
        call_data->finish_with_error(redirect_cntl->ErrorText());
        delete done;
        delete redirect_cntl;
        return;
      }
    }
  });
}

void XllmHttpServiceImpl::handle_v1_completions(
    std::shared_ptr<CompletionCallData> call_data,
    const std::string& req_attachment,
    const std::string& service_request_id,
    bool stream,
    const std::string& model,
    bool include_usage,
    const std::string& target_uri) {
  handle(call_data,
         req_attachment,
         service_request_id,
         stream,
         model,
         include_usage,
         target_uri,
         "/v1/completions");
}

void XllmHttpServiceImpl::handle_v1_chat_completions(
    std::shared_ptr<ChatCallData> call_data,
    const std::string& req_attachment,
    const std::string& service_request_id,
    bool stream,
    const std::string& model,
    bool include_usage,
    const std::string& target_uri) {
  handle(call_data,
         req_attachment,
         service_request_id,
         stream,
         model,
         include_usage,
         target_uri,
         "/v1/chat/completions");
}

void XllmHttpServiceImpl::post_serving(
    const std::string& serving_method,
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  assert(initialized_);
  ClosureGuard done_guard(done);
  auto cntl = reinterpret_cast<brpc::Controller*>(controller);

  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    cntl->SetFailed("brpc request | respose | controller is null");
    return;
  }

  nlohmann::json json_value;
  try {
    json_value = nlohmann::json::parse(cntl->request_attachment().to_string());
  } catch (const std::exception& e) {
    std::stringstream ss;
    ss << "Json parse request failed, error: " << e.what() << std::endl;
    LOG(ERROR) << ss.str();
    cntl->SetFailed(ss.str());
    return;
  }
  bool stream = false;
  if (json_value.contains("stream")) {
    try {
      stream = json_value.at("stream").get<bool>();
    } catch (const std::exception& e) {
      LOG(ERROR)
          << "Invalid args(stream) type in request, required bool type value.";
      cntl->SetFailed(
          "Invalid args(stream) type in request, required bool type value.");
      return;
    }
  }
  std::string model = json_value.at("model").get<std::string>();
  bool include_usage = false;
  if (json_value.contains("stream_options")) {
    try {
      include_usage =
          json_value["stream_options"].at("include_usage").get<bool>();
    } catch (const std::exception& e) {
      LOG(ERROR) << "Invalid args(include_usage) type in request, required "
                    "bool type value.";
      cntl->SetFailed(
          "Invalid args(include_usage) type in request, required "
          "bool type value.");
      return;
    }
  }
  // TODO: add `created_time` fileds etc.
  // create xllm_service request_id: service_request_id
  std::string service_request_id = generate_service_request_id(serving_method);
  json_value["service_request_id"] = service_request_id;

  std::function<void(const std::string&)> trace_callback;
  if (options_.enable_request_trace()) {
    trace_callback = [this, service_request_id](const std::string& message) {
      request_tracer_->log(service_request_id, message);
    };
  } else {
    trace_callback = nullptr;
  }

  ScheduleResult schedule_res;
  if (serving_method == "/v1/completions") {
    if (json_value.contains("prompt")) {
      if (!scheduler_->schedule(json_value.at("prompt").get<std::string>(),
                                &schedule_res)) {
        cntl->SetFailed("Schedule fail!");
        LOG(ERROR) << "XllmRpcServiceImpl::schedule error!";
        return;
      }
    } else {
      cntl->SetFailed("Input has no prompt!");
      LOG(ERROR) << "Input has no prompt!";
      return;
    }
    json_value["token_ids"] = std::move(schedule_res.token_ids);
    json_value["routing"] = std::move(schedule_res.routing.serialize_to_json());

    std::string req_attachment = json_value.dump();
    auto arena = response->GetArena();
    auto resp_pb =
        google::protobuf::Arena::CreateMessage<llm::proto::CompletionResponse>(
            arena);
    auto call_data = std::make_shared<CompletionCallData>(
        cntl, stream, done_guard.release(), resp_pb);
    handle_v1_completions(call_data,
                          req_attachment,
                          service_request_id,
                          stream,
                          model,
                          include_usage,
                          schedule_res.routing.prefill_name);
  } else if (serving_method == "/v1/chat/completions") {
    if (json_value.contains("messages") && json_value["messages"].is_array()) {
      ChatMessages messages;
      try {
        const auto& msgs = json_value["messages"];
        messages.reserve(msgs.size());
        for (const auto& msg : msgs) {
          if (msg.contains("role") && msg["role"].is_string() &&
              msg.contains("content") && msg["content"].is_string()) {
            messages.emplace_back(msg["role"].get<std::string>(),
                                  msg["content"].get<std::string>());
          }
        }
      } catch (const nlohmann::json::exception& e) {
        cntl->SetFailed("Parse request fail, Invalid messages!");
        LOG(ERROR) << "Parse request fail, Invalid messages!";
        return;
      }

      if (!scheduler_->schedule(messages, &schedule_res)) {
        cntl->SetFailed("Schedule fail!");
        LOG(ERROR) << "XllmRpcServiceImpl::schedule error!";
        return;
      }
    } else {
      cntl->SetFailed("Input has no messages!");
      LOG(ERROR) << "Input has no messages!";
      return;
    }
    json_value["token_ids"] = std::move(schedule_res.token_ids);
    json_value["routing"] = std::move(schedule_res.routing.serialize_to_json());

    std::string req_attachment = json_value.dump();
    auto arena = response->GetArena();
    auto resp_pb =
        google::protobuf::Arena::CreateMessage<llm::proto::ChatResponse>(arena);
    auto call_data = std::make_shared<ChatCallData>(
        cntl, stream, done_guard.release(), resp_pb);
    handle_v1_chat_completions(call_data,
                               req_attachment,
                               service_request_id,
                               stream,
                               model,
                               include_usage,
                               schedule_res.routing.prefill_name);
  } else {
    LOG(ERROR) << "Not supported method: " << serving_method;
    cntl->SetFailed("Not supported method: " + serving_method);
    return;
  }
}

namespace {
void handle_get_response(brpc::Controller* cntl,
                         std::shared_ptr<CompletionCallData> call_data,
                         google::protobuf::Closure* done) {
  std::unique_ptr<brpc::Controller> cntl_guard(cntl);
  std::unique_ptr<google::protobuf::Closure> done_guard(done);
  if (cntl->Failed()) {
    LOG(WARNING) << "Fail to send stream generation, " << cntl->ErrorText();
    return;
  }
  call_data->write_and_finish(cntl->response_attachment().to_string());
}
}  // namespace

void XllmHttpServiceImpl::get_serving(
    const std::string& serving_method,
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  assert(initialized_);
  ClosureGuard done_guard(done);
  auto cntl = reinterpret_cast<brpc::Controller*>(controller);

  if (!request || !response || !controller) {
    LOG(ERROR) << "brpc request | respose | controller is null";
    cntl->SetFailed("brpc request | respose | controller is null");
    return;
  }

  // auto call_data = std::make_shared<StreamCallData>(cntl, false,
  // done_guard.release());
  auto call_data = std::make_shared<CompletionCallData>(
      cntl, false, done_guard.release(), nullptr);

  ScheduleResult schedule_res;
  if (!scheduler_->schedule("", &schedule_res)) {
    cntl->SetFailed("Schedule fail!");
    LOG(ERROR) << "XllmRpcServiceImpl::schedule error!";
    return;
  }

  brpc::Channel* channel_ptr =
      scheduler_->get_channel(schedule_res.routing.prefill_name).get();
  std::string target_uri = schedule_res.routing.prefill_name + serving_method;

  thread_pool_->schedule(
      [/*req_attachment, */ call_data, cntl, channel_ptr, target_uri]() {
        brpc::Controller* redirect_cntl = new brpc::Controller();
        redirect_cntl->http_request().uri() = target_uri.c_str();
        redirect_cntl->http_request().set_method(brpc::HTTP_METHOD_GET);

        google::protobuf::Closure* done = brpc::NewCallback(
            &handle_get_response, redirect_cntl, call_data, done);

        // Because `done'(last parameter) is NULL, this function waits until
        // the response comes back or error occurs(including timeout).
        channel_ptr->CallMethod(NULL, redirect_cntl, NULL, NULL, done);
        if (redirect_cntl->Failed()) {
          LOG(ERROR) << "Redirect to instance error: "
                     << redirect_cntl->ErrorText();
          call_data->finish_with_error(redirect_cntl->ErrorText());
          delete done;
          delete redirect_cntl;
        }
      });
}

void XllmHttpServiceImpl::Completions(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  post_serving("/v1/completions", controller, request, response, done);
}

void XllmHttpServiceImpl::ChatCompletions(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  post_serving("/v1/chat/completions", controller, request, response, done);
}

void XllmHttpServiceImpl::Embeddings(
    ::google::protobuf::RpcController* controller,
    const proto::HttpRequest* request,
    proto::HttpResponse* response,
    ::google::protobuf::Closure* done) {
  post_serving("/v1/embeddings", controller, request, response, done);
}

void XllmHttpServiceImpl::Models(::google::protobuf::RpcController* controller,
                                 const proto::HttpRequest* request,
                                 proto::HttpResponse* response,
                                 ::google::protobuf::Closure* done) {
  get_serving("/v1/models", controller, request, response, done);
}

void XllmHttpServiceImpl::Metrics(::google::protobuf::RpcController* controller,
                                  const proto::HttpRequest* request,
                                  proto::HttpResponse* response,
                                  ::google::protobuf::Closure* done) {
  get_serving("/metrics", controller, request, response, done);
}

}  // namespace xllm_service
