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
#include "xllm_rpc_service.pb.h"

namespace xllm_service {

using CompletionCallData = StreamCallData<llm::proto::CompletionRequest,
                                          llm::proto::CompletionResponse>;

using ChatCallData =
    StreamCallData<llm::proto::ChatRequest, llm::proto::ChatResponse>;

using OutputCallback = std::function<bool(llm::RequestOutput output)>;

class ResponseHandler final {
 public:
  ResponseHandler() = default;
  ~ResponseHandler() = default;

  bool send_delta_to_client(std::shared_ptr<ChatCallData> call_data,
                            std::unordered_set<size_t>* first_message_sent,
                            bool include_usage,
                            const std::string& request_id,
                            int64_t created_time,
                            const std::string& model,
                            const llm::RequestOutput& output);
  bool send_result_to_client(std::shared_ptr<ChatCallData> call_data,
                             const std::string& request_id,
                             int64_t created_time,
                             const std::string& model,
                             const llm::RequestOutput& req_output);

  bool send_delta_to_client(std::shared_ptr<CompletionCallData> call_data,
                            bool include_usage,
                            const std::string& request_id,
                            int64_t created_time,
                            const std::string& model,
                            const llm::RequestOutput& output);
  bool send_result_to_client(std::shared_ptr<CompletionCallData> call_data,
                             const std::string& request_id,
                             int64_t created_time,
                             const std::string& model,
                             const llm::RequestOutput& req_output);
};

}  // namespace xllm_service
