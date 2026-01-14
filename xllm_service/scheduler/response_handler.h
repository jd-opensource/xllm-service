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

#include "common/call_data.h"
#include "common/threadpool.h"
#include "common/xllm/output.h"
#include "common/xllm/status.h"

namespace xllm_service {

class ResponseHandler final {
 public:
  ResponseHandler() = default;
  ~ResponseHandler() = default;

  bool send_delta_to_client(std::shared_ptr<ChatCallData> call_data,
                            bool include_usage,
                            int64_t created_time,
                            const std::string& model,
                            const xllm::RequestOutput& output);
  bool send_result_to_client(std::shared_ptr<ChatCallData> call_data,
                             int64_t created_time,
                             const std::string& model,
                             const xllm::RequestOutput& req_output);

  bool send_delta_to_client(std::shared_ptr<CompletionCallData> call_data,
                            bool include_usage,
                            int64_t created_time,
                            const std::string& model,
                            const xllm::RequestOutput& output);
  bool send_result_to_client(std::shared_ptr<CompletionCallData> call_data,
                             int64_t created_time,
                             const std::string& model,
                             const xllm::RequestOutput& req_output);
};

}  // namespace xllm_service
