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

#include <brpc/server.h>

#include <thread>

#include "api_service/api_service.h"
#include "common/options.h"
#include "instances/instance_service.h"
#include "runtime/executor.h"

namespace xllm_service {

class Master {
 public:
  explicit Master(const Options& options);
  ~Master();

  bool start();
  void stop();

 private:
  bool start_api_server();
  bool start_instance_server();

 private:
  Options options_;

  // Executor for executing requests and instances
  std::unique_ptr<Executor> executor_;

  // 1.For api service
  std::string api_server_address_;
  std::unique_ptr<xllm_service::XllmHttpServiceImpl> api_service_;
  brpc::Server api_server_;
  std::unique_ptr<std::thread> api_server_thread_;

  // 2.For instance service
  std::string instance_server_address_;
  std::unique_ptr<xllm_service::XllmRpcService> instance_service_;
  brpc::Server instance_server_;
  std::unique_ptr<std::thread> instance_server_thread_;
};

}  // namespace xllm_service
