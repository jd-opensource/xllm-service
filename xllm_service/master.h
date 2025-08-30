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

#include "common/options.h"
#include "http_service/service.h"
#include "rpc_service/service.h"
#include "scheduler/scheduler.h"

namespace xllm_service {

class Master {
 public:
  explicit Master(const Options& options);
  ~Master();

  bool start();
  void stop();

 private:
  bool start_http_server();
  bool start_rpc_server();

 private:
  Options options_;

  // Scheduler for scheduling requests and instances
  std::unique_ptr<Scheduler> scheduler_;

  // 1.For http service
  std::string http_server_address_;
  std::unique_ptr<xllm_service::XllmHttpServiceImpl> http_service_;
  brpc::Server http_server_;
  std::unique_ptr<std::thread> http_server_thread_;

  // 2.For rpc service
  std::string rpc_server_address_;
  std::unique_ptr<xllm_service::XllmRpcService> rpc_service_;
  brpc::Server rpc_server_;
  std::unique_ptr<std::thread> rpc_server_thread_;
};

}  // namespace xllm_service
