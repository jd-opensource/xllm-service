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

#include "http_service/service.h"
#include "rpc_service/service.h"

namespace xllm_service {

struct ServerOptions {
  // http server options
  std::string http_server_host = "";
  int32_t http_port = 9999;
  int32_t http_idle_timeout_s = -1;
  int32_t http_num_threads = 32;
  int32_t http_max_concurrency = 128;
  bool enable_request_trace = false;

  // rpc server options
  std::string rpc_server_host = "";
  int32_t rpc_port = 9999;
  int32_t rpc_idle_timeout_s = -1;
  int32_t rpc_num_threads = 32;
  int32_t rpc_max_concurrency = 128;
  std::string etcd_addr = "";
  std::string disagg_pd_policy = "RR";
  int32_t detect_disconnected_instance_interval = 15;
  int32_t block_size = 16;
  std::string model_type = "chatglm";
  std::string tokenizer_path = "";
};

class Master {
 public:
  explicit Master(const ServerOptions& server_options);
  ~Master();

  bool start();
  void stop();

 private:
  bool start_http_server();
  bool start_rpc_server();

 private:
  ServerOptions server_options_;

  // 1.For http service
  std::string http_server_address_;
  std::unique_ptr<xllm_service::XllmHttpServiceImpl> http_service_;
  brpc::Server http_server_;
  std::unique_ptr<std::thread> http_server_thread_;

  // 2.For rpc service
  std::string rpc_server_address_;
  std::shared_ptr<xllm_service::XllmRpcServiceImpl> rpc_service_impl_;
  std::unique_ptr<xllm_service::XllmRpcService> rpc_service_;
  brpc::Server rpc_server_;
  std::unique_ptr<std::thread> rpc_server_thread_;
};

}  // namespace xllm_service
