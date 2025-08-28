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

#include <gflags/gflags.h>

DECLARE_string(http_server_host);

DECLARE_int32(http_server_port);

DECLARE_int32(http_server_idle_timeout_s);

DECLARE_int32(http_server_num_threads);

DECLARE_int32(http_server_max_concurrency);

DECLARE_string(rpc_server_host);

DECLARE_int32(rpc_server_port);

DECLARE_int32(rpc_server_idle_timeout_s);

DECLARE_int32(rpc_server_num_threads);

DECLARE_int32(rpc_server_max_concurrency);

DECLARE_string(test_instance_addr);

DECLARE_int32(timeout_ms);

DECLARE_string(listen_addr);

DECLARE_int32(port);

DECLARE_int32(idle_timeout_s);

DECLARE_int32(num_threads);

DECLARE_int32(max_concurrency);

DECLARE_string(etcd_addr);

DECLARE_string(disagg_pd_policy);

DECLARE_int32(detect_disconnected_instance_interval);

DECLARE_int32(block_size);

DECLARE_string(tokenizer_path);

DECLARE_string(model_type);

DECLARE_bool(enable_request_trace);
