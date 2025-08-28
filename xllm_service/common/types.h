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

#include <glog/logging.h>

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace xllm_service {

struct HttpServiceConfig {
  int num_threads = 16;
  int timeout_ms = -1;
  std::string test_instance_addr = "";
  bool enable_request_trace = false;
};

struct RpcServiceConfig {
  std::string etcd_addr = "";
  std::string disagg_pd_policy = "";
  int detect_disconnected_instance_interval = 15;  // seconds
};

// instances pair for prefill and decode in disagg PD mode.
struct InstancesPair {
  std::string prefill_instance_http_addr = "";
  // empty means no decode instance, only prefill instance is available
  std::string decode_instance_http_addr = "";
};

enum class ErrorCode : int32_t {
  OK = 0,
  INTERNAL_ERROR = 1,
  INSTANCE_EXISTED = 2,
  INSTANCE_NOT_EXISTED = 3,
};

class ConvertErrorCode {
 public:
  static int32_t to_int(ErrorCode code) noexcept {
    return static_cast<int32_t>(code);
  }

  static ErrorCode from_int(int32_t code) noexcept {
    return static_cast<ErrorCode>(code);
  }
};

enum class InstanceType : int8_t {
  DEFAULT = 0,
  // prefill instance
  PREFILL = 1,
  // decode instance
  DECODE = 2,
};

struct InstanceMetaInfo {
 public:
  InstanceMetaInfo() { set_init_timestamp(); }
  InstanceMetaInfo(const std::string& inst_name, const std::string rpc_addr)
      : name(inst_name), rpc_address(rpc_addr) {
    set_init_timestamp();
  }
  InstanceMetaInfo(const std::string& inst_name,
                   const std::string rpc_addr,
                   const InstanceType& inst_type)
      : name(inst_name), rpc_address(rpc_addr), type(inst_type) {
    set_init_timestamp();
  }

  std::string name = "";
  std::string rpc_address = "";
  InstanceType type = InstanceType::DEFAULT;
  std::vector<uint64_t> cluster_ids;
  std::vector<std::string> addrs;
  std::vector<uint64_t> k_cache_ids;
  std::vector<uint64_t> v_cache_ids;
  int32_t dp_size;

  // latest heatbeat timestamp
  uint64_t latest_timestamp = 0;

 private:
  void set_init_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            now.time_since_epoch())
                            .count();
    latest_timestamp = timestamp_ms;
  }
};

// the info be stored in etcd
struct InstanceIdentityInfo {
  std::string instance_addr;
  std::string rpc_addr;
  int8_t instance_type;  // convert to InstanceType

  const std::string debug_string() const {
    std::string debug_str =
        "instance_addr: " + instance_addr + ", rpc_addr: " + rpc_addr +
        ", instance_type: " + std::to_string((int)(instance_type));
    return debug_str;
  }
};

// Function call related types
struct JsonFunction {
  std::string name;
  std::string description;
  nlohmann::json parameters;

  JsonFunction() = default;
  JsonFunction(const std::string& func_name,
               const std::string& desc,
               const nlohmann::json& params)
      : name(func_name), description(desc), parameters(params) {}
};

struct JsonTool {
  std::string type;  // "function"
  JsonFunction function;

  JsonTool() : type("function") {}
  JsonTool(const std::string& tool_type, const JsonFunction& func)
      : type(tool_type), function(func) {}
};

}  // namespace xllm_service
