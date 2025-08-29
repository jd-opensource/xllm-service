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
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/hash_util.h"
#include "nlohmann/json.hpp"

namespace xllm_service {

struct CacheLocations;
using Murmur3KeyCacheMap = std::unordered_map<Murmur3Key,
                                              CacheLocations,
                                              FixedStringKeyHash,
                                              FixedStringKeyEqual>;

struct HttpServiceConfig {
  int num_threads = 16;
  int timeout_ms = -1;
  std::string test_instance_addr = "";
  bool enable_request_trace = false;
};

struct RpcServiceConfig {
  std::string etcd_addr = "";
  std::string load_balance_policy = "";
  int detect_disconnected_instance_interval = 15;  // seconds
  std::string service_name = "";
};

struct ModelConfig {
  int32_t block_size = 16;
  std::string model_type = "chatglm";
  std::string tokenizer_path = "";
};

struct Routing {
  std::string prefill_name;
  std::string decode_name;

  nlohmann::json serialize_to_json() const {
    nlohmann::json json_val;
    json_val["prefill_name"] = prefill_name;
    json_val["decode_name"] = decode_name;
    return json_val;
  }

  std::string debug_string() const { return serialize_to_json().dump(2); }
};

struct ScheduleResult {
  std::vector<int32_t> token_ids;
  Routing routing;
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

struct LoadMetrics {
  LoadMetrics() : waiting_requests_num(0), gpu_cache_usage_perc(0) {};
  LoadMetrics(const uint64_t& waiting_reqs_num, const float& usage)
      : waiting_requests_num(waiting_reqs_num), gpu_cache_usage_perc(usage) {};

  uint64_t waiting_requests_num;
  float gpu_cache_usage_perc;

  nlohmann::json serialize_to_json() const {
    nlohmann::json json_val;
    json_val["waiting_requests_num"] = waiting_requests_num;
    json_val["gpu_cache_usage_perc"] = gpu_cache_usage_perc;
    return json_val;
  }

  std::string debug_string() const { return serialize_to_json().dump(2); }

  bool parse_from_json(const std::string& json_str) {
    try {
      nlohmann::json json_value = nlohmann::json::parse(json_str);

      waiting_requests_num =
          json_value.at("waiting_requests_num").get<uint64_t>();
      gpu_cache_usage_perc = json_value.at("gpu_cache_usage_perc").get<float>();

    } catch (const std::exception& e) {
      LOG(ERROR) << "json str:" << json_str
                 << ", parse to loadmetrics error: " << e.what();
      return false;
    }
    return true;
  }

  bool empty() const { return false; }
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

  uint64_t instance_index = -1;

  nlohmann::json serialize_to_json() const {
    nlohmann::json json_val;
    json_val["name"] = name;
    json_val["rpc_address"] = rpc_address;
    json_val["type"] = int8_t(type);
    json_val["addrs"] = addrs;
    json_val["cluster_ids"] = cluster_ids;
    json_val["k_cache_ids"] = k_cache_ids;
    json_val["v_cache_ids"] = v_cache_ids;
    json_val["dp_size"] = dp_size;
    return json_val;
  }

  std::string debug_string() const { return serialize_to_json().dump(2); }

  bool parse_from_json(const std::string& json_str) {
    try {
      nlohmann::json json_value = nlohmann::json::parse(json_str);
      name = json_value.at("name").get<std::string>();
      rpc_address = json_value.at("rpc_address").get<std::string>();
      type = static_cast<InstanceType>(json_value.at("type").get<int8_t>());

      for (const auto& item :
           json_value.at("cluster_ids").get<std::vector<uint64_t>>()) {
        cluster_ids.push_back(item);
      }

      for (const auto& item :
           json_value.at("k_cache_ids").get<std::vector<uint64_t>>()) {
        k_cache_ids.push_back(item);
      }

      for (const auto& item :
           json_value.at("addrs").get<std::vector<std::string>>()) {
        addrs.push_back(item);
      }

      for (const auto& item :
           json_value.at("v_cache_ids").get<std::vector<uint64_t>>()) {
        v_cache_ids.push_back(item);
      }

      dp_size = json_value.at("dp_size").get<int32_t>();

      set_init_timestamp();
    } catch (const std::exception& e) {
      LOG(ERROR) << "json str:" << json_str
                 << ", parse to instancemetainfo error: " << e.what();
      return false;
    }
    return true;
  }

  bool empty() const { return rpc_address == ""; }

 private:
  void set_init_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            now.time_since_epoch())
                            .count();
    latest_timestamp = timestamp_ms;
  }
};

struct CacheLocations {
  std::unordered_set<std::string> hbm_instance_set;
  std::unordered_set<std::string> dram_instance_set;
  std::unordered_set<std::string> ssd_instance_set;

  nlohmann::json serialize_to_json() const {
    nlohmann::json json_val;
    json_val["hbm_instance_set"] = hbm_instance_set;
    json_val["dram_instance_set"] = dram_instance_set;
    json_val["ssd_instance_set"] = ssd_instance_set;
    return json_val;
  }

  std::string debug_string() { return serialize_to_json().dump(2); }

  bool parse_from_json(const std::string& json_str) {
    try {
      nlohmann::json json_value = nlohmann::json::parse(json_str);
      for (const auto& item :
           json_value.at("hbm_instance_set").get<std::vector<std::string>>()) {
        hbm_instance_set.insert(item);
      }

      for (const auto& item :
           json_value.at("dram_instance_set").get<std::vector<std::string>>()) {
        dram_instance_set.insert(item);
      }

      for (const auto& item :
           json_value.at("ssd_instance_set").get<std::vector<std::string>>()) {
        ssd_instance_set.insert(item);
      }

    } catch (const std::exception& e) {
      LOG(ERROR) << "json str:" << json_str
                 << ", parse to cachelocation error: " << e.what();
      return false;
    }
    return true;
  }

  bool empty() const {
    return hbm_instance_set.empty() && dram_instance_set.empty() &&
           ssd_instance_set.empty();
  }
};

/**
 * @brief Records the prefix cache match lengths for different instances on
 * current request
 *
 * This struct stores and manages prefix cache matching information across
 * multiple instances, supporting different storage types (HBM, DRAM, SSD) for
 * match length recording, and tracks information about the best matching
 * instance.
 */
struct OverlapScores {
  // Set of matched instance names
  std::unordered_set<std::string> instances;
  // HBM storage type instance match length mapping (instance name -> match
  // length)
  std::unordered_map<std::string, uint32_t> hbm_instance_score;
  // DRAM storage type instance match length mapping (instance name -> match
  // length)
  std::unordered_map<std::string, uint32_t> dram_instance_score;
  // SSD storage type instance match length mapping (instance name -> match
  // length)
  std::unordered_map<std::string, uint32_t> ssd_instance_score;
  uint32_t max_block_num = 0;
  uint32_t max_matched_block_num = 0;
  std::string max_matched_instance_name = "";

  std::string debug_string() {
    nlohmann::json json_val;
    json_val["instances"] = instances;
    json_val["hbm_instance_score"] = hbm_instance_score;
    json_val["dram_instance_score"] = dram_instance_score;
    json_val["ssd_instance_score"] = ssd_instance_score;
    json_val["max_block_num"] = max_block_num;
    json_val["max_matched_block_num"] = max_matched_block_num;
    json_val["max_matched_instance_name"] = max_matched_instance_name;
    return json_val.dump(2);
  }
};

struct LoadBalanceInfos {
  OverlapScores overlap_scores;
  std::unordered_map<std::string, LoadMetrics> prefill_load_metrics;
  std::unordered_map<std::string, LoadMetrics> decode_load_metrics;
  uint64_t prefill_max_waiting_requests_num = 0;
  uint64_t decode_max_waiting_requests_num = 0;

  std::string debug_string() {
    nlohmann::json json_val;

    json_val["overlap_scores"] =
        nlohmann::json::parse(overlap_scores.debug_string());

    nlohmann::json prefill_json;
    for (auto& [key, metrics] : prefill_load_metrics) {
      prefill_json[key] = nlohmann::json::parse(metrics.debug_string());
    }
    json_val["prefill_load_metrics"] = prefill_json;

    nlohmann::json decode_json;
    for (auto& [key, metrics] : decode_load_metrics) {
      decode_json[key] = nlohmann::json::parse(metrics.debug_string());
    }
    json_val["decode_load_metrics"] = decode_json;

    json_val["prefill_max_waiting_requests_num"] =
        prefill_max_waiting_requests_num;
    json_val["decode_max_waiting_requests_num"] =
        decode_max_waiting_requests_num;

    return json_val.dump(2);
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
