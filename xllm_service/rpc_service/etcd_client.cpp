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

#include "etcd_client.h"

#include <glog/logging.h>

#include <nlohmann/json.hpp>

namespace xllm_service {

EtcdClient::EtcdClient(const std::string& etcd_addr)
    : client_(etcd_addr), etcd_addr_(etcd_addr) {
  auto response = client_.put("XLLM_PING", "PING");
  if (!response.is_ok()) {
    LOG(FATAL) << "etcd connect to etcd server failed: "
               << response.error_message();
  }
}

EtcdClient::~EtcdClient() {}

bool EtcdClient::get(const std::string& key, InstanceIdentityInfo& value) {
  auto response = client_.get(key);
  if (!response.is_ok()) {
    LOG(ERROR) << "etcd get " << key << " failed: " << response.error_message();
    return false;
  }
  auto json_str = response.value().as_string();
  try {
    nlohmann::json json_value = nlohmann::json::parse(json_str);
    value.instance_addr = json_value.at("instance_addr").get<std::string>();
    value.instance_type = json_value.at("instance_type").get<int8_t>();
  } catch (const std::exception& e) {
    LOG(ERROR) << "etcd get " << key
               << " failed: json parse error: " << e.what();
    return false;
  }

  return true;
}

bool EtcdClient::get_prefix(const std::string& key_prefix,
                            std::vector<InstanceIdentityInfo>& values) {
  auto response = client_.ls(key_prefix);
  if (!response.is_ok()) {
    LOG(ERROR) << "etcd get " << key_prefix
               << " failed: " << response.error_message();
    return false;
  }
  for (const auto& v : response.values()) {
    InstanceIdentityInfo value;
    auto json_str = v.as_string();
    try {
      nlohmann::json json_value = nlohmann::json::parse(json_str);
      value.instance_addr = json_value.at("instance_addr").get<std::string>();
      value.instance_type = json_value.at("instance_type").get<int8_t>();
      values.emplace_back(value);
    } catch (const std::exception& e) {
      LOG(ERROR) << "etcd get " << key_prefix
                 << " failed: json parse error: " << e.what();
      return false;
    }
  }

  return true;
}

bool EtcdClient::set(const std::string& key,
                     const InstanceIdentityInfo& value) {
  std::string json_str;
  try {
    nlohmann::json json_value;
    json_value["instance_addr"] = value.instance_addr;
    json_value["instance_type"] = value.instance_type;
    json_str = json_value.dump();
  } catch (const std::exception& e) {
    LOG(ERROR) << "etcd set " << key
               << " failed: json dump error: " << e.what();
    return false;
  }

  auto response = client_.put(key, json_str);
  if (!response.is_ok()) {
    LOG(ERROR) << "etcd set " << key << " failed: " << response.error_message();
    return false;
  }

  return true;
}

bool EtcdClient::rm(const std::string& key) {
  auto response = client_.rm(key);
  if (!response.is_ok()) {
    LOG(ERROR) << "etcd rm " << key << " failed: " << response.error_message();
    return false;
  }

  return true;
}

}  // namespace xllm_service
