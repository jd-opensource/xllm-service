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
  LOG(INFO) << "EtcdClient init put start!";
  auto response = client_.put("XLLM_PING", "PING");
  LOG(INFO) << "EtcdClient init put end!";
  if (!response.is_ok()) {
    LOG(FATAL) << "etcd connect to etcd server failed: "
               << response.error_message();
  }
}

EtcdClient::~EtcdClient() { stop_watch(); }

bool EtcdClient::set(const std::string& key, const std::string& value) {
  auto response = client_.put(key, value);
  if (!response.is_ok()) {
    LOG(ERROR) << "etcd set " << key << " failed: " << response.error_message();
    return false;
  }

  return true;
}

bool EtcdClient::set(const std::string& key,
                     const std::string& value,
                     const int ttl) {
  auto keep_alive = std::make_shared<etcd::KeepAlive>(client_, ttl);
  etcdv3::Transaction transaction;
  transaction.add_compare_create(key, 0);
  transaction.add_success_put(key, value, keep_alive->Lease());
  etcd::Response response = client_.txn(transaction);
  if (response.is_ok()) {
    keep_alives_.emplace_back(std::move(keep_alive));
    return true;
  } else {
    keep_alive->Cancel();
    return false;
  }
}

bool EtcdClient::set(const std::string& key_prefix,
                     const Murmur3KeyCacheMap& values) {
  bool rt = true;
  for (const auto& iter : values) {
    if (iter.second.empty()) {
      rt = rt && client_.rm(key_prefix + iter.first.to_string()).is_ok();
    } else {
      rt = rt && client_
                     .put(key_prefix + iter.first.to_string(),
                          iter.second.serialize_to_json().dump())
                     .is_ok();
    }
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

bool EtcdClient::rm(const std::string& key_prefix,
                    const std::unordered_set<std::string>& keys) {
  etcdv3::Transaction transaction;
  transaction.add_compare_version(
      "XLLM:SERVICE:MASTER", etcdv3::CompareResult::GREATER, -1);
  for (const auto& iter : keys) {
    transaction.add_success_delete(key_prefix + iter);
  }
  return client_.txn(transaction).is_ok();
}

bool EtcdClient::get(const std::string& key, std::string* value) {
  auto response = client_.get(key);
  if (!response.is_ok()) {
    LOG(ERROR) << "etcd get " << key << " failed: " << response.error_message();
    return false;
  }
  if (value) {
    *value = response.value().as_string();
  }
  return true;
}

bool EtcdClient::get_prefix(const std::string& key_prefix,
                            Murmur3KeyCacheMap* values) {
  auto response = client_.ls(key_prefix);
  if (!response.is_ok()) {
    LOG(ERROR) << "etcd get " << key_prefix
               << " failed: " << response.error_message();
    return false;
  }

  for (int i = 0; i < response.keys().size(); i++) {
    Murmur3Key key(response.key(i).substr(key_prefix.size()).c_str());
    auto json_str = response.value(i).as_string();

    CacheLocations value;
    if (!value.parse_from_json(json_str)) {
      LOG(ERROR) << "Parse json fail: " << json_str;
      continue;
    }

    values->insert_or_assign(std::move(key), std::move(value));
  }
  return true;
}

bool EtcdClient::get_prefix(
    const std::string& key_prefix,
    std::unordered_map<std::string, std::string>* values) {
  auto response = client_.ls(key_prefix);
  if (!response.is_ok()) {
    LOG(ERROR) << "etcd get " << key_prefix
               << " failed: " << response.error_message();
    return false;
  }

  for (int i = 0; i < response.keys().size(); i++) {
    auto key_str = response.key(i).substr(key_prefix.size());
    auto str = response.value(i).as_string();

    values->insert_or_assign(std::move(key_str), std::move(str));
  }
  return true;
}

void EtcdClient::add_watch(const std::string& key_prefix,
                           Callback callback,
                           bool recursive) {
  std::lock_guard<std::mutex> lock(watchers_mutex_);

  if (watchers_.find(key_prefix) != watchers_.end()) {
    watchers_[key_prefix].watcher->Cancel();
  }
  auto watcher = std::make_unique<etcd::Watcher>(
      client_,
      key_prefix,
      [callback, key_prefix](etcd::Response response) {
        callback(response, uint64_t(key_prefix.size()));
      },
      recursive);

  watchers_[key_prefix] = {std::move(watcher), callback};
}

void EtcdClient::remove_watch(const std::string& key_prefix) {
  std::lock_guard<std::mutex> lock(watchers_mutex_);

  auto it = watchers_.find(key_prefix);
  if (it != watchers_.end()) {
    it->second.watcher->Cancel();
    watchers_.erase(it);
  }
}

void EtcdClient::stop_watch() {
  std::lock_guard<std::mutex> lock(watchers_mutex_);

  for (auto& pair : watchers_) {
    pair.second.watcher->Cancel();
  }

  watchers_.clear();
}

}  // namespace xllm_service
