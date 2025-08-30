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

#include <etcd/KeepAlive.hpp>
#include <etcd/SyncClient.hpp>
#include <etcd/Watcher.hpp>
#include <etcd/v3/Transaction.hpp>
#include <string>
#include <unordered_set>

#include "common/hash_util.h"
#include "common/types.h"

namespace xllm_service {

using Callback = std::function<void(const etcd::Response&, const uint64_t&)>;

class EtcdClient {
 public:
  EtcdClient(const std::string& etcd_addr);
  ~EtcdClient();

  template <typename T>
  bool set(const std::string& key, const T& value) {
    auto response = client_.put(key, value.serialize_to_json().dump());
    if (!response.is_ok()) {
      LOG(ERROR) << "etcd set " << key
                 << " failed: " << response.error_message();
      return false;
    }

    return true;
  }

  template <typename T>
  bool set(const std::string& key_prefix,
           const unordered_map<std::string, T>& values) {
    bool rt = true;
    for (const auto& iter : values) {
      if (iter.second.empty()) {
        rt = rt && client_.rm(key_prefix + iter.first).is_ok();
      } else {
        rt = rt && client_
                       .put(key_prefix + iter.first,
                            iter.second.serialize_to_json().dump())
                       .is_ok();
      }
    }
    return true;
  }

  bool set(const std::string& key_prefix, const Murmur3KeyCacheMap& values);

  bool set(const std::string& key, const std::string& value);

  // create key-value with lease and transaction
  bool set(const std::string& key, const std::string& value, const int ttl);

  bool rm(const std::string& key);

  bool rm(const std::string& key_prefix,
          const std::unordered_set<std::string>& keys);

  template <typename T>
  bool get(const std::string& key, T* value) {
    auto response = client_.get(key);
    if (!response.is_ok()) {
      LOG(ERROR) << "etcd get " << key
                 << " failed: " << response.error_message();
      return false;
    }
    if (value) {
      return value->parse_from_json(response.value().as_string());
    } else {
      return true;
    }
  }

  bool get(const std::string& key_prefix, std::string* value);

  template <typename T>
  bool get_prefix(const std::string& key_prefix,
                  std::unordered_map<std::string, T>* values) {
    auto response = client_.ls(key_prefix);
    if (!response.is_ok()) {
      LOG(ERROR) << "etcd get " << key_prefix
                 << " failed: " << response.error_message();
      return false;
    }

    for (int i = 0; i < response.keys().size(); i++) {
      auto key_str = response.key(i).substr(key_prefix.size());
      auto json_str = response.value(i).as_string();

      T value;
      if (!value.parse_from_json(json_str)) {
        LOG(ERROR) << "Parse json fail: " << json_str;
        continue;
      }

      values->insert_or_assign(std::move(key_str), std::move(value));
    }
    return true;
  }

  bool get_prefix(const std::string& key_prefix, Murmur3KeyCacheMap* values);

  bool get_prefix(const std::string& key_prefix,
                  std::unordered_map<std::string, std::string>* values);

  void add_watch(const std::string& key_prefix,
                 Callback callback,
                 bool recursive = true);

  void remove_watch(const std::string& key_prefix);

  void stop_watch();

 private:
  struct WatcherInfo {
    std::unique_ptr<etcd::Watcher> watcher;
    Callback callback;
  };

  etcd::SyncClient client_;
  std::string etcd_addr_;
  std::mutex watchers_mutex_;
  std::map<std::string, WatcherInfo> watchers_;
  std::vector<std::shared_ptr<etcd::KeepAlive>> keep_alives_;
};

}  // namespace xllm_service
