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

#include <etcd/SyncClient.hpp>
#include <string>

#include "common/types.h"

namespace xllm_service {

// the format is:
// key: XLLM:PREFILL:inst_id -> value
// or
// key: XLLM:DECODE:inst_id -> value
class EtcdClient {
 public:
  EtcdClient(const std::string& etcd_addr);
  ~EtcdClient();

  bool get(const std::string& key, InstanceIdentityInfo& value);
  // get all keys with prefix
  bool get_prefix(const std::string& key_prefix,
                  std::vector<InstanceIdentityInfo>& values);
  bool set(const std::string& key, const InstanceIdentityInfo& value);
  bool rm(const std::string& key);

 private:
  etcd::SyncClient client_;
  std::string etcd_addr_;
};

}  // namespace xllm_service
