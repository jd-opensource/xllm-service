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
