#pragma once

#include <shared_mutex>
#include <thread>

#include "common/hash_util.h"
#include "common/macros.h"
#include "common/slice.h"
#include "common/threadpool.h"
#include "common/types.h"
#include "etcd_client.h"
#include "xllm_rpc_service.pb.h"

namespace xllm_service {

class GlobalKVCacheMgr final {
 public:
  explicit GlobalKVCacheMgr(const std::shared_ptr<EtcdClient>& etcd_client,
                            const ModelConfig& model_config,
                            const bool is_master_service);
  ~GlobalKVCacheMgr();

  void match(const Slice<int32_t>& token_ids, OverlapScores* overlap_scores);

  void record_updated_kvcaches(const std::string& instance_name,
                               const proto::KvCacheEvent& kvcache_event);
  bool upload_kvcache();

  void set_as_master();

 private:
  DISALLOW_COPY_AND_ASSIGN(GlobalKVCacheMgr);

  void update_kvcache(const etcd::Response& response,
                      const uint64_t prefix_len);

 private:
  ModelConfig model_config_;
  std::atomic_bool is_master_service_ = false;
  bool exited_ = false;
  std::shared_mutex kvcache_mutex_;
  Murmur3KeyCacheMap kvcache_infos_;
  std::shared_ptr<EtcdClient> etcd_client_;  // not own

  std::mutex update_mutex_;
  Murmur3KeyCacheMap updated_kvcaches_;

  ThreadPool threadpool_;
};

}  // namespace xllm_service
