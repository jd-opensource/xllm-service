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

#include <atomic>
#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "common/macros.h"
#include "common/options.h"
#include "common/threadpool.h"
#include "common/time_predictor.h"
#include "common/types.h"
#include "request/request.h"
#include "scheduler/etcd_client/etcd_client.h"
#include "scheduler/managers/instance_topology.h"
#include "xllm_rpc_service.pb.h"

namespace xllm_service {

// Point-in-time copy of metrics-side state (no TimePredictor; use predict_*).
struct MetricsSnapshot {
  std::unordered_map<std::string, LoadMetrics> load_metrics;
  std::unordered_map<std::string, RequestMetrics> request_metrics;
  std::unordered_map<std::string, LatencyMetrics> latency_metrics;
};

// Per-instance load / request / latency metrics, predictors, and etcd upload.
class InstanceMetrics {
 public:
  virtual ~InstanceMetrics() = default;

  virtual void get_load_metrics(LoadBalanceInfos* infos,
                                const TopologySnapshot& topology) = 0;

  virtual void record_load_metrics_update(
      const std::string& instance_name,
      const proto::LoadMetrics& load_metrics) = 0;

  virtual bool upload_load_metrics() = 0;

  virtual void update_latency_metrics(
      const std::string& instance_name,
      const proto::LatencyMetrics& latency_metrics) = 0;

  virtual void update_request_metrics(std::shared_ptr<Request> request,
                                      RequestAction action) = 0;

  virtual MetricsSnapshot snapshot() const = 0;

  virtual double predict_ttft(const std::string& instance_name,
                              int32_t token_len) = 0;

  virtual double predict_tpot(const std::string& instance_name,
                              int32_t total_length,
                              int32_t batch_size) = 0;

  virtual void set_as_master() = 0;

  virtual void add_instance_metrics(const std::string& name,
                                    const InstanceMetaInfo& info) = 0;

  virtual void remove_instance_metrics(const std::string& name) = 0;
};

// Default implementation (extracted from InstanceMgr). Call set_topology()
// after InstanceTopologyImpl is constructed.
class InstanceMetricsImpl final : public InstanceMetrics {
 public:
  InstanceMetricsImpl(const Options& options,
                      const std::shared_ptr<EtcdClient>& etcd_client,
                      bool is_master_service);

  ~InstanceMetricsImpl() override;

  void set_topology(InstanceTopologyImpl* topology);

  void load_initial_load_metrics_from_etcd();

  void shutdown();

  void get_load_metrics(LoadBalanceInfos* infos,
                        const TopologySnapshot& topology) override;

  void record_load_metrics_update(
      const std::string& instance_name,
      const proto::LoadMetrics& load_metrics) override;

  bool upload_load_metrics() override;

  void update_latency_metrics(
      const std::string& instance_name,
      const proto::LatencyMetrics& latency_metrics) override;

  void update_request_metrics(std::shared_ptr<Request> request,
                              RequestAction action) override;

  MetricsSnapshot snapshot() const override;

  double predict_ttft(const std::string& instance_name,
                      int32_t token_len) override;

  double predict_tpot(const std::string& instance_name,
                      int32_t total_length,
                      int32_t batch_size) override;

  void set_as_master() override;

  void add_instance_metrics(const std::string& name,
                            const InstanceMetaInfo& info) override;

  void remove_instance_metrics(const std::string& name) override;

 private:
  friend class InstanceMgr;

  DISALLOW_COPY_AND_ASSIGN(InstanceMetricsImpl);

  // Caller must hold metrics_mutex_; used by InstanceMgr SLO path under
  // cluster+metrics locks (predict_* would deadlock).
  TimePredictor& time_predictor_unlocked(const std::string& instance_name);

  void update_load_metrics(const etcd::Response& response,
                           const uint64_t& prefix_len);

  Options options_;
  std::shared_ptr<EtcdClient> etcd_client_;
  std::atomic_bool is_master_service_;
  std::atomic_bool exited_{false};

  InstanceTopologyImpl* topology_impl_ = nullptr;

  ThreadPool load_metrics_threadpool_;
  mutable std::shared_mutex metrics_mutex_;
  std::unordered_map<std::string, LoadMetrics> load_metrics_;
  std::unordered_map<std::string, LoadMetrics> updated_metrics_;
  std::unordered_set<std::string> removed_instance_;
  std::unordered_map<std::string, TimePredictor> time_predictors_;
  std::unordered_map<std::string, LatencyMetrics> latency_metrics_;
  std::unordered_map<std::string, RequestMetrics> request_metrics_;
};

}  // namespace xllm_service
