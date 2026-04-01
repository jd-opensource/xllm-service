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

#include "instance_mgr.h"

#include <glog/logging.h>

#include <limits>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/global_gflags.h"
#include "common/types.h"
#include "instance_kvcache.h"

namespace xllm_service {

InstanceMgr::InstanceMgr(
    const Options& options,
    const std::shared_ptr<EtcdClient>& etcd_client,
    const bool is_master_service,
    OnInstanceDeregisteredCallback on_instance_deregistered)
    : etcd_client_(etcd_client),
      metrics_impl_(std::make_unique<InstanceMetricsImpl>(options,
                                                          etcd_client,
                                                          is_master_service)),
      topology_impl_(std::make_unique<InstanceTopologyImpl>(
          options,
          etcd_client,
          std::move(on_instance_deregistered),
          [this](const std::string& n, const InstanceMetaInfo& i) {
            metrics_impl_->add_instance_metrics(n, i);
          },
          [this](const std::string& n) {
            metrics_impl_->remove_instance_metrics(n);
          })),
      kvcache_(std::make_unique<InstanceKVCache>(options,
                                                 etcd_client,
                                                 is_master_service)) {
  metrics_impl_->set_topology(topology_impl_.get());
  init();
}

void InstanceMgr::init() {
  topology_impl_->init_from_etcd_register_all();
  metrics_impl_->load_initial_load_metrics_from_etcd();
}

InstanceMgr::~InstanceMgr() { metrics_impl_->shutdown(); }

InstanceMetaInfo InstanceMgr::get_instance_info(
    const std::string& instance_name) {
  return topology_impl_->get_instance_info(instance_name);
}

bool InstanceMgr::validate_scheduled_routing(const Request& request) {
  const std::string& prefill = request.routing.prefill_name;
  const std::string& decode = request.routing.decode_name;

  if (prefill.empty() || decode.empty()) {
    LOG(WARNING)
        << "validate_scheduled_routing: empty prefill_name or decode_name";
    return false;
  }

  if (request.prefill_incarnation_id.empty() ||
      request.decode_incarnation_id.empty()) {
    LOG(WARNING) << "validate_scheduled_routing: empty prefill or decode "
                    "incarnation_id";
    return false;
  }

  const InstanceMetaInfo prefill_info = get_instance_info(prefill);
  const InstanceMetaInfo decode_info = get_instance_info(decode);

  if (prefill_info.name.empty()) {
    LOG(WARNING)
        << "validate_scheduled_routing: prefill instance not registered: "
        << prefill;
    return false;
  }
  if (decode_info.name.empty()) {
    LOG(WARNING)
        << "validate_scheduled_routing: decode instance not registered: "
        << decode;
    return false;
  }

  if (prefill_info.runtime_state != InstanceRuntimeState::ACTIVE) {
    LOG(WARNING) << "validate_scheduled_routing: prefill not ACTIVE: "
                 << prefill
                 << " state=" << runtime_state_name(prefill_info.runtime_state);
    return false;
  }
  if (decode_info.runtime_state != InstanceRuntimeState::ACTIVE) {
    LOG(WARNING) << "validate_scheduled_routing: decode not ACTIVE: " << decode
                 << " state=" << runtime_state_name(decode_info.runtime_state);
    return false;
  }

  if (prefill_info.incarnation_id != request.prefill_incarnation_id) {
    LOG(WARNING)
        << "validate_scheduled_routing: prefill incarnation_id mismatch: "
        << prefill;
    return false;
  }
  if (decode_info.incarnation_id != request.decode_incarnation_id) {
    LOG(WARNING)
        << "validate_scheduled_routing: decode incarnation_id mismatch: "
        << decode;
    return false;
  }

  if (!get_channel(prefill)) {
    LOG(WARNING) << "validate_scheduled_routing: prefill brpc channel missing: "
                 << prefill;
    return false;
  }
  if (!get_channel(decode)) {
    LOG(WARNING) << "validate_scheduled_routing: decode brpc channel missing: "
                 << decode;
    return false;
  }

  return true;
}

std::vector<std::string> InstanceMgr::get_static_decode_list() {
  return topology_impl_->get_static_decode_list();
}

std::vector<std::string> InstanceMgr::get_static_prefill_list() {
  return topology_impl_->get_static_prefill_list();
}

bool InstanceMgr::prepare_load_balance_candidates(
    const std::function<bool(const InstanceMetaInfo&)>& is_schedulable,
    LoadBalanceCandidates* candidates) {
  std::shared_lock<std::shared_mutex> topo_lock(topology_impl_->cluster_mutex_);
  std::shared_lock<std::shared_mutex> metrics_lock(
      metrics_impl_->metrics_mutex_);

  topology_impl_->collect_load_balance_lists_locked(
      &candidates->prefill_candidates,
      &candidates->decode_candidates,
      &candidates->load_balance_infos.instance_infos,
      is_schedulable);

  if (candidates->prefill_candidates.empty() ||
      candidates->decode_candidates.empty()) {
    candidates->prefill_candidates.clear();
    candidates->decode_candidates.clear();
    candidates->load_balance_infos = LoadBalanceInfos{};
    return false;
  }

  metrics_impl_->fill_load_balance_infos_no_lock(
      candidates->prefill_candidates,
      candidates->decode_candidates,
      &candidates->load_balance_infos);
  return true;
}

double InstanceMgr::predict_tpot(const std::string& instance_name,
                                 int32_t total_length,
                                 int32_t batch_size) {
  return metrics_impl_->predict_tpot(instance_name, total_length, batch_size);
}

void InstanceMgr::kvcache_match(const Slice<int32_t>& token_ids,
                                OverlapScores* overlap_scores) {
  kvcache_->match(token_ids, overlap_scores);
}

bool InstanceMgr::upload_master_state_to_etcd() {
  const bool kvcache_ok = kvcache_->upload_kvcache();
  const bool load_ok = metrics_impl_->upload_load_metrics();
  return kvcache_ok && load_ok;
}

void InstanceMgr::set_as_master() {
  metrics_impl_->set_as_master();
  kvcache_->set_as_master();
}

std::shared_ptr<brpc::Channel> InstanceMgr::get_channel(
    const std::string& instance_name) {
  return topology_impl_->get_channel(instance_name);
}

bool InstanceMgr::on_instance_heartbeat(const proto::HeartbeatRequest& req) {
  if (!topology_impl_->record_instance_heartbeat(req.name(),
                                                 req.incarnation_id())) {
    return false;
  }
  kvcache_->record_updated_kvcaches(req.name(), req.cache_event());
  metrics_impl_->record_load_metrics_update(req.name(), req.load_metrics());
  metrics_impl_->update_latency_metrics(req.name(), req.latency_metrics());
  return true;
}

void InstanceMgr::update_request_metrics(std::shared_ptr<Request> request,
                                         RequestAction action) {
  metrics_impl_->update_request_metrics(request, action);
}

}  // namespace xllm_service
