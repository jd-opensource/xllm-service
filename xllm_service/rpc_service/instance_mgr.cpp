#include "instance_mgr.h"

#include <absl/strings/str_join.h>
#include <glog/logging.h>

#include <chrono>
#include <iostream>

#include "common/types.h"
#include "common/utils.h"
namespace xllm_service {

// magic number, TODO: move to config file or env var
static constexpr int kDetectIntervals = 15;  // 15seconds
static std::unordered_map<InstanceType, std::string> ETCD_KEYS_PREFIX_MAP = {
    {InstanceType::DEFAULT, "XLLM:DEFAULT:"},
    {InstanceType::PREFILL, "XLLM:PREFILL:"},
    {InstanceType::DECODE, "XLLM:DECODE:"},
};
static std::string ETCD_ALL_KEYS_PREFIX = "XLLM:";
static std::string DEFAULT_DISAGG_PD_POLICY = "RR";

InstanceMgr::InstanceMgr(const RpcServiceConfig& config) : config_(config) {
  if (config.etcd_addr.empty()) {
    LOG(INFO) << "Disable etcd meta server";
    use_etcd_ = false;
  } else {
    LOG(INFO) << "Connect to etcd meta server: " << config.etcd_addr;
    use_etcd_ = true;
    etcd_client_ = std::make_unique<EtcdClient>(config.etcd_addr);
  }

  internal_init();
}

void InstanceMgr::internal_init() {
  std::string pd_policy = config_.disagg_pd_policy;
  if (config_.disagg_pd_policy.empty()) {
    LOG(WARNING) << "Not specify diasgg pd policy, use `RR` policy as default.";
    pd_policy = DEFAULT_DISAGG_PD_POLICY;
  }
  if (pd_policy == "RR") {
    disagg_pd_policy_ = std::make_unique<RoundRobinDisaggPdPolicy>();
  } else {
    LOG(FATAL) << "Not supported diasgg pd policy: " << pd_policy;
    return;
  }

  heartbeat_thread_ = std::make_unique<std::thread>(
      &InstanceMgr::detect_disconnected_instances, this);
}

InstanceMgr::~InstanceMgr() {
  exited_ = true;
  if (heartbeat_thread_) {
    heartbeat_thread_->join();
  }
}

void InstanceMgr::detect_disconnected_instances() {
  while (!exited_) {
    std::this_thread::sleep_for(std::chrono::seconds(kDetectIntervals));
    {
      std::lock_guard<std::mutex> guard(inst_mutex_);
      auto now = std::chrono::system_clock::now();
      auto timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                              now.time_since_epoch())
                              .count();
      std::vector<std::string> disconnected_instances_name;
      for (const auto& [name, info] : instances_) {
        if (timestamp_ms - info.latest_timestamp > kDetectIntervals * 1000) {
          LOG(WARNING) << "Instance maybe disconnected, instance_name: " << name
                       << ", last heartbeat interval(s): "
                       << (timestamp_ms - info.latest_timestamp) / 1000.0;
          disconnected_instances_name.emplace_back(name);
        }
      }

      // no instances disconnected, return
      if (disconnected_instances_name.empty()) {
        continue;
      }

      if (utils::enable_debug_log()) {
        const auto instance_names =
            absl::StrJoin(disconnected_instances_name, ", ");
        LOG(WARNING) << "Detect disconnected instance, instance_name: "
                     << instance_names;
      }
      // detele instance metainfo from etcd
      delete_persistence_metainfo(disconnected_instances_name);
      for (const auto& name : disconnected_instances_name) {
        disagg_pd_policy_->remove_instance(name, instances_[name].type);
        instances_.erase(name);
      }
    }
  }
}

ErrorCode InstanceMgr::register_instance(const std::string& instance_name) {
  std::lock_guard<std::mutex> guard(inst_mutex_);
  if (utils::enable_debug_log()) {
    LOG(WARNING) << "Register instance, instance_name: " << instance_name;
  }
  if (instances_.find(instance_name) != instances_.end()) {
    // update_instance_timestamp(instance_name);
    LOG(ERROR) << "Instance is already registered, instance_name: "
               << instance_name;
    return ErrorCode::INSTANCE_EXISTED;
  }

  InstanceMetaInfo default_info(instance_name, "");
  instances_[instance_name] = default_info;
  disagg_pd_policy_->insert_instance(instance_name,
                                     &(instances_[instance_name]));
  // save instance metainfo to etcd
  save_persistence_metainfo(default_info);
  return ErrorCode::OK;
}

ErrorCode InstanceMgr::register_instance(const std::string& instance_name,
                                         const InstanceMetaInfo& metainfo) {
  std::lock_guard<std::mutex> guard(inst_mutex_);
  if (utils::enable_debug_log()) {
    LOG(WARNING) << "Register instance, instance_name: " << instance_name;
  }
  if (instances_.find(instance_name) != instances_.end()) {
    // update_instance_timestamp(instance_name);
    LOG(ERROR) << "Instance is already registered, instance_name: "
               << instance_name;
    return ErrorCode::INSTANCE_EXISTED;
  }

  instances_[instance_name] = metainfo;
  disagg_pd_policy_->insert_instance(instance_name,
                                     &(instances_[instance_name]));
  // save instance metainfo to etcd
  save_persistence_metainfo(metainfo);
  return ErrorCode::OK;
}

ErrorCode InstanceMgr::update_instance_metainfo(
    const std::string& instance_name,
    const InstanceMetaInfo& metainfo) {
  std::lock_guard<std::mutex> guard(inst_mutex_);
  if (utils::enable_debug_log()) {
    LOG(WARNING) << "Update instance metainfo, instance_name: "
                 << instance_name;
  }
  if (instances_.find(instance_name) == instances_.end()) {
    LOG(ERROR) << "Instance is not registered, instance_name: "
               << instance_name;
    return ErrorCode::INSTANCE_NOT_EXISTED;
  }

  instances_[instance_name] = metainfo;
  update_instance_timestamp(instance_name);
  disagg_pd_policy_->update_instance(instance_name,
                                     &(instances_[instance_name]));
  return ErrorCode::OK;
}

void InstanceMgr::save_persistence_metainfo(const InstanceMetaInfo& metainfo) {
  if (!use_etcd_) {
    return;
  }
  std::string key = ETCD_KEYS_PREFIX_MAP[metainfo.type] + metainfo.name;
  InstanceIdentityInfo value;
  value.instance_addr = metainfo.name;
  value.rpc_addr = metainfo.rpc_address;
  value.instance_type = static_cast<int8_t>(metainfo.type);
  bool ok = etcd_client_->set(key, value);
  if (!ok) {
    LOG(ERROR) << "Save instance metainfo to etcd failed, key: " << key;
    return;
  }

  if (utils::enable_debug_log()) {
    InstanceIdentityInfo debug_value;
    bool ok = etcd_client_->get(key, debug_value);
    if (!ok) {
      LOG(ERROR) << "Get instance metainfo from etcd failed, key: " << key;
      return;
    }
    LOG(WARNING) << "Instance after put: " << debug_value.debug_string();
  }
}

void InstanceMgr::delete_persistence_metainfo(
    const std::vector<std::string>& instance_names) {
  if (!use_etcd_ || instance_names.empty()) {
    return;
  }
  // TODO: use batch delete later
  for (const auto& name : instance_names) {
    InstanceMetaInfo& metainfo = instances_[name];
    std::string key = ETCD_KEYS_PREFIX_MAP[metainfo.type] + metainfo.name;
    bool ok = etcd_client_->rm(key);
    if (!ok) {
      LOG(ERROR) << "Delete instance metainfo from etcd failed, key: " << key;
    }
  }

  if (utils::enable_debug_log()) {
    std::vector<InstanceIdentityInfo> debug_values;
    bool ok = etcd_client_->get_prefix(ETCD_ALL_KEYS_PREFIX, debug_values);
    if (!ok) {
      LOG(ERROR) << "Get instance metainfo from etcd failed, key: "
                 << ETCD_ALL_KEYS_PREFIX;
      return;
    }
    std::string concat_debug_str;
    for (const auto& v : debug_values) {
      concat_debug_str += v.debug_string();
      concat_debug_str += "\n";
    }
    LOG(WARNING) << "Instances after delete: " << concat_debug_str;
  }
}

ErrorCode InstanceMgr::heartbeat(const std::string& instance_name) {
  std::lock_guard<std::mutex> guard(inst_mutex_);
  if (utils::enable_debug_log()) {
    LOG(WARNING) << "Receive heartbeat, instance_name: " << instance_name;
  }
  if (instances_.find(instance_name) == instances_.end()) {
    LOG(ERROR) << "Instance is not registered, instance_name: "
               << instance_name;
    return ErrorCode::INSTANCE_NOT_EXISTED;
  }

  update_instance_timestamp(instance_name);

  return ErrorCode::OK;
}

void InstanceMgr::update_instance_timestamp(const std::string& inst_name) {
  auto now = std::chrono::system_clock::now();
  auto timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          now.time_since_epoch())
                          .count();
  instances_[inst_name].latest_timestamp = timestamp_ms;
}

InstancesPair InstanceMgr::select_instances_pair(bool only_prefill) {
  return disagg_pd_policy_->select_instances_pair(only_prefill);
}

InstanceMetaInfo InstanceMgr::get_instance_info(
    const std::string& instance_name) {
  std::lock_guard<std::mutex> guard(inst_mutex_);
  if (instances_.find(instance_name) == instances_.end()) {
    LOG(ERROR) << "Get instance info failed, instance is not registered, "
                  "instance_name: "
               << instance_name;
    return InstanceMetaInfo();
  }
  return instances_[instance_name];
}

// TODO: refactor later, currently return all decode instances
std::vector<std::string> InstanceMgr::get_static_decode_list(
    const std::string& instance_name) {
  std::vector<std::string> decode_list;
  std::lock_guard<std::mutex> guard(inst_mutex_);
  for (auto& inst : instances_) {
    if (inst.second.type == InstanceType::DECODE) {
      decode_list.emplace_back(inst.second.name);
    }
  }

  return decode_list;
}

}  // namespace xllm_service
