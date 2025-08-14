#pragma once

#include <brpc/channel.h>
#include <butil/time.h>

#include <string>
#include <thread>

#include "common/types.h"
#include "xllm_rpc_service.pb.h"

namespace xllm_service {

struct ChannelOptions {
  std::string protocol = "baidu_std";
  std::string connection_type = "";
  std::string load_balancer = "";
  int timeout_ms = 100;
  int max_retry = 3;
  int interval_ms = 1000;
};

class XllmRpcClient {
 public:
  XllmRpcClient(const std::string& instace_name,
                const std::string& master_addr,
                const ChannelOptions& options);
  ~XllmRpcClient();

  ErrorCode register_instance();
  ErrorCode register_instance(const InstanceMetaInfo& metainfo);

 private:
  void heartbeat();

 private:
  bool exited_ = false;
  bool register_inst_done_ = false;
  // instance rdma address or other info: ip port
  std::string instance_name_;
  std::string master_addr_;
  brpc::Channel master_channel_;
  std::unique_ptr<proto::XllmRpcService_Stub> master_stub_;
  std::unique_ptr<std::thread> heartbeat_thread_;
};

}  // namespace xllm_service
