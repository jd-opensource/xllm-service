#include <brpc/server.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "common/global_gflags.h"
#include "common/utils.h"
#include "rpc_service/service.h"

int main(int argc, char* argv[]) {
  // Initialize gflags
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Initialize glog
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true;

  LOG(INFO) << "Starting xllm rpc service, port: " << FLAGS_port;

  if (!xllm_service::utils::is_port_available(FLAGS_port)) {
    LOG(ERROR) << "Port " << FLAGS_port << " is already in use. "
               << "Please specify a different port using --port flag.";
    return 1;
  }

  xllm_service::RpcServiceConfig config;
  config.etcd_addr = FLAGS_etcd_addr;
  config.disagg_pd_policy = FLAGS_disagg_pd_policy;
  config.detect_disconnected_instance_interval =
      FLAGS_detect_disconnected_instance_interval;

  // create xllm service
  auto xllm_service_impl =
      std::make_shared<xllm_service::XllmRpcServiceImpl>(config);
  xllm_service::XllmRpcService service(xllm_service_impl);

  // Initialize brpc server
  std::string server_address = "0.0.0.0:" + std::to_string(FLAGS_port);
  brpc::Server server;
  if (server.AddService(&service, brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
    LOG(ERROR) << "Failed to add service to server";
    return -1;
  }

  butil::EndPoint endpoint;
  if (!FLAGS_listen_addr.empty()) {
    if (butil::str2endpoint(FLAGS_listen_addr.c_str(), &endpoint) < 0) {
      LOG(ERROR) << "Invalid listen address:" << FLAGS_listen_addr;
      return -1;
    }
  } else {
    endpoint = butil::EndPoint(butil::IP_ANY, FLAGS_port);
  }

  // Start the server.
  brpc::ServerOptions options;
  options.idle_timeout_sec = FLAGS_idle_timeout_s;
  options.num_threads = FLAGS_num_threads;
  options.max_concurrency = FLAGS_max_concurrency;
  options.idle_timeout_sec = FLAGS_idle_timeout_s;
  if (server.Start(endpoint, &options) != 0) {
    LOG(ERROR) << "Fail to start Brpc rpc server";
    return -1;
  }

  LOG(INFO) << "Xllm rpc service listening on " << server_address;

  // Wait until Ctrl-C is pressed, then Stop() and Join() the server.
  server.RunUntilAskedToQuit();

  return 0;
}
