include(cc_library)
include(cc_test)

cc_library(
  NAME
   managers
  HDRS
    instance_mgr.h
    instance_metrics.h
    instance_topology.h
    instance_kvcache.h
  SRCS
    instance_mgr.cpp
    instance_metrics.cpp
    instance_topology.cpp
    instance_kvcache.cpp
  DEPS
    :chat_template
    :common
    :etcd_client
    :request
    absl::random_random
    absl::strings
    glog::glog
    proto::proto_rpc_service
    proto_xllm
)
target_link_libraries(managers PRIVATE brpc-static)
