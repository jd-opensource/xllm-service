include(cc_library)
include(cc_test)

cc_library(
  NAME
  instance_service
  HDRS
    instance_client.h
    instance_service.h
  SRCS
    instance_client.cpp
    instance_service.cpp
  DEPS
    :common
    :runtime
    absl::random_random
    absl::strings
    glog::glog
    proto::proto_rpc_service
    proto_xllm
)
target_link_libraries(instance_service PRIVATE brpc-static)

cc_library(
  NAME
   instances
  HDRS
    instance_mgr.h
    global_kvcache_mgr.h
  SRCS
    instance_mgr.cpp
    global_kvcache_mgr.cpp
  DEPS
    :chat_template
    :common
    :request
    absl::random_random
    absl::strings
    glog::glog
    proto::proto_rpc_service
    proto_xllm
)
target_link_libraries(instances PRIVATE brpc-static)
