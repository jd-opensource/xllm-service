include(cc_binary)
include(cc_library)
include(cc_test)
add_subdirectory(loadbalance_policy)

cc_library(
  NAME
  xllm_rpc_service
  HDRS
    scheduler.h
    etcd_client.h
    instance_mgr.h
    global_kvcache_mgr.h
    response_handler.h
    service.h
  SRCS
    scheduler.cpp
    etcd_client.cpp
    instance_mgr.cpp
    global_kvcache_mgr.cpp
    response_handler.cpp
    service.cpp
  DEPS
    :common
    :loadbalance_policy
    absl::random_random
    absl::strings
    cpprest
    etcd-cpp-api
    glog::glog
    nlohmann_json::nlohmann_json
    proto::proto_rpc_service
    proto_xllm
    tokenizer
    chat_template
)
target_link_libraries(xllm_rpc_service PRIVATE brpc-static)

cc_binary(
  NAME 
    xllm_rpc_service_test
  SRCS 
    rpc_service_test.cpp
  DEPS
    :xllm_rpc_service
    gflags::gflags
    glog::glog
    GTest::gtest_main
)
add_test(NAME XllmRpcServiceTest COMMAND xllm_rpc_service_test)

cc_binary(
  NAME 
    xllm_rpc_serving
  SRCS 
    main.cpp
  DEPS
    :xllm_rpc_service
    gflags::gflags   
)

cc_library(
  NAME
  xllm_rpc_client
  HDRS
    client.h
  SRCS
    client.cpp
  DEPS
    :common
    glog::glog
    proto::proto_rpc_service
)
target_link_libraries(xllm_rpc_client PRIVATE brpc-static)
