include(cc_binary)

cc_binary(
  NAME 
    rpc_hello_client
  SRCS 
    rpc_hello_client.cpp
  DEPS
    :xllm_rpc_service
    gflags::gflags
    glog::glog
)

cc_binary(
  NAME 
    rpc_client_test
  SRCS 
    rpc_client_test.cpp
  DEPS
    :xllm_rpc_client
    gflags::gflags
    glog::glog
)

cc_binary(
  NAME
    http_client_test
  SRCS
    http_client_test.cpp
  DEPS
    gflags::gflags
)
target_link_libraries(http_client_test PRIVATE brpc-static)
