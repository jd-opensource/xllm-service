include(cc_library)
include(cc_test)

cc_library(
  NAME
  etcd_client
  HDRS
    etcd_client.h
  SRCS
    etcd_client.cpp
  DEPS
    :common
    cpprest
    etcd-cpp-api
    glog::glog
    nlohmann_json::nlohmann_json
)