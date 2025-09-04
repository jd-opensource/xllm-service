include(cc_library)

add_subdirectory(etcd_client)
add_subdirectory(managers)
add_subdirectory(loadbalance_policy)

cc_library(
  NAME
    scheduler
  HDRS
    response_handler.h
    scheduler.h
  SRCS
    response_handler.cpp
    scheduler.cpp
  DEPS
    :chat_template
    :common
    :etcd_client
    :loadbalance_policy
    :managers
    :request
    cpprest
    etcd-cpp-api
    glog::glog
    nlohmann_json::nlohmann_json
)