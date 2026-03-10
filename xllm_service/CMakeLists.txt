include(cc_binary)
include(cc_library)
include(cc_test)

add_subdirectory(proto)
add_subdirectory(common)
add_subdirectory(request)
add_subdirectory(instances)
add_subdirectory(scheduler)
add_subdirectory(runtime)
add_subdirectory(api_service)

cc_binary(
  NAME
    xllm_master_serving
  HDRS
    xllm_service.h
  SRCS
    xllm_service.cpp
  DEPS
    :api_service
    :instance_service
    glog::glog
)

add_subdirectory(examples)
