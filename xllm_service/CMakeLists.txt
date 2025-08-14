include(cc_binary)
include(cc_library)
include(cc_test)

add_subdirectory(proto)
add_subdirectory(common)
add_subdirectory(rpc_service)
add_subdirectory(tokenizer)
add_subdirectory(chat_template)
add_subdirectory(http_service)

cc_binary(
  NAME
    xllm_master_serving
  HDRS
    master.h
  SRCS
    master.cpp
  DEPS
    :xllm_http_service
    :xllm_rpc_service
)

add_subdirectory(examples)
