include(grpc_proto_library)
include(proto_library)

proto_library(
  NAME
    proto_rpc_service
  SRCS
    xllm_rpc_service.proto
)

proto_library(
  NAME
    proto_http_service
  SRCS
    xllm_http_service.proto
)

proto_library(
  NAME
    proto_xllm
  SRCS
    ${CMAKE_SOURCE_DIR}/third_party/xllm/xllm/proto/chat.proto
    ${CMAKE_SOURCE_DIR}/third_party/xllm/xllm/proto/common.proto
    ${CMAKE_SOURCE_DIR}/third_party/xllm/xllm/proto/completion.proto
    ${CMAKE_SOURCE_DIR}/third_party/xllm/xllm/proto/rec.proto
    ${CMAKE_SOURCE_DIR}/third_party/xllm/xllm/proto/disagg_pd.proto
)
