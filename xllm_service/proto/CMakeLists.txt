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
    xllm/chat.proto
    xllm/common.proto
    xllm/completion.proto
)
