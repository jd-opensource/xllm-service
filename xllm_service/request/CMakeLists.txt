include(cc_library)

add_subdirectory(tokenizer)
add_subdirectory(chat_template)

cc_library(
  NAME
    request
  HDRS
    request.h
    request_tracer.h
  SRCS
    request_tracer.cpp
  DEPS
    :common
)