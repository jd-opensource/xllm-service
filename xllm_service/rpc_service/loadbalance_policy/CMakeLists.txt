include(cc_binary)
include(cc_library)
include(cc_test)

cc_library(
  NAME
  loadbalance_policy
  HDRS
    loadbalance_policy.h
  SRCS
    loadbalance_policy.cpp
  DEPS
    :common
)
