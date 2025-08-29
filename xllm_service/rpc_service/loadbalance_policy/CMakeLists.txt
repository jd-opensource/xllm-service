include(cc_binary)
include(cc_library)
include(cc_test)

cc_library(
  NAME
  loadbalance_policy
  HDRS
    loadbalance_policy.h
    round_robin.h
    cache_aware_routing.h
  SRCS
    round_robin.cpp
    cache_aware_routing.cpp
  DEPS
    :common
    :managers
)
