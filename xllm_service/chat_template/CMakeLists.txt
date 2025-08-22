include(cc_library)
include(cc_test)

cc_library (
  NAME
    chat_template
  HDRS
    jinja_chat_template.h
    chat_template_factory.h
  SRCS
    jinja_chat_template.cpp
    chat_template_factory.cpp
  DEPS
    :minja
    :tokenizer
    nlohmann_json::nlohmann_json
    glog::glog
)

cc_test (
  NAME
    chat_template_test
  SRCS
    jinja_chat_template_test.cpp
  DEPS
    :chat_template
    GTest::gtest_main
)
