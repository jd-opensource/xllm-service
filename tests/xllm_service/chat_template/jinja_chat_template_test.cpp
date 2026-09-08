/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm-service/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "chat_template/jinja_chat_template.h"

#include <gtest/gtest.h>

#include <memory>

#include "chat_template/chat_template.h"

namespace xllm_service {

TEST(JinjaChatTemplate, EncodeAddsSpecialTokensViaBase) {
  TokenizerArgs args;
  args.chat_template("{{ messages[0]['content'] }}");
  args.bos_token("");
  args.eos_token("");

  std::unique_ptr<ChatTemplate> tmpl =
      std::make_unique<JinjaChatTemplate>(args);
  EXPECT_TRUE(tmpl->encode_add_special_tokens());
}

TEST(JinjaChatTemplate, OpenChatModel) {
  // clang-format off
  const std::string template_str =
      "<s>"
      "{% for message in messages %}"
        "{{ 'GPT4 Correct ' + message['role'] + ': ' + message['content'] + '<|end_of_turn|>'}}"
      "{% endfor %}"
      "{% if add_generation_prompt %}{{ 'GPT4 Correct Assistant:' }}{% endif %}";

  nlohmann::ordered_json messages = {
      {{"role", "system"}, {"content", "you are a helpful assistant."}},
      {{"role", "user"}, {"content", "hi"}},
      {{"role", "assistant"}, {"content", "what i can do for you?"}},
      {{"role", "user"}, {"content", "how are you?"}}};
  const std::string expected =
    "<s>"
    "GPT4 Correct system: you are a helpful assistant.<|end_of_turn|>"
    "GPT4 Correct user: hi<|end_of_turn|>"
    "GPT4 Correct assistant: what i can do for you?<|end_of_turn|>"
    "GPT4 Correct user: how are you?<|end_of_turn|>"
    "GPT4 Correct Assistant:";
  // clang-format on

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("<|end_of_turn|>");
  JinjaChatTemplate template_(args);
  auto result = template_.apply(messages);
  ASSERT_TRUE(result.has_value());

  EXPECT_EQ(result.value(), expected);
}

TEST(JinjaChatTemplate, ApplyChatTemplateKwargs) {
  const std::string template_str =
      "{% if enable_thinking %}<think>{% endif %}"
      "{% for message in messages %}{{ message['content'] }}{% endfor %}";

  nlohmann::ordered_json messages = {{{"role", "user"}, {"content", "hello"}}};
  nlohmann::ordered_json chat_template_kwargs = {{"enable_thinking", false}};

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  JinjaChatTemplate template_(args);

  auto result = template_.apply(
      messages, nlohmann::ordered_json::array(), chat_template_kwargs);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), "hello");
}

TEST(JinjaChatTemplate, SupportsUndefinedTestForOptionalKwargs) {
  const std::string template_str =
      "{% if enable_thinking is undefined or enable_thinking is true %}"
      "thinking"
      "{% else %}no_thinking{% endif %}"
      "{{ messages[0]['content'] }}";

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  JinjaChatTemplate template_(args);

  nlohmann::ordered_json messages = {{{"role", "user"}, {"content", "hello"}}};
  const nlohmann::ordered_json tools = nlohmann::ordered_json::array();

  const nlohmann::ordered_json empty_kwargs = nlohmann::json::object();
  auto missing_value_result = template_.apply(messages, tools, empty_kwargs);
  ASSERT_TRUE(missing_value_result.has_value());
  EXPECT_EQ(missing_value_result.value(), "thinkinghello");

  const nlohmann::ordered_json false_kwargs = {{"enable_thinking", false}};
  auto false_value_result = template_.apply(messages, tools, false_kwargs);
  ASSERT_TRUE(false_value_result.has_value());
  EXPECT_EQ(false_value_result.value(), "no_thinkinghello");
}

TEST(JinjaChatTemplate, SupportsWhitespaceInUndefinedTests) {
  const std::string template_str =
      "{% if enable_thinking  is\nundefined %}undefined"
      "{% elif enable_thinking is not\tundefined %}defined"
      "{% else %}unexpected{% endif %}";

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  JinjaChatTemplate template_(args);

  nlohmann::ordered_json messages = nlohmann::json::array();
  const nlohmann::ordered_json tools = nlohmann::json::array();

  const nlohmann::ordered_json empty_kwargs = nlohmann::json::object();
  auto missing_value_result = template_.apply(messages, tools, empty_kwargs);
  ASSERT_TRUE(missing_value_result.has_value());
  EXPECT_EQ(missing_value_result.value(), "undefined");

  const nlohmann::ordered_json false_kwargs = {{"enable_thinking", false}};
  auto false_value_result = template_.apply(messages, tools, false_kwargs);
  ASSERT_TRUE(false_value_result.has_value());
  EXPECT_EQ(false_value_result.value(), "defined");
}

TEST(JinjaChatTemplate, SupportsUndefinedTestAfterDelimiterInStringLiteral) {
  const std::string template_str =
      "{% if 'value%}' == 'value%}' and enable_thinking is undefined %}"
      "undefined{% else %}defined{% endif %}";

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  JinjaChatTemplate template_(args);

  nlohmann::ordered_json messages = nlohmann::json::array();
  const nlohmann::ordered_json tools = nlohmann::json::array();
  const nlohmann::ordered_json kwargs = nlohmann::json::object();

  auto result = template_.apply(messages, tools, kwargs);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), "undefined");
}

TEST(JinjaChatTemplate, SupportsPunctuationBeforeUndefinedTest) {
  const std::string template_str =
      "{% if (enable_thinking)is undefined %}missing"
      "{% else %}unexpected{% endif %}";

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  JinjaChatTemplate template_(args);

  nlohmann::ordered_json messages = nlohmann::json::array();
  const nlohmann::ordered_json tools = nlohmann::json::array();

  const nlohmann::ordered_json empty_kwargs = nlohmann::json::object();
  auto missing_result = template_.apply(messages, tools, empty_kwargs);
  ASSERT_TRUE(missing_result.has_value());
  EXPECT_EQ(missing_result.value(), "missing");
}

TEST(JinjaChatTemplate, PreservesNullInUndefinedTests) {
  const std::string template_str =
      "{% if enable_thinking is undefined %}missing"
      "{% elif enable_thinking is not undefined %}defined"
      "{% else %}unexpected{% endif %}";

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  JinjaChatTemplate template_(args);

  nlohmann::ordered_json messages = nlohmann::json::array();
  const nlohmann::ordered_json tools = nlohmann::json::array();
  const nlohmann::ordered_json null_kwargs = {{"enable_thinking", nullptr}};
  auto null_result = template_.apply(messages, tools, null_kwargs);
  ASSERT_TRUE(null_result.has_value());
  EXPECT_EQ(null_result.value(), "defined");
}

TEST(JinjaChatTemplate, PreservesUnaryNotPrecedenceInUndefinedTests) {
  const std::string template_str =
      "{% if not enable_thinking is undefined %}defined"
      "{% else %}missing{% endif %}";

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  JinjaChatTemplate template_(args);

  nlohmann::ordered_json messages = nlohmann::json::array();
  const nlohmann::ordered_json tools = nlohmann::json::array();

  const nlohmann::ordered_json empty_kwargs = nlohmann::json::object();
  auto missing_result = template_.apply(messages, tools, empty_kwargs);
  ASSERT_TRUE(missing_result.has_value());
  EXPECT_EQ(missing_result.value(), "missing");

  const nlohmann::ordered_json null_kwargs = {{"enable_thinking", nullptr}};
  auto null_result = template_.apply(messages, tools, null_kwargs);
  ASSERT_TRUE(null_result.has_value());
  EXPECT_EQ(null_result.value(), "defined");
}

TEST(JinjaChatTemplate, PreservesNestedNullInUndefinedTests) {
  const std::string template_str =
      "{% if options.enable_thinking is undefined %}missing"
      "{% elif options.enable_thinking is not undefined %}defined"
      "{% else %}unexpected{% endif %}";

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  JinjaChatTemplate template_(args);

  nlohmann::ordered_json messages = nlohmann::json::array();
  const nlohmann::ordered_json tools = nlohmann::json::array();
  const nlohmann::ordered_json null_kwargs = {
      {"options", {{"enable_thinking", nullptr}}}};
  auto null_result = template_.apply(messages, tools, null_kwargs);
  ASSERT_TRUE(null_result.has_value());
  EXPECT_EQ(null_result.value(), "defined");
}

TEST(JinjaChatTemplate, PreservesLocalNullInUndefinedTests) {
  const std::string template_str =
      "{% set value = none %}"
      "{% if value is undefined %}missing"
      "{% else %}defined{% endif %}";

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  JinjaChatTemplate template_(args);

  nlohmann::ordered_json messages = nlohmann::json::array();
  const nlohmann::ordered_json tools = nlohmann::json::array();
  const nlohmann::ordered_json kwargs = nlohmann::json::object();

  auto result = template_.apply(messages, tools, kwargs);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), "defined");
}

TEST(JinjaChatTemplate, PreservesWhitespaceControlledLocalNull) {
  const std::string template_str =
      "{% set value = none -%}"
      "{% if value is undefined %}missing"
      "{% else %}defined{% endif %}";

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  JinjaChatTemplate template_(args);

  nlohmann::ordered_json messages = nlohmann::json::array();
  const nlohmann::ordered_json tools = nlohmann::json::array();
  const nlohmann::ordered_json kwargs = nlohmann::json::object();

  auto result = template_.apply(messages, tools, kwargs);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), "defined");
}

TEST(JinjaChatTemplate, PreservesWhitespaceControlAfterLocalNullInjection) {
  const std::string template_str =
      "{% set value = none -%}\n"
      "{% if value is undefined %}missing{% else %}Hello{% endif %}";

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  JinjaChatTemplate template_(args);

  nlohmann::ordered_json messages = nlohmann::json::array();
  const nlohmann::ordered_json tools = nlohmann::json::array();
  const nlohmann::ordered_json kwargs = nlohmann::json::object();

  auto result = template_.apply(messages, tools, kwargs);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), "Hello");
}

TEST(JinjaChatTemplate, DoesNotTreatNoneLiteralAsUndefined) {
  const std::string template_str =
      "{% if none is undefined %}unexpected"
      "{% else %}defined{% endif %}";

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  JinjaChatTemplate template_(args);

  nlohmann::ordered_json messages = nlohmann::json::array();
  const nlohmann::ordered_json tools = nlohmann::json::array();
  const nlohmann::ordered_json kwargs = nlohmann::json::object();

  auto result = template_.apply(messages, tools, kwargs);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), "defined");
}

TEST(JinjaChatTemplate, DoesNotTreatUppercaseNoneLiteralAsUndefined) {
  const std::string template_str =
      "{% if None is undefined %}unexpected"
      "{% else %}defined{% endif %}";

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  JinjaChatTemplate template_(args);

  nlohmann::ordered_json messages = nlohmann::json::array();
  const nlohmann::ordered_json tools = nlohmann::json::array();
  const nlohmann::ordered_json kwargs = nlohmann::json::object();

  auto result = template_.apply(messages, tools, kwargs);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), "defined");
}

TEST(JinjaChatTemplate, SupportsUndefinedTestCallSyntax) {
  const std::string template_str =
      "{% if enable_thinking is undefined() %}missing"
      "{% else %}defined{% endif %}";

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  JinjaChatTemplate template_(args);

  nlohmann::ordered_json messages = nlohmann::json::array();
  const nlohmann::ordered_json tools = nlohmann::json::array();
  const nlohmann::ordered_json kwargs = nlohmann::json::object();

  auto result = template_.apply(messages, tools, kwargs);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), "missing");
}

TEST(JinjaChatTemplate, RemovesPresenceAfterLocalBecomesUndefined) {
  const std::string template_str =
      "{% set value = none %}"
      "{% set value = missing %}"
      "{% if value is undefined %}missing"
      "{% else %}defined{% endif %}";

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  JinjaChatTemplate template_(args);

  nlohmann::ordered_json messages = nlohmann::json::array();
  const nlohmann::ordered_json tools = nlohmann::json::array();
  const nlohmann::ordered_json kwargs = nlohmann::json::object();

  auto result = template_.apply(messages, tools, kwargs);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), "missing");
}

TEST(JinjaChatTemplate, PreservesNullAssignedFromContextPath) {
  const std::string template_str =
      "{% set value = source %}"
      "{% if value is undefined %}missing"
      "{% else %}defined{% endif %}";

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  JinjaChatTemplate template_(args);

  nlohmann::ordered_json messages = nlohmann::json::array();
  const nlohmann::ordered_json tools = nlohmann::json::array();
  const nlohmann::ordered_json kwargs = {{"source", nullptr}};

  auto result = template_.apply(messages, tools, kwargs);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), "defined");
}

TEST(JinjaChatTemplate, RemovesDescendantPresenceAfterLocalOverwrite) {
  const std::string template_str =
      "{% set options = {} %}"
      "{% if options.flag is undefined %}missing"
      "{% else %}defined{% endif %}";

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  JinjaChatTemplate template_(args);

  nlohmann::ordered_json messages = nlohmann::json::array();
  const nlohmann::ordered_json tools = nlohmann::json::array();
  const nlohmann::ordered_json kwargs = {{"options", {{"flag", nullptr}}}};

  auto result = template_.apply(messages, tools, kwargs);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), "missing");
}

TEST(JinjaChatTemplate, PropagatesDescendantPresenceThroughLocalAlias) {
  const std::string template_str =
      "{% set alias = options %}"
      "{% if alias.flag is undefined %}missing"
      "{% else %}defined{% endif %}";

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  JinjaChatTemplate template_(args);

  nlohmann::ordered_json messages = nlohmann::json::array();
  const nlohmann::ordered_json tools = nlohmann::json::array();
  const nlohmann::ordered_json kwargs = {{"options", {{"flag", nullptr}}}};

  auto result = template_.apply(messages, tools, kwargs);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), "defined");
}

TEST(JinjaChatTemplate, PreservesUnderscoreBindingDuringPresenceTracking) {
  const std::string template_str =
      "{% set _ = 'visible' %}"
      "{% set value = none %}"
      "{{ _ }}";

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  JinjaChatTemplate template_(args);

  nlohmann::ordered_json messages = nlohmann::json::array();
  const nlohmann::ordered_json tools = nlohmann::json::array();
  const nlohmann::ordered_json kwargs = nlohmann::json::object();

  auto result = template_.apply(messages, tools, kwargs);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), "visible");
}

TEST(JinjaChatTemplate, SupportsNestedBracesBeforeUndefinedTest) {
  const std::string template_str = "{{ {\"a\": {}} or opt is undefined }}";

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  JinjaChatTemplate template_(args);

  nlohmann::ordered_json messages = nlohmann::json::array();
  const nlohmann::ordered_json tools = nlohmann::json::array();
  const nlohmann::ordered_json kwargs = nlohmann::json::object();

  auto result = template_.apply(messages, tools, kwargs);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), "{'a': {}}");
}

TEST(JinjaChatTemplate, PreservesBinaryOperandInUndefinedTest) {
  const std::string template_str =
      "{% if 1 + value is undefined %}missing"
      "{% else %}defined{% endif %}";

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  JinjaChatTemplate template_(args);

  nlohmann::ordered_json messages = nlohmann::json::array();
  const nlohmann::ordered_json tools = nlohmann::json::array();
  const nlohmann::ordered_json kwargs = {{"value", 2}};

  auto result = template_.apply(messages, tools, kwargs);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), "defined");
}

TEST(JinjaChatTemplate, DoesNotConflateLiteralDotsInPresencePaths) {
  const std::string template_str =
      "{% if options.enable_thinking is undefined %}missing"
      "{% else %}defined{% endif %}";

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  JinjaChatTemplate template_(args);

  nlohmann::ordered_json messages = nlohmann::json::array();
  const nlohmann::ordered_json tools = nlohmann::json::array();
  const nlohmann::ordered_json kwargs = {{"options", nlohmann::json::object()},
                                         {"options.enable_thinking", false}};

  auto result = template_.apply(messages, tools, kwargs);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), "missing");
}

TEST(JinjaChatTemplate, DoesNotTreatFilterNameAsContextPath) {
  const std::string template_str =
      "{% if value | default is undefined %}missing"
      "{% else %}defined{% endif %}";

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  JinjaChatTemplate template_(args);

  nlohmann::ordered_json messages = nlohmann::json::array();
  const nlohmann::ordered_json tools = nlohmann::json::array();
  const nlohmann::ordered_json kwargs = nlohmann::json::object();

  auto result = template_.apply(messages, tools, kwargs);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), "missing");
}

TEST(JinjaChatTemplate, RejectsNonObjectChatTemplateKwargs) {
  TokenizerArgs args;
  args.chat_template("{{ messages[0]['content'] }}");
  args.bos_token("");
  args.eos_token("");
  JinjaChatTemplate template_(args);

  nlohmann::ordered_json messages = {{{"role", "user"}, {"content", "hello"}}};
  const nlohmann::ordered_json tools = nlohmann::json::array();
  const nlohmann::ordered_json invalid_kwargs = nlohmann::json::array();

  auto result = template_.apply(messages, tools, invalid_kwargs);
  EXPECT_FALSE(result.has_value());
}

TEST(JinjaChatTemplate, RejectsReservedChatTemplateKwargsKey) {
  TokenizerArgs args;
  args.chat_template("{{ messages[0]['content'] }}");
  args.bos_token("");
  args.eos_token("");
  JinjaChatTemplate template_(args);

  nlohmann::ordered_json messages = {{{"role", "user"}, {"content", "hello"}}};
  const nlohmann::ordered_json tools = nlohmann::json::array();
  const nlohmann::ordered_json kwargs = {
      {"__xllm_chat_template_kwargs_presence", "user_value"}};

  auto result = template_.apply(messages, tools, kwargs);
  EXPECT_FALSE(result.has_value());
}

TEST(JinjaChatTemplate, MakesThinkingAndEffortAvailableInExtraContext) {
  TokenizerArgs args;
  args.chat_template(
      "{% if thinking %}thinking{% endif %}:{{ reasoning_effort }}");
  args.bos_token("");
  args.eos_token("");
  JinjaChatTemplate template_(args);

  nlohmann::ordered_json messages = nlohmann::ordered_json::array();
  nlohmann::ordered_json tools = nlohmann::ordered_json::array();
  nlohmann::ordered_json kwargs = {{"thinking", true},
                                   {"reasoning_effort", "high"}};
  auto result = template_.apply(messages, tools, kwargs);

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), "thinking:high");
}

TEST(JinjaChatTemplate, RendersAnthropicOrderedBlocks) {
  const std::string template_str =
      "{% for item in messages[0]['content'] %}"
      "{{ item['type'] }}:"
      "{% if item['type'] == 'thinking' %}{{ item['thinking'] }}:{{ "
      "item['signature'] }}{% endif %}"
      "{% if item['type'] == 'tool_use' %}{{ item['id'] }}:{{ item['name'] "
      "}}:{{ item['input']['path'] }}{% endif %}"
      "{% if item['type'] == 'redacted_thinking' %}{{ item['data'] }}{% endif "
      "%}"
      "{% if item['type'] == 'text' %}{{ item['text'] }}{% endif %}|"
      "{% endfor %}"
      "reasoning={{ messages[0]['reasoning_content'] }}";

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  JinjaChatTemplate template_(args);

  Message::MMContent thinking("thinking");
  thinking.thinking = "need tool";
  thinking.signature = "sig_1";

  Message::MMContent tool_use("tool_use");
  tool_use.id = "toolu_1";
  tool_use.name = "Read";
  tool_use.input = {{"path", "a.txt"}};

  Message::MMContent redacted("redacted_thinking");
  redacted.data = "opaque";

  Message::MMContent text("text", "done");

  Message message("assistant",
                  Message::MMContentVec{thinking, tool_use, redacted, text});
  message.reasoning_content = "need tool";
  ChatMessages messages{message};

  auto result = template_.apply(messages);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result.value(),
            "thinking:need tool:sig_1|tool_use:toolu_1:Read:a.txt|"
            "redacted_thinking:opaque|text:done|reasoning=need tool");
}

TEST(JinjaChatTemplate, RendersPreservedThinkingForStringTemplates) {
  const std::string template_str =
      "{% for message in messages %}"
      "{{ message['role'] }}:"
      "{% if message['content'] is string %}{{ message['content'] }}{% endif "
      "%}|"
      "{% if message.get('tool_calls') %}"
      "{% for tool_call in message['tool_calls'] %}"
      "call={{ tool_call['function']['arguments'] }}|"
      "{% endfor %}"
      "{% endif %}"
      "{% endfor %}";

  TokenizerArgs args;
  args.chat_template(template_str);
  args.bos_token("");
  args.eos_token("");
  JinjaChatTemplate template_(args);

  Message::MMContent thinking("thinking");
  thinking.thinking = "PTB_marker";
  thinking.signature = "sig_1";

  Message::MMContent tool_use("tool_use");
  tool_use.id = "toolu_1";
  tool_use.name = "preserve_probe";
  tool_use.input = {{"marker", "TIB_marker"}};

  Message::MMContent text("text", "done");

  Message assistant("assistant",
                    Message::MMContentVec{thinking, tool_use, text});
  assistant.reasoning_content = "PTB_marker";
  Message::ToolCall tool_call;
  tool_call.id = "toolu_1";
  tool_call.type = "function";
  tool_call.function.name = "preserve_probe";
  tool_call.function.arguments = R"({"marker":"TIB_marker"})";
  assistant.tool_calls = Message::ToolCallVec{tool_call};

  Message tool("tool", "TRB_marker");
  tool.tool_call_id = "toolu_1";

  auto result = template_.apply(ChatMessages{assistant, tool});
  ASSERT_TRUE(result.has_value());
  EXPECT_NE(result->find("<think>PTB_marker</think>"), std::string::npos);
  EXPECT_NE(result->find("TIB_marker"), std::string::npos);
  EXPECT_NE(result->find("TRB_marker"), std::string::npos);
}

}  // namespace xllm_service
