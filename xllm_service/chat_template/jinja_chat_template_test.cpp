#include "chat_template/jinja_chat_template.h"

#include <gtest/gtest.h>

namespace xllm_service {

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

}  // namespace xllm_service
