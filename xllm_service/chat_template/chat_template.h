#pragma once
#include <optional>
#include <string>
#include <vector>

namespace xllm_service {

struct Message {
  Message() = default;
  Message(const std::string& role, const std::string& content)
      : role(role), content(content) {}

  std::string role;
  std::string content;
};
using ChatMessages = std::vector<Message>;

// ChatTemplate only supports 'system', 'user' and 'assistant' roles.
// start with system message, then user and assistant message. (u/a/u/a/u...)
class ChatTemplate {
 public:
  virtual ~ChatTemplate() = default;

  // apply the chat template to the messages
  // return std::nullopt if the template is not applicable.
  virtual std::optional<std::string> apply(
      const ChatMessages& messages) const = 0;
};

}  // namespace xllm_service