#pragma once

#include <optional>
#include <sstream>
#include <string>

#include "chat_template/chat_template.h"

namespace xllm_service {

// A chat template implementation that embeds template logic in the code.
class CodedChatTemplate : public ChatTemplate {
 public:
  CodedChatTemplate() = default;

  std::optional<std::string> apply(const ChatMessages& messages) const override;

  // generate prompt from dialogs
  virtual std::optional<std::string> get_prompt(
      const std::string_view& system_message,
      const std::vector<std::string_view>& messages) const = 0;
};

class ChatGLMChatTemplate final : public CodedChatTemplate {
 public:
  // generate prompt from dialogs
  // https://github.com/THUDM/ChatGLM3/blob/main/PROMPT.md
  std::optional<std::string> get_prompt(
      const std::string_view& system_message,
      const std::vector<std::string_view>& messages) const override {
    // at least one user message
    if (messages.size() % 2 == 0) {
      return std::nullopt;
    }

    std::stringstream ss;
    if (!system_message.empty()) {
      ss << "<|system|>\n" << system_message << "\n";
    }

    // then user and assistant message pairs (u/a/u/a/u...)
    for (size_t i = 0; i < messages.size(); ++i) {
      const char* role = (i % 2) == 0 ? "user" : "assistant";
      ss << "<|" << role << "|>\n" << messages[i] << "\n";
    }
    // end with assistant message
    ss << "<|assistant|>\n";
    return ss.str();
  }
};

class ChatGLM4ChatTemplate final : public CodedChatTemplate {
 public:
  // generate prompt from dialogs
  // ref to
  // https://huggingface.co/THUDM/glm-4-9b/blob/main/tokenization_chatglm.py#L144
  std::optional<std::string> get_prompt(
      const std::string_view& system_message,
      const std::vector<std::string_view>& messages) const override {
    // at least one user message
    if (messages.size() % 2 == 0) {
      return std::nullopt;
    }

    // TODO: support function calls.
    // message format: <|{role}|>{metadata}\n{message}
    std::stringstream ss;
    if (!system_message.empty()) {
      ss << "<|system|>\n" << system_message << "\n";
    }

    // then user and assistant message pairs (u/a/u/a/u...)
    for (size_t i = 0; i < messages.size(); ++i) {
      const char* role = (i % 2) == 0 ? "user" : "assistant";
      ss << "<|" << role << "|>\n" << messages[i] << "\n";
    }
    // end with assistant message
    ss << "<|assistant|>\n";
    return ss.str();
  }
};

class MiniCPMVChatTemplate final : public CodedChatTemplate {
 public:
  std::optional<std::string> get_prompt(
      const std::string_view& system_message,
      const std::vector<std::string_view>& messages) const override {
    // at least one user message
    if (messages.size() % 2 == 0) {
      return std::nullopt;
    }

    std::stringstream ss;
    if (!system_message.empty()) {
      ss << "<|im_start|>system\n" << system_message << "<|im_end|>\n";
    } else {
      ss << "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
    }

    // then user and assistant message pairs (u/a/u/a/u...)
    for (size_t i = 0; i < messages.size(); ++i) {
      const char* role = (i % 2) == 0 ? "user" : "assistant";
      ss << "<|im_start|>" << role << "\n" << messages[i] << "<|im_end|>\n";
    }
    // end with assistant message
    ss << "<|im_start|>assistant\n";
    return ss.str();
  }
};

class QwenChatTemplate final : public CodedChatTemplate {
 public:
  // Prompt template:
  // <|im_start|>user\n {message} <|im_end|>\n
  // <|im_start|>assistant\n
  std::optional<std::string> get_prompt(
      const std::string_view& system_message,
      const std::vector<std::string_view>& messages) const override {
    // at least one user message
    if (messages.size() % 2 == 0) {
      return std::nullopt;
    }

    std::stringstream ss;
    if (!system_message.empty()) {
      ss << "<|im_start|>system\n" << system_message << "<|im_end|>\n";
    }

    // then user and assistant message pairs (u/a/u/a/u...)
    for (size_t i = 0; i < messages.size(); ++i) {
      const char* role = (i % 2) == 0 ? "user" : "assistant";
      ss << "<|im_start|>" << role << "\n" << messages[i] << "<|im_end|>\n";
    }
    // end with assistant message
    ss << "<|im_start|>assistant\n";
    return ss.str();
  }
};

}  // namespace xllm_service