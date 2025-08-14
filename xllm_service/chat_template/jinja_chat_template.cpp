#include "chat_template/jinja_chat_template.h"

#include <glog/logging.h>
#include <unistd.h>

#include <optional>
#include <string>

namespace xllm_service {

JinjaChatTemplate::JinjaChatTemplate(const TokenizerArgs& args) : args_(args) {
  try {
    template_ = std::make_unique<minja::chat_template>(
        args_.chat_template(), args_.bos_token(), args_.eos_token());
    LOG(INFO) << "Jinja chat template init succeed.";

  } catch (const std::exception& e) {
    LOG(FATAL) << "Failed to parse jinja chat template, TokenizerArgs: "
               << args_ << std::endl
               << "Error message: " << e.what();
  }
}

std::optional<std::string> JinjaChatTemplate::apply(
    const ChatMessages& messages) const {
  // convert the messages to json object
  nlohmann::ordered_json messages_json = nlohmann::json::array();
  for (const auto& message : messages) {
    nlohmann::ordered_json message_json;
    message_json["role"] = message.role;
    message_json["content"] = message.content;
    messages_json.push_back(message_json);
  }
  // apply the template
  return apply(messages_json);
}

std::optional<std::string> JinjaChatTemplate::apply(
    nlohmann::ordered_json& messages) const {
  minja::chat_template_inputs input;
  input.messages = messages;
  input.add_generation_prompt = true;
  minja::chat_template_options options;

  return template_->apply(input, options);
}

}  // namespace xllm_service