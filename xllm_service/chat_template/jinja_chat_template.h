#pragma once

#include <minja/chat-template.hpp>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>

#include "chat_template.h"
#include "tokenizer/tokenizer_args.h"

namespace xllm_service {

// A chat template implementation that uses jinja2 as the template engine.
class JinjaChatTemplate : public ChatTemplate {
 public:
  JinjaChatTemplate(const TokenizerArgs& args);

  std::optional<std::string> apply(const ChatMessages& messages) const override;

  // expose this function for testing
  // apply the template to the values in the json object
  std::optional<std::string> apply(nlohmann::ordered_json& messages) const;

 private:
  TokenizerArgs args_;
  std::unique_ptr<minja::chat_template> template_;
};

}  // namespace xllm_service
