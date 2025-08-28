/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#pragma once

#include <minja/chat-template.hpp>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "common/types.h"
#include "tokenizer/tokenizer_args.h"

namespace xllm_service {

struct Message {
  struct MMUrl {
    std::string url;
  };

  struct MMContent {
    MMContent(const std::string& type) : type(type) {}
    MMContent(const std::string& type, const std::string& text)
        : type(type), text(text) {}

    std::string type;

    std::string text;
    MMUrl image_url;  // image place holder

    MMUrl video_url;  // video place holder
    MMUrl audio_url;  // audio place holder
  };

  using MMContentVec = std::vector<MMContent>;
  using Content = std::variant<std::string, MMContentVec>;

  Message() = default;
  Message(const std::string& role, const std::string& content)
      : role(role), content(content) {}

  Message(const std::string& role, const MMContentVec& content)
      : role(role), content(content) {}

  std::string role;
  Content content;
};
using ChatMessages = std::vector<Message>;

// A chat template implementation that uses jinja2 as the template engine.
class JinjaChatTemplate {
 public:
  JinjaChatTemplate(const TokenizerArgs& args);

  std::optional<std::string> apply(const ChatMessages& messages) const;

  std::optional<std::string> apply(
      const ChatMessages& messages,
      const std::vector<xllm_service::JsonTool>& json_tools) const;

  // expose this function for testing
  // apply the template to the values in the json object
  std::optional<std::string> apply(nlohmann::ordered_json& messages) const;

  std::optional<std::string> apply(nlohmann::ordered_json& messages,
                                   const nlohmann::ordered_json& tools) const;

 private:
  nlohmann::ordered_json get_mm_content(const Message::MMContentVec& vec) const;

 private:
  TokenizerArgs args_;
  std::unique_ptr<minja::chat_template> template_;
};

}  // namespace xllm_service
