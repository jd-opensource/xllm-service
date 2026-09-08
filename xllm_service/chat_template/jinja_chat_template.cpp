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

#include "jinja_chat_template.h"

#include <glog/logging.h>
#include <unistd.h>

#include <cctype>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace xllm_service {
namespace {

constexpr std::string_view kChatTemplateKwargsPresence =
    "__xllm_chat_template_kwargs_presence";
constexpr std::string_view kChatTemplateKwargsPresenceState =
    "__xllm_chat_template_kwargs_presence_state";

std::string_view trim_whitespace(std::string_view text);

std::string escape_json_pointer_token(std::string_view token) {
  std::string escaped;
  for (const char character : token) {
    if (character == '~') {
      escaped += "~0";
    } else if (character == '/') {
      escaped += "~1";
    } else {
      escaped += character;
    }
  }
  return escaped;
}

std::string presence_path_for_context_path(std::string_view context_path) {
  std::string path;
  size_t component_start = 0;
  while (component_start < context_path.size()) {
    const size_t component_end = context_path.find('.', component_start);
    const std::string_view component =
        context_path.substr(component_start, component_end - component_start);
    path += "/" + escape_json_pointer_token(component);
    if (component_end == std::string_view::npos) {
      break;
    }
    component_start = component_end + 1;
  }
  return path;
}

void append_present_paths(const nlohmann::ordered_json& value,
                          const std::string& path,
                          nlohmann::ordered_json& paths) {
  if (value.is_object()) {
    for (const auto& item : value.items()) {
      const std::string item_path =
          path + "/" + escape_json_pointer_token(item.key());
      paths.push_back(item_path);
      append_present_paths(item.value(), item_path, paths);
    }
  } else if (value.is_array()) {
    for (size_t index = 0; index < value.size(); ++index) {
      const std::string item_path = path + "/" + std::to_string(index);
      paths.push_back(item_path);
      append_present_paths(value[index], item_path, paths);
    }
  }
}

bool get_str_field(const nlohmann::ordered_json& json,
                   const char* key,
                   std::string* value) {
  if (!json.contains(key) || !json[key].is_string()) {
    return false;
  }
  *value = json[key].get<std::string>();
  return true;
}

std::vector<std::string> thinking_texts(
    const nlohmann::ordered_json& messages) {
  std::vector<std::string> texts;
  for (const auto& message : messages) {
    if (!message.contains("content") || !message["content"].is_array()) {
      continue;
    }
    for (const auto& item : message["content"]) {
      std::string type;
      std::string thinking;
      if (get_str_field(item, "type", &type) && type == "thinking" &&
          get_str_field(item, "thinking", &thinking) && !thinking.empty()) {
        texts.emplace_back(std::move(thinking));
      }
    }
  }
  return texts;
}

bool has_all_thinking(const std::string& prompt,
                      const nlohmann::ordered_json& messages) {
  for (const auto& thinking : thinking_texts(messages)) {
    if (prompt.find(thinking) == std::string::npos) {
      return false;
    }
  }
  return true;
}

std::string join_parts(const std::vector<std::string>& parts) {
  std::string text;
  for (const auto& part : parts) {
    if (part.empty()) {
      continue;
    }
    if (!text.empty()) {
      text += '\n';
    }
    text += part;
  }
  return text;
}

std::string prompt_content(const nlohmann::ordered_json& content) {
  std::vector<std::string> parts;
  for (const auto& item : content) {
    std::string type;
    if (!get_str_field(item, "type", &type)) {
      continue;
    }
    if (type == "thinking") {
      std::string thinking;
      if (get_str_field(item, "thinking", &thinking) && !thinking.empty()) {
        parts.emplace_back("<think>" + thinking + "</think>");
      }
    } else if (type == "text") {
      std::string text;
      if (get_str_field(item, "text", &text) && !text.empty()) {
        parts.emplace_back(std::move(text));
      }
    }
  }
  return join_parts(parts);
}

bool has_thinking_block(const nlohmann::ordered_json& content) {
  for (const auto& item : content) {
    std::string type;
    if (get_str_field(item, "type", &type) && type == "thinking") {
      return true;
    }
  }
  return false;
}

nlohmann::ordered_json with_thinking_fallback(
    const nlohmann::ordered_json& messages) {
  auto fallback = messages;
  for (auto& message : fallback) {
    if (!message.contains("content") || !message["content"].is_array()) {
      continue;
    }
    if (!has_thinking_block(message["content"])) {
      continue;
    }
    auto text = prompt_content(message["content"]);
    if (!text.empty()) {
      message["content"] = std::move(text);
    }
  }
  return fallback;
}

void replace_undefined_tests(std::string& block) {
  constexpr std::string_view kIs = "is";
  constexpr std::string_view kNot = "not";
  constexpr std::string_view kUndefined = "undefined";
  constexpr std::string_view kIsNone = "is none";
  constexpr std::string_view kIsNotNone = "is not none";

  const auto is_whitespace = [](char character) {
    return std::isspace(static_cast<unsigned char>(character)) != 0;
  };
  const auto is_identifier_character = [](char character) {
    return std::isalnum(static_cast<unsigned char>(character)) != 0 ||
           character == '_';
  };
  const auto is_context_path_character =
      [&is_identifier_character](char character) {
        return is_identifier_character(character) || character == '.';
      };
  const auto skip_whitespace = [&is_whitespace](const std::string& text,
                                                size_t pos) {
    while (pos < text.size() && is_whitespace(text[pos])) {
      ++pos;
    }
    return pos;
  };
  const auto context_path_before = [&is_whitespace, &is_context_path_character](
                                       const std::string& text, size_t pos) {
    while (pos > 0 && is_whitespace(text[pos - 1])) {
      --pos;
    }
    const size_t path_end = pos;
    while (pos > 0 && is_context_path_character(text[pos - 1])) {
      --pos;
    }
    if (pos == path_end || text[pos] == '.' || text[path_end - 1] == '.') {
      return std::string_view();
    }
    size_t preceding_pos = pos;
    while (preceding_pos > 0 && is_whitespace(text[preceding_pos - 1])) {
      --preceding_pos;
    }
    if (preceding_pos > 0) {
      const char preceding = text[preceding_pos - 1];
      if (preceding == '|' || preceding == '+' || preceding == '-' ||
          preceding == '*' || preceding == '/' || preceding == '%' ||
          preceding == '~') {
        return std::string_view();
      }
    }
    return std::string_view(text.data() + pos, path_end - pos);
  };

  char quote = '\0';
  for (size_t pos = 0; pos < block.size();) {
    const char current = block[pos];
    if (quote != '\0') {
      if (current == '\\') {
        pos += 2;
      } else if (current == quote) {
        quote = '\0';
        ++pos;
      } else {
        ++pos;
      }
      continue;
    }
    if (current == '\'' || current == '"') {
      quote = current;
      ++pos;
      continue;
    }

    if (block.compare(pos, kIs.size(), kIs) != 0 ||
        (pos > 0 && is_identifier_character(block[pos - 1]))) {
      ++pos;
      continue;
    }

    size_t token_end = pos + kIs.size();
    const size_t test_pos = skip_whitespace(block, token_end);
    if (test_pos == token_end) {
      ++pos;
      continue;
    }

    bool has_not = false;
    if (block.compare(test_pos, kNot.size(), kNot) == 0) {
      token_end = test_pos + kNot.size();
      const size_t undefined_pos = skip_whitespace(block, token_end);
      if (undefined_pos == token_end) {
        ++pos;
        continue;
      }
      token_end = undefined_pos;
      has_not = true;
    } else {
      token_end = test_pos;
    }

    if (block.compare(token_end, kUndefined.size(), kUndefined) != 0 ||
        (token_end + kUndefined.size() < block.size() &&
         is_identifier_character(block[token_end + kUndefined.size()]))) {
      ++pos;
      continue;
    }

    const std::string_view context_path = context_path_before(block, pos);
    size_t replace_end = token_end + kUndefined.size();
    const size_t call_pos = skip_whitespace(block, replace_end);
    if (call_pos < block.size() && block[call_pos] == '(') {
      const size_t close_paren_pos = skip_whitespace(block, call_pos + 1);
      if (close_paren_pos < block.size() && block[close_paren_pos] == ')') {
        replace_end = close_paren_pos + 1;
      }
    }
    if (context_path.empty()) {
      const std::string_view replacement = has_not ? kIsNotNone : kIsNone;
      block.replace(pos, replace_end - pos, replacement);
      pos += replacement.size();
      continue;
    }

    const size_t context_path_pos =
        static_cast<size_t>(context_path.data() - block.data());
    if (context_path == "none" || context_path == "None" ||
        context_path == "true" || context_path == "True" ||
        context_path == "false" || context_path == "False") {
      const std::string_view replacement = has_not ? "true" : "false";
      block.replace(
          context_path_pos, replace_end - context_path_pos, replacement);
      pos = context_path_pos + replacement.size();
      continue;
    }
    std::string replacement = "(" + std::string(context_path) + " ";
    replacement += has_not ? kIsNotNone : kIsNone;
    replacement += has_not ? " or '" : " and '";
    replacement += presence_path_for_context_path(context_path);
    replacement += has_not ? "' in " : "' not in ";
    replacement += kChatTemplateKwargsPresence;
    replacement += ")";
    block.replace(
        context_path_pos, replace_end - context_path_pos, replacement);
    pos = context_path_pos + replacement.size();
  }
}

struct LocalAssignment {
  std::string name;
  std::string value_expression;
};

std::optional<LocalAssignment> local_assignment(const std::string& block) {
  std::string_view content(block.data() + 2, block.size() - 4);
  content = trim_whitespace(content);
  if (!content.empty() && content.front() == '-') {
    content.remove_prefix(1);
    content = trim_whitespace(content);
  }
  constexpr std::string_view kSet = "set";
  if (content.substr(0, kSet.size()) != kSet ||
      (content.size() > kSet.size() &&
       std::isspace(static_cast<unsigned char>(content[kSet.size()])) == 0)) {
    return std::nullopt;
  }
  content.remove_prefix(kSet.size());
  content = trim_whitespace(content);
  size_t name_end = 0;
  while (name_end < content.size() &&
         (std::isalnum(static_cast<unsigned char>(content[name_end])) != 0 ||
          content[name_end] == '_')) {
    ++name_end;
  }
  if (name_end == 0) {
    return std::nullopt;
  }
  const std::string name(content.substr(0, name_end));
  content.remove_prefix(name_end);
  content = trim_whitespace(content);
  if (content.empty() || content.front() != '=') {
    return std::nullopt;
  }
  content.remove_prefix(1);
  content = trim_whitespace(content);
  if (!content.empty() && content.back() == '-') {
    content.remove_suffix(1);
    content = trim_whitespace(content);
  }
  return LocalAssignment{std::move(name), std::string(content)};
}

bool is_context_path(std::string_view expression) {
  if (expression.empty() || expression.front() == '.' ||
      expression.back() == '.') {
    return false;
  }
  for (const char character : expression) {
    if (std::isalnum(static_cast<unsigned char>(character)) == 0 &&
        character != '_' && character != '.') {
      return false;
    }
  }
  return true;
}

bool is_defined_literal(std::string_view expression) {
  while (expression.size() >= 2 && expression.front() == '(' &&
         expression.back() == ')') {
    expression.remove_prefix(1);
    expression.remove_suffix(1);
    expression = trim_whitespace(expression);
  }
  return expression == "none" || expression == "None" || expression == "true" ||
         expression == "True" || expression == "false" || expression == "False";
}

size_t find_block_end(const std::string& chat_template,
                      size_t content_pos,
                      std::string_view close_delimiter) {
  char quote = '\0';
  const bool is_output_block = close_delimiter == "}}";
  int32_t brace_depth = 0;
  for (size_t pos = content_pos; pos < chat_template.size(); ++pos) {
    const char current = chat_template[pos];
    if (quote != '\0') {
      if (current == '\\') {
        ++pos;
      } else if (current == quote) {
        quote = '\0';
      }
      continue;
    }
    if (current == '\'' || current == '"') {
      quote = current;
      continue;
    }
    if (chat_template.compare(pos, close_delimiter.size(), close_delimiter) ==
            0 &&
        (!is_output_block || brace_depth == 0)) {
      return pos;
    }
    if (is_output_block && current == '{') {
      ++brace_depth;
    } else if (is_output_block && current == '}' && brace_depth > 0) {
      --brace_depth;
    }
  }
  return std::string::npos;
}

std::string_view trim_whitespace(std::string_view text) {
  while (!text.empty() &&
         std::isspace(static_cast<unsigned char>(text.front())) != 0) {
    text.remove_prefix(1);
  }
  while (!text.empty() &&
         std::isspace(static_cast<unsigned char>(text.back())) != 0) {
    text.remove_suffix(1);
  }
  return text;
}

bool is_statement(const std::string& block, std::string_view statement) {
  std::string_view content(block.data() + 2, block.size() - 4);
  content = trim_whitespace(content);
  if (!content.empty() && content.front() == '-') {
    content.remove_prefix(1);
    content = trim_whitespace(content);
  }
  if (content.size() < statement.size() ||
      content.substr(0, statement.size()) != statement) {
    return false;
  }
  if (content.size() == statement.size()) {
    return true;
  }
  const char next = content[statement.size()];
  return std::isspace(static_cast<unsigned char>(next)) != 0 || next == '-';
}

size_t find_raw_end(const std::string& chat_template, size_t search_pos) {
  while (search_pos < chat_template.size()) {
    const size_t block_pos = chat_template.find("{%", search_pos);
    if (block_pos == std::string::npos) {
      return std::string::npos;
    }
    const size_t close_pos = chat_template.find("%}", block_pos + 2);
    if (close_pos == std::string::npos) {
      return std::string::npos;
    }
    const std::string block = chat_template.substr(
        block_pos, close_pos + std::string_view("%}").size() - block_pos);
    if (is_statement(block, "endraw")) {
      return close_pos + std::string_view("%}").size();
    }
    search_pos = block_pos + 2;
  }
  return std::string::npos;
}

std::string normalize_minja_tests(std::string chat_template) {
  size_t search_pos = 0;
  while (search_pos < chat_template.size()) {
    const size_t block_pos = chat_template.find('{', search_pos);
    if (block_pos == std::string::npos) {
      break;
    }

    if (block_pos + 1 >= chat_template.size()) {
      break;
    }

    const char block_type = chat_template[block_pos + 1];
    if (block_type == '#') {
      const size_t close_pos = chat_template.find("#}", block_pos + 2);
      if (close_pos == std::string::npos) {
        break;
      }
      search_pos = close_pos + std::string_view("#}").size();
      continue;
    }
    if (block_type != '{' && block_type != '%') {
      search_pos = block_pos + 1;
      continue;
    }

    const std::string_view close = block_type == '{' ? "}}" : "%}";
    const size_t close_pos =
        find_block_end(chat_template, block_pos + 2, close);
    if (close_pos == std::string::npos) {
      break;
    }

    std::string block =
        chat_template.substr(block_pos, close_pos + close.size() - block_pos);
    if (block_type == '%' && is_statement(block, "raw")) {
      const size_t raw_end_pos = find_raw_end(chat_template, close_pos + 2);
      if (raw_end_pos == std::string::npos) {
        break;
      }
      search_pos = raw_end_pos;
      continue;
    }

    // Qwen templates use Jinja's `is undefined` test for optional arguments.
    // Minja represents missing arguments as null and supports `is none`.
    replace_undefined_tests(block);
    if (block_type == '%') {
      const auto assignment = local_assignment(block);
      if (assignment.has_value()) {
        const bool trim_trailing_whitespace =
            block.size() >= 3 && block.compare(block.size() - 3, 3, "-%}") == 0;
        if (trim_trailing_whitespace) {
          block.erase(block.size() - 3, 1);
        }
        const std::string presence_path =
            presence_path_for_context_path(assignment->name);
        const std::string presence_name(kChatTemplateKwargsPresence);
        const std::string state_name(kChatTemplateKwargsPresenceState);
        block += "{% set " + state_name +
                 " = namespace(paths=[]) %}"
                 "{% for path in " +
                 presence_name +
                 " %}"
                 "{% if path != '" +
                 presence_path + "' and not path.startswith('" + presence_path +
                 "/') %}"
                 "{% set " +
                 state_name + ".unused = " + state_name +
                 ".paths.append(path) %}{% endif %}{% endfor %}";
        if (is_defined_literal(assignment->value_expression)) {
          block += "{% set " + state_name + ".unused = " + state_name +
                   ".paths.append('" + presence_path + "') %}";
        } else if (is_context_path(assignment->value_expression)) {
          const std::string source_path =
              presence_path_for_context_path(assignment->value_expression);
          block += "{% for path in " + presence_name +
                   " %}"
                   "{% if path == '" +
                   source_path + "' %}{% set " + state_name +
                   ".unused = " + state_name + ".paths.append('" +
                   presence_path + "') %}{% elif path.startswith('" +
                   source_path +
                   "/') %}"
                   "{% set " +
                   state_name + ".unused = " + state_name + ".paths.append('" +
                   presence_path + "' + path.replace('" + source_path +
                   "', '', 1)) %}{% endif %}{% endfor %}";
        }
        block += "{% set " + presence_name + " = " + state_name + ".paths" +
                 (trim_trailing_whitespace ? " -%}" : " %}");
      }
    }
    chat_template.replace(
        block_pos, close_pos + close.size() - block_pos, block);
    search_pos = block_pos + block.size();
  }
  return chat_template;
}

}  // namespace

JinjaChatTemplate::JinjaChatTemplate(const TokenizerArgs& args) : args_(args) {
  try {
    template_ = std::make_unique<minja::chat_template>(
        normalize_minja_tests(args_.chat_template()),
        args_.bos_token(),
        args_.eos_token());
    LOG(INFO) << "Jinja chat template init succeed.";

  } catch (const std::exception& e) {
    LOG(FATAL) << "Failed to parse jinja chat template, TokenizerArgs: "
               << args_ << std::endl
               << "Error message: " << e.what();
  }
}

std::optional<std::string> JinjaChatTemplate::apply(
    const ChatMessages& messages) const {
  const std::vector<xllm_service::JsonTool> empty_tools;
  const nlohmann::ordered_json chat_template_kwargs = nlohmann::json::object();
  return apply(messages, empty_tools, chat_template_kwargs);
}

std::optional<std::string> JinjaChatTemplate::apply(
    nlohmann::ordered_json& messages) const {
  // Call the overloaded method with empty tools
  nlohmann::ordered_json empty_tools = nlohmann::json::array();
  const nlohmann::ordered_json chat_template_kwargs = nlohmann::json::object();
  return apply(messages, empty_tools, chat_template_kwargs);
}

std::optional<std::string> JinjaChatTemplate::apply(
    const ChatMessages& messages,
    const std::vector<xllm_service::JsonTool>& json_tools) const {
  const nlohmann::ordered_json chat_template_kwargs = nlohmann::json::object();
  return apply(messages, json_tools, chat_template_kwargs);
}

std::optional<std::string> JinjaChatTemplate::apply(
    const ChatMessages& messages,
    const std::vector<xllm_service::JsonTool>& json_tools,
    const nlohmann::ordered_json& chat_template_kwargs) const {
  // convert the messages to json object
  nlohmann::ordered_json messages_json = nlohmann::json::array();
  for (const auto& message : messages) {
    nlohmann::ordered_json message_json;
    message_json["role"] = message.role;

    if (std::holds_alternative<std::string>(message.content)) {
      message_json["content"] = std::get<std::string>(message.content);
    } else if (std::holds_alternative<Message::MMContentVec>(message.content)) {
      message_json["content"] =
          get_mm_content(std::get<Message::MMContentVec>(message.content));
    }
    if (message.tool_calls.has_value()) {
      nlohmann::ordered_json tool_calls_json = nlohmann::json::array();
      for (const auto& tool_call : *message.tool_calls) {
        // Tool-call arguments arrive as a JSON string (OpenAI wire format), but
        // chat templates such as GLM iterate `arguments.items()`, which only
        // works on a JSON object. A non-object value (truncated/malformed JSON,
        // or a JSON array/scalar) makes minja throw "Unknown method: items",
        // which would abort the whole process. Always hand the template an
        // object: parse when possible and fall back to an empty object
        // otherwise, so rendering can never throw on this field.
        nlohmann::ordered_json arguments_json =
            nlohmann::ordered_json::object();
        if (!tool_call.function.arguments.empty()) {
          try {
            auto parsed = nlohmann::json::parse(tool_call.function.arguments);
            if (parsed.is_object()) {
              arguments_json = std::move(parsed);
            } else {
              LOG(WARNING) << "Tool-call arguments are not a JSON object, "
                              "rendering empty args: "
                           << tool_call.function.arguments;
            }
          } catch (const nlohmann::json::exception& e) {
            LOG(WARNING) << "Failed to parse tool-call arguments, rendering "
                            "empty args: "
                         << e.what();
          }
        }
        tool_calls_json.emplace_back(nlohmann::ordered_json{
            {"id", tool_call.id},
            {"type", tool_call.type},
            {"function",
             {{"name", tool_call.function.name},
              {"arguments", std::move(arguments_json)}}}});
      }
      message_json["tool_calls"] = std::move(tool_calls_json);
    }
    if (message.reasoning_content.has_value()) {
      message_json["reasoning_content"] = message.reasoning_content.value();
    }
    if (!message.tool_call_id.empty()) {
      message_json["tool_call_id"] = message.tool_call_id;
    }

    messages_json.push_back(message_json);
  }

  nlohmann::ordered_json tools_json = nlohmann::json::array();
  for (const auto& json_tool : json_tools) {
    nlohmann::ordered_json tool_json;
    tool_json["type"] = json_tool.type;

    nlohmann::ordered_json function_json;
    function_json["name"] = json_tool.function.name;
    function_json["description"] = json_tool.function.description;
    function_json["parameters"] = json_tool.function.parameters;

    tool_json["function"] = function_json;
    tools_json.push_back(tool_json);
  }
  // apply the template
  auto prompt = apply(messages_json, tools_json, chat_template_kwargs);
  if (!prompt.has_value() || has_all_thinking(prompt.value(), messages_json)) {
    return prompt;
  }

  auto fallback_messages = with_thinking_fallback(messages_json);
  auto fallback_prompt =
      apply(fallback_messages, tools_json, chat_template_kwargs);
  if (fallback_prompt.has_value()) {
    return fallback_prompt;
  }
  return prompt;
}

std::optional<std::string> JinjaChatTemplate::apply(
    nlohmann::ordered_json& messages,
    const nlohmann::ordered_json& tools) const {
  const nlohmann::ordered_json chat_template_kwargs = nlohmann::json::object();
  return apply(messages, tools, chat_template_kwargs);
}

std::optional<std::string> JinjaChatTemplate::apply(
    nlohmann::ordered_json& messages,
    const nlohmann::ordered_json& tools,
    const nlohmann::ordered_json& chat_template_kwargs) const {
  if (!chat_template_kwargs.is_object()) {
    LOG(ERROR) << "chat_template_kwargs must be a JSON object.";
    return std::nullopt;
  }
  if (chat_template_kwargs.contains(kChatTemplateKwargsPresence) ||
      chat_template_kwargs.contains(kChatTemplateKwargsPresenceState)) {
    LOG(ERROR) << "chat_template_kwargs contains reserved key: "
               << (chat_template_kwargs.contains(kChatTemplateKwargsPresence)
                       ? kChatTemplateKwargsPresence
                       : kChatTemplateKwargsPresenceState);
    return std::nullopt;
  }

  minja::chat_template_inputs input;
  input.messages = messages;
  input.tools = tools;
  input.add_generation_prompt = true;
  nlohmann::ordered_json extra_context = chat_template_kwargs;
  nlohmann::ordered_json extra_context_keys = nlohmann::json::array();
  append_present_paths(chat_template_kwargs, "", extra_context_keys);
  extra_context[kChatTemplateKwargsPresence] = std::move(extra_context_keys);
  input.extra_context = std::move(extra_context);
  minja::chat_template_options options;

  // minja throws std::runtime_error on any rendering failure (e.g. calling
  // `.items()` on a non-object value). The OpenAI HTTP path does not wrap the
  // scheduler call in a try/catch, so an uncaught exception here would escape
  // the brpc handler and terminate the whole process. Contain it and degrade
  // to an empty result, letting callers report a normal "failed to construct
  // prompt" error instead of crashing.
  try {
    return template_->apply(input, options);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to apply chat template: " << e.what();
    return std::nullopt;
  }
}

nlohmann::ordered_json JinjaChatTemplate::get_mm_content(
    const Message::MMContentVec& vec) const {
  nlohmann::ordered_json content_json = nlohmann::json::array();

  for (const auto& item : vec) {
    nlohmann::ordered_json item_json;
    item_json["type"] = item.type;

    if (item.type == "text") {
      item_json["text"] = item.text;
    } else if (item.type == "thinking") {
      item_json["thinking"] = item.thinking;
      if (!item.signature.empty()) {
        item_json["signature"] = item.signature;
      }
    } else if (item.type == "redacted_thinking") {
      item_json["data"] = item.data;
    } else if (item.type == "tool_use") {
      item_json["id"] = item.id;
      item_json["name"] = item.name;
      item_json["input"] = item.input;
    } else {
      item_json[item.type] = "mm place holder";
    }

    content_json.emplace_back(item_json);
  }

  return std::move(content_json);
}

}  // namespace xllm_service
