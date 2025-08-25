#include "chat_template/chat_template_factory.h"

#include <glog/logging.h>

#include "chat_template/coded_chat_template.h"
#include "chat_template/common_chat_template.h"
#include "chat_template/jinja_chat_template.h"

namespace xllm_service {

constexpr std::array<std::string_view, 5> JINJA_CHAT_TEMPLATE_MODELS{
    "deepseek_v3_mtp",
    "deepseek_v2",
    "deepseek_v3",
    "qwen2",
    "qwen3"};

constexpr bool is_jinja_model(std::string_view model) {
  for (auto m : JINJA_CHAT_TEMPLATE_MODELS) {
    if (m == model) return true;
  }
  return false;
}

std::unique_ptr<ChatTemplate> create_chat_template(
    const std::string& model_type,
    const TokenizerArgs& tokenizer_args) {
  if (is_jinja_model(model_type)) {
    return std::make_unique<JinjaChatTemplate>(tokenizer_args);
  } else if (model_type == "chatglm") {
    return std::make_unique<ChatGLMChatTemplate>();
  } else if (model_type == "chatglm4") {
    return std::make_unique<ChatGLM4ChatTemplate>();
  } else if (model_type == "llama") {
    return std::make_unique<Llama2ChatTemplate>();
  } else if (model_type == "llama3") {
    return std::make_unique<Llama3ChatTemplate>();
  } else if (model_type == "rhino") {
    return std::make_unique<RhinoChatTemplate>();
  } else if (model_type == "minicpmv") {
    return std::make_unique<MiniCPMVChatTemplate>();
  } else if (model_type == "qwen") {
    return std::make_unique<QwenChatTemplate>();
  } else {
    LOG(FATAL) << "Unknow model: " << model_type
               << ", create ChatTemplate fail!";
  }
}

}  // namespace xllm_service
