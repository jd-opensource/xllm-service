#pragma once

#include "chat_template.h"
#include "tokenizer/tokenizer_args.h"

namespace xllm_service {

std::unique_ptr<ChatTemplate> create_chat_template(
    const std::string& model_type,
    const TokenizerArgs& tokenizer_args);

}  // namespace xllm_service