#pragma once

#include "common/types.h"
#include "tokenizer.h"
#include "tokenizer_args.h"

namespace xllm_service {

std::unique_ptr<Tokenizer> create_tokenizer(const ModelConfig& model_config,
                                            TokenizerArgs* tokenizer_args);

}  // namespace xllm_service
