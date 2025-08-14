#pragma once
#include "common/json_reader.h"
#include "tokenizer_args.h"

namespace xllm_service {

class TokenizerArgsLoader {
 public:
  static void load(const std::string& model_type,
                   const std::string& tokenizer_args_file_path,
                   TokenizerArgs* tokenizer_args);

 private:
  static void load_chatglm_args(TokenizerArgs* args);

  static void load_chatglm4_args(TokenizerArgs* args);

  static void load_Yi_args(TokenizerArgs* args);

  static void load_qwen_args(TokenizerArgs* args);
};

}  // namespace xllm_service
