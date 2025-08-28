/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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
