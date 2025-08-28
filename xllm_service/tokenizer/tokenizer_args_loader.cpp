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

#include "tokenizer/tokenizer_args_loader.h"

#include <glog/logging.h>

namespace xllm_service {
#define SET_ARG(arg_name, value) [&] { args->arg_name() = value; }()

static std::string CHATGLM = "chatglm";
static std::string CHATGLM4 = "chatglm4";
static std::string YI = "Yi";
static std::string QWEN = "qwen";

void TokenizerArgsLoader::load(const std::string& model_type,
                               const std::string& tokenizer_args_file_path,
                               TokenizerArgs* tokenizer_args) {
  JsonReader tokenizer_reader;
  if (tokenizer_reader.parse(tokenizer_args_file_path)) {
    // read chat template if exists
    if (auto v = tokenizer_reader.value<std::string>("chat_template")) {
      tokenizer_args->chat_template() = v.value();
    }
    if (auto v = tokenizer_reader.value<bool>("add_bos_token")) {
      tokenizer_args->add_bos_token() = v.value();
    }
    if (auto v = tokenizer_reader.value<bool>("add_eos_token")) {
      tokenizer_args->add_eos_token() = v.value();
    }
    if (auto v = tokenizer_reader.value<std::string>("tokenizer_class")) {
      tokenizer_args->tokenizer_class() = v.value();
    }
    // read bos_token
    if (auto v = tokenizer_reader.value<std::string>("bos_token.content")) {
      tokenizer_args->bos_token() = v.value();
    } else if (auto v = tokenizer_reader.value<std::string>("bos_token")) {
      tokenizer_args->bos_token() = v.value();
    }
    // read eos_token
    if (auto v = tokenizer_reader.value<std::string>("eos_token.content")) {
      tokenizer_args->eos_token() = v.value();
    } else if (auto v = tokenizer_reader.value<std::string>("eos_token")) {
      tokenizer_args->eos_token() = v.value();
    }
    // read pad_token
    if (auto v = tokenizer_reader.value<std::string>("pad_token.content")) {
      tokenizer_args->pad_token() = v.value();
    } else if (auto v = tokenizer_reader.value<std::string>("pad_token")) {
      tokenizer_args->pad_token() = v.value();
    }
  }

  if (model_type == CHATGLM) {
    load_chatglm_args(tokenizer_args);
  } else if (model_type == CHATGLM) {
    load_chatglm_args(tokenizer_args);
  } else if (model_type == CHATGLM4) {
    load_chatglm4_args(tokenizer_args);
  } else if (model_type == YI) {
    load_Yi_args(tokenizer_args);
  } else if (model_type == QWEN) {
    load_qwen_args(tokenizer_args);
  } else {
    LOG(ERROR) << "unrecognized model type: " << model_type;
  }
}

void TokenizerArgsLoader::load_chatglm_args(TokenizerArgs* args) {
  SET_ARG(tokenizer_type, "sentencepiece");
  SET_ARG(vocab_file, "tokenizer.model");

  // set special tokens
  // ref to:
  // https://huggingface.co/THUDM/chatglm3-6b/blob/main/tokenizer_config.json
  const std::vector<SpecialToken> special_tokens({{"[MASK]", 64789},
                                                  {"[gMASK]", 64790},
                                                  {"[sMASK]", 64791},
                                                  {"sop", 64792},
                                                  {"eop", 64793},
                                                  {"<|system|>", 64794},
                                                  {"<|user|>", 64795},
                                                  {"<|assistant|>", 64796},
                                                  {"<|observation|>", 64797}});
  SET_ARG(special_tokens, special_tokens);
  SET_ARG(prefix_tokens, std::vector<std::string>({"[gMASK]", "sop"}));
}

void TokenizerArgsLoader::load_chatglm4_args(TokenizerArgs* args) {
  SET_ARG(tokenizer_type, "tiktoken");
  SET_ARG(vocab_file, "tokenizer.model");

  // set special tokens
  // ref to:
  // https://huggingface.co/THUDM/glm-4-9b/blob/main/tokenizer_config.json
  const std::vector<SpecialToken> special_tokens(
      {{"<|endoftext|>", 151329},
       {"[MASK]", 151330},
       {"[gMASK]", 151331},
       {"[sMASK]", 151332},
       {"<sop>", 151333},
       {"<eop>", 151334},
       {"<|system|>", 151335},
       {"<|user|>", 151336},
       {"<|assistant|>", 151337},
       {"<|observation|>", 151338},
       {"<|begin_of_image|>", 151339},
       {"<|end_of_image|>", 151340},
       {"<|begin_of_video|>", 151341},
       {"<|end_of_video|>", 151342}});
  SET_ARG(special_tokens, special_tokens);

  SET_ARG(prefix_tokens, std::vector<std::string>({"[gMASK]", "<sop>"}));

  // set regex pattern for tiktoken tokenizer.
  // ref to:
  // https://huggingface.co/THUDM/glm-4-9b/blob/main/tokenization_chatglm.py#L27
  // N.B. replaced '\s+(?!\S)' with '\s+[^\s]' to avoid regex error
  const std::string pattern =
      R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+[^\S]|\s+)";
  SET_ARG(pattern, pattern);
}

void TokenizerArgsLoader::load_Yi_args(TokenizerArgs* args) {
  SET_ARG(tokenizer_type, "sentencepiece");
  SET_ARG(vocab_file, "tokenizer.model");

  // set special tokens
  // ref to:
  // https://huggingface.co/01-ai/Yi-34B-Chat-4bits/blob/main/tokenizer_config.json
  const std::vector<SpecialToken> special_tokens({{"<unk>", 0},
                                                  {"<|startoftext|>", 1},
                                                  {"<|endoftext|>", 2},
                                                  {"<|im_start|>", 6},
                                                  {"<|im_end|>", 7},
                                                  {"<|im_sep|>", 8}});
  SET_ARG(special_tokens, special_tokens);
}

void TokenizerArgsLoader::load_qwen_args(TokenizerArgs* args) {
  SET_ARG(tokenizer_type, "tiktoken");
  // adapted from
  // https://huggingface.co/Qwen/Qwen-14B-Chat-Int4/blob/main/tokenization_qwen.py
  SET_ARG(vocab_file, "qwen.tiktoken");

  // set special tokens
  std::vector<SpecialToken> special_tokens;
  int32_t next_id = 151643;
  special_tokens.emplace_back("<|endoftext|>", next_id++);
  special_tokens.emplace_back("<|im_start|>", next_id++);
  special_tokens.emplace_back("<|im_end|>", next_id++);
  for (int32_t i = 0; i < 205; ++i) {
    special_tokens.emplace_back("<|extra_" + std::to_string(i) + "|>",
                                next_id++);
  }
  SET_ARG(special_tokens, special_tokens);

  // set regex pattern for tiktoken tokenizer.
  const std::string pattern =
      R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+[^\S]|\s+)";
  SET_ARG(pattern, pattern);
}

}  // namespace xllm_service
