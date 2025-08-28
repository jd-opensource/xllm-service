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

// The C API
#ifdef __cplusplus
extern "C" {
#endif

// The C interface to the hf-tokenizers library
// ported from https://github.com/mlc-ai/tokenizers-cpp
#include <stddef.h>
#include <stdint.h>

using TokenizerHandle = void*;

TokenizerHandle tokenizer_from_file(const char* path);
// TokenizerHandle tokenizer_from_pretrained(const char* identifier);

void tokenizer_encode(TokenizerHandle handle,
                      const char* data,
                      size_t len,
                      bool add_special_tokens);

void tokenizer_decode(TokenizerHandle handle,
                      const uint32_t* data,
                      size_t len,
                      bool skip_special_tokens);

void tokenizer_get_decode_str(TokenizerHandle handle,
                              const char** data,
                              size_t* len);

void tokenizer_get_encode_ids(TokenizerHandle handle,
                              const uint32_t** id_data,
                              size_t* len);

void tokenizer_id_to_token(TokenizerHandle handle,
                           uint32_t id,
                           const char** data,
                           size_t* len);

// -1 if token is not in vocab
int32_t tokenizer_token_to_id(TokenizerHandle handle,
                              const char* token,
                              size_t len);

void tokenizer_free(TokenizerHandle handle);

size_t tokenizer_get_vocab_size(TokenizerHandle handle, bool with_added_tokens);

#ifdef __cplusplus
}
#endif
