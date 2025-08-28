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

#include "uuid.h"

#include <absl/random/distributions.h>

namespace xllm_service {
namespace llm {

std::string ShortUUID::random(size_t len) {
  if (len == 0) {
    len = 22;
  }

  std::string uuid(len, ' ');
  for (size_t i = 0; i < len; i++) {
    const size_t rand = absl::Uniform<size_t>(
        absl::IntervalClosedOpen, gen_, 0, alphabet_.size());
    uuid[i] = alphabet_[rand];
  }
  return uuid;
}

}  // namespace llm
}  // namespace xllm_service
