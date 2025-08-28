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
#include <absl/random/random.h>

#include <string>

namespace xllm_service {
namespace llm {

class ShortUUID {
 public:
  ShortUUID() = default;

  std::string random(size_t len = 0);

 private:
  std::string alphabet_ =
      "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
      "abcdefghijkmnopqrstuvwxyz";
  absl::BitGen gen_;
};

}  // namespace llm
}  // namespace xllm_service
