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

#include "common/json_reader.h"

#include <glog/logging.h>

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
namespace xllm_service {

bool JsonReader::parse(const std::string& json_file_path) {
  if (!std::filesystem::exists(json_file_path)) {
    return false;
  }

  std::ifstream ifs(json_file_path);
  if (!ifs.is_open()) {
    return false;
  }

  data_ = nlohmann::json::parse(ifs);
  return true;
}

bool JsonReader::contains(const std::string& key) const {
  // slipt the key by '.' then traverse the json object
  std::vector<std::string> keys = absl::StrSplit(key, '.');
  nlohmann::json data = data_;
  for (const auto& k : keys) {
    if (!data.contains(k)) {
      return false;
    }
    data = data[k];
  }
  return true;
}

}  // namespace xllm_service