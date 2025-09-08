/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include <Eigen/Dense>

namespace xllm_service {

// Predictor for predicting TTFT based on input length
class TtftPredictor final {
 public:
  TtftPredictor(
      const std::vector<std::pair<int32_t, int64_t>>& ttft_profiling_data);
  ~TtftPredictor() = default;

  int64_t predict_ttft(int32_t length);

 private:
  Eigen::VectorXd coefficients_;
};

}  // namespace xllm_service