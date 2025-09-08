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

#include "ttft_predictor.h"

static constexpr int32_t kDegree = 2;

namespace xllm_service {

TtftPredictor::TtftPredictor(
    const std::vector<std::pair<int32_t, int64_t>>& ttft_profiling_data) {
  if (!ttft_profiling_data.empty()) {
    // construct Vandermonde matrix
    int32_t m = ttft_profiling_data.size();
    int32_t n = kDegree + 1;
    Eigen::MatrixXd matrix(m, n);
    for (int32_t i = 0; i < m; ++i) {
      for (int32_t j = 0; j < n; ++j) {
        matrix(i, j) = std::pow(ttft_profiling_data[i].first, j);
      }
    }

    // construct target vector
    Eigen::VectorXd target(m);
    for (int32_t i = 0; i < m; ++i) {
      target(i) = ttft_profiling_data[i].second;
    }

    // get coefficients
    coefficients_ = matrix.colPivHouseholderQr().solve(target);
  } else {
    coefficients_ = Eigen::VectorXd::Zero(1);
  }
}

int64_t TtftPredictor::predict_ttft(int32_t length) {
  double result = 0.0;
  double power = 1.0;
  for (int32_t i = 0; i < coefficients_.size(); ++i) {
    result += coefficients_(i) * power;
    power *= length;
  }

  return static_cast<int64_t>(result);
}

}  // namespace xllm_service