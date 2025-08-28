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

#include "common/utils.h"

#include <glog/logging.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <mutex>

namespace xllm_service {
namespace utils {

bool enable_debug_log() {
  static bool debug_log_enabled = false;
  static std::once_flag debug_flag;
  std::call_once(debug_flag, []() {
    const char* enable_debug_env = std::getenv("ENABLE_XLLM_DEBUG_LOG");
    if (enable_debug_env != nullptr && std::string(enable_debug_env) == "1") {
      debug_log_enabled = true;
    }
  });

  return debug_log_enabled;
}

bool is_port_available(int port) {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    LOG(ERROR) << "create socket failed.";
    return false;
  }

  int opt = 1;
  if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
    LOG(WARNING) << "set socket options failed.";
  }

  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(port);
  if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
    return false;
  }
  close(fd);

  return true;
}

bool get_bool_env(const std::string& key, bool defaultValue) {
  const char* val = std::getenv(key.c_str());
  if (val == nullptr) {
    return defaultValue;
  }
  std::string strVal(val);
  return (strVal == "1" || strVal == "true" || strVal == "TRUE" ||
          strVal == "True");
}

}  // namespace utils
}  // namespace xllm_service
