#pragma once

#include <string>

namespace xllm_service {
namespace utils {

bool enable_debug_log();
bool is_port_available(int port);
bool get_bool_env(const std::string& key, bool defaultValue);

}  // namespace utils
}  // namespace xllm_service
