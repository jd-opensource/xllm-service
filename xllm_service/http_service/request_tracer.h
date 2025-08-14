#pragma once
#include <fstream>
#include <mutex>
#include <string>

namespace xllm_service {

class RequestTracer {
 public:
  RequestTracer(bool enable_request_trace);
  ~RequestTracer() = default;
  RequestTracer(const RequestTracer&) = delete;
  RequestTracer& operator=(const RequestTracer&) = delete;
  RequestTracer(RequestTracer&&) = delete;
  RequestTracer& operator=(RequestTracer&&) = delete;
  void log(const std::string& service_request_id,
           const std::string& input_or_output);

 private:
  std::ofstream log_stream_;
  std::mutex mutex_;
  bool enable_request_trace_ = false;
};
}  // namespace xllm_service