#pragma once

#include <google/protobuf/service.h>

#include "butil/macros.h"

namespace xllm_service {

// RAII: Call Run() of the closure on destruction.
class ClosureGuard {
 public:
  ClosureGuard() : done_(nullptr) {}

  // Constructed with a closure which will be Run() inside dtor.
  explicit ClosureGuard(google::protobuf::Closure* done) : done_(done) {}

  // Run internal closure if it's not NULL.
  ~ClosureGuard() {
    if (done_) {
      done_->Run();
    }
  }

  // Run internal closure if it's not NULL and set it to `done'.
  void reset(google::protobuf::Closure* done) {
    if (done_) {
      done_->Run();
    }
    done_ = done;
  }

  // Return and set internal closure to NULL.
  google::protobuf::Closure* release() {
    google::protobuf::Closure* const prev_done = done_;
    done_ = nullptr;
    return prev_done;
  }

  // True if no closure inside.
  bool empty() const { return done_ == nullptr; }

  // Exchange closure with another guard.
  void swap(ClosureGuard& other) { std::swap(done_, other.done_); }

 private:
  // Copying this object makes no sense.
  DISALLOW_COPY_AND_ASSIGN(ClosureGuard);

  google::protobuf::Closure* done_ = nullptr;
};

}  // namespace xllm_service
