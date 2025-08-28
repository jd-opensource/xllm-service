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

namespace xllm_service {
// a central place to define common macros for the project
// clang-format off
#define DEFINE_ARG(T, name)                                       \
 public:                                                          \
  inline auto name(const T& name) ->decltype(*this) {             \
    this->name##_ = name;                                         \
    return *this;                                                 \
  }                                                               \
  inline const T& name() const noexcept { return this->name##_; } \
  inline T& name() noexcept { return this->name##_; }             \
                                                                  \
  T name##_

#define DEFINE_PTR_ARG(T, name)                             \
 public:                                                    \
  inline auto name(T* name) ->decltype(*this) {             \
    this->name##_ = name;                                   \
    return *this;                                           \
  }                                                         \
  inline T* name() const noexcept { return this->name##_; } \
                                                            \
  T* name##_

// clang-format on

#ifndef UNUSED_PARAMETER
#define UNUSED_PARAMETER(x) ((void)(x))
#endif

#if __has_attribute(guarded_by)
#define GUARDED_BY(x) __attribute__((guarded_by(x)))
#else
#define GUARDED_BY(x)
#endif

// concatenate two strings
#define LLM_STR_CAT(s1, s2) s1##s2

// create an anonymous variable
#define LLM_ANON_VAR(str) LLM_STR_CAT(str, __LINE__)

#define REQUIRES(...) std::enable_if_t<(__VA_ARGS__)>* = nullptr

#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;      \
  void operator=(const TypeName&) = delete

// Define a macro to simplify adding elements from a vector to a repeated field
#define ADD_VECTOR_TO_PROTO(proto_field, vec) \
  do {                                        \
    proto_field->Reserve(vec.size());         \
    for (const auto& value : vec) {           \
      *proto_field->Add() = value;            \
    }                                         \
  } while (0)

#define CALLBACK_WITH_ERROR(CODE, MSG) callback(Status{CODE, MSG});

}  // namespace xllm_service
