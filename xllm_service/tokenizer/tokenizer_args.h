#pragma once
#include <absl/strings/escaping.h>
#include <absl/strings/str_join.h>

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "common/macros.h"

namespace xllm_service {

using SpecialToken = std::pair<std::string, int32_t>;

struct TokenizerArgs {
  // Type of tokenizer to use. valid values are "sentencepiece" and "tiktoken".
  DEFINE_ARG(std::string, tokenizer_type) = "sentencepiece";

  // Vocab file name.
  DEFINE_ARG(std::string, vocab_file) = "tokenizer.model";

  // Special tokens to add to the vocabulary.
  DEFINE_ARG(std::vector<SpecialToken>, special_tokens);

  // Regex pattern used by tiktok tokenizer only.
  DEFINE_ARG(std::string, pattern);

  // tokens to add to the beginning of the input sequence.
  DEFINE_ARG(std::vector<std::string>, prefix_tokens);

  // chat template
  DEFINE_ARG(std::string, chat_template);

  // add_bos_token
  DEFINE_ARG(bool, add_bos_token) = false;

  // add_eos_token
  DEFINE_ARG(bool, add_eos_token) = false;

  // bos_token
  DEFINE_ARG(std::string, bos_token);

  // eos_token
  DEFINE_ARG(std::string, eos_token);

  // pad_token
  DEFINE_ARG(std::string, pad_token);

  // tokenizer_class
  DEFINE_ARG(std::string, tokenizer_class);
};

inline std::ostream& operator<<(std::ostream& os, const TokenizerArgs& args) {
  os << "TokenizerArgs: [";
  os << "tokenizer_type: " << args.tokenizer_type();
  //  os << ", chat_template: " << args.chat_template();
  os << ", add_bos_token: " << args.add_bos_token();
  os << ", add_eos_token: " << args.add_eos_token();
  os << ", bos_token: " << args.bos_token();
  os << ", eos_token: " << args.eos_token();
  os << ", pad_token: " << args.pad_token();
  os << ", tokenizer_class: " << args.tokenizer_class();
  if (!args.special_tokens().empty()) {
    os << ", special_tokens: [";
    for (const auto& [token, id] : args.special_tokens()) {
      os << "(" << token << ", " << id << ") ";
    }
    os << "]";
  }
  os << ", pattern: " << absl::CEscape(args.pattern());
  if (!args.prefix_tokens().empty()) {
    os << ", prefix_tokens: [" << absl::StrJoin(args.prefix_tokens(), ", ")
       << "]";
  }
  os << "]";
  return os;
}

}  // namespace xllm_service
