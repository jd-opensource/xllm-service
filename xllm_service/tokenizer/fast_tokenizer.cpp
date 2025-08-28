#include "fast_tokenizer.h"

#include <glog/logging.h>

namespace xllm_service {

FastTokenizer::FastTokenizer(const std::string& tokenizer_json_path)
    : tokenizer_json_path_(tokenizer_json_path) {
  handle_ = tokenizers_new_from_path(tokenizer_json_path.c_str());
  CHECK(handle_ != nullptr)
      << "Failed to load tokenizer from file: " << tokenizer_json_path;
}

std::unique_ptr<Tokenizer> FastTokenizer::clone() const {
  return std::make_unique<FastTokenizer>(tokenizer_json_path_);
}

FastTokenizer::~FastTokenizer() { tokenizers_free(handle_); }

bool FastTokenizer::encode(const std::string_view& text,
                           std::vector<int32_t>* ids) const {
  TokenizerEncodeResult result;
  tokenizers_encode(
      handle_, text.data(), text.size(), /*add_special_tokens=*/1, &result);

  std::vector<int32_t> ret(result.token_ids, result.token_ids + result.len);
  *ids = std::move(ret);

  return true;
}

std::string FastTokenizer::decode(const Slice<int32_t>& ids,
                                  bool skip_special_tokens) const {
  const char* data = nullptr;
  size_t len = 0;
  tokenizers_decode(handle_,
                    reinterpret_cast<const uint32_t*>(ids.data()),
                    ids.size(),
                    skip_special_tokens,
                    &data,
                    &len);
  return {data, len};
}

std::optional<int32_t> FastTokenizer::token_to_id(
    const std::string_view& token) const {
  int32_t id = -1;
  tokenizers_token_to_id(handle_, token.data(), token.size(), &id);
  return id == -1 ? std::optional<int32_t>(std::nullopt)
                  : std::optional<int32_t>(id);
}

std::string FastTokenizer::id_to_token(int32_t id) const {
  const char* data = nullptr;
  size_t len = 0;
  tokenizers_id_to_token(handle_, id, &data, &len);
  return {data, len};
}

size_t FastTokenizer::vocab_size() const {
  size_t size;
  tokenizers_get_vocab_size(handle_, &size);
  CHECK(size > 0) << "vocab_size must be greater than 0.";
  return size;
}

}  // namespace xllm_service
