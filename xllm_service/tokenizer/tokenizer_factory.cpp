#include "tokenizer_factory.h"

#include "hf_tokenizer.h"
#include "sentencepiece_tokenizer.h"
#include "tiktoken_tokenizer.h"
#include "tokenizer_args_loader.h"

namespace xllm_service {
std::unique_ptr<Tokenizer> create_tokenizer(const ModelConfig& model_config,
                                            TokenizerArgs* tokenizer_args) {
  TokenizerArgsLoader::load(
      model_config.model_type,
      model_config.tokenizer_path + "/tokenizer_config.json",
      tokenizer_args);

  const std::string tokenizer_path =
      model_config.tokenizer_path + "/tokenizer.json";

  if (std::filesystem::exists(tokenizer_path)) {
    LOG(INFO) << "Using fast tokenizer.";
    // load fast tokenizer
    return HFTokenizer::from_file(tokenizer_path);

  } else if (tokenizer_args->tokenizer_type() == "tiktoken") {
    // fallback to sentencepiece/tiktoken tokenizer if no fast tokenizer
    // exists
    LOG(INFO) << "Using Tiktoken tokenizer.";
    return std::make_unique<TiktokenTokenizer>(model_config.tokenizer_path,
                                               *tokenizer_args);
  } else {
    LOG(INFO) << "Using SentencePiece tokenizer.";
    return std::make_unique<SentencePieceTokenizer>(model_config.tokenizer_path,
                                                    *tokenizer_args);
  }
}

}  // namespace xllm_service
