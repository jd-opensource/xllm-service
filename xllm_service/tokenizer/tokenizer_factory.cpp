#include "tokenizer_factory.h"

#include <filesystem>

namespace xllm_service {

std::unique_ptr<Tokenizer> TokenizerFactory::create_tokenizer(
    const std::string& model_weights_path,
    TokenizerArgs tokenizer_args) {
  const std::string tokenizer_json_path =
      model_weights_path + "/tokenizer.json";
  if (std::filesystem::exists(tokenizer_json_path)) {
    // 1. fast tokenizer
    LOG(INFO) << "Create fast tokenizer.";
    return std::make_unique<FastTokenizer>(tokenizer_json_path);
  } else if (tokenizer_args.tokenizer_type() == "tiktoken" ||
             tokenizer_args.tokenizer_class() == "TikTokenTokenizer") {
    // 2. create tiktoken tokenizer
    LOG(INFO) << "Create Tiktoken tokenizer.";
    return std::make_unique<TiktokenTokenizer>(model_weights_path,
                                               tokenizer_args);
  } else {
    // 3. create sentencepiece tokenizer
    LOG(INFO) << "Create SentencePiece tokenizer.";
    return std::make_unique<SentencePieceTokenizer>(model_weights_path,
                                                    tokenizer_args);
  }
}

}  // namespace xllm_service
