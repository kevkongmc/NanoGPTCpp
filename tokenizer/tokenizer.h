#ifndef TOKENIZER_H_
#define TOKENIZER_H_

#include <torch/torch.h>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

// Examples:
// Google uses SentencePiece: https://github.com/google/sentencepiece
// Openai uses tiktoken: https://github.com/openai/tiktoken
class Tokenizer {
public:
  Tokenizer(torch::TensorOptions options) : options_(options) {
  }

  absl::StatusOr<torch::Tensor> Encode(absl::string_view s);
  absl::StatusOr<std::string> Decode(const torch::Tensor& v);

private:
  torch::TensorOptions options_;
  absl::flat_hash_map<char, int8_t> encode_dict_;
  absl::flat_hash_map<int8_t, char> decode_dict_;
};

#endif  // TOKENIZER_H_
