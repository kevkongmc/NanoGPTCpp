#ifndef BIGRAM_LANGUAGE_MODEL_H_
#define BIGRAM_LANGUAGE_MODEL_H_

#include <torch/nn/module.h>
#include <torch/nn/modules/embedding.h>
#include <torch/torch.h>

#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

class BigramLanguageModel : public torch::nn::Module {
public:
  // Vocab size to vocab size is basically puts every word in its own dimension
  // In actuality, at scale, we can do much less.
  // see https://youtu.be/KJtZARuO3JY?t=1270
  explicit BigramLanguageModel(int vocab_size)
      : token_embedding_table_(
            register_module("token_embedding_table",
                            torch::nn::Embedding(vocab_size, vocab_size))) {
  }

  // Input shape: (batch_size, seq_len)
  // Target shape: (batch_size, seq_len)
  // Returns: pair of (logits, loss)
  // Logits shape: (batch_size, seq_len, vocab_size)
  // Loss shape: singleton
  std::pair<torch::Tensor, torch::Tensor> Forward(torch::Tensor input,
                                                  torch::Tensor targets);

  // Forward pass without targets (for inference)
  // Returns: logits, shape of (batch_size, seq_len, vocab_size)
  torch::Tensor Forward(torch::Tensor input);

  absl::flat_hash_map<absl::string_view, int> EstimateLoss(
      torch::Tensor input, torch::Tensor targets);

  // Generate new tokens based on input context
  // Input shape: (batch_size, seq_len): current context of characters in batch
  // Output shape: (batch_size, seq_len + max_new_tokens): context + generated
  // characters
  absl::StatusOr<torch::Tensor> Generate(torch::Tensor input,
                                         int max_new_tokens);

private:
  torch::nn::Embedding token_embedding_table_;
};

#endif  // BIGRAM_LANGUAGE_MODEL_H_