#include "bigram_language_model.h"

#include <torch/nn/functional/activation.h>

std::pair<torch::Tensor, torch::Tensor> BigramLanguageModel::Forward(
    torch::Tensor input, torch::Tensor targets) {
  // Get the logits for the given indices
  // Shape: (batch_size, seq_len, vocab_size)
  torch::Tensor logits = token_embedding_table_(input);
  auto batch_size = logits.size(0);  // aka B
  auto seq_len = logits.size(1);     // aka T
  auto vocab_size = logits.size(2);  // aka C

  // Flatten for cross entropy loss
  auto flat_logits = logits.view({batch_size * seq_len, vocab_size});
  auto flat_targets = targets.view({batch_size * seq_len});

  // Calculate loss on flattened tensors
  torch::Tensor loss =
      torch::nn::functional::cross_entropy(flat_logits, flat_targets);

  // Return unflattened logits and loss
  return std::make_pair(logits, loss);
}

torch::Tensor BigramLanguageModel::Forward(torch::Tensor input) {
  return token_embedding_table_(input);
}

absl::StatusOr<torch::Tensor> BigramLanguageModel::Generate(
    torch::Tensor input, int max_new_tokens) {
  if (max_new_tokens < 1) {
    return absl::InvalidArgumentError("max_new_tokens must be at least 1");
  }
  for (int i = 0; i < max_new_tokens; ++i) {
    // Get the logits (probability of next token) for the given indices
    // Shape: (batch_size, seq_len, vocab_size)
    torch::Tensor logits = Forward(input);

    // Get the logits for the last token in the sequences
    // Shape: (batch_size, vocab_size)
    torch::Tensor logits_last = logits.select(1, -1);

    // Apply a softmax function so that the logits sum to 100%.
    // Shape: (batch_size, vocab_size)
    torch::Tensor probabilities = torch::softmax(logits_last, -1);

    // Sample from the distribution of next tokens
    // Shape: (batch_size)
    torch::Tensor next_tokens = torch::multinomial(probabilities, 1);

    // Append the next tokens to the input
    input = torch::cat({input, next_tokens}, 1);
  }
  return input;
}
