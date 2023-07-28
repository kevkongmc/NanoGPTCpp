#ifndef BIGRAM_LANGUAGE_MODEL_H_
#define BIGRAM_LANGUAGE_MODEL_H_

#include "torch/torch.h"

class BigramLanguageModel : public torch::nn::Module {
public:
  BigramLanguageModel(int vocab_size)
      : token_embedding_table_(vocab_size, vocab_size) {}

  // 
  void Forward() {
    token_embedding_table_.
  }

private:
  torch::nn::EmbeddingImpl token_embedding_table_;
}

#endif // BIGRAM_LANGUAGE_MODEL_H_