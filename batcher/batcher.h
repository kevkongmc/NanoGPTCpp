#ifndef NANOGPT_CPP_BATCHER_BATCHER_H_
#define NANOGPT_CPP_BATCHER_BATCHER_H_

#include <torch/torch.h>

class Batcher {
public:
    Batcher(torch::TensorOptions options, int batch_size, int block_size)
        : options_(options), batch_size_(batch_size), block_size_(block_size) {}

    std::pair<torch::Tensor, torch::Tensor> Batch(torch::Tensor data);

private:
  torch::TensorOptions options_;
  int batch_size_;
  int block_size_;
};

#endif // NANOGPT_CPP_BATCHER_BATCHER_H_