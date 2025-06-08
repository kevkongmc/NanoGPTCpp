#include "batcher.h"

#include <torch/torch.h>

std::pair<torch::Tensor, torch::Tensor> Batcher::Batch(torch::Tensor data) {
  data = data.view(-1);

  // Generate random starting indices for each sequence in the batch
  torch::Tensor ix =
      torch::randint(0, data.size(0) - block_size_, {batch_size_}, options_);

  // Create index tensors for x and y (target is always next token)
  torch::Tensor x_indices =
      torch::arange(block_size_, options_).expand({batch_size_, block_size_}) +
      ix.unsqueeze(1).to(torch::kLong);
  torch::Tensor y_indices = x_indices + 1;

  // Reshape to 1D for indexing and gather the sequences
  x_indices = x_indices.view(-1);
  y_indices = y_indices.view(-1);

  // Index into the data tensor
  torch::Tensor x = data.index({x_indices}).view({batch_size_, block_size_});
  torch::Tensor y = data.index({y_indices}).view({batch_size_, block_size_});
  x = x.to(options_.device());
  y = y.to(options_.device());

  return std::make_pair(x, y);
}
