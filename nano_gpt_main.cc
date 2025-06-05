#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <torch/torch.h>
#include <ATen/ATen.h>

#include "tokenizer.h"

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"

ABSL_FLAG(int, max_block_size, 8,
          "Maximum chunk/context size used to train the transformer model");
ABSL_FLAG(int, max_batch_size, 4,
          "Maximum number of chunks to process simultaneously");

// data should be single-dimensional
std::pair<torch::Stack, torch::Stack> getBatch(torch::Tensor data) {
  auto ix = torch::randint(data.size(0) - absl::GetFlag(FLAGS_max_block_size),
                           absl::GetFlag(FLAGS_max_batch_size));
  torch::Stack x;
  
}

int main() {
  const torch::Device device(torch::cuda::is_available() ? torch::kCUDA
                                                         : torch::kCPU);
  torch::TensorOptions tensor_options =
      torch::TensorOptions().dtype(torch::kInt32).device(device);

  Tokenizer *tokenizer = new Tokenizer();

  std::ifstream stream;
  stream.open("tiny_shakespeare.txt");
  std::stringstream buf;
  buf << stream.rdbuf();
  stream.close();

  std::string tiny_shakespeare = buf.str();
  // TODO source status macros from somewhere
  std::vector<int> ts_encoded = tokenizer->Encode(tiny_shakespeare).value();
  torch::Tensor ts_tensor = torch::from_blob(ts_encoded.data(),
                                    {long(ts_encoded.size())}, tensor_options);

  std::vector<int> training_set(ts_encoded.begin(),
                                ts_encoded.begin() + ts_encoded.size() * 0.9);
  std::vector<int> validation_set(ts_encoded.begin() + ts_encoded.size() * 0.9,
                                  ts_encoded.end());

  torch::Tensor training_tensor = ts_tensor.slice(0, 0, ts_tensor.size(0) * 0.9);
  torch::Tensor validation_tensor =
      ts_tensor.slice(0, ts_tensor.size(0) * 0.9, ts_tensor.size(0));

  // Ensure we have enough data for the requested block size
  if (training_tensor.size(0) <= absl::GetFlag(FLAGS_max_block_size)) {
    std::cerr << "Error: Not enough training data for the requested block size." << std::endl;
    return 1;
  }
  // An extra index is appended at the end in order to provide the target
  // character in the training example
  torch::Tensor x = training_tensor.slice(0, 0, absl::GetFlag(FLAGS_max_block_size));
  torch::Tensor y = training_tensor.slice(0, 1, absl::GetFlag(FLAGS_max_block_size) + 1);
  for (int i = 0; i < absl::GetFlag(FLAGS_max_block_size); ++i) {
    std::cout << "When the context is " << x.slice(0, 0, i)
              << " then the output is " << y[i] << std::endl;
  }

  return absl::OkStatus().raw_code();
}
