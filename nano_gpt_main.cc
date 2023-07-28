#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "tokenizer.h"
#include "torch/torch.h"

#include "absl/container/flat_hash_map.h"
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/strings/str_join.h"
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
  auto tensor_options =
      torch::TensorOptions().dtype(torch::kInt32).device(device);

  Tokenizer *tokenizer = new Tokenizer();

  std::ifstream stream;
  stream.open("tiny_shakespeare.txt");
  std::stringstream buf;
  buf << stream.rdbuf();
  stream.close();

  auto tiny_shakespeare = buf.str();
  // TODO source status macros from somewhere
  std::vector<int> ts_encoded = tokenizer->Encode(tiny_shakespeare).value();
  auto ts_tensor = torch::from_blob(ts_encoded.data(),
                                    {long(ts_encoded.size())}, tensor_options);

  std::vector<int> training_set(ts_encoded.begin(),
                                ts_encoded.begin() + ts_encoded.size() * 0.9);
  std::vector<int> validation_set(ts_encoded.begin() + ts_encoded.size() * 0.9,
                                  ts_encoded.end());

  auto training_tensor = ts_tensor.slice(0, 0, ts_tensor.size(0) * 0.9);
  auto validation_tensor =
      ts_tensor.slice(0, ts_tensor.size(0) * 0.9, ts_tensor.size(0));

  // An extra index is appended at the end in order to provide the target
  // character in the training example
  auto x = training_tensor.slice(0, 0, absl::GetFlag(FLAGS_max_block_size));
  auto y = training_tensor.slice(0, 1, absl::GetFlag(FLAGS_max_block_size) + 1);
  for (int i = 0; i < absl::GetFlag(FLAGS_max_block_size); ++i) {
    std::cout << "When the context is " << x.slice(0, 0, i)
              << " then the output is " << y[i] << std::endl;
  }

  return absl::OkStatus().raw_code();
}
