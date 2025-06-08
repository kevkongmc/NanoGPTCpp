#include <torch/torch.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "batcher/batcher.h"
#include "bigram_language_model.h"
#include "tokenizer/tokenizer.h"

ABSL_FLAG(int, block_size, 8,
          "Maximum chunk/context size used to train the transformer model");
ABSL_FLAG(int, batch_size, 32,
          "Maximum number of chunks to process simultaneously");
ABSL_FLAG(int, max_iterations, 16384,
          "Maximum number of iterations to train the transformer model");
ABSL_FLAG(int, eval_interval, 1024, "Interval at which to evaluate the model");
ABSL_FLAG(float, learning_rate, 1e-3, "Learning rate for the optimizer");

int main() {
  const torch::Device device(torch::cuda::is_available() ? torch::kCUDA
                                                         : torch::kCPU);
  torch::TensorOptions tensor_options =
      torch::TensorOptions().dtype(torch::kInt64).device(device);

  Tokenizer* tokenizer = new Tokenizer(tensor_options);

  std::ifstream stream;
  stream.open("tiny_shakespeare.txt");
  std::stringstream buf;
  buf << stream.rdbuf();
  stream.close();

  std::string tiny_shakespeare = buf.str();
  int tiny_shakespeare_size = tiny_shakespeare.size();
  std::cout << "Tiny Shakespeare size: " << tiny_shakespeare_size << std::endl;

  std::string first_thousand_chars = tiny_shakespeare.substr(0, 1000);
  std::cout << "First thousand chars: " << first_thousand_chars << std::endl;

  absl::flat_hash_set<char> vocab;
  for (char c : tiny_shakespeare) {
    vocab.insert(c);
  }

  std::cout << "Vocabulary: ";
  for (char c : vocab) {
    std::cout << c;
  }
  std::cout << std::endl;

  std::cout << "Vocabulary size: " << vocab.size() << std::endl;
  auto ts_tensor = tokenizer->Encode(tiny_shakespeare).value();

  // torch::Tensor first_thousand_encoded =
  //     tokenizer->Encode(first_thousand_chars).value();

  // std::cout << "First thousand encoded: " << first_thousand_encoded
  //           << std::endl;

  torch::Tensor training_set = ts_tensor.slice(0, 0, ts_tensor.size(0) * 0.9);
  // The other 10% is reserved for validation
  torch::Tensor validation_set =
      ts_tensor.slice(0, ts_tensor.size(0) * 0.9, ts_tensor.size(0));

  torch::Tensor x = training_set.slice(0, 0, absl::GetFlag(FLAGS_block_size));
  // We right-shift by 1 in order to provide the target character in the
  // training example
  torch::Tensor y =
      training_set.slice(0, 1, absl::GetFlag(FLAGS_block_size) + 1);
  for (int i = 0; i < absl::GetFlag(FLAGS_block_size); ++i) {
    // This will mostly come out to be consecutive numbers, as the vocabulary
    // is encoded on a first-come-first-serve basis.
    std::cout << "When the context is " << x.slice(0, 0, i)
              << " then the output is " << y[i] << std::endl;
  }

  Batcher batcher(tensor_options, absl::GetFlag(FLAGS_batch_size),
                  absl::GetFlag(FLAGS_block_size));

  std::pair<torch::Tensor, torch::Tensor> batch = batcher.Batch(training_set);

  auto batch_input = batch.first;    // xb
  auto batch_target = batch.second;  // yb

  std::cout << "Batch input shape: " << batch_input.sizes() << std::endl;
  std::cout << "Batch input: " << batch_input << std::endl;
  std::cout << "Batch target shape: " << batch_target.sizes() << std::endl;
  std::cout << "Batch target: " << batch_target << std::endl;

  for (int i = 0; i < absl::GetFlag(FLAGS_batch_size); ++i) {
    auto curr_batch_input = batch_input.slice(0, i, i + 1);
    auto curr_batch_target = batch_target.slice(0, i, i + 1);
    for (int j = 0; j < absl::GetFlag(FLAGS_block_size); ++j) {
      std::cout << "When the context is " << curr_batch_input.slice(1, 0, j + 1)
                << " then the output is "
                << curr_batch_target.slice(1, j, j + 1) << std::endl;
    }
  }

  BigramLanguageModel model(vocab.size());
  model.to(device);

  auto [logits, loss] = model.Forward(batch_input, batch_target);
  // Logits shape: [batch_size, seq_len, vocab_size]
  std::cout << "Logits shape: " << logits.sizes() << std::endl;

  // Loss shape: singleton.
  std::cout << "Loss: " << loss << std::endl;

  absl::StatusOr<torch::Tensor> batch_output = model.Generate(batch_input, 1);

  if (!batch_output.ok()) {
    std::cerr << batch_output.status() << std::endl;
    return batch_output.status().raw_code();
  }

  // std::cout << "Batch output shape: " << batch_output->sizes() << std::endl;

  for (int i = 0; i < batch_input.size(0); ++i) {
    torch::Tensor input_slice = batch_input.slice(0, i, i + 1).flatten();
    torch::Tensor output_slice = batch_output->slice(0, i, i + 1).flatten();

    std::cout << "Batch input decoded: "
              << tokenizer->Decode(input_slice).value() << std::endl;
    std::cout << "Batch output decoded: "
              << tokenizer->Decode(output_slice).value() << std::endl;
  }

  auto optimizer =
      torch::optim::AdamW(model.parameters(), torch::optim::AdamWOptions(1e-3));

  for (int i = 0; i < 16384; ++i) {
    auto [logits, loss] = model.Forward(batch_input, batch_target);
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
    // std::cout << "Loss: " << loss << std::endl;
  }

  torch::Tensor blank_input = torch::zeros({1, 1}, tensor_options);

  absl::StatusOr<torch::Tensor> trained_model_output =
      model.Generate(blank_input, 512);

  if (!trained_model_output.ok()) {
    std::cerr << trained_model_output.status() << std::endl;
    return trained_model_output.status().raw_code();
  }

  std::cout << "Trained model output: " << *trained_model_output << std::endl;

  std::cout << "Trained model output decoded: "
            << tokenizer->Decode(trained_model_output->flatten()).value()
            << std::endl;

  return absl::OkStatus().raw_code();
}
