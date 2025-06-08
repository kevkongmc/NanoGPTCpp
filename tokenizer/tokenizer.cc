#include "tokenizer.h"

#include <string>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

// Using the ascii values of the characters as the token ids will lead to
// index out of bounds when we try to access the token embedding table,
// since the token embedding table is initialized with the vocabulary size.
// Therefore, we use a flat_hash_map of char to int8_t to map the characters
// to our custom-assigned token ids.
absl::StatusOr<torch::Tensor> Tokenizer::Encode(absl::string_view s) {
  for (char c : s) {
    if (encode_dict_.find(c) == encode_dict_.end()) {
      int current_id = encode_dict_.size();
      encode_dict_[c] = current_id;
      decode_dict_[current_id] = c;
    }
  }

  torch::Tensor result =
      torch::empty({static_cast<int64_t>(s.size())}, options_);
  torch::TensorAccessor<int64_t, 1> result_a = result.accessor<int64_t, 1>();

  // Use parallel_for to process the string in parallel
  torch::parallel_for(0, s.size(), 0, [&](int64_t start, int64_t end) {
    for (int64_t i = start; i < end; ++i) {
      // This is thread-safe for reading since we've already populated all
      // characters
      result_a[i] = encode_dict_.at(s[i]);
    }
  });

  return result;
}

absl::StatusOr<std::string> Tokenizer::Decode(const torch::Tensor& v) {
  std::string out;
  torch::TensorAccessor<int64_t, 1> accessor = v.accessor<int64_t, 1>();
  for (int64_t i = 0; i < accessor.size(0); ++i) {
    out.push_back(decode_dict_.at(accessor[i]));
  }
  return out;
}
