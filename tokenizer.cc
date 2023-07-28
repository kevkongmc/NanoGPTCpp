#include "tokenizer.h"

#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"

absl::StatusOr<std::vector<int>> Tokenizer::Encode(absl::string_view s) {
  // FIXME ensure that this can take tensors for parallel operations for char <-> int conversions
  std::vector<int> out;
  for (char c : s) {
    out.push_back(int(c));
  }

  return out;
}

absl::StatusOr<std::string> Tokenizer::Decode(std::vector<int> &v) {
  std::string out;
  for (int i : v) {
    out.push_back(char(i));
  }

  return out;
}
