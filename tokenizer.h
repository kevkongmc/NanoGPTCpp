#ifndef TOKENIZER_H_
#define TOKENIZER_H_

#include <fstream>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/status/statusor.h"

class Tokenizer {
public:
  Tokenizer() {}

  absl::StatusOr<std::vector<int>> Encode(absl::string_view s);
  absl::StatusOr<std::string> Decode(std::vector<int>& v);
};

#endif // TOKENIZER_H_
