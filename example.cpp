#include <iostream>

#include "torch/torch.h"

int main() {
  torch::Device device(torch::cuda::is_available() ? torch::kCUDA
                                                   : torch::kCPU);
  at::Tensor a = at::ones({2, 2}, at::kInt);
  at::Tensor b = at::randn({2, 2});
  at::stack();
  auto c = a + b.to(at::kInt);
  c.size(0);
  c = c.cpu();
  std::cout << c << std::endl;
}
