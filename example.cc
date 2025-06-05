#include <iostream>
#include <torch/torch.h>
#include <ATen/ATen.h>

int main() {
  torch::Device device(torch::cuda::is_available() ? torch::kCUDA
                                                   : torch::kCPU);
  at::Tensor a = at::ones({2, 2}, at::kInt);
  at::Tensor b = at::randn({2, 2});
  // Stack tensors a and b along dimension 0
  at::Tensor c = at::stack({a, b.to(at::kInt)}, 0);
  // Sum the stacked tensors along dimension 0
  c = c.sum(0);
  c.size(0);
  c = c.cpu();
  std::cout << c << std::endl;
}
