#include <torch/extension.h>

torch::Tensor softmax_cuda(torch::Tensor x);
torch::Tensor attention_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v, bool causal);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax", &softmax_cuda, "Fused fp16 softmax (CUDA)");
    m.def("attention", &attention_cuda, "Online-softmax fp16 attention (CUDA)");
}
