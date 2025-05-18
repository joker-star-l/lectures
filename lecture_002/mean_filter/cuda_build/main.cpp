#include <torch/extension.h>
torch::Tensor mean_filter(torch::Tensor image, int radius);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("mean_filter", torch::wrap_pybind_function(mean_filter), "mean_filter");
}