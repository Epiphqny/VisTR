#include <torch/extension.h>
#include "deform_conv.h"

#if defined(WITH_CUDA) || defined(WITH_HIP)
extern int get_cudart_version();
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  m.def("deform_conv_forward", &deform_conv_forward, "deform_conv_forward");
  m.def(
      "deform_conv_backward_input",
      &deform_conv_backward_input,
      "deform_conv_backward_input");
  m.def(
      "deform_conv_backward_filter",
      &deform_conv_backward_filter,
      "deform_conv_backward_filter");
  m.def(
      "modulated_deform_conv_forward",
      &modulated_deform_conv_forward,
      "modulated_deform_conv_forward");
  m.def(
      "modulated_deform_conv_backward",
      &modulated_deform_conv_backward,
      "modulated_deform_conv_backward");
}