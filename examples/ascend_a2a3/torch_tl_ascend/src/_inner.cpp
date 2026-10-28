#include <ATen/Operators.h>
#include <Python.h>
#include <torch/all.h>
#include <torch/library.h>

#include "torch_npu/csrc/core/npu/NPUStream.h"

extern "C" {
typedef void *aclrtStream;
void call(uint8_t *Q_handle, uint8_t *K_handle, uint8_t *V_handle,
          uint8_t *Output_handle, uint8_t *workspace_1_handle,
          uint8_t *workspace_2_handle, uint8_t *workspace_3_handle,
          aclrtStream stream);
}

at::Tensor flash_attention_wrapper(const at::Tensor &Q, const at::Tensor &K,
                                   const at::Tensor &V) {
  constexpr int64_t block_M = 64;
  constexpr int64_t block_N = 64;

  auto dtype = at::kHalf;
  auto accum_dtype = at::kFloat;

  TORCH_CHECK(Q.dim() == 4,
              "tl_ascend.flash_attention: Q must be a 4-D tensor");
  TORCH_CHECK(K.dim() == 4,
              "tl_ascend.flash_attention: K must be a 4-D tensor");
  TORCH_CHECK(V.dim() == 4,
              "tl_ascend.flash_attention: V must be a 4-D tensor");

  // libop.so is compiled for this one static BHSD signature. Reject a
  // different request before allocation, stream lookup or the raw ABI call.
  for (const auto *tensor : {&Q, &K, &V}) {
    TORCH_CHECK(tensor->size(0) == 4 && tensor->size(1) == 16 &&
                    tensor->size(2) == 4096 && tensor->size(3) == 128,
                "tl_ascend.flash_attention: compiled BHSD shape must be [4, "
                "16, 4096, 128]");
    TORCH_CHECK(tensor->scalar_type() == at::kHalf,
                "tl_ascend.flash_attention: compiled dtype must be float16");
    TORCH_CHECK(tensor->is_contiguous(),
                "tl_ascend.flash_attention: Q, K and V must be contiguous");
    TORCH_CHECK(tensor->device().type() == c10::DeviceType::PrivateUse1,
                "tl_ascend.flash_attention: Q, K and V must be NPU tensors");
    TORCH_CHECK(
        tensor->device() == Q.device(),
        "tl_ascend.flash_attention: Q, K and V must be on the same NPU device");
  }

  // Q, K, V: [batch, heads, seq_len, dim]
  const int64_t batch = Q.size(0);
  const int64_t heads = Q.size(1);
  const int64_t seq_len = Q.size(2);
  const int64_t dim = Q.size(3);

  const int64_t block_num = (seq_len / block_M) * heads * batch;

  at::Tensor Output = at::empty_like(Q);
  at::Tensor workspace_1 =
      at::empty({block_num, block_M, block_N}, Q.options().dtype(accum_dtype));
  at::Tensor workspace_2 =
      at::empty({block_num, block_M, block_N}, Q.options().dtype(dtype));
  at::Tensor workspace_3 =
      at::empty({block_num, block_M, dim}, Q.options().dtype(accum_dtype));

  aclrtStream stream = c10_npu::getCurrentNPUStream().stream(false);

  TORCH_CHECK(stream != nullptr,
              "tl_ascend.flash_attention: Get current NPU stream failed.");

  call(reinterpret_cast<uint8_t *>(Q.data_ptr()),
       reinterpret_cast<uint8_t *>(K.data_ptr()),
       reinterpret_cast<uint8_t *>(V.data_ptr()),
       reinterpret_cast<uint8_t *>(Output.data_ptr()),
       reinterpret_cast<uint8_t *>(workspace_1.data_ptr()),
       reinterpret_cast<uint8_t *>(workspace_2.data_ptr()),
       reinterpret_cast<uint8_t *>(workspace_3.data_ptr()), stream);
  return Output;
}

TORCH_LIBRARY(tl_ascend, m) {
  m.def("flash_attention(Tensor Q, Tensor K, Tensor V) -> Tensor");
}

TORCH_LIBRARY_IMPL(tl_ascend, PrivateUse1, m) {
  m.impl("flash_attention", &flash_attention_wrapper);
}

TORCH_LIBRARY_IMPL(tl_ascend, Meta, m) {
  m.impl(
      "flash_attention",
      [](const at::Tensor &, const at::Tensor &,
         const at::Tensor &) -> at::Tensor {
        TORCH_CHECK(
            false,
            "tl_ascend.flash_attention: Meta/torch.compile is not supported; "
            "this extension requires the fixed compiled NPU signature");
      });
}

extern "C" {
/* Creates a dummy empty _inner module that can be imported from Python.
   The import from Python will load the .so consisting of this file
   in this extension, so that the TORCH_LIBRARY static initializers
   below are run. */
PyObject *PyInit__inner(void) {
  static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT,
      "_inner", /* name of module */
      NULL,     /* module documentation, may be NULL */
      -1,       /* size of per-interpreter state of the module,
                    or -1 if the module keeps state in global variables. */
      NULL,     /* methods */
  };
  return PyModule_Create(&module_def);
}
}
