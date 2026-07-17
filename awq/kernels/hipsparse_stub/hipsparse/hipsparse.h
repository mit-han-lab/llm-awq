/* ROCm-LLM-AWQ: minimal hipsparse stub for builds where hipsparse is not in include path.
 * PyTorch's ATen/hip/HIPContextLight.h includes hipsparse.h but only uses it for
 * hipsparseHandle_t in sparse tensor operations — not needed for AWQ inference.
 * KB entry: awq_hipsparse_stub_for_torch_hip_context (new)
 */
#pragma once
#ifndef HIPSPARSE_H
#define HIPSPARSE_H
#include <hip/hip_runtime.h>
typedef void* hipsparseHandle_t;
typedef void* hipsparseSpMatDescr_t;
typedef void* hipsparseDnVecDescr_t;
typedef void* hipsparseDnMatDescr_t;
typedef int   hipsparseStatus_t;
#define HIPSPARSE_STATUS_SUCCESS 0
#endif // HIPSPARSE_H
