// include/basalt/kernels/prefetch.cuh - an L2 prefetch hint that compiles on every architecture.
//
// `cp.async.bulk.prefetch.L2.global` is a Hopper (sm_90) instruction.  On Ada (sm_89) and Ampere the hint is
// dropped: it only warms L2, so leaving it out changes performance, never results.  The call sites keep the
// same shape on every architecture.
#pragma once

#include <cuda_runtime.h>

namespace basalt::kernels {

/// Ask L2 to start fetching `bytes` at `p`.  A hint: no effect on the values a kernel computes.
__device__ __forceinline__ void prefetch_l2(const void* p, unsigned bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes) : "memory");
#else
    (void) p;
    (void) bytes;
#endif
}

}  // namespace basalt::kernels
