// src/prefill/fp8_gemm_stub.cu - the sm_89 (Ada) build's stand-in for the FP8 E4M3 tensor-core products.
//
// fp8_gemm.cu's block-scaled mma is an sm_120a instruction, so the Ada build has no FP8 tensor-core path.  Nothing in
// the inkling2 path calls these (its prompt products go through MMQ, and the dense module is not built), but the
// symbols must exist for anything that links prefill_fp8.  Each entry point reports the missing path and exits.
#include "basalt/prefill/fp8_gemm.hpp"

#include <cstdio>
#include <cstdlib>

namespace basalt::prefill::fp8 {

namespace {
[[noreturn]] void unavailable(const char* what) {
    std::fprintf(stderr, "basalt: prefill fp8: %s needs the sm_120a FP8 tensor cores (this is the sm_89 build)\n", what);
    std::exit(1);
}
}  // namespace

size_t act_bytes(int64_t rows, int64_t cols) { return (size_t) rows * (size_t) cols + (size_t) rows * 4 + 16; }
float* act_scales(void* act, int64_t rows, int64_t cols) { return (float*) ((uint8_t*) act + (size_t) rows * (size_t) cols); }

void quantize(const float*, int64_t, int64_t, int64_t, void*, void*) { unavailable("quantize"); }
void rms_norm_quantize(const float*, const float*, float, int64_t, int64_t, void*, float*, void*) { unavailable("rms_norm_quantize"); }
void rms_norm_quantize(const uint16_t*, const float*, float, int64_t, int64_t, void*, uint16_t*, void*) { unavailable("rms_norm_quantize"); }
void gemm(const uint8_t*, int64_t, int64_t, const float*, const void*, int64_t, void*, int64_t, Epilogue, void*) { unavailable("gemm"); }
void gemm_segmented(const Segment*, int, int64_t, const void*, int64_t, Epilogue, void*) { unavailable("gemm_segmented"); }

}  // namespace basalt::prefill::fp8
