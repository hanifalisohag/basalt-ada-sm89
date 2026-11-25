// src/prefill/moe_mxfp8_stub.cu - the sm_89 (Ada) build's stand-in for the MXFP8 expert products.
//
// moe_mxfp8.cu's block-scaled mma (kind::mxf8f6f4) is an sm_120a instruction, so the Ada build has no MXFP8
// tensor-core path.  The inkling2 prompt products go through MMQ instead (prompt.cpp's mxfp8_shape() is only
// consulted when the MXFP8 path is built), but the symbols must exist for anything that links prefill_mxfp8.
// Each entry point reports the missing path and exits.
#include "basalt/prefill/moe_mxfp8.hpp"

#include <cstdio>
#include <cstdlib>

namespace basalt::prefill::mxfp8 {

namespace {
[[noreturn]] void unavailable(const char* what) {
    std::fprintf(stderr, "basalt: prefill mxfp8: %s needs the sm_120a FP8 block-scaled tensor cores (this is the sm_89 build)\n", what);
    std::exit(1);
}
}  // namespace

size_t act_bytes(int64_t rows, int64_t cols) { return (size_t) rows * (size_t) cols + (size_t) rows * (size_t) (cols / 32) + 16; }

void plan_blocks(const int32_t*, int, int32_t*, void*) { unavailable("plan_blocks"); }
void quantize_tokens(const float*, int64_t, int64_t, int64_t, void*, void*) { unavailable("quantize_tokens"); }
void quantize_f16(const uint16_t*, int64_t, int64_t, int64_t, void*, void*) { unavailable("quantize_f16"); }
void run(const Product&, void*) { unavailable("run"); }

}  // namespace basalt::prefill::mxfp8
