// include/basalt/kernels/narrow_bf16.cuh - FP8 E4M3 and FP4 E2M1 codes as bf16x2 for a kernel's mma operands.  Device
// code only (no kernel): include from a .cu.
//
// Every E4M3 and E2M1 value is exact in bf16.  `cvt.rn.bf16x2.e4m3x2` and `cvt.rn.bf16x2.e2m1x2` (one instruction
// each) need PTX ISA 9.2, CUDA 13.2: the ptxas of CUDA 13.0 and 13.1 rejects them.  There the codes go to f16x2
// (`cvt.rn.f16x2.*`, also exact) and each half widens through f32 to bf16, which gives the same bits, NaNs included.
// tests/kernels/narrow_bf16_parity.cu compares the two ways on every input.
#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#if defined(__CUDACC_VER_MAJOR__) && \
    (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && \
    (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 900)
#define BASALT_CVT_BF16X2_NARROW 1
#else
#define BASALT_CVT_BF16X2_NARROW 0
#endif

// `cvt.rn.f16x2.e4m3x2` and `cvt.rn.f16x2.e2m1x2` are sm_120a instructions: on Ada (sm_89) and Ampere the header
// converts in software instead.  Both formats are tiny and every value is exact in bf16, so the fallback is a few
// shifts and adds and gives the same bits, NaNs included (tests/kernels/narrow_bf16_parity.cu checks it).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1200
#define BASALT_CVT_NARROW_SOFTWARE 1
#else
#define BASALT_CVT_NARROW_SOFTWARE 0
#endif

namespace basalt::kernels {

/// f16x2 as bf16x2, for values bf16 holds exactly (each half through f32)
__device__ __forceinline__ unsigned f16x2_bf16x2(unsigned f) {
    unsigned h;
    asm("{\n"
        ".reg .f16 l, u;\n"
        ".reg .f32 a, b;\n"
        "mov.b32 {l, u}, %1;\n"
        "cvt.f32.f16 a, l;\n"
        "cvt.f32.f16 b, u;\n"
        "cvt.rn.bf16x2.f32 %0, b, a;\n"
        "}\n"
        : "=r"(h)
        : "r"(f));
    return h;
}

/// one E4M3 byte as bf16 bits, in software: sign, 4 exponent bits (bias 7) and 3 mantissa bits, all of which bf16
/// holds exactly.  A zero exponent with a zero mantissa is a signed zero; with a nonzero mantissa it is subnormal
/// (value = mantissa * 2^-9); 0x7f/0xff are NaN.
__device__ __forceinline__ unsigned e4m3_bf16_soft(unsigned b) {
    const unsigned s = (b >> 7) & 1u, e = (b >> 3) & 0xfu, m = b & 7u;
    unsigned bits;
    if (e == 0) {
        if (m == 0) bits = 0;   // signed zero
        else {
            // subnormal: m * 2^-9, normalise to a bf16 exponent
            int shift = 0;
            unsigned mm = m;
            while ((mm & 4u) == 0) { mm <<= 1; ++shift; }   // mm in [4,7], value = mm * 2^(-9-shift)
            const int exp = -9 - shift + 2;                 // mm has an implicit leading 1 at bit 2
            bits = (unsigned)(exp + 127) << 7 | ((mm & 3u) << 5);
        }
    } else if (e == 0xfu && m == 7u) {
        bits = 0x7f80u | (m ? 1u : 0u);   // NaN (0x7f/0xff); the payload is not significant
    } else {
        bits = (unsigned)((int)e - 7 + 127) << 7 | (m << 4);
    }
    return (s << 15) | bits;
}

/// two E4M3 bytes (the low 16 bits of x, the low byte the lower half) as bf16x2, in software
__device__ __forceinline__ unsigned e4m3x2_bf16x2_soft(unsigned x) {
    return e4m3_bf16_soft(x & 0xffu) | (e4m3_bf16_soft((x >> 8) & 0xffu) << 16);
}

/// one E2M1 nibble as bf16 bits, in software: sign, 2 exponent bits (bias 1) and 1 mantissa bit; the eight
/// magnitudes are 0, 0.5, 1, 1.5, 2, 3, 4, 6, all exact in bf16.
__device__ __forceinline__ unsigned e2m1_bf16_soft(unsigned n) {
    static const unsigned mag[8] = {0x0000u, 0x3f00u, 0x3f80u, 0x3fc0u, 0x4000u, 0x4040u, 0x4080u, 0x40c0u};
    return ((n & 8u) << 12) | mag[n & 7u];
}

/// two E4M3 bytes (the low 16 bits of x, the low byte the lower half) as bf16x2 through f16x2
__device__ __forceinline__ unsigned e4m3x2_bf16x2_via_f16(unsigned x) {
#if BASALT_CVT_NARROW_SOFTWARE
    return e4m3x2_bf16x2_soft(x);
#else
    unsigned f;
    const unsigned short v = (unsigned short) x;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f) : "h"(v));
    return f16x2_bf16x2(f);
#endif
}

/// the four bytes of w, each two E2M1 codes (the low nibble the lower half), as bf16x2 through f16x2
__device__ __forceinline__ void e2m1x8_bf16x2_via_f16(unsigned w, unsigned (&h)[4]) {
#if BASALT_CVT_NARROW_SOFTWARE
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const unsigned b = (w >> (8 * k)) & 0xffu;
        h[k] = e2m1_bf16_soft(b & 0xfu) | (e2m1_bf16_soft((b >> 4) & 0xfu) << 16);
    }
#else
    unsigned f[4];
    asm("{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        "mov.b32 {b0, b1, b2, b3}, %4;\n"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
        "cvt.rn.f16x2.e2m1x2 %1, b1;\n"
        "cvt.rn.f16x2.e2m1x2 %2, b2;\n"
        "cvt.rn.f16x2.e2m1x2 %3, b3;\n"
        "}\n"
        : "=r"(f[0]), "=r"(f[1]), "=r"(f[2]), "=r"(f[3])
        : "r"(w));
#pragma unroll
    for (int i = 0; i < 4; ++i) h[i] = f16x2_bf16x2(f[i]);
#endif
}

#if BASALT_CVT_NARROW_SOFTWARE
// bf16x2 multiply without mul.rn.bf16x2 (sm_90+): unpack to float, multiply, repack.  Exact for bf16 inputs.
__device__ __forceinline__ unsigned bf16x2_mul(unsigned a, unsigned b) {
    const float a0 = __bfloat162float(__ushort_as_bfloat16((unsigned short) (a & 0xffffu)));
    const float a1 = __bfloat162float(__ushort_as_bfloat16((unsigned short) (a >> 16)));
    const float b0 = __bfloat162float(__ushort_as_bfloat16((unsigned short) (b & 0xffffu)));
    const float b1 = __bfloat162float(__ushort_as_bfloat16((unsigned short) (b >> 16)));
    const unsigned short r0 = __bfloat16_as_ushort(__float2bfloat16_rn(a0 * b0));
    const unsigned short r1 = __bfloat16_as_ushort(__float2bfloat16_rn(a1 * b1));
    return (unsigned) r0 | ((unsigned) r1 << 16);
}
#else
__device__ __forceinline__ unsigned bf16x2_mul(unsigned a, unsigned b) {
    unsigned d;
    asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
    return d;
}
#endif

/// two E4M3 bytes (the low 16 bits of x, the low byte the lower half) as bf16x2 (exact)
__device__ __forceinline__ unsigned e4m3x2_bf16x2(unsigned x) {
#if BASALT_CVT_NARROW_SOFTWARE
    return e4m3x2_bf16x2_soft(x);
#elif BASALT_CVT_BF16X2_NARROW
    unsigned h;
    const unsigned short v = (unsigned short) x;
    asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"(v));
    return h;
#else
    return e4m3x2_bf16x2_via_f16(x);
#endif
}

/// the four bytes of w, each two E2M1 codes (the low nibble the lower half), as bf16x2 times the bf16x2 scale sc (exact:
/// a code times a UE4M3 scale has at most five significant bits)
__device__ __forceinline__ void e2m1x8_bf16x2_scaled(unsigned w, unsigned sc, unsigned (&h)[4]) {
#if BASALT_CVT_NARROW_SOFTWARE
    unsigned t[4];
    e2m1x8_bf16x2_via_f16(w, t);
#pragma unroll
    for (int i = 0; i < 4; ++i) h[i] = bf16x2_mul(t[i], sc);
#elif BASALT_CVT_BF16X2_NARROW
    asm("{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        ".reg .b32 t0, t1, t2, t3;\n"
        "mov.b32 {b0, b1, b2, b3}, %4;\n"
        "cvt.rn.bf16x2.e2m1x2 t0, b0;\n"
        "cvt.rn.bf16x2.e2m1x2 t1, b1;\n"
        "cvt.rn.bf16x2.e2m1x2 t2, b2;\n"
        "cvt.rn.bf16x2.e2m1x2 t3, b3;\n"
        "mul.rn.bf16x2 %0, t0, %5;\n"
        "mul.rn.bf16x2 %1, t1, %5;\n"
        "mul.rn.bf16x2 %2, t2, %5;\n"
        "mul.rn.bf16x2 %3, t3, %5;\n"
        "}\n"
        : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3])
        : "r"(w), "r"(sc));
#else
    unsigned t[4];
    e2m1x8_bf16x2_via_f16(w, t);
#pragma unroll
    for (int i = 0; i < 4; ++i) h[i] = bf16x2_mul(t[i], sc);
#endif
}

}  // namespace basalt::kernels
