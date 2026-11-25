// src/prefill/fp4_dense_ada.cu - the sm_89 (Ada) build's implementation of the fp4_dense API (fp4_dense.hpp).
//
// fp4_dense.cu's kernel is TMA + mbarrier + the block-scaled m16n8k64 mma, all sm_120a; on Ada none of them exist, so
// the dense stack's prompt products had no path and the qwen35 model could not link.  This file gives the same API a
// portable implementation: the weights (NVFP4 planes, nvfp4_planes.hpp) and the activations (the quantize layout below)
// are decoded to bf16 in registers and multiplied on mma.m16n8k16 (bf16 in, fp32 accumulate), which Ada has.  The
// decode is the one the decode window already uses (window_gemm.cuh, narrow_bf16.cuh), so a prompt chunk and a decode
// window read the same bytes the same way.
//
// The activation layout is fp4_dense.cu's, byte for byte, so a buffer quantized here is read by the decode path too:
//   codes   [rows][cols / 2] E2M1 nibbles at offset 0 (byte b of a row: value 2b low, 2b + 1 high);
//   scales  padded(rows) * (cols / 16) UE4M3 bytes after the codes, in 256-token x 16-group tiles (kScaleTileBytes),
//           tile (r / 256) * (cols / 256) + ktile, within a tile (r % 256) * 16 + (b & 15);
//   gs      one fp32 per row: the row's largest |value| / (6 * 448).
#include "basalt/prefill/fp4_dense.hpp"

#include "basalt/kernels/narrow_bf16.cuh"
#include "basalt/kernels/silu.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "basalt/log.hpp"

namespace basalt::prefill::fp4_dense {
namespace {

using namespace basalt::kernels;

constexpr int kBT = 256;    // tokens per block (the sm_120a kernel's tile, so the activation layout matches)
constexpr int kBR = 128;    // weight rows per block
constexpr int kStages = 2;  // K stages in flight (3 would need 108 KB of smem for the SwiGLU pair, past Ada's 100 KB)
constexpr int kThreads = 256;
constexpr int kKStage = 128;   // K values per stage
constexpr int kScaleTileBytes = kBT * 16;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        basalt::log::error("basalt", "fp4 dense: %s: %s", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

__host__ __device__ int64_t padded(int64_t tokens) { return (tokens + kBT - 1) / kBT * kBT; }

// ------------------------------------------------------------------------------------------------- device

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
    const unsigned s = (unsigned) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void mma_bf16(float (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                 "{%0, %1, %2, %3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// The activation quantizer: fp4_dense.cu's quantize_kernel, unchanged in what it writes.
template <bool NORM, bool AMAX = false, class In = float, class Xn = float>
__global__ void quantize_kernel(const In* __restrict__ x, uint8_t* __restrict__ codes, uint8_t* __restrict__ scales,
                                float* __restrict__ gs, int cols, int64_t ld, int64_t rows,
                                const float* __restrict__ nw = nullptr, float eps = 0.0f, Xn* __restrict__ xn = nullptr,
                                const float* __restrict__ rowmax = nullptr) {
    const int lane = threadIdx.x & 31;
    const int64_t r = (int64_t) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int nb = cols / 16, ktiles = cols / 256;
    const int64_t tile_base = (r / kBT) * ktiles;
    uint8_t* srow = scales + (r % kBT) * 16;
    if (r >= rows) {
        for (int kt = lane; kt < ktiles; kt += 32)
            *reinterpret_cast<uint4*>(srow + (tile_base + kt) * kScaleTileBytes) = make_uint4(0, 0, 0, 0);
        return;
    }
    const In* xr = x + r * ld;
    float inv_rms = 1.0f;
    if (NORM) {
        float ss = 0.0f;
        for (int b = lane; b < nb; b += 32) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float4 v;
                if constexpr (std::is_same_v<In, float>) {
                    v = *(const float4*) (xr + 16 * b + 4 * i);
                } else {
                    const uint2 u = *(const uint2*) (xr + 16 * b + 4 * i);
                    const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
                    const float2 c = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
                    v = make_float4(a.x, a.y, c.x, c.y);
                }
                ss = fmaf(v.x, v.x, fmaf(v.y, v.y, fmaf(v.z, v.z, fmaf(v.w, v.w, ss))));
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        inv_rms = rsqrtf(ss / (float) cols + eps);
    }
    auto val4 = [&](int c) -> float4 {
        float4 v;
        if constexpr (std::is_same_v<In, float>) {
            v = *(const float4*) (xr + c);
        } else {
            const uint2 u = *(const uint2*) (xr + c);
            const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
            const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
            v = make_float4(a.x, a.y, b.x, b.y);
        }
        if (NORM) {
            const float4 w = *(const float4*) (nw + c);
            v = make_float4(v.x * inv_rms * w.x, v.y * inv_rms * w.y, v.z * inv_rms * w.z, v.w * inv_rms * w.w);
        }
        return v;
    };
    float amax = 0.0f;
    if (AMAX) {
        amax = rowmax[r];
    } else {
        for (int b = lane; b < nb; b += 32) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 v = val4(16 * b + 4 * i);
                if (NORM && xn) {
                    if constexpr (std::is_same_v<Xn, float>) {
                        *(float4*) (xn + r * cols + 16 * b + 4 * i) = v;
                    } else {
                        const __nv_bfloat162 a = __floats2bfloat162_rn(v.x, v.y);
                        const __nv_bfloat162 c = __floats2bfloat162_rn(v.z, v.w);
                        *reinterpret_cast<uint2*>(xn + r * cols + 16 * b + 4 * i) =
                            make_uint2(*reinterpret_cast<const unsigned*>(&a), *reinterpret_cast<const unsigned*>(&c));
                    }
                }
                amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    }
    const float g = amax * (1.0f / (6.0f * 448.0f));
    if (lane == 0) gs[r] = g;
    const float inv_g = amax > 0.0f ? 1.0f / g : 0.0f;
    for (int b = lane; b < nb; b += 32) {
        float v[16];
        float m = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float4 t = val4(16 * b + 4 * i);
            v[4 * i] = t.x; v[4 * i + 1] = t.y; v[4 * i + 2] = t.z; v[4 * i + 3] = t.w;
            m = fmaxf(m, fmaxf(fmaxf(fabsf(t.x), fabsf(t.y)), fmaxf(fabsf(t.z), fabsf(t.w))));
        }
        const __nv_fp8_storage_t sb = __nv_cvt_float_to_fp8(m * inv_g * (1.0f / 6.0f), __NV_SATFINITE, __NV_E4M3);
        const float sd = __half2float(__half(__nv_cvt_fp8_to_halfraw(sb, __NV_E4M3)));
        const float inv = sd > 0.0f ? 1.0f / (sd * g) : 0.0f;
        uint32_t w[2] = {0, 0};
        // E2M1 to nearest, ties to the even code, saturating (cvt.rn.satfinite.e2m1x2.f32 is sm_120a)
        auto e2m1 = [](float x) -> uint32_t {
            const float a = fabsf(x);
            const uint32_t c = a <= 0.25f ? 0 : a < 0.75f ? 1 : a <= 1.25f ? 2 : a < 1.75f ? 3 : a <= 2.5f ? 4 : a < 3.5f ? 5 : a <= 5.0f ? 6 : 7;
            return c | (x < 0.0f && c ? 8u : 0u);
        };
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint32_t cc = e2m1(v[2 * j] * inv) | (e2m1(v[2 * j + 1] * inv) << 4);
            w[j >> 2] |= cc << (8 * (j & 3));
        }
        *reinterpret_cast<uint2*>(codes + r * (cols / 2) + 8 * b) = make_uint2(w[0], w[1]);
        srow[(tile_base + b / 16) * kScaleTileBytes + (b & 15)] = (uint8_t) sb;
    }
}

// ------------------------------------------------------------------------------------------------- the product

// One block: kBR weight rows x kBT tokens, K streamed in kKStage stages.  The weights (NVFP4 planes) and the
// activations (the quantize layout) are decoded to bf16 and multiplied on mma.m16n8k16.  A warp owns 16 weight rows
// (kBR / 8 warps) and all kBT tokens in 8-row tiles; the activation column of a tile is the mma's B operand.
//
// SWIGLU: w0 is the gate matrix and w1 the up matrix, both rows x K; the block's kBR output rows read kBR gate rows
// and kBR up rows, and the output is silu(s_gate g) * (s_up u).  ADD: dst += the product.  BF: dst is bf16.
template <bool SWIGLU, bool ADD, bool BF = false>
__global__ __launch_bounds__(kThreads, 1) void product_kernel(const uint8_t* __restrict__ w0,
                                                             const uint8_t* __restrict__ w1,
                                                             const float* __restrict__ ws0,
                                                             const float* __restrict__ ws1,
                                                             int64_t rows, int64_t K, const uint8_t* __restrict__ act,
                                                             const float* __restrict__ gs, int tokens, void* __restrict__ dst,
                                                             int64_t ld, float* __restrict__ amax) {
    constexpr int kWarps = kThreads / 32;
    constexpr int kRowsPerWarp = kBR / kWarps;   // 16
    constexpr int kTokTiles = kBT / 8;           // 32 tiles of 8 tokens: a warp owns its rows for every token
    constexpr int kCodeBytes = kBR * (kKStage / 2);   // weight codes per stage, per matrix
    constexpr int kScaleBytes = (kKStage / 64) * kBR * 4;
    constexpr int kActCodes = kBT * (kKStage / 2);
    constexpr int kActScales = kBT * (kKStage / 16);
    constexpr int kMats = SWIGLU ? 2 : 1;

    extern __shared__ __align__(16) uint8_t smem[];
    uint8_t* wc = smem;                                   // [kMats][kStages][kCodeBytes]
    uint8_t* wsc = wc + kMats * kStages * kCodeBytes;     // [kMats][kStages][kScaleBytes]
    uint8_t* ac = wsc + kMats * kStages * kScaleBytes;    // [kStages][kActCodes]
    uint8_t* asc = ac + kStages * kActCodes;              // [kStages][kActScales]

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    const int64_t row0 = (int64_t) blockIdx.y * kBR;
    const int tok0 = (int) blockIdx.x * kBT;
    const int ktiles = (int) (K / kKStage);
    const int64_t row_bytes = K / 2;
    const int64_t scale_off = rows * (K / 2);
    const int64_t act_scale_off = (int64_t) tokens * (K / 2);

    // stage s: the weight codes/scales of rows row0.. and the activation codes/scales of tokens tok0..
    auto issue = [&](int kt, int s) {
        for (int m = 0; m < kMats; ++m) {
            const uint8_t* w = m == 0 ? w0 : w1;
            uint8_t* wcs = wc + (m * kStages + s) * kCodeBytes;
            uint8_t* wss = wsc + (m * kStages + s) * kScaleBytes;
            for (int p = threadIdx.x; p < kBR * (kKStage / 2) / 16; p += kThreads) {
                const int r = p / (kKStage / 32), q = p % (kKStage / 32);
                const int64_t row = min(row0 + r, rows - 1);
                cp_async16(wcs + r * (kKStage / 2) + q * 16, w + row * row_bytes + (int64_t) kt * (kKStage / 2) + q * 16);
            }
            // the weight scale plane: 512-byte tiles of 128 rows x 4 groups, tiles row-block major.  A row's four
            // scale bytes sit at a 4-byte offset, so this is a plain 4-byte copy (cp.async needs 16).
            for (int p = threadIdx.x; p < (kKStage / 64) * kBR; p += kThreads) {
                const int kb = p / kBR, r = p % kBR;
                const int64_t row = row0 + r;
                const int64_t tbase = (row / 128) * (K / 64);
                const int quart = (int) ((row & 31) * 16 + ((row & 127) >> 5) * 4);
                const uint32_t v = *reinterpret_cast<const uint32_t*>(
                    w + scale_off + (tbase + kt * (kKStage / 64) + kb) * 512 + quart);
                *reinterpret_cast<uint32_t*>(wss + kb * kBR * 4 + r * 4) = v;
            }
        }
        uint8_t* acs = ac + s * kActCodes;
        for (int p = threadIdx.x; p < kBT * (kKStage / 2) / 16; p += kThreads) {
            const int r = p / (kKStage / 32), q = p % (kKStage / 32);
            cp_async16(acs + r * (kKStage / 2) + q * 16, act + (int64_t) (tok0 + r) * (K / 2) + (int64_t) kt * (kKStage / 2) + q * 16);
        }
        uint8_t* ass = asc + s * kActScales;
        // The activation scale plane tiles 256 tokens x 16 groups in 4096-byte blocks, tiles group-major: a token's
        // group g sits at (r % 256) * 16 + (tile + g / 16) * 4096 + (g & 15).  kKStage / 16 groups per stage span
        // kKStage / 256 tiles, so gather them a group at a time.
        for (int p = threadIdx.x; p < kBT * (kKStage / 16); p += kThreads) {
            const int r = p / (kKStage / 16), g = p % (kKStage / 16);
            const int64_t tbase = ((tok0 + r) / kBT) * (K / 256);
            const int gg = kt * (kKStage / 16) + g;   // the global 16-value group
            ass[r * (kKStage / 16) + g] = act[act_scale_off + (tbase + gg / 16) * kScaleTileBytes +
                                             (r % kBT) * 16 + (gg % 16)];
        }
    };

#pragma unroll
    for (int s = 0; s < kStages - 1; ++s)
        if (s < ktiles) issue(s, s);
    cp_commit();
    float acc[kMats][kTokTiles][4];
#pragma unroll
    for (int m = 0; m < kMats; ++m)
#pragma unroll
        for (int t = 0; t < kTokTiles; ++t) acc[m][t][0] = acc[m][t][1] = acc[m][t][2] = acc[m][t][3] = 0.0f;

    const int r0l = warp * kRowsPerWarp + g, r1l = r0l + 8;
    for (int kt = 0; kt < ktiles; ++kt) {
        const int s = kt % kStages;
        if (kt + kStages - 1 < ktiles) issue(kt + kStages - 1, (kt + kStages - 1) % kStages);
        cp_commit();
        cp_wait<kStages - 1>();
        __syncthreads();
        const uint8_t* acs = ac + s * kActCodes;
        const uint8_t* ass = asc + s * kActScales;
#pragma unroll
        for (int b = 0; b < kKStage / 64; ++b) {
#pragma unroll
            for (int m = 0; m < kMats; ++m) {
                const uint8_t* wcs = wc + (m * kStages + s) * kCodeBytes;
                const uint8_t* wss = wsc + (m * kStages + s) * kScaleBytes;
                const uint8_t* q0 = wcs + r0l * (kKStage / 2);
                const uint8_t* q1 = wcs + r1l * (kKStage / 2);
                unsigned a0[4], a1[4], a2[4], a3[4];
#pragma unroll
                for (int st = 0; st < 4; ++st) {
                    const int wg = 4 * b + st;   // the 16-value group this thread's window covers
                    const unsigned s0 = wss[(wg / 4) * kBR * 4 + r0l * 4 + (wg % 4)];
                    const unsigned s1 = wss[(wg / 4) * kBR * 4 + r1l * 4 + (wg % 4)];
                    const unsigned sc0 = e4m3x2_bf16x2(s0 | (s0 << 8)), sc1 = e4m3x2_bf16x2(s1 | (s1 << 8));
                    const int off = b * 32 + st * 8 + c;   // codes 64b + 16st + 2c .. +1 and +8 .. +9
                    const uint32_t wa = (uint32_t) q0[off] | ((uint32_t) q0[off + 1] << 8) |
                                        ((uint32_t) q0[off + 2] << 16) | ((uint32_t) q0[off + 3] << 24);
                    const uint32_t wb = (uint32_t) q0[off + 4];
                    const uint32_t wc = (uint32_t) q1[off] | ((uint32_t) q1[off + 1] << 8) |
                                        ((uint32_t) q1[off + 2] << 16) | ((uint32_t) q1[off + 3] << 24);
                    const uint32_t wd = (uint32_t) q1[off + 4];
                    unsigned h0[4], h1[4], h2[4], h3[4];
                    e2m1x8_bf16x2_scaled(wa, sc0, h0);
                    e2m1x8_bf16x2_scaled(wb, sc0, h2);
                    e2m1x8_bf16x2_scaled(wc, sc1, h1);
                    e2m1x8_bf16x2_scaled(wd, sc1, h3);
                    a0[st] = h0[0];
                    a1[st] = h1[0];
                    a2[st] = h2[0];
                    a3[st] = h3[0];
                }
#pragma unroll
                for (int t = 0; t < kTokTiles; ++t) {
                    const uint8_t* xr = acs + (t * 8 + g) * (kKStage / 2);
                    unsigned xh[8];
#pragma unroll
                    for (int st = 0; st < 4; ++st) {
                        const unsigned xs0 = ass[(t * 8 + g) * (kKStage / 16) + 4 * b + st];
                        const unsigned xsc = e4m3x2_bf16x2(xs0 | (xs0 << 8));
                        const int off = b * 32 + st * 8 + c;   // codes 64b + 16st + 2c .. +1 and +8 .. +9
                        const uint32_t xa = (uint32_t) xr[off] | ((uint32_t) xr[off + 1] << 8) |
                                            ((uint32_t) xr[off + 2] << 16) | ((uint32_t) xr[off + 3] << 24);
                        const uint32_t xb = (uint32_t) xr[off + 4] | ((uint32_t) xr[off + 5] << 8);
                        unsigned xh0[4], xh1[4];
                        e2m1x8_bf16x2_scaled(xa, xsc, xh0);
                        e2m1x8_bf16x2_scaled(xb, xsc, xh1);
                        xh[st] = xh0[0];
                        xh[4 + st] = xh1[0];
                    }
#pragma unroll
                    for (int st = 0; st < 4; ++st)
                        mma_bf16(acc[m][t], a0[st], a1[st], a2[st], a3[st], xh[st], xh[4 + st]);
                }
            }
        }
        __syncthreads();
    }

    // ---- epilogue
    const float sg = ws0 ? *ws0 : 1.0f;
    const float su = ws1 ? *ws1 : 1.0f;
    if constexpr (SWIGLU) {
        __nv_bfloat16* hdst = reinterpret_cast<__nv_bfloat16*>(dst);
#pragma unroll
        for (int t = 0; t < kTokTiles; ++t)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int col = t * 8 + 2 * c + (j & 1);
                const int rl = warp * kRowsPerWarp + g + 8 * (j >> 1);
                if (tok0 + col >= tokens || row0 + rl >= rows) continue;
                const float gt = gs ? gs[tok0 + col] : 1.0f;
                const float gv = __fmul_rn(acc[0][t][j], __fmul_rn(gt, sg));
                const float uv = __fmul_rn(acc[1][t][j], __fmul_rn(gt, su));
                const float hv = __fmul_rn(basalt::kernels::silu_quotient(gv), uv);   // silu(g) * u
                const __nv_bfloat16 hb = __float2bfloat16_rn(hv);
                hdst[(int64_t) (tok0 + col) * ld + row0 + rl] = hb;
                if (amax) atomicMax(reinterpret_cast<int*>(amax + tok0 + col),
                                    __float_as_int(fabsf(__bfloat162float(hb))));   // the amax is of the stored values
            }
    } else {
#pragma unroll
        for (int t = 0; t < kTokTiles; ++t)
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int col = t * 8 + 2 * c + (j & 1);
                const int rl = warp * kRowsPerWarp + g + 8 * (j >> 1);
                if (tok0 + col >= tokens || row0 + rl >= rows) continue;
                const float v = __fmul_rn(acc[0][t][j], __fmul_rn(gs ? gs[tok0 + col] : 1.0f, sg));
                if constexpr (BF) {
                    uint16_t* d16 = reinterpret_cast<uint16_t*>(dst) + (int64_t) (tok0 + col) * ld + row0 + rl;
                    if (ADD)
                        *d16 = __bfloat16_as_ushort(__float2bfloat16_rn(
                            __fadd_rn(__bfloat162float(__ushort_as_bfloat16(*d16)), v)));
                    else
                        *d16 = __bfloat16_as_ushort(__float2bfloat16_rn(v));
                } else {
                    float* d = reinterpret_cast<float*>(dst) + (int64_t) (tok0 + col) * ld + row0 + rl;
                    if (ADD) *d += v;
                    else *d = v;
                }
            }
    }
}

// ------------------------------------------------------------------------------------------------- host

void check(const void* a, const void* b, int64_t rows, int64_t K, int64_t tokens, const void* act, const void* dst,
           int64_t ld, const char* what, bool bf16 = false, const float* gs = nullptr) {
    const auto al = [](const void* v, size_t n) { return (reinterpret_cast<uintptr_t>(v) & (n - 1)) == 0; };
    if (rows <= 0 || rows % 128 || K <= 0 || K % 256 || tokens <= 0 || ld < rows || (ld & (bf16 ? 7 : 1)) || !al(a, 16) ||
        !al(b, 16) || !al(act, 16) || !al(dst, bf16 ? 16 : 8) || (bf16 && !al(gs, 16))) {
        basalt::log::error("basalt", "fp4 dense: %s: rows %lld K %lld tokens %lld ld %lld", what, (long long) rows,
                           (long long) K, (long long) tokens, (long long) ld);
        std::exit(1);
    }
}

template <bool SWIGLU, bool ADD, bool BF = false>
void launch(const uint8_t* w0, const uint8_t* w1, const float* ws0, const float* ws1, int64_t rows, int64_t K,
            const uint8_t* act, const float* gs, int64_t tokens, void* dst, int64_t ld, float* amax, void* stream) {
    constexpr int kCodeBytes = kBR * (kKStage / 2);
    constexpr int kScaleBytes = (kKStage / 64) * kBR * 4;
    constexpr int kActCodes = kBT * (kKStage / 2);
    constexpr int kActScales = kBT * (kKStage / 16);
    constexpr int kMats = SWIGLU ? 2 : 1;
    const int smem = kMats * kStages * (kCodeBytes + kScaleBytes) + kStages * (kActCodes + kActScales);
    static bool configured = false;
    if (!configured) {
        ck(cudaFuncSetAttribute(product_kernel<SWIGLU, ADD, BF>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem),
           "smem attribute");
        configured = true;
    }
    const dim3 grid((unsigned) ((tokens + kBT - 1) / kBT), (unsigned) (rows / kBR));
    product_kernel<SWIGLU, ADD, BF><<<grid, kThreads, smem, (cudaStream_t) stream>>>(w0, w1, ws0, ws1, rows, K, act, gs,
                                                                                     (int) tokens, dst, ld, amax);
    ck(cudaGetLastError(), "product");
}

}  // namespace

size_t act_bytes(int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) (cols / 2) + (size_t) padded(rows) * (size_t) (cols / 16);
}

void quantize(const float* x, int64_t rows, int64_t cols, int64_t ld, void* act, float* gs, void* stream) {
    if (rows <= 0) return;
    if (cols % 256 || (ld & 3) || (reinterpret_cast<uintptr_t>(x) & 15) || (reinterpret_cast<uintptr_t>(act) & 15)) {
        basalt::log::error("basalt", "fp4 dense: quantize %lld columns, ld %lld", (long long) cols, (long long) ld);
        std::exit(1);
    }
    uint8_t* codes = (uint8_t*) act;
    uint8_t* scales = codes + (size_t) rows * (size_t) (cols / 2);
    const unsigned grid = (unsigned) ((padded(rows) + 7) / 8);
    quantize_kernel<false><<<grid, 256, 0, (cudaStream_t) stream>>>(x, codes, scales, gs, (int) cols, ld, rows);
    ck(cudaGetLastError(), "quantize");
}

void quantize_amax(const uint16_t* x, const float* rowmax, int64_t rows, int64_t cols, int64_t ld, void* act, float* gs,
                   void* stream) {
    if (rows <= 0) return;
    if (cols % 256 || (ld & 3) || (reinterpret_cast<uintptr_t>(x) & 15) || (reinterpret_cast<uintptr_t>(act) & 15)) {
        basalt::log::error("basalt", "fp4 dense: quantize %lld columns, ld %lld", (long long) cols, (long long) ld);
        std::exit(1);
    }
    uint8_t* codes = (uint8_t*) act;
    uint8_t* scales = codes + (size_t) rows * (size_t) (cols / 2);
    const unsigned grid = (unsigned) ((padded(rows) + 7) / 8);
    quantize_kernel<false, true, __nv_bfloat16><<<grid, 256, 0, (cudaStream_t) stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x), codes, scales, gs, (int) cols, ld, rows, nullptr, 0.0f,
        static_cast<float*>(nullptr), rowmax);
    ck(cudaGetLastError(), "quantize_amax");
}

void rms_norm_quantize(const float* x, const float* w, float eps, int64_t rows, int64_t cols, void* act, float* gs,
                       float* xn, void* stream) {
    if (rows <= 0) return;
    if (cols % 256 || (reinterpret_cast<uintptr_t>(x) & 15) || (reinterpret_cast<uintptr_t>(act) & 15)) {
        basalt::log::error("basalt", "fp4 dense: rms_norm_quantize %lld columns", (long long) cols);
        std::exit(1);
    }
    uint8_t* codes = (uint8_t*) act;
    uint8_t* scales = codes + (size_t) rows * (size_t) (cols / 2);
    const unsigned grid = (unsigned) ((padded(rows) + 7) / 8);
    quantize_kernel<true><<<grid, 256, 0, (cudaStream_t) stream>>>(x, codes, scales, gs, (int) cols, cols, rows, w, eps, xn);
    ck(cudaGetLastError(), "rms_norm_quantize");
}

void product(const void* w, const float* w_scale, int64_t rows, int64_t K, const void* act, const float* gs,
             int64_t tokens, float* dst, int64_t ld, bool add, void* stream) {
    check(w, w, rows, K, tokens, act, dst, ld, "product");
    if (add)
        launch<false, true>(reinterpret_cast<const uint8_t*>(w), nullptr, w_scale, nullptr, rows, K,
                            reinterpret_cast<const uint8_t*>(act), gs, tokens, dst, ld, nullptr, stream);
    else
        launch<false, false>(reinterpret_cast<const uint8_t*>(w), nullptr, w_scale, nullptr, rows, K,
                             reinterpret_cast<const uint8_t*>(act), gs, tokens, dst, ld, nullptr, stream);
}

void product_bf16(const void* w, const float* w_scale, int64_t rows, int64_t K, const void* act, const float* gs,
                  int64_t tokens, uint16_t* dst, int64_t ld, void* stream) {
    check(w, w, rows, K, tokens, act, dst, ld, "product_bf16", true, gs);
    launch<false, false, true>(reinterpret_cast<const uint8_t*>(w), nullptr, w_scale, nullptr, rows, K,
                               reinterpret_cast<const uint8_t*>(act), gs, tokens, dst, ld, nullptr, stream);
}

void product_add_bf16(const void* w, const float* w_scale, int64_t rows, int64_t K, const void* act, const float* gs,
                      int64_t tokens, uint16_t* dst, int64_t ld, void* stream) {
    check(w, w, rows, K, tokens, act, dst, ld, "product_add_bf16", true, gs);
    launch<false, true, true>(reinterpret_cast<const uint8_t*>(w), nullptr, w_scale, nullptr, rows, K,
                              reinterpret_cast<const uint8_t*>(act), gs, tokens, dst, ld, nullptr, stream);
}

void rms_norm_quantize(const uint16_t* x, const float* w, float eps, int64_t rows, int64_t cols, void* act, float* gs,
                       uint16_t* xn, void* stream) {
    if (rows <= 0) return;
    if (cols % 256 || (reinterpret_cast<uintptr_t>(x) & 15) || (reinterpret_cast<uintptr_t>(act) & 15) ||
        (xn && (reinterpret_cast<uintptr_t>(xn) & 15))) {
        basalt::log::error("basalt", "fp4 dense: rms_norm_quantize %lld columns", (long long) cols);
        std::exit(1);
    }
    uint8_t* codes = (uint8_t*) act;
    uint8_t* scales = codes + (size_t) rows * (size_t) (cols / 2);
    const unsigned grid = (unsigned) ((padded(rows) + 7) / 8);
    quantize_kernel<true, false, __nv_bfloat16, __nv_bfloat16><<<grid, 256, 0, (cudaStream_t) stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x), codes, scales, gs, (int) cols, cols, rows, w, eps,
        reinterpret_cast<__nv_bfloat16*>(xn));
    ck(cudaGetLastError(), "rms_norm_quantize");
}

void swiglu(const void* w_gate, const void* w_up, const float* s_gate, const float* s_up, int64_t rows, int64_t K,
            const void* act, const float* gs, int64_t tokens, uint16_t* dst, int64_t ld, void* stream, float* amax) {
    check(w_gate, w_up, rows, K, tokens, act, dst, ld, "swiglu", true, gs);
    if (amax && cudaMemsetAsync(amax, 0, (size_t) tokens * sizeof(float), (cudaStream_t) stream) != cudaSuccess)
        ck(cudaGetLastError(), "swiglu amax");
    launch<true, false>(reinterpret_cast<const uint8_t*>(w_gate), reinterpret_cast<const uint8_t*>(w_up), s_gate, s_up,
                        rows, K, reinterpret_cast<const uint8_t*>(act), gs, tokens, dst, ld, amax, stream);
}

}  // namespace basalt::prefill::fp4_dense
