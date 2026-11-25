// SPDX-License-Identifier: MIT
// Explicit instantiation of the MMQ kernel for MXFP8 (Basalt, the MXFP4 instantiation adapted: the E4M3 codes are
// quantized to int8 per 32-value sub-block, the UE8M0 scale folded into the float the epilogue multiplies by).
// Copyright (c) 2023-2026 The ggml authors.  MIT licence: see THIRD_PARTY_LICENSES.
#include "../mmq.cuh"

DECL_MMQ_CASE(GGML_TYPE_MXFP8);
