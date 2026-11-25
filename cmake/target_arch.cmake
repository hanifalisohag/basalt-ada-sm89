# cmake/target_arch.cmake - the two things a build is for: a CPU architecture and one GPU compute capability.
# Included right after project() by the root project and by the standalone tools/vision configuration.
#
#   BASALT_ARM64    ON when the target is Arm64: the CPU kernels are NEON. OFF on x86-64 (AVX2 / AVX-512).
#   BASALT_CUDA_CC  120 (discrete Blackwell) or 121 (the integrated Blackwell of the N1X and GB10 systems-on-chip).
#                   It follows CMAKE_CUDA_ARCHITECTURES, which defaults to 121 on Arm64 and 120 elsewhere. The two are
#                   independent: -DCMAKE_CUDA_ARCHITECTURES=120 on Arm64 builds for a discrete card in an Arm machine.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(ARM64|arm64|aarch64)$" OR "${CMAKE_CXX_COMPILER_ARCHITECTURE_ID}" MATCHES "^ARM64")
  set(BASALT_ARM64 ON)
else()
  set(BASALT_ARM64 OFF)
endif()

# _basalt_cuda_arch_default is set before project() when the caller named no architecture; compiler detection has
# since filled in the default of nvcc itself, which is replaced here.
if(_basalt_cuda_arch_default)
  if(BASALT_ARM64)
    set(CMAKE_CUDA_ARCHITECTURES 121 CACHE STRING "CUDA architectures" FORCE)
  else()
    set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures" FORCE)
  endif()
endif()

set(BASALT_CUDA_CC "")
foreach(_a IN LISTS CMAKE_CUDA_ARCHITECTURES)
  string(REGEX REPLACE "[-a].*$" "" _base "${_a}")     # 120-real / 121a are still 120 / 121
  if(NOT _base MATCHES "^(89|120|121)$" OR (BASALT_CUDA_CC AND NOT _base STREQUAL BASALT_CUDA_CC))
    message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES is '${CMAKE_CUDA_ARCHITECTURES}': basalt builds for one compute capability, 8.9 (Ada), 12.0 or 12.1 (peer-arch kernels are compiled separately)")
  endif()
  set(BASALT_CUDA_CC ${_base})
endforeach()
# the architecture of the targets that use the FP4 / block-scaled tensor-core instructions
if(BASALT_CUDA_CC STREQUAL "89")
  set(BASALT_CUDA_ARCH_A "89")     # Ada has no 'a' (accelerated) variant
else()
  set(BASALT_CUDA_ARCH_A "${BASALT_CUDA_CC}a-real")
endif()
add_compile_definitions(BASALT_CUDA_CC=${BASALT_CUDA_CC})
# BASALT_ADA (sm_89) is the portable build: the Blackwell-only prefill products (moe_fp4, fp4_dense,
# fp8_gemm) are left out and the models route their prefill through the portable MMQ path instead, so the
# engine runs on Ada (RTX 4090) and Ampere. The decode kernels and the model code are already portable.
if(BASALT_CUDA_CC STREQUAL "89")
  set(BASALT_ADA ON)
  add_compile_definitions(BASALT_ADA=1)
  message(STATUS "basalt: Ada (sm_89) build - the Blackwell-only prefill products are left out, prefill uses MMQ")
else()
  set(BASALT_ADA OFF)
endif()
if(BASALT_ARM64)
  include(${CMAKE_CURRENT_LIST_DIR}/arm64.cmake)
  basalt_arm64_global_flags()
  message(STATUS "basalt: Arm64, NEON CPU kernels; CUDA arch ${CMAKE_CUDA_ARCHITECTURES}")
else()
  message(STATUS "basalt: x86-64, AVX2 / AVX-512 CPU kernels; CUDA arch ${CMAKE_CUDA_ARCHITECTURES}")
endif()
