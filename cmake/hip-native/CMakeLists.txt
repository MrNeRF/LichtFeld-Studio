# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
cmake_minimum_required(VERSION 3.30)
project(LichtFeldHipNative LANGUAGES NONE)
# Small SDK-only build for cross-target compilation checks. No host application,
# CUDA toolkit, Torch, display server, or AMD GPU is needed to compile it.
get_filename_component(_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
# Developer correctness probe: force the physical wave on a compatible RDNA
# device. This override is deliberately absent from normal application builds.
set(LFS_HIP_TEST_WAVE_SIZE "" CACHE STRING "SDK-only physical-wave test override: empty, 32 or 64")
if(LFS_HIP_TEST_WAVE_SIZE STREQUAL "32")
    set(LFS_HIP_TEST_WAVE_FLAGS -mno-wavefrontsize64)
elseif(LFS_HIP_TEST_WAVE_SIZE STREQUAL "64")
    # rocPRIM selects its compile-time wave and DPP algorithm from the
    # architecture family, independently of -mwavefrontsize64. On RDNA this
    # test override must also select its generic wave64 implementation.
    # Normal application builds and native GCN/CDNA targets are unaffected.
    set(LFS_HIP_TEST_WAVE_FLAGS -mwavefrontsize64 -DROCPRIM_NAVI=0 -DROCPRIM_DISABLE_DPP=1
        -include "${_root}/tests/hip/forced_wave64_runtime.hpp")
    # Current rocPRIM host dispatch derives the wave from the GPU family even
    # after a physical-wave compiler override. Adapt a private header copy;
    # never patch the installed SDK or ship this forced-mode test adaptation.
    set(_test_sdk "${CMAKE_CURRENT_BINARY_DIR}/forced-wave64-sdk")
    set(_source_config "${LFS_HIP_ROOT}/include/rocprim/device/config_types.hpp")
    file(SHA256 "${_source_config}" _sdk_hash)
    set(_stamp "${_test_sdk}/source.sha256")
    set(_previous_hash "")
    if(EXISTS "${_stamp}")
        file(READ "${_stamp}" _previous_hash)
    endif()
    set(_config "${_test_sdk}/rocprim/device/config_types.hpp")
    if(NOT _sdk_hash STREQUAL _previous_hash OR NOT EXISTS "${_config}")
        file(COPY "${LFS_HIP_ROOT}/include/rocprim" DESTINATION "${_test_sdk}")
        file(READ "${_config}" _text)
        string(REPLACE "\r\n" "\n" _text "${_text}")
        set(_anchor "constexpr arch::wavefront::target get_wavefront_size(const gen gen = gen::unknown)\n{")
        string(FIND "${_text}" "${_anchor}" _position)
        if(_position EQUAL -1)
            message(FATAL_ERROR "This rocPRIM version needs a reviewed forced-wave64 test adaptation")
        endif()
        string(REPLACE "${_anchor}" "${_anchor}\n    return arch::wavefront::target::size64; // SDK-only forced-wave64 test." _text "${_text}")
        file(WRITE "${_config}" "${_text}")
        file(WRITE "${_stamp}" "${_sdk_hash}")
    endif()
    # Binning and other generated units also instantiate host rocPRIM calls.
    list(APPEND LFS_HIP_TEST_WAVE_FLAGS "-I${_test_sdk}")
    set(LFS_HIP_TEST_RUNTIME_DEPENDS "${_root}/tests/hip/forced_wave64_runtime.hpp" "${_config}")
elseif(NOT LFS_HIP_TEST_WAVE_SIZE STREQUAL "")
    message(FATAL_ERROR "LFS_HIP_TEST_WAVE_SIZE must be empty, 32 or 64")
endif()
set(LFS_TENSOR_HIP OFF)
add_subdirectory("${_root}/src/core/tensor/backend/hip" runtime)
add_subdirectory("${_root}/src/training/hip" training)
