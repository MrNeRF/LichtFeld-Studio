/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#ifndef LFS_HAS_CUDA
#define LFS_HAS_CUDA 1
#endif

#if LFS_HAS_CUDA
#include <cuda_runtime.h>
#else
struct CUstream_st;
using cudaStream_t = CUstream_st*;
struct CUevent_st;
using cudaEvent_t = CUevent_st*;
enum cudaError : int;
using cudaError_t = cudaError;
#endif
