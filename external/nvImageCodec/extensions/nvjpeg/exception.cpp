/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "exception.h"

const std::map<nvjpegStatus_t, nvimgcodecStatus_t> nvjpeg_status_to_nvimgcodec_error_map =
    {{NVJPEG_STATUS_SUCCESS, NVIMGCODEC_STATUS_SUCCESS},
     {NVJPEG_STATUS_NOT_INITIALIZED, NVIMGCODEC_STATUS_EXTENSION_NOT_INITIALIZED},
     {NVJPEG_STATUS_INVALID_PARAMETER, NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER},
     {NVJPEG_STATUS_INTERNAL_ERROR, NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR},
     {NVJPEG_STATUS_INCOMPLETE_BITSTREAM, NVIMGCODEC_STATUS_EXTENSION_INCOMPLETE_BITSTREAM},
     {NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED, NVIMGCODEC_STATUS_EXTENSION_IMPLEMENTATION_NOT_SUPPORTED},
     {NVJPEG_STATUS_EXECUTION_FAILED, NVIMGCODEC_STATUS_EXTENSION_EXECUTION_FAILED},
     {NVJPEG_STATUS_BAD_JPEG, NVIMGCODEC_STATUS_EXTENSION_BAD_CODE_STREAM},
     {NVJPEG_STATUS_ARCH_MISMATCH, NVIMGCODEC_STATUS_EXTENSION_ARCH_MISMATCH},
     {NVJPEG_STATUS_ALLOCATOR_FAILURE, NVIMGCODEC_STATUS_EXTENSION_ALLOCATOR_FAILURE},
     {NVJPEG_STATUS_JPEG_NOT_SUPPORTED, NVIMGCODEC_STATUS_EXTENSION_CODESTREAM_UNSUPPORTED}};

namespace StatusStrings {
    const std::string sExtNotInit = "not initialized";
    const std::string sExtInvalidParam = "Invalid parameter";
    const std::string sExtBadJpeg = "Bad jpeg";
    const std::string sExtJpegUnSupp = "Jpeg not supported";
    const std::string sExtAllocFail = "allocator failure";
    const std::string sExtArchMis = "arch mismatch";
    const std::string sExtIntErr = "internal error";
    const std::string sExtImplNA = "implementation not supported";
    const std::string sExtIncBits = "incomplete bitstream";
    const std::string sExtExeFailed = "execution failed";
    const std::string sExtCudaCallError = "cuda call error";
} // namespace StatusStrings

const char* getErrorString(nvjpegStatus_t eStatus_) {
    switch (eStatus_) {
    case NVJPEG_STATUS_ALLOCATOR_FAILURE:
        return StatusStrings::sExtAllocFail.c_str();
    case NVJPEG_STATUS_ARCH_MISMATCH:
        return StatusStrings::sExtArchMis.c_str();
    case NVJPEG_STATUS_BAD_JPEG:
        return StatusStrings::sExtBadJpeg.c_str();
    case NVJPEG_STATUS_JPEG_NOT_SUPPORTED:
        return StatusStrings::sExtJpegUnSupp.c_str();
    case NVJPEG_STATUS_EXECUTION_FAILED:
        return StatusStrings::sExtExeFailed.c_str();
    case NVJPEG_STATUS_INCOMPLETE_BITSTREAM:
        return StatusStrings::sExtIncBits.c_str();
    case NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED:
        return StatusStrings::sExtImplNA.c_str();
    case NVJPEG_STATUS_INTERNAL_ERROR:
        return StatusStrings::sExtIntErr.c_str();
    case NVJPEG_STATUS_INVALID_PARAMETER:
        return StatusStrings::sExtInvalidParam.c_str();
    case NVJPEG_STATUS_NOT_INITIALIZED:
        return StatusStrings::sExtNotInit.c_str();
    default:
        return StatusStrings::sExtIntErr.c_str();
    }
}

NvJpegException NvJpegException::FromNvJpegError(nvjpegStatus_t status, const std::string& where) {
    NvJpegException e;
    e.status_ = NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    auto it = nvjpeg_status_to_nvimgcodec_error_map.find(status);
    if (it != nvjpeg_status_to_nvimgcodec_error_map.end())
        e.status_ = it->second;
    std::stringstream ss;
    ss << "nvjpeg error #" << static_cast<int>(status) << " (" << getErrorString(status) << ")" << " when running " << where;
    e.info_ = ss.str();
    return e;
}

NvJpegException NvJpegException::FromCUDAError(cudaError_t status, const std::string& where) {
    NvJpegException e;
    e.status_ = NVIMGCODEC_STATUS_EXTENSION_CUDA_CALL_ERROR;
    std::stringstream ss;
    ss << "CUDA error #" << static_cast<int>(status) << " when running " << where;
    e.info_ = ss.str();
    return e;
}
