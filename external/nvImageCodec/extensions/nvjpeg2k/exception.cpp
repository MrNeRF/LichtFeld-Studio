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

const std::map<nvjpeg2kStatus_t, nvimgcodecStatus_t> nvjpeg2k_to_nvimgcodec_error_map = 
    {{NVJPEG2K_STATUS_SUCCESS, NVIMGCODEC_STATUS_SUCCESS},    
    {NVJPEG2K_STATUS_NOT_INITIALIZED, NVIMGCODEC_STATUS_EXTENSION_NOT_INITIALIZED},
    {NVJPEG2K_STATUS_INVALID_PARAMETER, NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER},
    {NVJPEG2K_STATUS_INTERNAL_ERROR, NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR},    
    {NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED, NVIMGCODEC_STATUS_EXTENSION_IMPLEMENTATION_NOT_SUPPORTED},
    {NVJPEG2K_STATUS_EXECUTION_FAILED, NVIMGCODEC_STATUS_EXTENSION_EXECUTION_FAILED},
    {NVJPEG2K_STATUS_BAD_JPEG, NVIMGCODEC_STATUS_EXTENSION_BAD_CODE_STREAM},    
    {NVJPEG2K_STATUS_ALLOCATOR_FAILURE, NVIMGCODEC_STATUS_EXTENSION_ALLOCATOR_FAILURE},
    {NVJPEG2K_STATUS_JPEG_NOT_SUPPORTED, NVIMGCODEC_STATUS_EXTENSION_CODESTREAM_UNSUPPORTED}};

namespace StatusStrings {
const std::string sExtNotInit        = "not initialized";
const std::string sExtInvalidParam   = "Invalid parameter";
const std::string sExtBadJpeg        = "Bad jpeg";
const std::string sExtJpegUnSupp     = "Jpeg not supported";
const std::string sExtAllocFail      = "allocator failure";
const std::string sExtIntErr         = "internal error";
const std::string sExtImplNA         = "implementation not supported";
const std::string sExtExeFailed      = "execution failed";
const std::string sExtCudaCallError  = "cuda call error";
} // namespace StatusStrings

const char* getErrorString(nvjpeg2kStatus_t eStatus_)
{
    switch (eStatus_) {        
    case NVJPEG2K_STATUS_ALLOCATOR_FAILURE:
        return StatusStrings::sExtAllocFail.c_str();
    case NVJPEG2K_STATUS_BAD_JPEG:
        return StatusStrings::sExtBadJpeg.c_str();      
    case NVJPEG2K_STATUS_JPEG_NOT_SUPPORTED:
        return StatusStrings::sExtJpegUnSupp.c_str();
    case NVJPEG2K_STATUS_EXECUTION_FAILED:
        return StatusStrings::sExtExeFailed.c_str();
    case NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED:
        return StatusStrings::sExtImplNA.c_str();
    case NVJPEG2K_STATUS_INTERNAL_ERROR:
        return StatusStrings::sExtIntErr.c_str();
    case NVJPEG2K_STATUS_INVALID_PARAMETER:
        return StatusStrings::sExtInvalidParam.c_str();
    case NVJPEG2K_STATUS_NOT_INITIALIZED:
        return StatusStrings::sExtNotInit.c_str();
    default:
        return StatusStrings::sExtIntErr.c_str();

    }
}

NvJpeg2kException NvJpeg2kException::FromNvJpeg2kError(nvjpeg2kStatus_t status, const std::string& where)
{
    auto nvimgcodec_status = NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    auto it = nvjpeg2k_to_nvimgcodec_error_map.find(status);
    if (it != nvjpeg2k_to_nvimgcodec_error_map.end()) {
        nvimgcodec_status = it->second;
    }

    std::stringstream ss;
    ss << "nvjpeg2k error #" << static_cast<int>(status) << " (" << getErrorString(status) << ")" << " when running " << where;

    return {nvimgcodec_status, ss.str()};
}

NvJpeg2kException NvJpeg2kException::FromCUDAError(cudaError_t status, const std::string& where)
{
    std::stringstream ss;
    ss << "CUDA error #" << static_cast<int>(status) << " when running " << where;
    return {NVIMGCODEC_STATUS_EXTENSION_CUDA_CALL_ERROR, ss.str()};
}


