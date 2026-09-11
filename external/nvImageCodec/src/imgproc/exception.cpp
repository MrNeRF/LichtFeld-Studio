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

#include "imgproc/exception.h"

namespace nvimgcodec {

    namespace StatusStrings {
        const std::string sNotValidFormat = "Not valid format";
        const std::string sUnsupportedFormat = "Unsupported format";
        const std::string sBadStream = "Corrupted stream";
        const std::string sStreamParse = "Stream parse error";
        const std::string sAllocationError = "Memory allocator error";
        const std::string sInternalError = "Internal error";
        const std::string sParameterError = "Error in the API call";
        const std::string sCUDAError = "Error in the CUDA API call";
        const std::string sArchMismatch = "Architecture mismatch";
    } // namespace StatusStrings

    const char* getErrorString(Status eStatus_) {
        switch (eStatus_) {
        case NOT_VALID_FORMAT_STATUS:
            return StatusStrings::sNotValidFormat.c_str();
        case UNSUPPORTED_FORMAT_STATUS:
            return StatusStrings::sUnsupportedFormat.c_str();
        case BAD_FORMAT_STATUS:
            return StatusStrings::sBadStream.c_str();
        case PARSE_STATUS:
            return StatusStrings::sStreamParse.c_str();
        case ALLOCATION_ERROR:
            return StatusStrings::sAllocationError.c_str();
        case INVALID_PARAMETER:
            return StatusStrings::sParameterError.c_str();
        case CUDA_CALL_ERROR:
            return StatusStrings::sCUDAError.c_str();
        case ARCH_MISMATCH:
            return StatusStrings::sArchMismatch.c_str();
        case INTERNAL_ERROR:
        default:
            return StatusStrings::sInternalError.c_str();
        }
    }

    Exception::Exception(Status eStatus, const std::string& rMessage, const std::string& rLoc)
        : eStatus_(eStatus), sMessage_(getErrorString(eStatus_) + (rMessage.empty() ? "" : " - " + rMessage)), sLocation_(rLoc) {
        ;
    }

    const char* Exception::what() const throw() {
        return sMessage_.c_str();
    };

    Status Exception::status() const {
        return eStatus_;
    }

    const char* Exception::where() const {
        return sLocation_.c_str();
    }

} // namespace nvimgcodec
