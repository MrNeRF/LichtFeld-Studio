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

#include <nvimgcodec.h>
#include "cuda_decoder.h"
#if NVJPEG_LOSSLESS_SUPPORTED
    #include "lossless_decoder.h"
#endif
#include "cuda_encoder.h"
#include "errors_handling.h"
#include "hw_encoder.h"
#include "hw_decoder.h"
#include "log.h"
#include "nvjpeg_utils.h"

namespace nvjpeg {

struct NvJpegImgCodecsExtension
{
  public:
    explicit NvJpegImgCodecsExtension(const nvimgcodecFrameworkDesc_t* framework)
        : framework_(framework)
#if NVJPEG_HW_ENCODER_SUPPORTED
        , jpeg_hw_encoder_(framework)
        , jpeg_hw_encoder_registered_(false)
#endif
        , jpeg_hw_decoder_(framework)
        , jpeg_hw_decoder_registered_(false)
        , jpeg_cuda_decoder_(framework)
        , jpeg_cuda_encoder_(framework)
        , jpeg_cuda_decoder_registered_(false)
        , jpeg_cuda_encoder_registered_(false)
#if NVJPEG_LOSSLESS_SUPPORTED
        , jpeg_lossless_decoder_(framework)
        , jpeg_lossless_decoder_registered_(false)
#endif
    {
        // Load nvjpeg library and check version at plugin registration time
        // Minimum nvJPEG version is based on CUDA version since nvJPEG ships with CUDA toolkit
#if CUDA_VERSION_MAJOR >= 13
        const NvjpegVersion MIN_NVJPEG_VERSION(13, 0, 0);
#else
        const NvjpegVersion MIN_NVJPEG_VERSION(12, 1, 0);
#endif

        NvjpegVersion version = get_nvjpeg_version();
        if (!version) {
            NVIMGCODEC_LOG_ERROR(framework, "nvjpeg-ext", 
                "Failed to load nvJPEG library or retrieve version. "
                "Decoders and encoders will not be registered.");
            return;
        }
        
        NVIMGCODEC_LOG_INFO(framework, "nvjpeg-ext", 
            "nvJPEG version: " << version.major_ver << "." << version.minor_ver << "." << version.patch_ver
            << " (CUDA version is " << CUDA_VERSION_MAJOR << "." << CUDA_VERSION_MINOR << ")");
        
        if (version < MIN_NVJPEG_VERSION) {
            NVIMGCODEC_LOG_WARNING(framework, "nvjpeg-ext", 
                "nvJPEG version " << version << " is older than minimum required version " << MIN_NVJPEG_VERSION 
                << " for CUDA " << CUDA_VERSION_MAJOR << ". Decoders and encoders will not be registered.");
            return;
        }
        
#if NVJPEG_HW_ENCODER_SUPPORTED
        if (jpeg_hw_encoder_.isPlatformSupported()) {
            framework->registerEncoder(framework->instance, jpeg_hw_encoder_.getEncoderDesc(), NVIMGCODEC_PRIORITY_VERY_HIGH);
            jpeg_hw_encoder_registered_ = true;
        } else {
            NVIMGCODEC_LOG_INFO(framework, "nvjpeg-ext", "HW encoder not supported by this platform. Skip.");
        }
#endif
        framework->registerEncoder(framework->instance, jpeg_cuda_encoder_.getEncoderDesc(), NVIMGCODEC_PRIORITY_HIGH);
        jpeg_cuda_encoder_registered_ = true;
        if (jpeg_hw_decoder_.isPlatformSupported()) {
            framework->registerDecoder(framework->instance, jpeg_hw_decoder_.getDecoderDesc(), NVIMGCODEC_PRIORITY_VERY_HIGH);
            jpeg_hw_decoder_registered_ = true;
        } else {
            NVIMGCODEC_LOG_INFO(framework, "nvjpeg-ext", "HW decoder not supported by this platform. Skip.");
        }
        framework->registerDecoder(framework->instance, jpeg_cuda_decoder_.getDecoderDesc(), NVIMGCODEC_PRIORITY_HIGH);
        jpeg_cuda_decoder_registered_ = true;
#if NVJPEG_LOSSLESS_SUPPORTED
        framework->registerDecoder(framework->instance, jpeg_lossless_decoder_.getDecoderDesc(), NVIMGCODEC_PRIORITY_HIGH);
        jpeg_lossless_decoder_registered_ = true;
#endif
    }
    ~NvJpegImgCodecsExtension()
    {
#if NVJPEG_HW_ENCODER_SUPPORTED
        if (jpeg_hw_encoder_registered_)
            framework_->unregisterEncoder(framework_->instance, jpeg_hw_encoder_.getEncoderDesc());
#endif
        if (jpeg_cuda_encoder_registered_)
            framework_->unregisterEncoder(framework_->instance, jpeg_cuda_encoder_.getEncoderDesc());
        if (jpeg_hw_decoder_registered_)
            framework_->unregisterDecoder(framework_->instance, jpeg_hw_decoder_.getDecoderDesc());
        if (jpeg_cuda_decoder_registered_)
            framework_->unregisterDecoder(framework_->instance, jpeg_cuda_decoder_.getDecoderDesc());
#if NVJPEG_LOSSLESS_SUPPORTED
        if (jpeg_lossless_decoder_registered_)
            framework_->unregisterDecoder(framework_->instance, jpeg_lossless_decoder_.getDecoderDesc());
#endif
    }

    static nvimgcodecStatus_t nvjpeg_extension_create(void* instance, nvimgcodecExtension_t* extension, const nvimgcodecFrameworkDesc_t* framework)
    {
        try {
            XM_CHECK_NULL(framework)
            NVIMGCODEC_LOG_TRACE(framework, "nvjpeg-ext", "nvjpeg_extension_create");
            XM_CHECK_NULL(extension)
            *extension = reinterpret_cast<nvimgcodecExtension_t>(new nvjpeg::NvJpegImgCodecsExtension(framework));
        } catch (const NvJpegException& e) {
            return e.nvimgcodecStatus();
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    static nvimgcodecStatus_t nvjpeg_extension_destroy(nvimgcodecExtension_t extension)
    {
        try {
            XM_CHECK_NULL(extension)
            auto ext_handle = reinterpret_cast<nvjpeg::NvJpegImgCodecsExtension*>(extension);
            NVIMGCODEC_LOG_TRACE(ext_handle->framework_, "nvjpeg_ext", "nvjpeg_extension_destroy");
            delete ext_handle;
        } catch (const NvJpegException& e) {
            return e.nvimgcodecStatus();
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    }

  private:
    const nvimgcodecFrameworkDesc_t* framework_;
#if NVJPEG_HW_ENCODER_SUPPORTED
    NvJpegHwEncoderPlugin jpeg_hw_encoder_;
    bool jpeg_hw_encoder_registered_;
#endif
    NvJpegHwDecoderPlugin jpeg_hw_decoder_;
    bool jpeg_hw_decoder_registered_;
    NvJpegCudaDecoderPlugin jpeg_cuda_decoder_;
    NvJpegCudaEncoderPlugin jpeg_cuda_encoder_;
    bool jpeg_cuda_decoder_registered_;
    bool jpeg_cuda_encoder_registered_;
#if NVJPEG_LOSSLESS_SUPPORTED
    NvJpegLosslessDecoderPlugin jpeg_lossless_decoder_;
    bool jpeg_lossless_decoder_registered_;
#endif
};
} // namespace nvjpeg

  // clang-format off
nvimgcodecExtensionDesc_t nvjpeg_extension = {
    NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC,
    sizeof(nvimgcodecExtensionDesc_t),
    NULL,

    NULL,
    "nvjpeg_extension",
    NVIMGCODEC_VER,    
    NVIMGCODEC_VER,

    nvjpeg::NvJpegImgCodecsExtension::nvjpeg_extension_create,
    nvjpeg::NvJpegImgCodecsExtension::nvjpeg_extension_destroy
};
// clang-format on  

nvimgcodecStatus_t get_nvjpeg_extension_desc(nvimgcodecExtensionDesc_t* ext_desc)
{
    if (ext_desc == nullptr) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    if (ext_desc->struct_type != NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    *ext_desc = nvjpeg_extension;
    return NVIMGCODEC_STATUS_SUCCESS;
}
