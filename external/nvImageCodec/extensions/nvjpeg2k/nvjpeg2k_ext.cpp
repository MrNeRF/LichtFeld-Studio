/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <nvjpeg2k.h>
#include "cuda_decoder.h"
#include "cuda_encoder.h"
#include "error_handling.h"
#include "log.h"
#include "nvjpeg2k_utils.h"

namespace nvjpeg2k {

struct NvJpeg2kImgCodecsExtension
{
  public:
    explicit NvJpeg2kImgCodecsExtension(const nvimgcodecFrameworkDesc_t* framework)
        : framework_(framework)
        , jpeg2k_decoder_(framework)
        , jpeg2k_encoder_(framework)
        , decoder_registered_(false)
        , encoder_registered_(false)
    {
        // Load nvjpeg2k library and check version at plugin registration time
        // This ensures the DLL is loaded early and version requirements are met
        // Minimum nvJPEG2000 0.8.0 is required
        // Minimum nvJPEG2000 0.9.0 is required for encoder API compatibility (HT encoder, int16 encoding, quality and quantization step)
        const Nvjpeg2kVersion MIN_NVJPEG2K_VERSION(0, 8, 0);
        Nvjpeg2kVersion version = get_nvjpeg2k_version();
        if (!version) {
            NVIMGCODEC_LOG_ERROR(framework, "nvjpeg2k_ext", 
                "Failed to load nvJPEG2000 library or retrieve version. "
                "Decoders and encoders will not be registered.");
            return;
        }
        
        NVIMGCODEC_LOG_INFO(framework, "nvjpeg2k_ext", 
            "nvJPEG2000 version: " << version.major_ver << "." << version.minor_ver << "." << version.patch_ver);
        
        if (version < MIN_NVJPEG2K_VERSION) {
            NVIMGCODEC_LOG_WARNING(framework, "nvjpeg2k_ext", 
                "nvJPEG2000 version " << version << " is older than minimum required version " << MIN_NVJPEG2K_VERSION 
                << ". Decoders and encoders will not be registered.");
            return;
        }
        
        framework->registerEncoder(framework->instance, jpeg2k_encoder_.getEncoderDesc(), NVIMGCODEC_PRIORITY_HIGH);
        encoder_registered_ = true;
        framework->registerDecoder(framework->instance, jpeg2k_decoder_.getDecoderDesc(), NVIMGCODEC_PRIORITY_HIGH);
        decoder_registered_ = true;
    }
    ~NvJpeg2kImgCodecsExtension()
    {
        if (encoder_registered_)
            framework_->unregisterEncoder(framework_->instance, jpeg2k_encoder_.getEncoderDesc());
        if (decoder_registered_)
            framework_->unregisterDecoder(framework_->instance, jpeg2k_decoder_.getDecoderDesc());
    }

    static nvimgcodecStatus_t nvjpeg2k_extension_create(
        void* instance, nvimgcodecExtension_t* extension, const nvimgcodecFrameworkDesc_t* framework)
    {
        try {
            XM_CHECK_NULL(framework)
            NVIMGCODEC_LOG_TRACE(framework, "nvjpeg2k_ext", "nvjpeg2k_extension_create");
            XM_CHECK_NULL(extension)
            *extension = reinterpret_cast<nvimgcodecExtension_t>(new nvjpeg2k::NvJpeg2kImgCodecsExtension(framework));
        } catch (const NvJpeg2kException& e) {
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    static nvimgcodecStatus_t nvjpeg2k_extension_destroy(nvimgcodecExtension_t extension)
    {
        try {
            XM_CHECK_NULL(extension)
            auto ext_handle = reinterpret_cast<nvjpeg2k::NvJpeg2kImgCodecsExtension*>(extension);
            NVIMGCODEC_LOG_TRACE(ext_handle->framework_, "nvjpeg2k_ext", "nvjpeg2k_extension_destroy");
            delete ext_handle;
        } catch (const NvJpeg2kException& e) {
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    }

  private:
    const nvimgcodecFrameworkDesc_t* framework_;
    NvJpeg2kDecoderPlugin jpeg2k_decoder_;
    NvJpeg2kEncoderPlugin jpeg2k_encoder_;
    bool decoder_registered_;
    bool encoder_registered_;
};

} // namespace nvjpeg2k

// clang-format off
nvimgcodecExtensionDesc_t nvjpeg2k_extension = {
    NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC,
    sizeof(nvimgcodecExtensionDesc_t),
    NULL,

    NULL,
    "nvjpeg2k_extension",  
    NVIMGCODEC_VER,           
    NVIMGCODEC_VER,
    
    nvjpeg2k::NvJpeg2kImgCodecsExtension::nvjpeg2k_extension_create,
    nvjpeg2k::NvJpeg2kImgCodecsExtension::nvjpeg2k_extension_destroy
};
// clang-format on

nvimgcodecStatus_t get_nvjpeg2k_extension_desc(nvimgcodecExtensionDesc_t* ext_desc)
{
    if (ext_desc == nullptr) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    if (ext_desc->struct_type != NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    *ext_desc = nvjpeg2k_extension;
    return NVIMGCODEC_STATUS_SUCCESS;
}
