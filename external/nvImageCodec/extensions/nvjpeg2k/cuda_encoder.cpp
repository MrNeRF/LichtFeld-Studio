/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#define NOMINMAX
#include "cuda_encoder.h"
#include "error_handling.h"
#include "imgproc/device_buffer.h"
#include "imgproc/image_info_checks.h"
#include "imgproc/pinned_buffer.h"
#include "log.h"
#include "nvimgcodec.h"
#include "nvimgcodec_type_utils.h"
#include "nvjpeg2k_utils.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <future>
#include <iostream>
#include <nvjpeg2k.h>
#include <nvjpeg2k_version.h>
#include <nvtx3/nvtx3.hpp>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using nvimgcodec::PinnedBuffer;

namespace nvjpeg2k {

    struct EncoderImpl {
        EncoderImpl(
            const char* id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params, const char* options);
        ~EncoderImpl();

        static nvimgcodecStatus_t static_destroy(nvimgcodecEncoder_t encoder);

        nvimgcodecProcessingStatus_t canEncode(const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecImageDesc_t* image,
                                               const nvimgcodecEncodeParams_t* params, int thread_idx);
        static nvimgcodecProcessingStatus_t static_can_encode(nvimgcodecEncoder_t encoder, const nvimgcodecCodeStreamDesc_t* code_stream,
                                                              const nvimgcodecImageDesc_t* image, const nvimgcodecEncodeParams_t* params, int thread_idx) {
            try {
                XM_CHECK_NULL(encoder);
                auto handle = reinterpret_cast<EncoderImpl*>(encoder);
                return handle->canEncode(code_stream, image, params, thread_idx);
            } catch (const std::runtime_error& e) {
                return NVIMGCODEC_PROCESSING_STATUS_FAIL;
            }
        }

        nvimgcodecStatus_t encode(const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecImageDesc_t* image,
                                  const nvimgcodecEncodeParams_t* params, int thread_idx);
        static nvimgcodecStatus_t static_encode_sample(nvimgcodecEncoder_t encoder, const nvimgcodecCodeStreamDesc_t* code_stream,
                                                       const nvimgcodecImageDesc_t* image, const nvimgcodecEncodeParams_t* params, int thread_idx) {
            try {
                XM_CHECK_NULL(encoder);
                auto handle = reinterpret_cast<EncoderImpl*>(encoder);
                return handle->encode(code_stream, image, params, thread_idx);
            } catch (const std::runtime_error& e) {
                return NVIMGCODEC_STATUS_EXECUTION_FAILED;
            }
        }

        const char* plugin_id_;
        nvjpeg2kEncoder_t handle_;
        const nvimgcodecFrameworkDesc_t* framework_;

        nvimgcodecDeviceAllocator_t* device_allocator_ = nullptr;
        nvimgcodecPinnedAllocator_t* pinned_allocator_ = nullptr;
        int device_id_;

        struct PerThreadResources {
            const nvimgcodecFrameworkDesc_t* framework_;
            const char* plugin_id_;
            nvjpeg2kEncoder_t handle_;
            const nvimgcodecExecutionParams_t* exec_params_;

            PinnedBuffer pinned_buffer_;
            cudaEvent_t event_;
            nvjpeg2kEncodeState_t state_;

            PerThreadResources(const nvimgcodecFrameworkDesc_t* framework, const char* plugin_id, nvjpeg2kEncoder_t handle,
                               const nvimgcodecExecutionParams_t* exec_params)
                : framework_(framework), plugin_id_(plugin_id), handle_(handle), exec_params_(exec_params), pinned_buffer_(exec_params ? exec_params->pinned_allocator : nullptr), event_(nullptr), state_(nullptr) {
                XM_CHECK_CUDA(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
                XM_CHECK_NVJPEG2K(nvjpeg2kEncodeStateCreate(handle_, &state_));
            }

            ~PerThreadResources() {
                if (state_) {
                    XM_NVJPEG2K_LOG_DESTROY(nvjpeg2kEncodeStateDestroy(state_));
                }
                if (event_) {
                    XM_CUDA_LOG_DESTROY(cudaEventDestroy(event_));
                }
            }
        };

        struct Sample {
            nvimgcodecCodeStreamDesc_t* code_stream;
            nvimgcodecImageDesc_t* image;
            const nvimgcodecEncodeParams_t* params;
        };

        std::vector<PerThreadResources> per_thread_;
        std::vector<Sample> samples_;

        const nvimgcodecExecutionParams_t* exec_params_;
        std::string options_;
        Nvjpeg2kVersion nvjpeg2k_version_;
    };

    NvJpeg2kEncoderPlugin::NvJpeg2kEncoderPlugin(const nvimgcodecFrameworkDesc_t* framework)
        : encoder_desc_{NVIMGCODEC_STRUCTURE_TYPE_ENCODER_DESC, sizeof(nvimgcodecEncoderDesc_t), NULL, this, plugin_id_, "jpeg2k",
                        NVIMGCODEC_BACKEND_KIND_GPU_ONLY, static_create, EncoderImpl::static_destroy, EncoderImpl::static_can_encode,
                        EncoderImpl::static_encode_sample},
          framework_(framework) {
    }

    nvimgcodecEncoderDesc_t* NvJpeg2kEncoderPlugin::getEncoderDesc() {
        return &encoder_desc_;
    }

    nvimgcodecProcessingStatus_t EncoderImpl::canEncode(const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecImageDesc_t* image,
                                                        const nvimgcodecEncodeParams_t* params, int thread_idx) {
        nvimgcodecProcessingStatus_t status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
        try {
            NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvjpeg2k_can_encode");
            XM_CHECK_NULL(status);
            XM_CHECK_NULL(code_stream);
            XM_CHECK_NULL(params);

            XM_CHECK_NULL(image);
            nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
            auto ret = image->getImageInfo(image->instance, &image_info);
            if (ret != NVIMGCODEC_STATUS_SUCCESS) {
                return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
            }

            nvimgcodecImageInfo_t out_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
            ret = code_stream->getImageInfo(code_stream->instance, &out_image_info);
            if (ret != NVIMGCODEC_STATUS_SUCCESS) {
                return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
            }

            if (strcmp(out_image_info.codec_name, "jpeg2k") != 0) {
                return NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
            }

            nvimgcodecJpeg2kEncodeParams_t* j2k_encode_params = static_cast<nvimgcodecJpeg2kEncodeParams_t*>(params->struct_next);
            while (j2k_encode_params && j2k_encode_params->struct_type != NVIMGCODEC_STRUCTURE_TYPE_JPEG2K_ENCODE_PARAMS)
                j2k_encode_params = static_cast<nvimgcodecJpeg2kEncodeParams_t*>(j2k_encode_params->struct_next);
            if (j2k_encode_params) {
                if ((j2k_encode_params->code_block_w != 32 || j2k_encode_params->code_block_h != 32) &&
                    (j2k_encode_params->code_block_w != 64 || j2k_encode_params->code_block_h != 64)) {
                    status = NVIMGCODEC_PROCESSING_STATUS_ENCODING_UNSUPPORTED;
                    NVIMGCODEC_LOG_WARNING(framework_, plugin_id_,
                                           "Unsupported block size: " << j2k_encode_params->code_block_w << "x" << j2k_encode_params->code_block_h
                                                                      << "(Valid values: 32, 64)");
                }
                if (j2k_encode_params->num_resolutions > NVJPEG2K_MAXRES) {
                    NVIMGCODEC_LOG_WARNING(framework_, plugin_id_,
                                           "Unsupported number of resolutions: " << j2k_encode_params->num_resolutions << " (max = " << NVJPEG2K_MAXRES << ") ");
                    status = NVIMGCODEC_PROCESSING_STATUS_ENCODING_UNSUPPORTED;
                }
            }

            nvimgcodecTileGeometryInfo_t* tile_geometry = static_cast<nvimgcodecTileGeometryInfo_t*>(params->struct_next);
            while (tile_geometry && tile_geometry->struct_type != NVIMGCODEC_STRUCTURE_TYPE_TILE_GEOMETRY_INFO)
                tile_geometry = static_cast<nvimgcodecTileGeometryInfo_t*>(tile_geometry->struct_next);
            if (tile_geometry) {
                if (tile_geometry->num_tiles_x != 0 || tile_geometry->num_tiles_y != 0) {
                    NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "Tiling by number of tiles is not supported with this version of nvjpeg2k, please define tile width and height instead.");
                    status |= NVIMGCODEC_PROCESSING_STATUS_TILING_UNSUPPORTED;
                }
                if (tile_geometry->tile_width > image_info.plane_info[0].width || tile_geometry->tile_height > image_info.plane_info[0].height) {
                    NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Tile width and height must be smaller than image width and height correspondingly.");
                    status |= NVIMGCODEC_PROCESSING_STATUS_TILING_UNSUPPORTED;
                }
                if (tile_geometry->tile_width > 0 && tile_geometry->tile_height > 0) {
                    // Validate tile parameters against nvjpeg2k 65535 tile limit
                    size_t num_tiles_x = (image_info.plane_info[0].width + tile_geometry->tile_width - 1) / tile_geometry->tile_width;
                    size_t num_tiles_y = (image_info.plane_info[0].height + tile_geometry->tile_height - 1) / tile_geometry->tile_height;
                    size_t total_tiles = num_tiles_x * num_tiles_y;

                    if (total_tiles > 65535) {
                        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_,
                                             "Tile dimensions " << tile_geometry->tile_width << "x" << tile_geometry->tile_height
                                                                << " would result in " << total_tiles << " tiles, which exceeds the nvjpeg2k limit of 65535 tiles");
                        status |= NVIMGCODEC_PROCESSING_STATUS_TILING_UNSUPPORTED;
                    }
                }
                if (tile_geometry->tile_offset_x > 0 || tile_geometry->tile_offset_y > 0) {
                    NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Non-zero tile offsets are not supported");
                    status |= NVIMGCODEC_PROCESSING_STATUS_TILING_UNSUPPORTED;
                }
            }

            static const std::set<nvimgcodecColorSpec_t> supported_color_space{
                NVIMGCODEC_COLORSPEC_UNKNOWN, NVIMGCODEC_COLORSPEC_SRGB, NVIMGCODEC_COLORSPEC_GRAY, NVIMGCODEC_COLORSPEC_SYCC};
            if (supported_color_space.find(image_info.color_spec) == supported_color_space.end()) {
                NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "Unsupported color spec: " << image_info.color_spec);
                status |= NVIMGCODEC_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED;
            }
            static const std::set<nvimgcodecChromaSubsampling_t> supported_css{
                NVIMGCODEC_SAMPLING_444, NVIMGCODEC_SAMPLING_422, NVIMGCODEC_SAMPLING_420, NVIMGCODEC_SAMPLING_GRAY};
            if (supported_css.find(image_info.chroma_subsampling) == supported_css.end()) {
                NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "Unsupported chroma subsampling: " << image_info.chroma_subsampling);
                status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
            }
            if (out_image_info.chroma_subsampling != image_info.chroma_subsampling) {
                NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "Output chroma subsampling (" << out_image_info.chroma_subsampling << ") is different from input chroma subsampling (" << image_info.chroma_subsampling << ")");
                status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
            }

            static const std::set<nvimgcodecSampleFormat_t> supported_sample_format{
                NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED,
                NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED,
                NVIMGCODEC_SAMPLEFORMAT_P_RGB,
                NVIMGCODEC_SAMPLEFORMAT_I_RGB,
                NVIMGCODEC_SAMPLEFORMAT_P_Y,
                NVIMGCODEC_SAMPLEFORMAT_I_Y,
                NVIMGCODEC_SAMPLEFORMAT_P_YUV,
                NVIMGCODEC_SAMPLEFORMAT_I_YUV,
                NVIMGCODEC_SAMPLEFORMAT_P_RGBA,
                NVIMGCODEC_SAMPLEFORMAT_I_RGBA,
            };
            if (supported_sample_format.find(image_info.sample_format) == supported_sample_format.end()) {
                NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "Unsupported sample format: " << image_info.sample_format);
                status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
            }

            if (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_Y) {
                if ((image_info.chroma_subsampling != NVIMGCODEC_SAMPLING_GRAY) ||
                    (out_image_info.chroma_subsampling != NVIMGCODEC_SAMPLING_GRAY)) {
                    NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "Unsupported chroma subsampling " << image_info.chroma_subsampling << "for sample format P_Y - only GRAY chroma subsampling is allowed.");
                    status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
                    status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
                }
                if (image_info.color_spec != NVIMGCODEC_COLORSPEC_GRAY) {
                    NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "Unsupported color spec " << image_info.color_spec << "for sample format P_Y - only GRAY color spec is allowed.");
                    status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
                    status |= NVIMGCODEC_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED;
                }
            }

            static const std::set<nvimgcodecSampleDataType_t> supported_sample_type{
                NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8, NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16, NVIMGCODEC_SAMPLE_DATA_TYPE_INT16};
            for (uint32_t p = 0; p < image_info.num_planes; ++p) {
                auto sample_type = image_info.plane_info[p].sample_type;
                if (supported_sample_type.find(sample_type) == supported_sample_type.end()) {
                    status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
                    continue;
                }
                // 0 means "use full bitdepth"; otherwise must fit in the sample type.
                uint8_t precision = image_info.plane_info[p].precision;
                uint8_t dtype_bitdepth = static_cast<uint8_t>(sample_type_to_bytes_per_element(sample_type) * 8);
                if (precision != 0 && precision > dtype_bitdepth) {
                    NVIMGCODEC_LOG_WARNING(framework_, plugin_id_,
                                           "plane_info[" << p << "].precision=" << static_cast<int>(precision)
                                                         << " exceeds the bitdepth of the sample type ("
                                                         << static_cast<int>(dtype_bitdepth) << " bits).");
                    status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
                }
            }
            if (!nvimgcodec::check_planes_consistency(framework_, plugin_id_, image_info)) {
                status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
            }

            static const std::set<nvimgcodecQualityType_t> supported_quality_types{
                NVIMGCODEC_QUALITY_TYPE_DEFAULT,
                NVIMGCODEC_QUALITY_TYPE_LOSSLESS,
                NVIMGCODEC_QUALITY_TYPE_QUALITY,
                NVIMGCODEC_QUALITY_TYPE_QUANTIZATION_STEP,
                NVIMGCODEC_QUALITY_TYPE_PSNR};
            if (supported_quality_types.find(params->quality_type) == supported_quality_types.end()) {
                status |= NVIMGCODEC_PROCESSING_STATUS_QUALITY_TYPE_UNSUPPORTED;
            }

            if (params->quality_type == NVIMGCODEC_QUALITY_TYPE_QUALITY) {
                if (params->quality_value < 1 || params->quality_value > 100) {
                    status |= NVIMGCODEC_PROCESSING_STATUS_QUALITY_VALUE_UNSUPPORTED;
                }
                if (image_info.color_spec == NVIMGCODEC_COLORSPEC_SRGB && j2k_encode_params->mct_mode != 1) {
                    NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "Quality cannot be used if input color_spec is SRGB and mct_mode is not 1");
                    status |= NVIMGCODEC_PROCESSING_STATUS_QUALITY_TYPE_UNSUPPORTED;
                }
            } else if (params->quality_type == NVIMGCODEC_QUALITY_TYPE_QUANTIZATION_STEP) {
                if (params->quality_value < 0) {
                    status |= NVIMGCODEC_PROCESSING_STATUS_QUALITY_VALUE_UNSUPPORTED;
                }
            } else if (params->quality_type == NVIMGCODEC_QUALITY_TYPE_PSNR) {
                if (params->quality_value < 0) {
                    status |= NVIMGCODEC_PROCESSING_STATUS_QUALITY_VALUE_UNSUPPORTED;
                }

                if (j2k_encode_params->ht) {
                    NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "PSNR cannot be used with HT encoder.");
                    status |= NVIMGCODEC_PROCESSING_STATUS_QUALITY_TYPE_UNSUPPORTED;
                }
            }

            if (j2k_encode_params->mct_mode == 1) {
                if (!(image_info.color_spec == NVIMGCODEC_COLORSPEC_SRGB && out_image_info.color_spec == NVIMGCODEC_COLORSPEC_SRGB)) {
                    NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "mct_mode=1 is only supported when both input and output color_spec are SRGB.");
                    status |= NVIMGCODEC_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED;
                }
            }

        } catch (const NvJpeg2kException& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if nvjpeg2k can encode - " << e.info());
            return NVIMGCODEC_PROCESSING_STATUS_FAIL;
        }
        return status;
    }

    EncoderImpl::EncoderImpl(
        const char* id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params, const char* options)
        : plugin_id_(id), framework_(framework), device_id_(0), exec_params_(exec_params), options_(options), nvjpeg2k_version_(get_nvjpeg2k_version()) {
        if (!nvjpeg2k_version_) {
            throw NvJpeg2kException(NVIMGCODEC_STATUS_EXTENSION_EXECUTION_FAILED, "Failed to get nvJPEG2000 version");
        }
        if (nvjpeg2k_version_ < Nvjpeg2kVersion(0, 9, 0)) {
            throw NvJpeg2kException(NVIMGCODEC_STATUS_EXTENSION_IMPLEMENTATION_NOT_SUPPORTED, "Encoding is supported only with nvJPEG2000 v0.9.0 or newer.");
        }

        XM_CHECK_NVJPEG2K(nvjpeg2kEncoderCreateSimple(&handle_));

        auto executor = exec_params->executor;
        int num_threads = executor->getNumThreads(executor->instance);

        per_thread_.reserve(num_threads);
        while (per_thread_.size() < static_cast<size_t>(num_threads)) {
            per_thread_.emplace_back(framework_, plugin_id_, handle_, exec_params_);
        }
    }

    nvimgcodecStatus_t NvJpeg2kEncoderPlugin::create(
        nvimgcodecEncoder_t* encoder, const nvimgcodecExecutionParams_t* exec_params, const char* options) {
        try {
            NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvjpeg2k_create_encoder");
            XM_CHECK_NULL(encoder);
            XM_CHECK_NULL(exec_params);
            if (exec_params->device_id == NVIMGCODEC_DEVICE_CPU_ONLY)
                return NVIMGCODEC_STATUS_INVALID_PARAMETER;

            *encoder = reinterpret_cast<nvimgcodecEncoder_t>(new EncoderImpl(plugin_id_, framework_, exec_params, options));
            return NVIMGCODEC_STATUS_SUCCESS;
        } catch (const NvJpeg2kException& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not create nvjpeg2k encoder - " << e.info());
            return e.nvimgcodecStatus();
        }
    }

    nvimgcodecStatus_t NvJpeg2kEncoderPlugin::static_create(
        void* instance, nvimgcodecEncoder_t* encoder, const nvimgcodecExecutionParams_t* exec_params, const char* options) {
        if (!instance) {
            return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
        }

        auto handle = reinterpret_cast<NvJpeg2kEncoderPlugin*>(instance);
        return handle->create(encoder, exec_params, options);
    }

    EncoderImpl::~EncoderImpl() {
        try {
            NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvjpeg2k_destroy_encoder");
            per_thread_.clear();
            if (handle_)
                XM_CHECK_NVJPEG2K(nvjpeg2kEncoderDestroy(handle_));
        } catch (const NvJpeg2kException& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not properly destroy nvjpeg2k encoder - " << e.info());
        }
    }

    nvimgcodecStatus_t EncoderImpl::static_destroy(nvimgcodecEncoder_t encoder) {
        try {
            XM_CHECK_NULL(encoder);
            auto handle = reinterpret_cast<EncoderImpl*>(encoder);
            delete handle;
        } catch (const NvJpeg2kException& e) {
            return e.nvimgcodecStatus();
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    static void fill_encode_config(
        nvjpeg2kEncodeConfig_t& encode_config, const nvimgcodecEncodeParams_t* params, uint32_t height, uint32_t width) {
        nvimgcodecJpeg2kEncodeParams_t* j2k_encode_params = static_cast<nvimgcodecJpeg2kEncodeParams_t*>(params->struct_next);
        while (j2k_encode_params && j2k_encode_params->struct_type != NVIMGCODEC_STRUCTURE_TYPE_JPEG2K_ENCODE_PARAMS)
            j2k_encode_params = static_cast<nvimgcodecJpeg2kEncodeParams_t*>(j2k_encode_params->struct_next);
        if (j2k_encode_params) {
            encode_config.stream_type = static_cast<nvjpeg2kBitstreamType>(j2k_encode_params->stream_type);
            encode_config.code_block_w = j2k_encode_params->code_block_w;
            encode_config.code_block_h = j2k_encode_params->code_block_h;
            encode_config.prog_order = static_cast<nvjpeg2kProgOrder>(j2k_encode_params->prog_order);
            encode_config.num_resolutions = j2k_encode_params->num_resolutions;
            encode_config.mct_mode = j2k_encode_params->mct_mode;
            if (j2k_encode_params->ht) {
                encode_config.rsiz = 0x4000;
                encode_config.encode_modes = 64;
            }
        }
        uint32_t max_num_resolutions = static_cast<uint32_t>(log2(static_cast<float>(std::min(height, width)))) + 1;
        encode_config.num_resolutions = std::min(encode_config.num_resolutions, max_num_resolutions);

        nvimgcodecTileGeometryInfo_t* tile_geometry = static_cast<nvimgcodecTileGeometryInfo_t*>(params->struct_next);
        while (tile_geometry && tile_geometry->struct_type != NVIMGCODEC_STRUCTURE_TYPE_TILE_GEOMETRY_INFO)
            tile_geometry = static_cast<nvimgcodecTileGeometryInfo_t*>(tile_geometry->struct_next);
        if (tile_geometry) {
            if (tile_geometry->tile_width > 0 && tile_geometry->tile_height > 0) {
                encode_config.enable_tiling = 1;
                encode_config.tile_width = tile_geometry->tile_width;
                encode_config.tile_height = tile_geometry->tile_height;
            }
        }
    }

    static nvjpeg2kColorSpace_t nvimgcodec_to_nvjpeg2k_color_spec(nvimgcodecColorSpec_t nvimgcodec_color_spec) {
        switch (nvimgcodec_color_spec) {
        case NVIMGCODEC_COLORSPEC_UNKNOWN:
            return NVJPEG2K_COLORSPACE_UNKNOWN;
        case NVIMGCODEC_COLORSPEC_SRGB:
            return NVJPEG2K_COLORSPACE_SRGB;
        case NVIMGCODEC_COLORSPEC_GRAY:
            return NVJPEG2K_COLORSPACE_GRAY;
        case NVIMGCODEC_COLORSPEC_SYCC:
            return NVJPEG2K_COLORSPACE_SYCC;
        case NVIMGCODEC_COLORSPEC_CMYK:
            return NVJPEG2K_COLORSPACE_NOT_SUPPORTED;
        case NVIMGCODEC_COLORSPEC_YCCK:
            return NVJPEG2K_COLORSPACE_NOT_SUPPORTED;
        case NVIMGCODEC_COLORSPEC_UNSUPPORTED:
            return NVJPEG2K_COLORSPACE_NOT_SUPPORTED;
        default:
            return NVJPEG2K_COLORSPACE_UNKNOWN;
        }
    }

    static nvjpeg2kImageType_t to_nvjpeg2k_sample_type(nvimgcodecSampleDataType_t sample_type) {
        switch (sample_type) {
        case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8:
            return NVJPEG2K_UINT8;
        case NVIMGCODEC_SAMPLE_DATA_TYPE_INT16:
            return NVJPEG2K_INT16;
        case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16:
            return NVJPEG2K_UINT16;
        default:
            throw NvJpeg2kException::FromNvJpeg2kError(NVJPEG2K_STATUS_INVALID_PARAMETER, "data type check");
        }
    }

    static bool is_buffer_pinned(const void* buffer) {
        if (buffer == nullptr) {
            return false;
        }

        cudaPointerAttributes attr;
        XM_CHECK_CUDA(cudaPointerGetAttributes(&attr, buffer));

        return attr.type == cudaMemoryTypeHost;
    }

    nvimgcodecStatus_t EncoderImpl::encode(const nvimgcodecCodeStreamDesc_t* code_stream,
                                           const nvimgcodecImageDesc_t* image, const nvimgcodecEncodeParams_t* params, int thread_idx) {
        auto& t = per_thread_[thread_idx];
        auto state_handle = t.state_;

        try {
            XM_CHECK_NULL(image);
            nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
            auto ret = image->getImageInfo(image->instance, &image_info);
            if (ret != NVIMGCODEC_STATUS_SUCCESS) {
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
                return ret;
            }

            unsigned char* device_buffer = reinterpret_cast<unsigned char*>(image_info.buffer);

            nvjpeg2kEncodeParams_t encode_params_;
            XM_CHECK_NVJPEG2K(nvjpeg2kEncodeParamsCreate(&encode_params_));
            std::unique_ptr<std::remove_pointer<nvjpeg2kEncodeParams_t>::type, decltype(&nvjpeg2kEncodeParamsDestroy)> encode_params(
                encode_params_, &nvjpeg2kEncodeParamsDestroy);

            auto sample_type = image_info.plane_info[0].sample_type;
            nvjpeg2kImageType_t nvjpeg2k_sample_type = to_nvjpeg2k_sample_type(sample_type);
            size_t num_components = std::max(image_info.num_planes, image_info.plane_info[0].num_channels);
            std::vector<nvjpeg2kImageComponentInfo_t> image_comp_info(num_components);

            uint32_t width = image_info.plane_info[0].width;
            uint32_t height = image_info.plane_info[0].height;
            std::vector<unsigned char*> encode_input(image_info.num_planes);
            std::vector<size_t> pitch_in_bytes(image_info.num_planes);
            size_t plane_start = 0;
            for (uint32_t c = 0; c < image_info.num_planes; ++c) {
                encode_input[c] = device_buffer + plane_start;
                pitch_in_bytes[c] = image_info.plane_info[c].row_stride;
                plane_start += image_info.plane_info[c].height * image_info.plane_info[c].row_stride;
            }

            uint8_t sgn = (sample_type == NVIMGCODEC_SAMPLE_DATA_TYPE_INT8 || sample_type == NVIMGCODEC_SAMPLE_DATA_TYPE_INT16);
            // canEncode has already validated that any non-zero plane_info[c].precision fits
            // in the sample type, so we can use it verbatim here and fall back to the dtype's
            // full bitdepth only when the caller left it as 0.
            uint8_t default_precision = static_cast<uint8_t>(sample_type_to_bytes_per_element(sample_type) * 8);
            // supports only the same precision for all planes, checked in canEncode.
            uint8_t user_precision = image_info.plane_info[0].precision;
            uint8_t applied_precision = user_precision ? user_precision : default_precision;
            for (uint32_t c = 0; c < num_components; c++) {
                image_comp_info[c].component_width = width;
                image_comp_info[c].component_height = height;
                image_comp_info[c].precision = applied_precision;
                image_comp_info[c].sgn = sgn;
            }

            nvimgcodecImageInfo_t out_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), nullptr};
            ret = code_stream->getImageInfo(code_stream->instance, &out_image_info);
            if (ret != NVIMGCODEC_STATUS_SUCCESS) {
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
                return ret;
            }

            nvjpeg2kEncodeConfig_t encode_config;
            memset(&encode_config, 0, sizeof(encode_config));
            if (num_components == 1) {
                encode_config.color_space = NVJPEG2K_COLORSPACE_GRAY;
            } else {
                encode_config.color_space = nvimgcodec_to_nvjpeg2k_color_spec(image_info.color_spec);
            }

            encode_config.image_width = width;
            encode_config.image_height = height;
            encode_config.num_components = num_components;
            encode_config.image_comp_info = image_comp_info.data();
            encode_config.mct_mode = 0;

            // Defaults
            encode_config.stream_type = NVJPEG2K_STREAM_JP2; // the bitstream will be in JP2 container format
            encode_config.code_block_w = 64;
            encode_config.code_block_h = 64;
            encode_config.irreversible = 1;
            encode_config.prog_order = NVJPEG2K_LRCP;
            encode_config.num_resolutions = 6;
            encode_config.num_layers = 1;
            encode_config.enable_tiling = 0;
            encode_config.enable_SOP_marker = 0;
            encode_config.enable_EPH_marker = 0;
            encode_config.encode_modes = 0;
#if NVJPEG2K_VER_MAJOR >= 0 && NVJPEG2K_VER_MINOR >= 8
            encode_config.num_precincts_init = 0;
#else
            encode_config.enable_custom_precincts = 0;
#endif
            fill_encode_config(encode_config, params, height, width);

            if (params->quality_type == NVIMGCODEC_QUALITY_TYPE_LOSSLESS) {
                NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, " - lossless encoding - using reversible 5-3 transform.");
                encode_config.irreversible = 0;
            }

            XM_CHECK_NVJPEG2K(nvjpeg2kEncodeParamsSetEncodeConfig(encode_params.get(), &encode_config));

            if (params->quality_type == NVIMGCODEC_QUALITY_TYPE_QUALITY) {
                NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, " - encoding with Q-factor: " << params->quality_value);
                XM_CHECK_NVJPEG2K(nvjpeg2kEncodeParamsSpecifyQuality(encode_params.get(), NVJPEG2K_QUALITY_TYPE_Q_FACTOR, params->quality_value));
            } else if (params->quality_type == NVIMGCODEC_QUALITY_TYPE_DEFAULT) {
                if (image_info.color_spec == NVIMGCODEC_COLORSPEC_SRGB && encode_config.mct_mode == 0) {
                    NVIMGCODEC_LOG_INFO(framework_, plugin_id_, " Q-factor cannot be used if input color space is SRGB and output is not Y or SYCC.");
                    if (encode_config.rsiz == 0x4000) { // ht, psnr will not work
                        NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, " - using default quantization setting");
                        // use default quantization set by nvjpeg2k
                    } else {
                        NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, " - default encoding - using PSNR: 50");
                        XM_CHECK_NVJPEG2K(nvjpeg2kEncodeParamsSpecifyQuality(encode_params.get(), NVJPEG2K_QUALITY_TYPE_TARGET_PSNR, 50));
                    }
                } else {
                    NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, " - default encoding - using Q-factor: 75");
                    XM_CHECK_NVJPEG2K(nvjpeg2kEncodeParamsSpecifyQuality(encode_params.get(), NVJPEG2K_QUALITY_TYPE_Q_FACTOR, 75));
                }
            } else if (params->quality_type == NVIMGCODEC_QUALITY_TYPE_QUANTIZATION_STEP) {
                NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, " - encoding with Quantization step: " << params->quality_value);
                XM_CHECK_NVJPEG2K(nvjpeg2kEncodeParamsSpecifyQuality(encode_params.get(), NVJPEG2K_QUALITY_TYPE_QUANTIZATION_STEP, params->quality_value));
            } else if (params->quality_type == NVIMGCODEC_QUALITY_TYPE_PSNR) {
                NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, " - encoding with target PSNR: " << params->quality_value);
                XM_CHECK_NVJPEG2K(nvjpeg2kEncodeParamsSpecifyQuality(encode_params.get(), NVJPEG2K_QUALITY_TYPE_TARGET_PSNR, params->quality_value));
            }

            nvjpeg2kImage_t input_image;
            input_image.num_components = num_components;
            input_image.pixel_data = reinterpret_cast<void**>(&encode_input[0]);
            input_image.pitch_in_bytes = pitch_in_bytes.data();
            input_image.pixel_type = nvjpeg2k_sample_type;

            nvjpeg2kImageFormat_t format = image_info.num_planes == 1 ? NVJPEG2K_FORMAT_INTERLEAVED : NVJPEG2K_FORMAT_PLANAR;
            XM_CHECK_NVJPEG2K(nvjpeg2kEncodeParamsSetInputFormat(encode_params.get(), format));

            XM_CHECK_NVJPEG2K(nvjpeg2kEncode(handle_, state_handle, encode_params.get(), &input_image, image_info.cuda_stream));
            XM_CHECK_CUDA(cudaEventRecord(t.event_, image_info.cuda_stream));
            XM_CHECK_CUDA(cudaEventSynchronize(t.event_));

            size_t length;
            XM_CHECK_NVJPEG2K(nvjpeg2kEncodeRetrieveBitstream(handle_, state_handle, NULL, &length, image_info.cuda_stream));

            nvimgcodecIoStreamDesc_t* io_stream = code_stream->io_stream;

            io_stream->reserve(io_stream->instance, length);
            void* encoded_data = nullptr;
            if (io_stream->map(io_stream->instance, &encoded_data, 0, length) !=
                NVIMGCODEC_STATUS_SUCCESS) {
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
                return NVIMGCODEC_STATUS_EXECUTION_FAILED;
            }
            bool is_pinned = is_buffer_pinned(encoded_data);
            if (encoded_data == nullptr || !is_pinned) {
                NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "Output buffer is not pinned, copying to pinned buffer");
                t.pinned_buffer_.resize(length, image_info.cuda_stream);
                XM_CHECK_NVJPEG2K(nvjpeg2kEncodeRetrieveBitstream(
                    handle_, state_handle, static_cast<uint8_t*>(t.pinned_buffer_.data), &length, image_info.cuda_stream));

                XM_CHECK_CUDA(cudaStreamSynchronize(image_info.cuda_stream));

                size_t output_size;
                io_stream->seek(io_stream->instance, 0, SEEK_SET);
                io_stream->write(io_stream->instance, &output_size, t.pinned_buffer_.data, t.pinned_buffer_.size);
                io_stream->flush(io_stream->instance);
            } else {
                NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "Output buffer is pinned, using it");
                XM_CHECK_NVJPEG2K(nvjpeg2kEncodeRetrieveBitstream(
                    handle_, state_handle, static_cast<uint8_t*>(encoded_data), &length, image_info.cuda_stream));
                XM_CHECK_CUDA(cudaStreamSynchronize(image_info.cuda_stream));
            }

            io_stream->unmap(io_stream->instance, encoded_data, length);
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_SUCCESS);
            return NVIMGCODEC_STATUS_SUCCESS;
        } catch (const NvJpeg2kException& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not encode jpeg2k code stream - " << e.info());
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
    }

} // namespace nvjpeg2k
