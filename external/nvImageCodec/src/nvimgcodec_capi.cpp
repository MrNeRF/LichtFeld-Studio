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

#include <cstring>
#include <filesystem>
#include <iostream>
#include <nvimgcodec.h>
#include <stdexcept>
#include <string>
#include <utility>

#include "code_stream.h"
#include "codec_registry.h"
#include "file_ext_codec.h"
#include "icodec.h"
#include "iimage_decoder.h"
#include "iimage_encoder.h"
#include "image.h"
#include "image_generic_decoder.h"
#include "image_generic_encoder.h"
#include "imgproc/exception.h"
#include "log.h"
#include "nvimgcodec_director.h"
#include "nvimgcodec_type_utils.h"
#include "plugin_framework.h"
#include "processing_results.h"

namespace fs = std::filesystem;

using namespace nvimgcodec;

__inline__ nvimgcodecStatus_t getCAPICode(Status status) {
    nvimgcodecStatus_t code = NVIMGCODEC_STATUS_SUCCESS;
    switch (status) {
    case STATUS_OK:
        code = NVIMGCODEC_STATUS_SUCCESS;
        break;
    case NOT_VALID_FORMAT_STATUS:
    case PARSE_STATUS:
    case BAD_FORMAT_STATUS:
        code = NVIMGCODEC_STATUS_BAD_CODESTREAM;
        break;
    case UNSUPPORTED_FORMAT_STATUS:
        code = NVIMGCODEC_STATUS_CODESTREAM_UNSUPPORTED;
        break;
    case CUDA_CALL_ERROR:
        code = NVIMGCODEC_STATUS_EXECUTION_FAILED;
        break;
    case ALLOCATION_ERROR:
        code = NVIMGCODEC_STATUS_ALLOCATOR_FAILURE;
        break;
    case INTERNAL_ERROR:
        code = NVIMGCODEC_STATUS_INTERNAL_ERROR;
        break;
    case INVALID_PARAMETER:
        code = NVIMGCODEC_STATUS_INVALID_PARAMETER;
        break;
    case ARCH_MISMATCH:
        code = NVIMGCODEC_STATUS_ARCH_MISMATCH;
        break;
    default:
        code = NVIMGCODEC_STATUS_INTERNAL_ERROR;
        break;
    }
    return code;
}

static nvimgcodecStatus_t validateCodeStreamViewSelection(const nvimgcodecCodeStreamView_t* code_stream_view, const char* context) {
    if (code_stream_view && code_stream_view->bitstream_offset != 0 && code_stream_view->image_idx != 0) {
        NVIMGCODEC_LOG_ERROR(
            Logger::get_default(), "image_idx cannot be combined with bitstream_offset for " << context);
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

static nvimgcodecStatus_t validateCodeStreamCreationView(const nvimgcodecCodeStreamView_t* code_stream_view) {
    auto status = validateCodeStreamViewSelection(code_stream_view, "code stream creation views");
    if (status != NVIMGCODEC_STATUS_SUCCESS || !code_stream_view) {
        return status;
    }
    if (code_stream_view->image_idx != 0) {
        NVIMGCODEC_LOG_ERROR(
            Logger::get_default(), "image_idx is not supported for code stream creation views; use sub-code streams");
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    if (code_stream_view->region.ndim != 0) {
        NVIMGCODEC_LOG_ERROR(
            Logger::get_default(), "region is not supported for code stream creation views; use sub-code streams");
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

#ifndef NDEBUG
#define VERBOSE_ERRORS
#endif

#define NVIMGCODECAPI_TRY try

#ifndef VERBOSE_ERRORS
#define NVIMGCODECAPI_CATCH(a)                                                   \
    catch (const Exception& e) {                                                 \
        a = getCAPICode(e.status());                                             \
    }                                                                            \
    catch (const std::exception& e) {                                            \
        NVIMGCODEC_LOG_ERROR(Logger::get_default(), e.what());                   \
        a = NVIMGCODEC_STATUS_INTERNAL_ERROR;                                    \
    }                                                                            \
    catch (...) {                                                                \
        NVIMGCODEC_LOG_ERROR(Logger::get_default(), "Unknown NVIMGCODEC error"); \
        a = NVIMGCODEC_STATUS_INTERNAL_ERROR;                                    \
    }
#else
#define NVIMGCODECAPI_CATCH(a)                                                                                    \
    catch (const Exception& e) {                                                                                  \
        NVIMGCODEC_LOG_ERROR(Logger::get_default(),                                                               \
                             "Error status: " << e.status() << " Where: " << e.where() << " What: " << e.what()); \
        a = getCAPICode(e.status());                                                                              \
    }                                                                                                             \
    catch (const std::exception& e) {                                                                             \
        NVIMGCODEC_LOG_ERROR(Logger::get_default(), e.what());                                                    \
        a = NVIMGCODEC_STATUS_INTERNAL_ERROR;                                                                     \
    }                                                                                                             \
    catch (...) {                                                                                                 \
        NVIMGCODEC_LOG_ERROR(Logger::get_default(), "Unknown NVIMGCODEC error");                                  \
        a = NVIMGCODEC_STATUS_INTERNAL_ERROR;                                                                     \
    }
#endif

#define CHECK_STRUCT_TYPE(obj_ptr, enum_type)                                                                                              \
    if (obj_ptr->struct_type != enum_type) {                                                                                               \
        throw Exception(                                                                                                                   \
            INTERNAL_ERROR, "Expected an object of type " + std::string(#enum_type) + "(" + std::to_string(static_cast<int>(enum_type)) +  \
                                "), but got an object of type " + std::to_string(obj_ptr->struct_type) +                                   \
                                ". The application was probably built against an nvImageCodec version that is not compatible with the "    \
                                "one currently installed (" +                                                                              \
                                std::to_string(NVIMGCODEC_VER_MAJOR) + "." + std::to_string(NVIMGCODEC_VER_MINOR) + "." +                  \
                                std::to_string(NVIMGCODEC_VER_PATCH) +                                                                     \
                                "). Please downgrade or upgrade your nvimagecodec version to match the one required by the application."); \
    }

#define CHECK_STRUCT_SIZE(obj_ptr, type)                                                                                                   \
    if (obj_ptr->struct_size != sizeof(type)) {                                                                                            \
        throw Exception(                                                                                                                   \
            INTERNAL_ERROR, "obj_ptr->struct_size(" + std::to_string(obj_ptr->struct_size) + ") != sizeof(" + std::string(#type) + ") (" + \
                                std::to_string(sizeof(type)) +                                                                             \
                                ". The application was probably built against an nvImageCodec version that is not compatible with the "    \
                                "one currently installed (" +                                                                              \
                                std::to_string(NVIMGCODEC_VER_MAJOR) + "." + std::to_string(NVIMGCODEC_VER_MINOR) + "." +                  \
                                std::to_string(NVIMGCODEC_VER_PATCH) +                                                                     \
                                "). Please downgrade or upgrade your nvimagecodec version to match the one required by the application."); \
    }

#define CHECK_STRUCT(obj_ptr, enum_type, type) \
    CHECK_STRUCT_TYPE(obj_ptr, enum_type);     \
    CHECK_STRUCT_SIZE(obj_ptr, type);

#define CHECK_NULL_AND_STRUCT(obj_ptr, enum_type, type) \
    CHECK_NULL(obj_ptr);                                \
    CHECK_STRUCT(obj_ptr, enum_type, type);

struct nvimgcodecInstance {
    nvimgcodecInstance(const nvimgcodecInstanceCreateInfo_t* create_info)
        : director_(create_info) {
    }
    NvImgCodecDirector director_;
};

struct nvimgcodecFuture {
    ProcessingResultsPromise::FutureImpl handle_;
};

struct nvimgcodecDecoder {
    nvimgcodecInstance_t instance_;
    std::unique_ptr<ImageGenericDecoder> image_decoder_;
};

struct nvimgcodecEncoder {
    nvimgcodecInstance_t instance_;
    std::unique_ptr<ImageGenericEncoder> image_encoder_;
};

struct nvimgcodecDebugMessenger {
    nvimgcodecDebugMessenger(const nvimgcodecDebugMessengerDesc_t* desc)
        : debug_messenger_(desc), instance_(nullptr) {
    }

    DebugMessenger debug_messenger_;
    nvimgcodecInstance_t instance_;
};

struct nvimgcodecExtension {
    nvimgcodecInstance_t nvimgcodec_instance_;
    nvimgcodecExtension_t extension_ext_handle_;
};

struct nvimgcodecCodeStream {
    nvimgcodecInstance_t nvimgcodec_instance_;
    nvimgcodec::CodeStream code_stream_;

    nvimgcodecCodeStream(nvimgcodecCodeStream* parent_code_stream, const nvimgcodecCodeStreamView_t& code_stream_view)
        : nvimgcodec_instance_(parent_code_stream->nvimgcodec_instance_), code_stream_(parent_code_stream->code_stream_, code_stream_view) {}

    nvimgcodecCodeStream(nvimgcodecInstance_t instance, std::unique_ptr<nvimgcodec::CodeStream> code_stream)
        : nvimgcodec_instance_(instance), code_stream_(std::move(*code_stream)) {}

    nvimgcodecCodeStream(nvimgcodecCodeStream&& other) noexcept
        : nvimgcodec_instance_(std::move(other.nvimgcodec_instance_)), code_stream_(std::move(other.code_stream_)) {}

    nvimgcodecCodeStream& operator=(nvimgcodecCodeStream&& other) noexcept {
        if (this != &other) {
            nvimgcodec_instance_ = std::move(other.nvimgcodec_instance_);
            code_stream_ = std::move(other.code_stream_);
        }
        return *this;
    }
};

struct nvimgcodecImage {
    nvimgcodecInstance_t nvimgcodec_instance_;
    Image image_;
};

nvimgcodecStatus_t nvimgcodecGetProperties(nvimgcodecProperties_t* properties) {
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    NVIMGCODECAPI_TRY {
        CHECK_NULL(properties);
        if (properties->struct_type != NVIMGCODEC_STRUCTURE_TYPE_PROPERTIES) {
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
        properties->version = NVIMGCODEC_VER;
        properties->cudart_version = CUDART_VERSION;
    }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

nvimgcodecStatus_t nvimgcodecInstanceCreate(nvimgcodecInstance_t* instance, const nvimgcodecInstanceCreateInfo_t* create_info) {
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    nvimgcodecInstance_t nvimgcodec = nullptr;
    NVIMGCODECAPI_TRY {
        CHECK_NULL(instance);
        CHECK_NULL_AND_STRUCT(create_info, NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nvimgcodecInstanceCreateInfo_t);
        nvimgcodec = new nvimgcodecInstance(create_info);
        *instance = nvimgcodec;
    }
    NVIMGCODECAPI_CATCH(ret)

    if (ret != NVIMGCODEC_STATUS_SUCCESS) {
        if (nvimgcodec) {
            delete nvimgcodec;
        }
    }

    return ret;
}

nvimgcodecStatus_t nvimgcodecInstanceDestroy(nvimgcodecInstance_t instance) {
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    NVIMGCODECAPI_TRY {
        CHECK_NULL(instance);
        delete instance;
    }
    NVIMGCODECAPI_CATCH(ret)

    return ret;
}

nvimgcodecStatus_t nvimgcodecExtensionCreate(
    nvimgcodecInstance_t instance, nvimgcodecExtension_t* extension, nvimgcodecExtensionDesc_t* extension_desc) {
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    NVIMGCODECAPI_TRY {
        CHECK_NULL(instance);
        CHECK_NULL(extension);
        CHECK_NULL_AND_STRUCT(extension_desc, NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, nvimgcodecExtensionDesc_t);
        nvimgcodecExtension_t extension_ext_handle;
        ret = instance->director_.plugin_framework_.registerExtension(&extension_ext_handle, extension_desc);
        if (ret == NVIMGCODEC_STATUS_SUCCESS) {
            *extension = new nvimgcodecExtension();
            (*extension)->nvimgcodec_instance_ = instance;
            (*extension)->extension_ext_handle_ = extension_ext_handle;
        }
    }
    NVIMGCODECAPI_CATCH(ret)

    return ret;
}

nvimgcodecStatus_t nvimgcodecExtensionDestroy(nvimgcodecExtension_t extension) {
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    NVIMGCODECAPI_TRY {
        CHECK_NULL(extension);
        ret = extension->nvimgcodec_instance_->director_.plugin_framework_.unregisterExtension(extension->extension_ext_handle_);
        delete extension;
    }
    NVIMGCODECAPI_CATCH(ret)

    return ret;
}

static nvimgcodecStatus_t nvimgcodecStreamCreate(nvimgcodecInstance_t instance, nvimgcodecCodeStream_t* code_stream) {
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    bool create_new_codestream = false;
    NVIMGCODECAPI_TRY {
        CHECK_NULL(instance);
        CHECK_NULL(code_stream);
        create_new_codestream = *code_stream == nullptr;
        std::unique_ptr<nvimgcodec::CodeStream> internal_code_stream = instance->director_.createCodeStream();
        if (create_new_codestream) {
            *code_stream = new nvimgcodecCodeStream(instance, std::move(internal_code_stream));
        } else {
            NVIMGCODEC_LOG_DEBUG(Logger::get_default(), "*code_stream is non-null, reusing it.");
            **code_stream = nvimgcodecCodeStream(instance, std::move(internal_code_stream));
        }
    }
    NVIMGCODECAPI_CATCH(ret)

    if (ret != NVIMGCODEC_STATUS_SUCCESS) {
        if (code_stream && create_new_codestream) {
            delete *code_stream;
        }
    }
    return ret;
}

nvimgcodecStatus_t nvimgcodecCodeStreamCreateFromFile(
    nvimgcodecInstance_t instance, nvimgcodecCodeStream_t* code_stream, const char* file_name,
    const nvimgcodecCodeStreamView_t* code_stream_view) {
    nvimgcodecStatus_t view_status = validateCodeStreamCreationView(code_stream_view);
    if (view_status != NVIMGCODEC_STATUS_SUCCESS) {
        return view_status;
    }
    const bool created_code_stream = (code_stream != nullptr && *code_stream == nullptr);
    nvimgcodecStatus_t ret = nvimgcodecStreamCreate(instance, code_stream);
    if (ret != NVIMGCODEC_STATUS_SUCCESS) {
        return ret;
    }
    NVIMGCODECAPI_TRY {
        CHECK_NULL(file_name);

        if (code_stream_view) {
            (*code_stream)->code_stream_.setCodeStreamView(code_stream_view);
        }
        (*code_stream)->code_stream_.parseFromFile(std::string(file_name));

        // Eagerly resolve a creation-time bitstream_offset so an invalid offset fails
        // here rather than at first decode. Parses only the selected IFD; root
        // (offset-less) streams stay lazy.
        if (code_stream_view && code_stream_view->bitstream_offset != 0) {
            nvimgcodecCodeStreamInfo_t codestream_info{
                NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfo_t), nullptr};
            ret = (*code_stream)->code_stream_.getCodeStreamInfo(&codestream_info);
            if (ret != NVIMGCODEC_STATUS_SUCCESS) {
                // Don't leak the stream we just created; match the contract that
                // *code_stream is null/unchanged on failure.
                if (created_code_stream) {
                    delete *code_stream;
                    *code_stream = nullptr;
                }
                return ret;
            }
        }
    }
    NVIMGCODECAPI_CATCH(ret)

    return ret;
}

nvimgcodecStatus_t nvimgcodecCodeStreamCreateFromHostMem(
    nvimgcodecInstance_t instance, nvimgcodecCodeStream_t* code_stream, const unsigned char* data, size_t length,
    const nvimgcodecCodeStreamView_t* code_stream_view) {
    nvimgcodecStatus_t view_status = validateCodeStreamCreationView(code_stream_view);
    if (view_status != NVIMGCODEC_STATUS_SUCCESS) {
        return view_status;
    }
    const bool created_code_stream = (code_stream != nullptr && *code_stream == nullptr);
    nvimgcodecStatus_t ret = nvimgcodecStreamCreate(instance, code_stream);
    if (ret != NVIMGCODEC_STATUS_SUCCESS) {
        return ret;
    }
    NVIMGCODECAPI_TRY {
        CHECK_NULL(data);

        if (code_stream_view) {
            (*code_stream)->code_stream_.setCodeStreamView(code_stream_view);
        }
        (*code_stream)->code_stream_.parseFromMem(data, length);

        // Eagerly resolve a creation-time bitstream_offset so an invalid offset fails
        // here rather than at first decode. Parses only the selected IFD; root
        // (offset-less) streams stay lazy.
        if (code_stream_view && code_stream_view->bitstream_offset != 0) {
            nvimgcodecCodeStreamInfo_t codestream_info{
                NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfo_t), nullptr};
            ret = (*code_stream)->code_stream_.getCodeStreamInfo(&codestream_info);
            if (ret != NVIMGCODEC_STATUS_SUCCESS) {
                // Don't leak the stream we just created; match the contract that
                // *code_stream is null/unchanged on failure.
                if (created_code_stream) {
                    delete *code_stream;
                    *code_stream = nullptr;
                }
                return ret;
            }
        }
    }

    NVIMGCODECAPI_CATCH(ret)

    return ret;
}

nvimgcodecStatus_t nvimgcodecCodeStreamCreateToFile(
    nvimgcodecInstance_t instance, nvimgcodecCodeStream_t* code_stream, const char* file_name, const nvimgcodecImageInfo_t* image_info) {
    nvimgcodecStatus_t ret = nvimgcodecStreamCreate(instance, code_stream);
    if (ret != NVIMGCODEC_STATUS_SUCCESS) {
        return ret;
    }
    NVIMGCODECAPI_TRY {
        CHECK_NULL(instance);
        CHECK_NULL(code_stream);
        CHECK_NULL(file_name);
        CHECK_NULL_AND_STRUCT(image_info, NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, nvimgcodecImageInfo_t);

        (*code_stream)->code_stream_.setOutputToFile(file_name);
        (*code_stream)->code_stream_.setImageInfo(image_info);
    }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

nvimgcodecStatus_t nvimgcodecCodeStreamCreateToHostMem(nvimgcodecInstance_t instance, nvimgcodecCodeStream_t* code_stream, void* ctx,
                                                       nvimgcodecResizeBufferFunc_t get_buffer_func, const nvimgcodecImageInfo_t* image_info) {
    nvimgcodecStatus_t ret = nvimgcodecStreamCreate(instance, code_stream);
    if (ret != NVIMGCODEC_STATUS_SUCCESS) {
        return ret;
    }
    NVIMGCODECAPI_TRY {
        CHECK_NULL(instance);
        CHECK_NULL(code_stream);
        CHECK_NULL(get_buffer_func);
        CHECK_NULL_AND_STRUCT(image_info, NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, nvimgcodecImageInfo_t);

        (*code_stream)->code_stream_.setOutputToHostMem(ctx, get_buffer_func);
        (*code_stream)->code_stream_.setImageInfo(image_info);
    }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

nvimgcodecStatus_t nvimgcodecCodeStreamDestroy(nvimgcodecCodeStream_t code_stream) {
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    NVIMGCODECAPI_TRY {
        CHECK_NULL(code_stream);
        delete code_stream;
    }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

nvimgcodecStatus_t nvimgcodecCodeStreamGetSubCodeStream(nvimgcodecCodeStream_t code_stream,
                                                        nvimgcodecCodeStream_t* sub_code_stream, const nvimgcodecCodeStreamView_t* code_stream_view) {
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    NVIMGCODECAPI_TRY {
        CHECK_NULL(code_stream);
        CHECK_NULL(sub_code_stream);
        CHECK_NULL_AND_STRUCT(code_stream_view, NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_VIEW, nvimgcodecCodeStreamView_t);

        ret = validateCodeStreamViewSelection(code_stream_view, "code stream views");
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            return ret;
        }

        nvimgcodecCodeStream candidate(code_stream, *code_stream_view);
        if (code_stream_view->image_idx != 0 || code_stream_view->bitstream_offset != 0) {
            nvimgcodecCodeStreamInfo_t codestream_info{
                NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfo_t), nullptr};
            ret = candidate.code_stream_.getCodeStreamInfo(&codestream_info);
            if (ret != NVIMGCODEC_STATUS_SUCCESS) {
                return ret;
            }
        }

        if (*sub_code_stream == nullptr) {
            *sub_code_stream = new nvimgcodecCodeStream(std::move(candidate));
        } else {
            NVIMGCODEC_LOG_DEBUG(Logger::get_default(), "*sub_code_stream is non-null, reusing it.");
            **sub_code_stream = std::move(candidate);
        }
    }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

nvimgcodecStatus_t nvimgcodecCodeStreamGetCodeStreamInfo(nvimgcodecCodeStream_t code_stream, nvimgcodecCodeStreamInfo_t* codestream_info) {
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    NVIMGCODECAPI_TRY {
        CHECK_NULL(code_stream);
        CHECK_NULL_AND_STRUCT(codestream_info, NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, nvimgcodecCodeStreamInfo_t);
        return code_stream->code_stream_.getCodeStreamInfo(codestream_info);
    }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

nvimgcodecStatus_t nvimgcodecCodeStreamGetImageInfo(nvimgcodecCodeStream_t code_stream, nvimgcodecImageInfo_t* image_info) {
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    NVIMGCODECAPI_TRY {
        CHECK_NULL(code_stream);
        CHECK_NULL_AND_STRUCT(image_info, NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, nvimgcodecImageInfo_t);
        return code_stream->code_stream_.getImageInfo(image_info);
    }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

namespace v0_3_0_api {

    typedef struct
    {
        nvimgcodecStructureType_t struct_type;
        size_t struct_size;
        const void* struct_next;
        void* instance;
        nvimgcodecStatus_t (*launch)(
            void* instance, int device_id, int sample_idx, void* task_context, void (*task)(int thread_id, int sample_idx, void* task_context));
        int (*getNumThreads)(void* instance);
    } nvimgcodecExecutorDesc_t;

    typedef struct
    {
        nvimgcodecStructureType_t struct_type;
        size_t struct_size;
        void* struct_next;
        nvimgcodecDeviceAllocator_t* device_allocator;
        nvimgcodecPinnedAllocator_t* pinned_allocator;
        int max_num_cpu_threads;
        nvimgcodecExecutorDesc_t* executor;
        int device_id;
        int pre_init;
        int num_backends;
        const nvimgcodecBackend_t* backends;
    } nvimgcodecExecutionParams_t;

    typedef struct
    {
        nvimgcodecStructureType_t struct_type;
        size_t struct_size;
        void* struct_next;
        float load_hint;
    } nvimgcodecBackendParams_t;

    typedef struct
    {
        nvimgcodecStructureType_t struct_type;
        size_t struct_size;
        void* struct_next;
        nvimgcodecBackendKind_t kind;
        nvimgcodecBackendParams_t params;
    } nvimgcodecBackend_t;

} // namespace v0_3_0_api

// This is a patch to maintain backward compatibility with versions <0.4.0:
// - If an old executor API was given, reinterpret all the parameters.
inline void updateExecutionParams(nvimgcodecExecutionParams_t* out_exec_params, const nvimgcodecExecutionParams_t* in_exec_params,
                                  std::vector<nvimgcodecBackend_t>& backends_buf) {
    if (in_exec_params->executor && in_exec_params->executor->struct_size == sizeof(v0_3_0_api::nvimgcodecExecutorDesc_t)) {
        NVIMGCODEC_LOG_WARNING(Logger::get_default(), "Incompatible executor instance, will use the default executor instead");
        // Also, assume reinterpret the whole thing to the new struct
        const v0_3_0_api::nvimgcodecExecutionParams_t* in_exec_params_v0_3_0 =
            reinterpret_cast<const v0_3_0_api::nvimgcodecExecutionParams_t*>(in_exec_params);
        out_exec_params->struct_type = in_exec_params_v0_3_0->struct_type;
        out_exec_params->struct_size = sizeof(nvimgcodecExecutionParams_t);
        out_exec_params->struct_next = in_exec_params_v0_3_0->struct_next;
        out_exec_params->device_allocator = in_exec_params_v0_3_0->device_allocator;
        out_exec_params->pinned_allocator = in_exec_params_v0_3_0->pinned_allocator;
        out_exec_params->max_num_cpu_threads = in_exec_params_v0_3_0->max_num_cpu_threads;
        out_exec_params->executor = nullptr; // dropping the executor, as it is not compatible
        out_exec_params->device_id = in_exec_params_v0_3_0->device_id;
        out_exec_params->pre_init = in_exec_params_v0_3_0->pre_init;
        out_exec_params->skip_pre_sync = 0; // not present in the old API
        out_exec_params->num_backends = in_exec_params_v0_3_0->num_backends;
        backends_buf.resize(out_exec_params->num_backends);
        for (int b = 0; b < in_exec_params_v0_3_0->num_backends; b++) {
            const auto& in_backend = in_exec_params_v0_3_0->backends[b];
            backends_buf[b] =
                nvimgcodecBackend_t{in_backend.struct_type, sizeof(nvimgcodecBackend_t), in_backend.struct_next, in_backend.kind,
                                    nvimgcodecBackendParams_t{in_backend.params.struct_type, sizeof(nvimgcodecBackendParams_t),
                                                              in_backend.params.struct_next, in_backend.params.load_hint, NVIMGCODEC_LOAD_HINT_POLICY_FIXED}};
        }
        out_exec_params->backends = backends_buf.data();
    } else {
        *out_exec_params = *in_exec_params;
    }
}

inline void checkExecutionParams(const nvimgcodecExecutionParams_t* exec_params) {
    CHECK_NULL_AND_STRUCT(exec_params, NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, nvimgcodecExecutionParams_t);
    if (exec_params->device_allocator) {
        CHECK_STRUCT(exec_params->device_allocator, NVIMGCODEC_STRUCTURE_TYPE_DEVICE_ALLOCATOR, nvimgcodecDeviceAllocator_t);
    }
    if (exec_params->pinned_allocator) {
        CHECK_STRUCT(exec_params->pinned_allocator, NVIMGCODEC_STRUCTURE_TYPE_PINNED_ALLOCATOR, nvimgcodecPinnedAllocator_t);
    }
    if (exec_params->executor) {
        CHECK_STRUCT(exec_params->executor, NVIMGCODEC_STRUCTURE_TYPE_EXECUTOR_DESC, nvimgcodecExecutorDesc_t);
    }

    if (exec_params->num_backends > 0) {
        for (int b = 0; b < exec_params->num_backends; b++) {
            const auto* backend = exec_params->backends + b;
            const auto* params = &backend->params;
            CHECK_STRUCT(backend, NVIMGCODEC_STRUCTURE_TYPE_BACKEND, nvimgcodecBackend_t);
            CHECK_STRUCT(params, NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS, nvimgcodecBackendParams_t);
        }
    }
}

NVIMGCODECAPI nvimgcodecStatus_t nvimgcodecDecoderCreate(
    nvimgcodecInstance_t instance, nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* in_exec_params, const char* options) {
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;

    NVIMGCODECAPI_TRY {
        CHECK_NULL(instance);
        CHECK_NULL(decoder);
        CHECK_NULL(in_exec_params);

        nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), 0};
        std::vector<nvimgcodecBackend_t> backends_buf;
        updateExecutionParams(&exec_params, in_exec_params, backends_buf); // this is a backward compatibility fix for <v0.4.0

        checkExecutionParams(&exec_params);

        std::unique_ptr<ImageGenericDecoder> image_decoder = instance->director_.createGenericDecoder(&exec_params, options);
        *decoder = new nvimgcodecDecoder();
        (*decoder)->image_decoder_ = std::move(image_decoder);
        (*decoder)->instance_ = instance;
    }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

nvimgcodecStatus_t nvimgcodecDecoderDestroy(nvimgcodecDecoder_t decoder) {
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    NVIMGCODECAPI_TRY {
        CHECK_NULL(decoder);
        delete decoder;
    }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

NVIMGCODECAPI nvimgcodecStatus_t nvimgcodecDecoderGetMetadata(nvimgcodecDecoder_t decoder, nvimgcodecCodeStream_t code_stream, nvimgcodecMetadata_t** metadata, int* metadata_count) {
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    NVIMGCODECAPI_TRY {
        CHECK_NULL(decoder);
        CHECK_NULL(code_stream);
        CHECK_NULL(metadata_count);
        if (metadata && *metadata_count <= 0) { // when metadata is null, metadata_count is output parameter, so input value is ignored
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
        ret = decoder->image_decoder_->getMetadata(&code_stream->code_stream_, metadata, metadata_count);
    }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

NVIMGCODECAPI nvimgcodecStatus_t nvimgcodecDecoderCanDecode(nvimgcodecDecoder_t decoder, const nvimgcodecCodeStream_t* streams,
                                                            const nvimgcodecImage_t* images, int batch_size, const nvimgcodecDecodeParams_t* params,
                                                            nvimgcodecProcessingStatus_t* processing_status, int force_format) {
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    NVIMGCODECAPI_TRY {
        CHECK_NULL(decoder);
        CHECK_NULL(streams);
        CHECK_NULL(images);
        CHECK_NULL_AND_STRUCT(params, NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS, nvimgcodecDecodeParams_t);
        CHECK_NULL(processing_status)

        if (batch_size <= 0)
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;

        std::vector<nvimgcodec::ICodeStream*> internal_code_streams;
        std::vector<nvimgcodec::IImage*> internal_images;

        for (int i = 0; i < batch_size; ++i) {
            internal_code_streams.push_back(&streams[i]->code_stream_);
            internal_images.push_back(&images[i]->image_);
        }

        decoder->image_decoder_->canDecode(internal_code_streams, internal_images, params, processing_status, force_format);
    }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

nvimgcodecStatus_t nvimgcodecDecoderDecode(nvimgcodecDecoder_t decoder, const nvimgcodecCodeStream_t* streams,
                                           const nvimgcodecImage_t* images, int batch_size, const nvimgcodecDecodeParams_t* params, nvimgcodecFuture_t* future) {
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    NVIMGCODECAPI_TRY {
        CHECK_NULL(decoder);
        CHECK_NULL(streams);
        CHECK_NULL(images);
        CHECK_NULL_AND_STRUCT(params, NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS, nvimgcodecDecodeParams_t);
        CHECK_NULL(future);

        if (batch_size <= 0)
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;

        std::vector<nvimgcodec::ICodeStream*> internal_code_streams;
        std::vector<nvimgcodec::IImage*> internal_images;

        for (int i = 0; i < batch_size; ++i) {
            internal_code_streams.push_back(&streams[i]->code_stream_);
            internal_images.push_back(&images[i]->image_);
        }
        *future = new nvimgcodecFuture();

        (*future)->handle_ = std::move(decoder->image_decoder_->decode(internal_code_streams, internal_images, params));
    }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

nvimgcodecStatus_t nvimgcodecImageCreate(nvimgcodecInstance_t instance, nvimgcodecImage_t* image, const nvimgcodecImageInfo_t* image_info) {
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;

    NVIMGCODECAPI_TRY {
        CHECK_NULL(image);
        CHECK_NULL(instance);
        CHECK_NULL_AND_STRUCT(image_info, NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, nvimgcodecImageInfo_t)
        CHECK_NULL(image_info->buffer);
        if (image_info->buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_UNKNOWN ||
            image_info->buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_UNSUPPORTED) {
            NVIMGCODEC_LOG_ERROR(Logger::get_default(), "Unknown or unsupported buffer kind");
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }

        if (*image == nullptr) {
            *image = new nvimgcodecImage();
        } else {
            NVIMGCODEC_LOG_DEBUG(Logger::get_default(), "*image is non-null, reusing it.");
        }
        assert(*image != nullptr);
        (*image)->image_.setImageInfo(image_info);
        (*image)->nvimgcodec_instance_ = instance;
    }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

nvimgcodecStatus_t nvimgcodecImageDestroy(nvimgcodecImage_t image) {
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    NVIMGCODECAPI_TRY {
        CHECK_NULL(image)
        delete image;
    }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

nvimgcodecStatus_t nvimgcodecImageGetImageInfo(nvimgcodecImage_t image, nvimgcodecImageInfo_t* image_info) {
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    NVIMGCODECAPI_TRY {
        CHECK_NULL(image);
        CHECK_NULL_AND_STRUCT(image_info, NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, nvimgcodecImageInfo_t)
        image->image_.getImageInfo(image_info);
    }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

NVIMGCODECAPI nvimgcodecStatus_t nvimgcodecEncoderCreate(
    nvimgcodecInstance_t instance, nvimgcodecEncoder_t* encoder, const nvimgcodecExecutionParams_t* in_exec_params, const char* options) {
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;

    NVIMGCODECAPI_TRY {
        CHECK_NULL(instance);
        CHECK_NULL(encoder);
        CHECK_NULL(in_exec_params);

        nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), 0};
        std::vector<nvimgcodecBackend_t> backends_buf;
        updateExecutionParams(&exec_params, in_exec_params, backends_buf); // this is a backward compatibility fix for <v0.4.0

        checkExecutionParams(&exec_params);
        std::unique_ptr<ImageGenericEncoder> image_encoder = instance->director_.createGenericEncoder(&exec_params, options);
        *encoder = new nvimgcodecEncoder();
        (*encoder)->image_encoder_ = std::move(image_encoder);
        (*encoder)->instance_ = instance;
    }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

nvimgcodecStatus_t nvimgcodecEncoderDestroy(nvimgcodecEncoder_t encoder) {
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    NVIMGCODECAPI_TRY {
        CHECK_NULL(encoder)
        delete encoder;
    }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

NVIMGCODECAPI nvimgcodecStatus_t nvimgcodecEncoderCanEncode(nvimgcodecEncoder_t encoder, const nvimgcodecImage_t* images,
                                                            const nvimgcodecCodeStream_t* streams, int batch_size, const nvimgcodecEncodeParams_t* params,
                                                            nvimgcodecProcessingStatus_t* processing_status, int force_format) {
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;

    NVIMGCODECAPI_TRY {
        CHECK_NULL(encoder);
        CHECK_NULL(streams);
        CHECK_NULL(images);
        CHECK_NULL_AND_STRUCT(params, NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS, nvimgcodecEncodeParams_t)
        CHECK_NULL(processing_status);

        if (batch_size <= 0) {
            NVIMGCODEC_LOG_ERROR(Logger::get_default(), "batch_size cannot be 0 or negative");
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }

        std::vector<nvimgcodec::ICodeStream*> internal_code_streams;
        std::vector<nvimgcodec::IImage*> internal_images;

        for (int i = 0; i < batch_size; ++i) {
            internal_code_streams.push_back(&streams[i]->code_stream_);
            internal_images.push_back(&images[i]->image_);
        }

        encoder->image_encoder_->canEncode(internal_images, internal_code_streams, params, processing_status, force_format);
    }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

nvimgcodecStatus_t nvimgcodecEncoderEncode(nvimgcodecEncoder_t encoder, const nvimgcodecImage_t* images,
                                           const nvimgcodecCodeStream_t* streams, int batch_size, const nvimgcodecEncodeParams_t* params, nvimgcodecFuture_t* future) {
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;

    NVIMGCODECAPI_TRY {
        CHECK_NULL(encoder);
        CHECK_NULL(streams);
        CHECK_NULL(images);
        CHECK_NULL_AND_STRUCT(params, NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS, nvimgcodecEncodeParams_t)
        CHECK_NULL(future);

        if (batch_size <= 0) {
            NVIMGCODEC_LOG_ERROR(Logger::get_default(), "batch_size cannot be 0 or negative");
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }

        std::vector<nvimgcodec::ICodeStream*> internal_code_streams;
        std::vector<nvimgcodec::IImage*> internal_images;

        for (int i = 0; i < batch_size; ++i) {
            internal_code_streams.push_back(&streams[i]->code_stream_);
            internal_images.push_back(&images[i]->image_);
        }

        *future = new nvimgcodecFuture();

        (*future)->handle_ = std::move(encoder->image_encoder_->encode(internal_images, internal_code_streams, params));
    }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

nvimgcodecStatus_t nvimgcodecDebugMessengerCreate(
    nvimgcodecInstance_t instance, nvimgcodecDebugMessenger_t* dbgMessenger, const nvimgcodecDebugMessengerDesc_t* messengerDesc) {
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    NVIMGCODECAPI_TRY {
        CHECK_NULL(instance);
        CHECK_NULL(dbgMessenger);
        CHECK_NULL(messengerDesc);
        CHECK_NULL_AND_STRUCT(messengerDesc, NVIMGCODEC_STRUCTURE_TYPE_DEBUG_MESSENGER_DESC, nvimgcodecDebugMessengerDesc_t);
        *dbgMessenger = new nvimgcodecDebugMessenger(messengerDesc);
        (*dbgMessenger)->instance_ = instance;
        instance->director_.registerDebugMessenger(&(*dbgMessenger)->debug_messenger_);
    }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

nvimgcodecStatus_t nvimgcodecDebugMessengerDestroy(nvimgcodecDebugMessenger_t dbgMessenger) {
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    NVIMGCODECAPI_TRY {
        CHECK_NULL(dbgMessenger);
        dbgMessenger->instance_->director_.unregisterDebugMessenger(&dbgMessenger->debug_messenger_);
        delete dbgMessenger;
    }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

nvimgcodecStatus_t nvimgcodecFutureWaitForAll(nvimgcodecFuture_t future) {
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    NVIMGCODECAPI_TRY {
        CHECK_NULL(future)
        future->handle_.wait();
    }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

nvimgcodecStatus_t nvimgcodecFutureDestroy(nvimgcodecFuture_t future) {
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    NVIMGCODECAPI_TRY {
        CHECK_NULL(future)
        delete future;
    }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}

nvimgcodecStatus_t nvimgcodecFutureGetProcessingStatus(
    nvimgcodecFuture_t future, nvimgcodecProcessingStatus_t* processing_status, size_t* size) {
    nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
    NVIMGCODECAPI_TRY {
        CHECK_NULL(future)
        CHECK_NULL(processing_status)
        CHECK_NULL(size)

        auto results = future->handle_.get();
        *size = results.size();
        auto ptr = processing_status;
        for (auto r : results)
            *ptr++ = r;
    }
    NVIMGCODECAPI_CATCH(ret)
    return ret;
}
