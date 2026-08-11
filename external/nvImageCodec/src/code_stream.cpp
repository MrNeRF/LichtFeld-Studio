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

#include "code_stream.h"
#include "codec.h"
#include "codec_registry.h"
#include "image_parser.h"
#include "imgproc/exception.h"
#include "log.h"
#include <atomic>
#include <cstring>
#include <iostream>
#include <nvtx3/nvtx3.hpp>
#include <string>

namespace nvimgcodec {

    static std::atomic<uint64_t> s_id(0);

    CodeStream::CodeStream(ICodecRegistry* codec_registry, std::unique_ptr<IIoStreamFactory> io_stream_factory)
        : codec_registry_(codec_registry), parser_(nullptr), io_stream_factory_(std::move(io_stream_factory)), io_stream_(nullptr), io_stream_desc_{NVIMGCODEC_STRUCTURE_TYPE_IO_STREAM_DESC, sizeof(nvimgcodecIoStreamDesc_t), nullptr, this, s_id.fetch_add(1, std::memory_order_relaxed), read_static, write_static, putc_static, skip_static, seek_static, tell_static, size_static, reserve_static, flush_static, map_static, unmap_static}, code_stream_desc_{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_DESC, sizeof(nvimgcodecCodeStreamDesc_t), nullptr, this, &io_stream_desc_, static_get_codestream_info, static_get_image_info} {
    }

    CodeStream::CodeStream(const CodeStream& other, const nvimgcodecCodeStreamView_t& code_stream_view)
        : codec_registry_{other.codec_registry_}, parser_{other.parser_} // For now we assume there is the same codec so the same parser. It can change in the future.
          ,
          io_stream_factory_{nullptr},
          io_stream_{other.io_stream_},
          io_stream_desc_{NVIMGCODEC_STRUCTURE_TYPE_IO_STREAM_DESC, sizeof(nvimgcodecIoStreamDesc_t), nullptr, this, other.io_stream_desc_.id, read_static, write_static, putc_static, skip_static, seek_static, tell_static, size_static, reserve_static, flush_static, map_static, unmap_static},
          code_stream_desc_{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_DESC, sizeof(nvimgcodecCodeStreamDesc_t), nullptr, this, &io_stream_desc_, static_get_codestream_info, static_get_image_info},
          parse_status_{NVIMGCODEC_STATUS_NOT_INITIALIZED},
          image_info_status_{NVIMGCODEC_STATUS_NOT_INITIALIZED},
          code_stream_view_(code_stream_view),
          codestream_info_{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfo_t), nullptr, &code_stream_view_, ""},
          codestream_info_tiff_ext_{NVIMGCODEC_STRUCTURE_TYPE_TIFF_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfoTiffExt_t), nullptr, 0},
          tile_geometry_info_{NVIMGCODEC_STRUCTURE_TYPE_TILE_GEOMETRY_INFO, sizeof(nvimgcodecTileGeometryInfo_t), nullptr},
          jpeg_info_{NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO, sizeof(nvimgcodecJpegImageInfo_t), &tile_geometry_info_},
          image_info_{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), &jpeg_info_} {
        const auto* parent_view = other.codestream_info_.code_stream_view;
        const bool parent_region = parent_view && parent_view->region.ndim != 0;

        code_stream_view_.struct_next = nullptr;
        code_stream_view_.region.struct_next = nullptr;

        if (parent_region) {
            throw Exception(INVALID_PARAMETER, "Cannot create a sub code stream with nested regions. This is not supported.", "CodeStream::CodeStream");
        }
        if (parent_view && code_stream_view.image_idx != 0) {
            throw Exception(INVALID_PARAMETER, "Cannot apply nonzero image_idx to a sub code stream.",
                            "CodeStream::CodeStream");
        }

        // Default child views preserve the parent's selected image. Explicit
        // bitstream_offset still overrides the parent selection.
        if (code_stream_view_.bitstream_offset == 0 && parent_view) {
            if (parent_view->bitstream_offset != 0) {
                code_stream_view_.bitstream_offset = parent_view->bitstream_offset;
            } else if (code_stream_view_.image_idx == 0) {
                code_stream_view_.image_idx = parent_view->image_idx;
            }
        }

        if (code_stream_view_.bitstream_offset != 0 && code_stream_view_.image_idx != 0) {
            throw Exception(INVALID_PARAMETER, "image_idx cannot be combined with bitstream_offset for code stream views",
                            "CodeStream::CodeStream");
        }

        // Link codestream info extension
        codestream_info_.struct_next = &codestream_info_tiff_ext_;

        // Reuse parsed metadata only when the child changes ROI but keeps the same
        // selected image. Container-specific selection stays in the parser cache.
        const size_t parent_image_idx = parent_view ? parent_view->image_idx : 0;
        const size_t parent_bitstream_offset = parent_view ? parent_view->bitstream_offset : 0;
        const bool selects_same_image =
            code_stream_view_.image_idx == parent_image_idx &&
            code_stream_view_.bitstream_offset == parent_bitstream_offset;
        const bool can_reuse_parsed_info =
            other.parse_status_ == NVIMGCODEC_STATUS_SUCCESS &&
            other.image_info_status_ == NVIMGCODEC_STATUS_SUCCESS &&
            other.codestream_info_.num_images == 1 &&
            selects_same_image;
        if (can_reuse_parsed_info) {
            parse_status_ = other.parse_status_;
            image_info_status_ = other.image_info_status_;
            codestream_info_ = other.codestream_info_;
            codestream_info_tiff_ext_ = other.codestream_info_tiff_ext_;
            tile_geometry_info_ = other.tile_geometry_info_;
            jpeg_info_ = other.jpeg_info_;
            image_info_ = other.image_info_;
            codestream_info_.code_stream_view = &code_stream_view_;
            codestream_info_.struct_next = &codestream_info_tiff_ext_;
            image_info_.struct_next = &jpeg_info_;
            jpeg_info_.struct_next = &tile_geometry_info_;
        }
    }

    void CodeStream::moveImpl(CodeStream&& other) {
        codec_registry_ = other.codec_registry_;
        parser_ = std::move(other.parser_);
        io_stream_factory_ = std::move(other.io_stream_factory_);
        io_stream_ = std::move(other.io_stream_);
        // Update io_stream_desc_ with new instance pointer but preserve ID from other
        io_stream_desc_ = {NVIMGCODEC_STRUCTURE_TYPE_IO_STREAM_DESC, sizeof(nvimgcodecIoStreamDesc_t), nullptr, this, other.io_stream_desc_.id,
                           read_static, write_static, putc_static, skip_static, seek_static, tell_static, size_static, reserve_static, flush_static,
                           map_static, unmap_static};
        // Update code_stream_desc_ with new instance pointer
        code_stream_desc_ = {NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_DESC, sizeof(nvimgcodecCodeStreamDesc_t), nullptr, this, &io_stream_desc_,
                             static_get_codestream_info, static_get_image_info};
        parse_status_ = other.parse_status_;
        image_info_status_ = other.image_info_status_;
        code_stream_view_ = other.code_stream_view_;
        codestream_info_ = other.codestream_info_;
        codestream_info_tiff_ext_ = other.codestream_info_tiff_ext_;
        tile_geometry_info_ = other.tile_geometry_info_;
        jpeg_info_ = other.jpeg_info_;
        image_info_ = other.image_info_;
        // Update pointer references
        if (other.codestream_info_.code_stream_view) {
            codestream_info_.code_stream_view = &code_stream_view_;
        }
        codestream_info_.struct_next = &codestream_info_tiff_ext_;
        image_info_.struct_next = &jpeg_info_;
        jpeg_info_.struct_next = &tile_geometry_info_;
    }

    CodeStream::CodeStream(CodeStream&& other) {
        moveImpl(std::move(other));
    }

    CodeStream& CodeStream::operator=(CodeStream&& other) noexcept {
        moveImpl(std::move(other));
        return *this;
    }

    CodeStream::~CodeStream() {
    }

    IImageParser* CodeStream::getParser() {
        if (parser_ == nullptr) {
            auto parser = codec_registry_->getParser(&code_stream_desc_);
            if (!parser) {
                throw Exception(UNSUPPORTED_FORMAT_STATUS, "The encoded stream did not match any of the available format parsers",
                                "CodeStream::parse - Encoded stream parsing");
            }

            parser_ = std::move(parser);
        }
        return parser_.get();
    }

    void CodeStream::parseFromFile(const std::string& file_name) {
        assert(io_stream_factory_);
        io_stream_ = io_stream_factory_->createFileIoStream(file_name, false, true, false);
        try {
            getParser();
        } catch (...) {} // share one parser/cache across sub streams
    }

    void CodeStream::parseFromMem(const unsigned char* data, size_t size) {
        assert(io_stream_factory_);
        io_stream_ = io_stream_factory_->createMemIoStream(data, size);
        try {
            getParser();
        } catch (...) {} // share one parser/cache across sub streams
    }

    void CodeStream::setOutputToFile(const char* file_name) {
        assert(io_stream_factory_);
        io_stream_ = io_stream_factory_->createFileIoStream(file_name, false, false, true);
    }

    void CodeStream::setOutputToHostMem(void* ctx, nvimgcodecResizeBufferFunc_t resize_buffer_func) {
        assert(io_stream_factory_);
        io_stream_ = io_stream_factory_->createMemIoStream(ctx, resize_buffer_func);
    }

    void CodeStream::setCodeStreamView(const nvimgcodecCodeStreamView_t* view) {
        // Creation views only seed an absolute bitstream offset. Image and region
        // selection is represented by sub-code streams.
        if (!view || view->bitstream_offset == 0) {
            return;
        }

        code_stream_view_.bitstream_offset = view->bitstream_offset;
        codestream_info_.code_stream_view = &code_stream_view_;
        codestream_info_.struct_next = &codestream_info_tiff_ext_;
    }

    template <typename T>
    void copy(T& dst, const T& src) {
        void* struct_next = dst.struct_next;
        dst = src;
        dst.struct_next = struct_next;
    }

    nvimgcodecStatus_t CodeStream::ensureCodeStreamInfoParsed() {
        if (parse_status_ == NVIMGCODEC_STATUS_NOT_INITIALIZED) {
            IImageParser* parser = getParser();
            if (parser) {
                parse_status_ = parser->getCodeStreamInfo(&code_stream_desc_, &codestream_info_);
                if (parse_status_ == NVIMGCODEC_STATUS_SUCCESS) {
                    // TIFF image_idx views are resolved by the parser to a concrete
                    // IFD and intentionally exposed as a single-image code stream.
                    const bool has_resolved_tiff_ifd_view =
                        codestream_info_.code_stream_view && codestream_info_tiff_ext_.ifd_offset != 0;
                    if (codestream_info_.code_stream_view &&
                        codestream_info_.code_stream_view->image_idx >= codestream_info_.num_images &&
                        !has_resolved_tiff_ifd_view) {
                        NVIMGCODEC_LOG_ERROR(Logger::get_default(), "Image index #" << codestream_info_.code_stream_view->image_idx
                                                                                    << " out of range (0, " << (codestream_info_.num_images == 0 ? 0 : codestream_info_.num_images - 1) << ")");
                        parse_status_ = NVIMGCODEC_STATUS_INVALID_PARAMETER;
                        return parse_status_;
                    }
                }
            } else {
                return NVIMGCODEC_STATUS_CODESTREAM_UNSUPPORTED;
            }
        }
        return parse_status_;
    }

    nvimgcodecStatus_t CodeStream::ensureParsed() {
        if (image_info_status_ == NVIMGCODEC_STATUS_NOT_INITIALIZED && !codestream_info_.code_stream_view &&
            parse_status_ == NVIMGCODEC_STATUS_NOT_INITIALIZED) {
            IImageParser* parser = getParser();
            if (parser && parser->getCodecName() == "tiff") {
                nvimgcodecCodeStreamView_t first_image_view{
                    NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_VIEW,
                    sizeof(nvimgcodecCodeStreamView_t),
                    nullptr,
                    0,
                    {},
                    0};
                CodeStream first_image_stream(*this, first_image_view);
                image_info_status_ = first_image_stream.ensureParsed();
                if (image_info_status_ != NVIMGCODEC_STATUS_SUCCESS) {
                    return image_info_status_;
                }

                copy(tile_geometry_info_, first_image_stream.tile_geometry_info_);
                copy(jpeg_info_, first_image_stream.jpeg_info_);
                copy(image_info_, first_image_stream.image_info_);
                return image_info_status_;
            }
        }

        auto status = ensureCodeStreamInfoParsed();
        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            return status;
        }

        if (image_info_status_ == NVIMGCODEC_STATUS_NOT_INITIALIZED) {
            IImageParser* parser = getParser();
            image_info_status_ = parser->getImageInfo(&code_stream_desc_, &image_info_);
        }
        return image_info_status_;
    }

    nvimgcodecStatus_t CodeStream::getCodeStreamInfo(nvimgcodecCodeStreamInfo_t* codestream_info) {
        assert(codestream_info);

        if (ensureCodeStreamInfoParsed() != NVIMGCODEC_STATUS_SUCCESS) {
            return parse_status_;
        }

        // Fill size from the backing stream when the parser left it unset (e.g. encoder-created
        // streams). Sub-code-streams share the parent's io_stream_ and inherit the same backing size.
        if (codestream_info_.size == 0 && io_stream_) {
            codestream_info_.size = io_stream_->size();
        }

        void* struct_next = codestream_info->struct_next;
        *codestream_info = codestream_info_;
        codestream_info->struct_next = struct_next;

        // Copy extension struct data to caller's extension chain
        while (struct_next) {
            auto* base = reinterpret_cast<nvimgcodecCodeStreamInfoTiffExt_t*>(struct_next);
            auto* next_struct_next = base->struct_next;
            switch (base->struct_type) {
            case NVIMGCODEC_STRUCTURE_TYPE_TIFF_CODE_STREAM_INFO:
                // Copy whole struct and fix the pointer (future-proof for new members)
                *base = codestream_info_tiff_ext_;
                base->struct_next = next_struct_next;
                break;
            default:
                break;
            }
            struct_next = next_struct_next;
        }

        return NVIMGCODEC_STATUS_SUCCESS;
    }

    nvimgcodecStatus_t CodeStream::getImageInfo(nvimgcodecImageInfo_t* image_info) {
        assert(image_info);

        auto status = ensureParsed();
        if (status != NVIMGCODEC_STATUS_SUCCESS) {
            return status;
        }

        void* struct_next = image_info->struct_next;
        *image_info = image_info_;
        image_info->struct_next = struct_next;

        while (struct_next) {
            auto* ptr = reinterpret_cast<nvimgcodecImageInfo_t*>(struct_next);
            auto* next_struct_next = ptr->struct_next;
            switch (ptr->struct_type) {
            case NVIMGCODEC_STRUCTURE_TYPE_TILE_GEOMETRY_INFO:
                copy(*reinterpret_cast<nvimgcodecTileGeometryInfo_t*>(ptr), tile_geometry_info_);
                break;
            case NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO:
                copy(*reinterpret_cast<nvimgcodecJpegImageInfo_t*>(ptr), jpeg_info_);
                break;
            default:
                return NVIMGCODEC_STATUS_INVALID_PARAMETER;
            }
            struct_next = next_struct_next;
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    nvimgcodecStatus_t CodeStream::setImageInfo(const nvimgcodecImageInfo_t* image_info) {
        void* tmp_struct_next = image_info_.struct_next;
        image_info_ = *image_info;
        image_info_.struct_next = tmp_struct_next;

        void* struct_next = image_info->struct_next;
        while (struct_next) {
            auto* ptr = reinterpret_cast<nvimgcodecImageInfo_t*>(struct_next);
            switch (ptr->struct_type) {
            case NVIMGCODEC_STRUCTURE_TYPE_TILE_GEOMETRY_INFO:
                copy(tile_geometry_info_, *reinterpret_cast<nvimgcodecTileGeometryInfo_t*>(ptr));
                break;
            case NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO:
                copy(jpeg_info_, *reinterpret_cast<nvimgcodecJpegImageInfo_t*>(ptr));
                break;
            default:
                assert(false);
            }
            struct_next = ptr->struct_next;
        }
        // Code streams created for encoding (CreateToHostMem/CreateToFile) have a single image.
        // Populate codestream_info_ so getCodeStreamInfo() and get_sub_code_stream() work without parsing.
        codestream_info_.num_images = 1;
        codestream_info_.code_stream_view = &code_stream_view_;
        codestream_info_.struct_next = &codestream_info_tiff_ext_;
        if (image_info->codec_name[0] != '\0') {
            strncpy(codestream_info_.codec_name, image_info->codec_name, NVIMGCODEC_MAX_CODEC_NAME_SIZE - 1);
            codestream_info_.codec_name[NVIMGCODEC_MAX_CODEC_NAME_SIZE - 1] = '\0';
        }
        codestream_info_.size = 0; // Updated in getCodeStreamInfo from io_stream_ when available
        parse_status_ = NVIMGCODEC_STATUS_SUCCESS;
        image_info_status_ = NVIMGCODEC_STATUS_SUCCESS;
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    std::string CodeStream::getCodecName() const {
        if (image_info_status_ == NVIMGCODEC_STATUS_SUCCESS) {
            return std::string(image_info_.codec_name);
        } else {
            IImageParser* parser = const_cast<CodeStream*>(this)->getParser();
            if (parser) {
                return parser->getCodecName();
            } else {
                return "";
            }
        }
    }

    ICodec* CodeStream::getCodec() const {
        return codec_registry_->getCodecByName(getCodecName().c_str());
    }

    nvimgcodecIoStreamDesc_t* CodeStream::getInputStreamDesc() {
        return &io_stream_desc_;
    }

    nvimgcodecCodeStreamDesc_t* CodeStream::getCodeStreamDesc() {
        return &code_stream_desc_;
    }

    nvimgcodecStatus_t CodeStream::read(size_t* output_size, void* buf, size_t bytes) {
        try {
            assert(io_stream_);
            *output_size = io_stream_->read(buf, bytes);
            return NVIMGCODEC_STATUS_SUCCESS;
        } catch (std::exception& e) {
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
    }

    nvimgcodecStatus_t CodeStream::write(size_t* output_size, void* buf, size_t bytes) {
        try {
            assert(io_stream_);
            *output_size = io_stream_->write(buf, bytes);
            return NVIMGCODEC_STATUS_SUCCESS;
        } catch (std::exception& e) {
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
    }
    nvimgcodecStatus_t CodeStream::putc(size_t* output_size, unsigned char ch) {
        try {
            assert(io_stream_);
            *output_size = io_stream_->putc(ch);

            return *output_size == 1 ? NVIMGCODEC_STATUS_SUCCESS : NVIMGCODEC_STATUS_BAD_CODESTREAM;
        } catch (std::exception& e) {
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
    }

    nvimgcodecStatus_t CodeStream::skip(size_t count) {
        try {
            assert(io_stream_);
            io_stream_->skip(count);
            return NVIMGCODEC_STATUS_SUCCESS;
        } catch (std::exception& e) {
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
    }

    nvimgcodecStatus_t CodeStream::seek(ptrdiff_t offset, int whence) {
        try {
            assert(io_stream_);
            io_stream_->seek(offset, whence);
            return NVIMGCODEC_STATUS_SUCCESS;
        } catch (std::exception& e) {
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
    }

    nvimgcodecStatus_t CodeStream::tell(ptrdiff_t* offset) {
        try {
            assert(io_stream_);
            *offset = io_stream_->tell();
            return NVIMGCODEC_STATUS_SUCCESS;
        } catch (std::exception& e) {
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
    }

    nvimgcodecStatus_t CodeStream::size(size_t* size) {
        try {
            assert(io_stream_);
            *size = io_stream_->size();
            return NVIMGCODEC_STATUS_SUCCESS;
        } catch (std::exception& e) {
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
    }

    nvimgcodecStatus_t CodeStream::reserve(size_t bytes) {
        try {
            assert(io_stream_);
            io_stream_->reserve(bytes);
            return NVIMGCODEC_STATUS_SUCCESS;
        } catch (std::exception& e) {
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
    }

    nvimgcodecStatus_t CodeStream::flush() {
        try {
            assert(io_stream_);
            io_stream_->flush();
            return NVIMGCODEC_STATUS_SUCCESS;
        } catch (std::exception& e) {
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
    }

    nvimgcodecStatus_t CodeStream::map(void** addr, size_t offset, size_t size) {
        try {
            assert(io_stream_);
            *addr = io_stream_->map(offset, size);
            return NVIMGCODEC_STATUS_SUCCESS;
        } catch (std::exception& e) {
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
    }

    nvimgcodecStatus_t CodeStream::unmap(void* addr, size_t size) {
        try {
            assert(io_stream_);
            io_stream_->unmap(addr, size);
            return NVIMGCODEC_STATUS_SUCCESS;
        } catch (std::exception& e) {
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
    }

    nvimgcodecStatus_t CodeStream::read_static(void* instance, size_t* output_size, void* buf, size_t bytes) {
        assert(instance);
        CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
        return handle->read(output_size, buf, bytes);
    }

    nvimgcodecStatus_t CodeStream::write_static(void* instance, size_t* output_size, void* buf, size_t bytes) {
        assert(instance);
        CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
        return handle->write(output_size, buf, bytes);
    }

    nvimgcodecStatus_t CodeStream::putc_static(void* instance, size_t* output_size, unsigned char ch) {
        assert(instance);
        CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
        return handle->putc(output_size, ch);
    }

    nvimgcodecStatus_t CodeStream::skip_static(void* instance, size_t count) {
        assert(instance);
        CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
        return handle->skip(count);
    }

    nvimgcodecStatus_t CodeStream::seek_static(void* instance, ptrdiff_t offset, int whence) {
        assert(instance);
        CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
        return handle->seek(offset, whence);
    }

    nvimgcodecStatus_t CodeStream::tell_static(void* instance, ptrdiff_t* offset) {
        assert(instance);
        CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
        return handle->tell(offset);
    }

    nvimgcodecStatus_t CodeStream::size_static(void* instance, size_t* size) {
        assert(instance);
        CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
        return handle->size(size);
    }

    nvimgcodecStatus_t CodeStream::reserve_static(void* instance, size_t bytes) {
        assert(instance);
        CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
        return handle->reserve(bytes);
    }

    nvimgcodecStatus_t CodeStream::flush_static(void* instance) {
        assert(instance);
        CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
        return handle->flush();
    }

    nvimgcodecStatus_t CodeStream::map_static(void* instance, void** addr, size_t offset, size_t size) {
        assert(instance);
        CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
        return handle->map(addr, offset, size);
    }

    nvimgcodecStatus_t CodeStream::unmap_static(void* instance, void* addr, size_t size) {
        assert(instance);
        CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
        return handle->unmap(addr, size);
    }

    nvimgcodecStatus_t CodeStream::static_get_codestream_info(void* instance, nvimgcodecCodeStreamInfo_t* codestream_info) {
        assert(instance);
        CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
        handle->getCodeStreamInfo(codestream_info);
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    nvimgcodecStatus_t CodeStream::static_get_image_info(void* instance, nvimgcodecImageInfo_t* result) {
        assert(instance);
        CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
        handle->getImageInfo(result);
        return NVIMGCODEC_STATUS_SUCCESS;
    }

} // namespace nvimgcodec
