// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/logger.hpp"
#include "media/media_backends.hpp"
#include "nvcodec_image_loader.hpp"
#include "video/color_convert.cuh"
#include "video/cuda_frame_handoff.hpp"
#include <algorithm>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
namespace lfs::io {
    namespace {
        constexpr std::size_t MAX_JPEG_BATCH_FRAMES = 32;
        constexpr std::size_t JPEG_BATCH_BYTE_BUDGET = 256ULL * 1024 * 1024;
        constexpr std::size_t MIN_CUDA_MEMORY_HEADROOM = 256ULL * 1024 * 1024;
        void requireCudaSuccess(const cudaError_t result, const char* const operation) {
            if (result != cudaSuccess) {
                throw std::runtime_error(std::string(operation) + ": " +
                                         cudaGetErrorString(result));
            }
        }
        template <typename T>
        void freeCudaBuffer(T*& buffer, const char* const name) {
            if (!buffer)
                return;
            const cudaError_t result = cudaFree(buffer);
            if (result != cudaSuccess) {
                LOG_WARN("Failed to free {}: {}", name, cudaGetErrorString(result));
            }
            buffer = nullptr;
        }
        class CudaJpegEncoder final : public media::detail::GpuJpegEncoder {
            std::unique_ptr<NvCodecImageLoader> nvcodec;
            std::unique_ptr<video::CudaFrameHandoff> handoff;
            std::uint8_t* gpu_batch_buffer = nullptr;
            std::uint8_t* gpu_rgb_buffer = nullptr;
            std::uint8_t* gpu_rotated_buffer = nullptr;
            std::size_t jpeg_batch_size = 0, frame_size;
            int src_width, src_height, out_width, out_height, rotation;

        public:
            explicit CudaJpegEncoder(const media::detail::JpegSettings& s)
                : frame_size(static_cast<std::size_t>(s.output_width) * s.output_height * 3),
                  src_width(s.source_width),
                  src_height(s.source_height),
                  out_width(s.output_width),
                  out_height(s.output_height),
                  rotation(s.rotation) {
                const bool using_hw_decode = s.hardware_decode;
                const bool needs_scale = s.needs_scale;
                const bool convert_hdr_to_sdr = s.hdr_to_sdr;
                const int estimated_total = s.estimated_frames;
                std::size_t cuda_free_bytes = 0;
                std::size_t cuda_total_bytes = 0;
                const cudaError_t memory_info_result =
                    cudaMemGetInfo(&cuda_free_bytes, &cuda_total_bytes);
                if (memory_info_result != cudaSuccess) {
                    LOG_WARN(
                        "Failed to query CUDA memory for JPEG batching: {}; "
                        "falling back to CPU",
                        cudaGetErrorString(memory_info_result));
                } else {
                    const std::size_t headroom = std::max(
                        MIN_CUDA_MEMORY_HEADROOM, cuda_total_bytes / 10);
                    const std::size_t auxiliary_frame_count =
                        using_hw_decode && !needs_scale &&
                                !convert_hdr_to_sdr
                            ? (rotation == 0 ? 1 : 2)
                            : 0;
                    const std::size_t auxiliary_bytes =
                        auxiliary_frame_count == 0 ||
                                frame_size <=
                                    std::numeric_limits<std::size_t>::max() /
                                        auxiliary_frame_count
                            ? frame_size * auxiliary_frame_count
                            : cuda_free_bytes;
                    const std::size_t available_after_auxiliary =
                        auxiliary_bytes < cuda_free_bytes
                            ? cuda_free_bytes - auxiliary_bytes
                            : 0;
                    const std::size_t available_for_batch =
                        headroom < available_after_auxiliary
                            ? available_after_auxiliary - headroom
                            : 0;
                    jpeg_batch_size = std::min(
                        {MAX_JPEG_BATCH_FRAMES,
                         static_cast<std::size_t>(estimated_total),
                         JPEG_BATCH_BYTE_BUDGET / frame_size,
                         available_for_batch / frame_size});
                }

                if (jpeg_batch_size > 0) {
                    NvCodecImageLoader::Options opts;
                    nvcodec = std::make_unique<NvCodecImageLoader>(opts);
                    const cudaError_t allocation_result = cudaMalloc(
                        &gpu_batch_buffer, jpeg_batch_size * frame_size);
                    if (allocation_result != cudaSuccess) {
                        LOG_WARN(
                            "Failed to allocate {}-frame CUDA JPEG batch: {}; "
                            "falling back to CPU",
                            jpeg_batch_size,
                            cudaGetErrorString(allocation_result));
                        gpu_batch_buffer = nullptr;
                        jpeg_batch_size = 0;
                    }
                } else if (memory_info_result == cudaSuccess) {
                    LOG_WARN(
                        "Insufficient CUDA memory headroom for JPEG batching; "
                        "falling back to CPU");
                }

                if (using_hw_decode && gpu_batch_buffer && !needs_scale) {
                    const std::size_t src_frame_size =
                        static_cast<std::size_t>(src_width) * src_height * 3;
                    const cudaError_t allocation_result =
                        cudaMalloc(&gpu_rgb_buffer, src_frame_size);
                    if (allocation_result != cudaSuccess) {
                        LOG_WARN("Failed to allocate CUDA RGB buffer: {}",
                                 cudaGetErrorString(allocation_result));
                        gpu_rgb_buffer = nullptr;
                    }
                }

                if (using_hw_decode && gpu_batch_buffer && gpu_rgb_buffer &&
                    !needs_scale && !convert_hdr_to_sdr &&
                    rotation != 0) {
                    const cudaError_t allocation_result =
                        cudaMalloc(&gpu_rotated_buffer, frame_size);
                    if (allocation_result != cudaSuccess) {
                        LOG_WARN(
                            "Failed to allocate CUDA rotation buffer: {}; "
                            "using the CPU conversion path",
                            cudaGetErrorString(allocation_result));
                        gpu_rotated_buffer = nullptr;
                    }
                }
            }
            ~CudaJpegEncoder() override {
                handoff.reset();
                freeCudaBuffer(gpu_rgb_buffer, "CUDA RGB buffer");
                freeCudaBuffer(gpu_batch_buffer, "CUDA JPEG batch buffer");
                freeCudaBuffer(gpu_rotated_buffer, "CUDA rotation buffer");
            }
            std::size_t capacity() const noexcept override { return jpeg_batch_size; }
            bool canConvertHardware() const noexcept override {
                return gpu_rgb_buffer && (rotation == 0 || gpu_rotated_buffer);
            }
            void convertHardware(const AVFrame* frame, std::uint8_t* readback) override {
                handoff = std::make_unique<video::CudaFrameHandoff>(frame);
                video::nv12ToRgbCuda(frame->data[0], frame->data[1], gpu_rgb_buffer,
                                     src_width, src_height, frame->linesize[0], frame->linesize[1], nullptr);
                requireCudaSuccess(cudaGetLastError(), "CUDA NV12-to-RGB conversion failed");
                if (readback)
                    requireCudaSuccess(cudaMemcpy(readback, gpu_rgb_buffer, frame_size,
                                                  cudaMemcpyDeviceToHost),
                                       "CUDA sharpness readback failed");
            }
            void finishHardware() override {
                if (handoff)
                    handoff->finish();
                handoff.reset();
            }
            void* queueHardware(std::size_t index) override {
                const std::uint8_t* source = gpu_rgb_buffer;
                if (rotation) {
                    video::rotateRgbCuda(gpu_rgb_buffer, gpu_rotated_buffer, out_width, out_height, rotation, nullptr);
                    requireCudaSuccess(cudaGetLastError(), "CUDA RGB rotation failed");
                    source = gpu_rotated_buffer;
                }
                void* target = gpu_batch_buffer + index * frame_size;
                requireCudaSuccess(cudaMemcpy(target, source, frame_size, cudaMemcpyDeviceToDevice), "CUDA JPEG batch copy failed");
                finishHardware();
                return target;
            }
            void* queueHost(std::size_t index, const std::uint8_t* source) override {
                void* target = gpu_batch_buffer + index * frame_size;
                requireCudaSuccess(cudaMemcpy(target, source, frame_size, cudaMemcpyHostToDevice), "CUDA JPEG upload failed");
                return target;
            }
            std::vector<std::vector<std::uint8_t>> encode(const std::vector<void*>& frames,
                                                          int width, int height, int quality) override {
                return nvcodec->encode_batch_rgb_to_jpeg(frames, width, height, quality);
            }
        };
    } // namespace
    std::unique_ptr<media::detail::GpuJpegEncoder> createStudioJpegEncoder(const media::detail::JpegSettings& settings) {
        if (!NvCodecImageLoader::is_available())
            return nullptr;
        return std::make_unique<CudaJpegEncoder>(settings);
    }
} // namespace lfs::io
