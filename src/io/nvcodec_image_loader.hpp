/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include <filesystem>
#include <memory>
#include <utility>
#include <vector>

// Forward declarations to avoid including nvImageCodec headers in public API
typedef struct nvimgcodecInstance* nvimgcodecInstance_t;
typedef struct nvimgcodecDecoder* nvimgcodecDecoder_t;
typedef struct nvimgcodecEncoder* nvimgcodecEncoder_t;

namespace lfs::io {

    /**
     * @brief Output format for image decoding
     */
    enum class DecodeFormat {
        RGB,      // 3-channel RGB [C,H,W] or [H,W,C]
        Grayscale // 1-channel grayscale [H,W]
    };

    /**
     * @brief GPU-accelerated image loader using NVIDIA nvImageCodec
     *
     * Provides hardware-accelerated JPEG decoding with direct GPU output.
     * Falls back to CPU decoding for unsupported formats.
     */
    class NvCodecImageLoader {
    public:
        struct Options {
            int device_id = 0;
            int max_num_cpu_threads = 0;
            bool enable_fallback = true;
            size_t decoder_pool_size = 8;
        };

        explicit NvCodecImageLoader(const Options& options);
        ~NvCodecImageLoader();

        // Disable copying
        NvCodecImageLoader(const NvCodecImageLoader&) = delete;
        NvCodecImageLoader& operator=(const NvCodecImageLoader&) = delete;

        /**
         * @brief Load and decode a single image to GPU memory
         *
         * @param path Path to image file
         * @param resize_factor Downscale factor (1 = no scaling, 2 = half size, etc.)
         * @param max_width Maximum width/height (0 = no limit)
         * @param cuda_stream Optional CUDA stream for async operations
         * @param format Output format (RGB: [C,H,W], Grayscale: [H,W])
         * @param output_uint8 If true for RGB, return uint8 [C,H,W] instead of float32 [0-1]
         * @return Tensor in GPU memory
         */
        lfs::core::Tensor load_image_gpu(
            const std::filesystem::path& path,
            int resize_factor = 1,
            int max_width = 0,
            void* cuda_stream = nullptr,
            DecodeFormat format = DecodeFormat::RGB,
            bool output_uint8 = false);

        /**
         * @brief Decode JPEG from memory to GPU
         *
         * @param jpeg_data Raw JPEG bytes
         * @param resize_factor Downscale factor (1 = no scaling, 2 = half size, etc.)
         * @param max_width Maximum width/height (0 = no limit)
         * @param cuda_stream Optional CUDA stream for async operations
         * @param format Output format (RGB: [C,H,W], Grayscale: [H,W])
         * @param output_uint8 If true for RGB, return uint8 [C,H,W] instead of float32 [0-1]
         * @return Tensor in GPU memory
         */
        lfs::core::Tensor load_image_from_memory_gpu(
            const std::vector<uint8_t>& jpeg_data,
            int resize_factor = 1,
            int max_width = 0,
            void* cuda_stream = nullptr,
            DecodeFormat format = DecodeFormat::RGB,
            bool output_uint8 = false);

        // Load and decode multiple images in batch
        std::vector<lfs::core::Tensor> load_images_batch_gpu(
            const std::vector<std::filesystem::path>& paths,
            int resize_factor = 1,
            int max_width = 0);

        // Batch decode JPEG blobs from memory
        std::vector<lfs::core::Tensor> batch_decode_from_memory(
            const std::vector<std::vector<uint8_t>>& jpeg_blobs,
            void* cuda_stream = nullptr);

        // Batch decode from spans (zero-copy)
        std::vector<lfs::core::Tensor> batch_decode_from_spans(
            const std::vector<std::pair<const uint8_t*, size_t>>& jpeg_spans,
            void* cuda_stream = nullptr);

        // Encode GPU tensor to JPEG bytes (RGB)
        std::vector<uint8_t> encode_to_jpeg(
            const lfs::core::Tensor& image,
            int quality = 100,
            void* cuda_stream = nullptr);

        // Encode GPU tensor to JPEG2k bytes (RGB 16bits)
        std::vector<uint8_t> encode_to_jpeg2k(
            const lfs::core::Tensor& image,
            void* cuda_stream = nullptr,
            bool high_throughput = true);

        /**
         * @brief Encode grayscale GPU tensor to lossless 16-bit JPEG2000 bytes
         *
         * @param image Float32 CUDA tensor in [H,W] layout, normalized [0,1].
         * @param cuda_stream Optional CUDA stream for async operations.
         * @param high_throughput Use JPEG2000 HT block coding.
         * @return JPEG2000 bytes preserving uint16 sample precision.
         */
        std::vector<uint8_t> encode_grayscale_to_jpeg2k(
            const lfs::core::Tensor& image,
            void* cuda_stream = nullptr,
            bool high_throughput = true);

        /**
         * @brief Decode lossless 16-bit JPEG2000 bytes from memory to GPU float32
         *
         * Decodes UINT16 JPEG2000 into a normalized Float32 CUDA tensor without
         * truncating to 8 bits. Grayscale returns [H,W]; RGB returns interleaved
         * [H,W,3]. Values round-trip bit-exact for samples representable as
         * uint16 / 65535.0f.
         *
         * @param jpeg2k_data Raw JPEG2000 bytes.
         * @param cuda_stream Optional CUDA stream for async operations.
         * @param synchronize Wait for device completion before returning.
         * @return Float32 CUDA tensor, [H,W] for grayscale or [H,W,3] for RGB.
         */
        lfs::core::Tensor decode_jpeg2k_16bit_from_memory_gpu(
            const std::vector<uint8_t>& jpeg2k_data,
            void* cuda_stream = nullptr,
            bool synchronize = true);

        std::vector<lfs::core::Tensor> decode_jpeg2k_16bit_batch_from_spans(
            const std::vector<std::pair<const uint8_t*, size_t>>& jpeg2k_spans,
            void* cuda_stream = nullptr,
            bool synchronize = true);

        /**
         * @brief Encode grayscale GPU tensor to JPEG bytes
         *
         * @param image Tensor in GPU memory, format: [H,W] float32 normalized [0-1]
         * @param quality JPEG quality (1-100)
         * @param cuda_stream Optional CUDA stream for async operations
         * @return JPEG bytes
         */
        std::vector<uint8_t> encode_grayscale_to_jpeg(
            const lfs::core::Tensor& image,
            int quality = 100,
            void* cuda_stream = nullptr);

        /**
         * @brief Batch encode raw RGB uint8 GPU data to JPEG bytes
         *
         * @param gpu_ptrs Vector of GPU device pointers to RGB24 data (HWC, uint8)
         * @param width Image width (all images must have same dimensions)
         * @param height Image height
         * @param quality JPEG quality (1-100)
         * @param cuda_stream Optional CUDA stream for async operations
         * @return Vector of JPEG byte vectors
         */
        std::vector<std::vector<uint8_t>> encode_batch_rgb_to_jpeg(
            const std::vector<void*>& gpu_ptrs,
            int width,
            int height,
            int quality = 95,
            void* cuda_stream = nullptr);

        /**
         * @brief Check if nvImageCodec is available and working
         */
        static bool is_available();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;

        std::vector<uint8_t> read_file(const std::filesystem::path& path);
    };

} // namespace lfs::io
