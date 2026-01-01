/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include <filesystem>
#include <memory>
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
         * @return Tensor in GPU memory, float32, normalized [0-1]
         */
        lfs::core::Tensor load_image_gpu(
            const std::filesystem::path& path,
            int resize_factor = 1,
            int max_width = 0,
            void* cuda_stream = nullptr,
            DecodeFormat format = DecodeFormat::RGB);

        /**
         * @brief Decode JPEG from memory to GPU
         *
         * @param jpeg_data Raw JPEG bytes
         * @param resize_factor Downscale factor (1 = no scaling, 2 = half size, etc.)
         * @param max_width Maximum width/height (0 = no limit)
         * @param cuda_stream Optional CUDA stream for async operations
         * @param format Output format (RGB: [C,H,W], Grayscale: [H,W])
         * @return Tensor in GPU memory, float32, normalized [0-1]
         */
        lfs::core::Tensor load_image_from_memory_gpu(
            const std::vector<uint8_t>& jpeg_data,
            int resize_factor = 1,
            int max_width = 0,
            void* cuda_stream = nullptr,
            DecodeFormat format = DecodeFormat::RGB);

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
         * @brief Check if nvImageCodec is available and working
         */
        static bool is_available();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;

        std::vector<uint8_t> read_file(const std::filesystem::path& path);
    };

} // namespace lfs::io
