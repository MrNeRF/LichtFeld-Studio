/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include "core/executable_path.hpp"
#include "core/path_utils.hpp"
#include "core/tensor.hpp"
#include "io/cuda/image_format_kernels.cuh"
#include "io/nvcodec_image_loader.hpp"

#include <OpenImageIO/imageio.h>
#include <cuda_runtime.h>
#include <stb_image.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

    namespace fs = std::filesystem;

    constexpr int kWidth = 3840;
    constexpr int kHeight = 2160;
    constexpr int kChannels = 3;

    struct SourceImage {
        int width = 0;
        int height = 0;
        std::vector<uint8_t> rgb;
    };

    template <typename Func>
    double measure_ms(Func&& func) {
        const auto start = std::chrono::steady_clock::now();
        func();
        const auto end = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    std::string path_string(const fs::path& path) {
        return path.string();
    }

    std::string nvjpeg2k_runtime_diagnostics() {
        std::ostringstream out;
        const fs::path extensions_dir = lfs::core::getExtensionsDir();
        out << "extensions_dir=" << lfs::core::path_to_utf8(extensions_dir);
        out << "\nextensions_dir_exists=" << (fs::exists(extensions_dir) ? "yes" : "no");
        out << "\nextensions_dir_entries=";
        if (fs::exists(extensions_dir)) {
            bool first = true;
            for (const auto& entry : fs::directory_iterator(extensions_dir)) {
                if (!first) {
                    out << ",";
                }
                first = false;
                out << entry.path().filename().string();
            }
            if (first) {
                out << "<empty>";
            }
        } else {
            out << "<missing>";
        }
        return out.str();
    }

    SourceImage load_source_image() {
        const fs::path path = fs::path(PROJECT_ROOT_PATH) / "data/bicycle/images_8/_DSC8739.JPG";
        std::unique_ptr<OIIO::ImageInput> input(OIIO::ImageInput::open(path_string(path)));
        if (!input) {
            throw std::runtime_error("Failed to open source image: " + path_string(path) + ": " + OIIO::geterror());
        }

        const OIIO::ImageSpec spec = input->spec();
        SourceImage image;
        image.width = spec.width;
        image.height = spec.height;
        image.rgb.resize(static_cast<size_t>(image.width) * image.height * kChannels);
        if (!input->read_image(0, 0, 0, kChannels, OIIO::TypeDesc::UINT8, image.rgb.data())) {
            const std::string error = input->geterror();
            throw std::runtime_error("Failed to read source image: " + (error.empty() ? OIIO::geterror() : error));
        }
        input->close();
        return image;
    }

    uint8_t sample_bilinear_u8(const SourceImage& source, int x, int y, int c) {
        const float sx = (static_cast<float>(x) + 0.5f) *
                             static_cast<float>(source.width) / static_cast<float>(kWidth) -
                         0.5f;
        const float sy = (static_cast<float>(y) + 0.5f) *
                             static_cast<float>(source.height) / static_cast<float>(kHeight) -
                         0.5f;
        const int x0 = std::clamp(static_cast<int>(std::floor(sx)), 0, source.width - 1);
        const int y0 = std::clamp(static_cast<int>(std::floor(sy)), 0, source.height - 1);
        const int x1 = std::min(x0 + 1, source.width - 1);
        const int y1 = std::min(y0 + 1, source.height - 1);
        const float tx = std::clamp(sx - static_cast<float>(x0), 0.0f, 1.0f);
        const float ty = std::clamp(sy - static_cast<float>(y0), 0.0f, 1.0f);

        const auto at = [&](int px, int py) {
            const size_t idx = (static_cast<size_t>(py) * source.width + px) * kChannels + c;
            return static_cast<float>(source.rgb[idx]);
        };
        const float top = at(x0, y0) * (1.0f - tx) + at(x1, y0) * tx;
        const float bottom = at(x0, y1) * (1.0f - tx) + at(x1, y1) * tx;
        return static_cast<uint8_t>(std::clamp(top * (1.0f - ty) + bottom * ty + 0.5f, 0.0f, 255.0f));
    }

    uint8_t noise8(int x, int y, int c) {
        uint32_t h = static_cast<uint32_t>(x) * 73856093u;
        h ^= static_cast<uint32_t>(y) * 19349663u;
        h ^= static_cast<uint32_t>(c) * 83492791u;
        h ^= h >> 13u;
        h *= 1274126177u;
        return static_cast<uint8_t>(h & 0xffu);
    }

    void build_16bit_test_images(std::vector<uint16_t>& gray, std::vector<uint16_t>& rgb) {
        const SourceImage source = load_source_image();
        gray.resize(static_cast<size_t>(kWidth) * kHeight);
        rgb.resize(static_cast<size_t>(kWidth) * kHeight * kChannels);

        for (int y = 0; y < kHeight; ++y) {
            for (int x = 0; x < kWidth; ++x) {
                uint8_t channels[kChannels];
                for (int c = 0; c < kChannels; ++c) {
                    channels[c] = sample_bilinear_u8(source, x, y, c);
                    const size_t rgb_idx = (static_cast<size_t>(y) * kWidth + x) * kChannels + c;
                    rgb[rgb_idx] = static_cast<uint16_t>((static_cast<uint16_t>(channels[c]) << 8u) |
                                                         noise8(x, y, c));
                }

                const uint16_t luma = static_cast<uint16_t>(
                    (static_cast<uint32_t>(channels[0]) * 77u +
                     static_cast<uint32_t>(channels[1]) * 150u +
                     static_cast<uint32_t>(channels[2]) * 29u) >>
                    8u);
                gray[static_cast<size_t>(y) * kWidth + x] =
                    static_cast<uint16_t>((luma << 8u) | noise8(x, y, 7));
            }
        }

        gray.front() = 0u;
        gray.back() = 65535u;
        rgb.front() = 0u;
        rgb.back() = 65535u;
    }

    void build_smooth_gray_gradient(std::vector<uint16_t>& gray) {
        gray.resize(static_cast<size_t>(kWidth) * kHeight);
        for (int y = 0; y < kHeight; ++y) {
            const uint32_t gy = static_cast<uint32_t>(y) * 65535u / static_cast<uint32_t>(kHeight - 1);
            for (int x = 0; x < kWidth; ++x) {
                const uint32_t gx = static_cast<uint32_t>(x) * 65535u / static_cast<uint32_t>(kWidth - 1);
                gray[static_cast<size_t>(y) * kWidth + x] = static_cast<uint16_t>((gx * 3u + gy) / 4u);
            }
        }
        gray.front() = 0u;
        gray.back() = 65535u;
    }

    lfs::core::Tensor uint16_to_float32_tensor(
        const std::vector<uint16_t>& input,
        const lfs::core::TensorShape& shape,
        size_t height,
        size_t width,
        size_t channels) {

        using namespace lfs::core;
        Tensor packed = Tensor::empty(shape, Device::CUDA, DataType::Float16);
        const cudaError_t copy_status = cudaMemcpy(
            packed.data_ptr(), input.data(), input.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
        if (copy_status != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemcpy failed: ") + cudaGetErrorString(copy_status));
        }

        Tensor output = Tensor::empty(shape, Device::CUDA, DataType::Float32);
        lfs::io::cuda::launch_uint16_hwc_to_float32_hwc(
            reinterpret_cast<const uint16_t*>(packed.data_ptr()),
            output.ptr<float>(),
            height,
            width,
            channels);
        const cudaError_t sync_status = cudaDeviceSynchronize();
        if (sync_status != cudaSuccess) {
            throw std::runtime_error(std::string("CUDA sync failed: ") + cudaGetErrorString(sync_status));
        }
        return output;
    }

    void expect_bit_exact_float_tensor(const lfs::core::Tensor& expected,
                                       const lfs::core::Tensor& actual,
                                       const char* label) {
        ASSERT_EQ(actual.dtype(), lfs::core::DataType::Float32) << label;
        ASSERT_EQ(actual.device(), lfs::core::Device::CUDA) << label;
        ASSERT_EQ(actual.shape(), expected.shape()) << label;

        const std::vector<float> expected_values = expected.cpu().to_vector();
        const std::vector<float> actual_values = actual.cpu().to_vector();
        ASSERT_EQ(actual_values.size(), expected_values.size()) << label;

        for (size_t i = 0; i < expected_values.size(); ++i) {
            if (actual_values[i] != expected_values[i]) {
                FAIL() << label << " mismatch at index " << i
                       << ": expected " << std::setprecision(10) << expected_values[i]
                       << ", got " << actual_values[i];
            }
        }
    }

    fs::path write_gray_png_reference(const std::vector<uint16_t>& gray) {
        fs::path path = fs::temp_directory_path() / "lfs_nvcodec_j2k_gray16_ref.png";
        std::unique_ptr<OIIO::ImageOutput> output(OIIO::ImageOutput::create(path_string(path)));
        if (!output) {
            throw std::runtime_error("Failed to create PNG output: " + OIIO::geterror());
        }

        OIIO::ImageSpec spec(kWidth, kHeight, 1, OIIO::TypeDesc::UINT16);
        spec.attribute("png:compressionLevel", 1);
        if (!output->open(path_string(path), spec)) {
            const std::string error = output->geterror();
            throw std::runtime_error("Failed to open PNG output: " + (error.empty() ? OIIO::geterror() : error));
        }
        if (!output->write_image(OIIO::TypeDesc::UINT16, gray.data())) {
            const std::string error = output->geterror();
            throw std::runtime_error("Failed to write PNG output: " + (error.empty() ? OIIO::geterror() : error));
        }
        output->close();
        return path;
    }

    double measure_stb_png_decode_ms(const std::vector<uint16_t>& gray) {
        const fs::path png_path = write_gray_png_reference(gray);
        int width = 0;
        int height = 0;
        int channels = 0;
        uint16_t* decoded = nullptr;
        const double ms = measure_ms([&] {
            decoded = stbi_load_16(path_string(png_path).c_str(), &width, &height, &channels, 1);
        });
        fs::remove(png_path);
        if (!decoded) {
            throw std::runtime_error(std::string("stbi_load_16 failed: ") + stbi_failure_reason());
        }
        stbi_image_free(decoded);
        if (width != kWidth || height != kHeight) {
            throw std::runtime_error("stbi_load_16 returned unexpected dimensions");
        }
        return ms;
    }

    struct Jpeg2kTimingRow {
        std::string name;
        double encode_ms = 0.0;
        double decode_ms = 0.0;
        size_t compressed_bytes = 0;
        size_t raw_bytes = 0;
    };

    Jpeg2kTimingRow round_trip_gray_jpeg2k(
        lfs::io::NvCodecImageLoader& loader,
        const lfs::core::Tensor& tensor,
        const std::string& name,
        bool high_throughput,
        size_t raw_bytes) {

        std::vector<uint8_t> encoded;
        double encode_ms = 0.0;
        try {
            encode_ms = measure_ms([&] {
                encoded = loader.encode_grayscale_to_jpeg2k(tensor, nullptr, high_throughput);
            });
        } catch (const std::exception& e) {
            throw std::runtime_error(name + " encode failed: " + e.what() + "\n" +
                                     nvjpeg2k_runtime_diagnostics());
        }

        lfs::core::Tensor decoded;
        double decode_ms = 0.0;
        try {
            decode_ms = measure_ms([&] {
                decoded = loader.decode_jpeg2k_16bit_from_memory_gpu(encoded);
            });
        } catch (const std::exception& e) {
            throw std::runtime_error(name + " decode failed: " + e.what() + "\n" +
                                     nvjpeg2k_runtime_diagnostics());
        }

        expect_bit_exact_float_tensor(tensor, decoded, name.c_str());
        return {name, encode_ms, decode_ms, encoded.size(), raw_bytes};
    }

    Jpeg2kTimingRow round_trip_rgb_jpeg2k(
        lfs::io::NvCodecImageLoader& loader,
        const lfs::core::Tensor& tensor,
        const std::string& name,
        bool high_throughput,
        size_t raw_bytes) {

        std::vector<uint8_t> encoded;
        double encode_ms = 0.0;
        try {
            encode_ms = measure_ms([&] {
                encoded = loader.encode_to_jpeg2k(tensor, nullptr, high_throughput);
            });
        } catch (const std::exception& e) {
            throw std::runtime_error(name + " encode failed: " + e.what() + "\n" +
                                     nvjpeg2k_runtime_diagnostics());
        }

        lfs::core::Tensor decoded;
        double decode_ms = 0.0;
        try {
            decode_ms = measure_ms([&] {
                decoded = loader.decode_jpeg2k_16bit_from_memory_gpu(encoded);
            });
        } catch (const std::exception& e) {
            throw std::runtime_error(name + " decode failed: " + e.what() + "\n" +
                                     nvjpeg2k_runtime_diagnostics());
        }

        expect_bit_exact_float_tensor(tensor, decoded, name.c_str());
        return {name, encode_ms, decode_ms, encoded.size(), raw_bytes};
    }

} // namespace

TEST(NvCodecImageLoaderJpeg2k16Bit, RoundTrips2160pGrayAndRgbLossless) {
    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    ASSERT_GT(device_count, 0);

    std::vector<uint16_t> gray_u16;
    std::vector<uint16_t> rgb_u16;
    build_16bit_test_images(gray_u16, rgb_u16);
    std::vector<uint16_t> smooth_gray_u16;
    build_smooth_gray_gradient(smooth_gray_u16);

    const auto gray_tensor = uint16_to_float32_tensor(
        gray_u16,
        lfs::core::TensorShape({static_cast<size_t>(kHeight), static_cast<size_t>(kWidth)}),
        kHeight,
        kWidth,
        1);
    const auto rgb_tensor = uint16_to_float32_tensor(
        rgb_u16,
        lfs::core::TensorShape({static_cast<size_t>(kHeight), static_cast<size_t>(kWidth), static_cast<size_t>(kChannels)}),
        kHeight,
        kWidth,
        kChannels);
    const auto smooth_gray_tensor = uint16_to_float32_tensor(
        smooth_gray_u16,
        lfs::core::TensorShape({static_cast<size_t>(kHeight), static_cast<size_t>(kWidth)}),
        kHeight,
        kWidth,
        1);

    lfs::io::NvCodecImageLoader::Options options;
    options.decoder_pool_size = 1;
    std::unique_ptr<lfs::io::NvCodecImageLoader> loader;
    try {
        loader = std::make_unique<lfs::io::NvCodecImageLoader>(options);
    } catch (const std::exception& e) {
        FAIL() << "nvImageCodec/nvjpeg2k runtime initialization failed: " << e.what()
               << "\nextensions_dir=" << lfs::core::path_to_utf8(lfs::core::getExtensionsDir());
    }

    const double stb_png_decode_ms = measure_stb_png_decode_ms(gray_u16);

    const size_t gray_raw_size = gray_u16.size() * sizeof(uint16_t);
    const size_t rgb_raw_size = rgb_u16.size() * sizeof(uint16_t);
    const size_t smooth_gray_raw_size = smooth_gray_u16.size() * sizeof(uint16_t);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "\nJPEG2000 16-bit codec timings (2160x3840)\n";
    std::cout << "case,encode_ms,decode_ms,compressed_bytes,raw_bytes,compressed_percent\n";
    const auto print_row = [](const Jpeg2kTimingRow& row) {
        std::cout << row.name << "," << row.encode_ms << "," << row.decode_ms << ","
                  << row.compressed_bytes << "," << row.raw_bytes << ","
                  << (100.0 * static_cast<double>(row.compressed_bytes) /
                      static_cast<double>(row.raw_bytes))
                  << "\n";
    };

    try {
        print_row(round_trip_gray_jpeg2k(
            *loader, gray_tensor, "gray_noise_ht0", false, gray_raw_size));
        print_row(round_trip_rgb_jpeg2k(
            *loader, rgb_tensor, "rgb_noise_ht0", false, rgb_raw_size));
        print_row(round_trip_gray_jpeg2k(
            *loader, gray_tensor, "gray_noise_ht1", true, gray_raw_size));
        print_row(round_trip_rgb_jpeg2k(
            *loader, rgb_tensor, "rgb_noise_ht1", true, rgb_raw_size));
        print_row(round_trip_gray_jpeg2k(
            *loader, smooth_gray_tensor, "gray_smooth_ht1", true, smooth_gray_raw_size));
    } catch (const std::exception& e) {
        FAIL() << e.what();
    }

    std::cout << "stb_png16_gray_decode_ms," << stb_png_decode_ms << "\n";
    std::cout << "nvjpeg2k_runtime_loaded,yes\n";
    std::cout << "round_trip_bit_exact,yes\n";
}

TEST(NvCodecImageLoaderJpeg2k8Bit, MaskRoundTripIsLossless) {
    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    ASSERT_GT(device_count, 0);

    constexpr size_t height = 64;
    constexpr size_t width = 96;
    std::vector<float> values(height * width);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = (i % 7 == 0) ? 1.0f : ((i % 5 == 0) ? 0.0f : 0.5f);
    }
    auto host = lfs::core::Tensor::from_blob(
        values.data(), lfs::core::TensorShape({height, width}),
        lfs::core::Device::CPU, lfs::core::DataType::Float32);
    auto input = host.to(lfs::core::Device::CUDA);

    lfs::io::NvCodecImageLoader::Options options;
    options.decoder_pool_size = 1;
    std::unique_ptr<lfs::io::NvCodecImageLoader> loader;
    try {
        loader = std::make_unique<lfs::io::NvCodecImageLoader>(options);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "nvImageCodec unavailable: " << e.what();
    }

    const auto encoded = loader->encode_grayscale_to_jpeg2k(input, nullptr, true, true);
    ASSERT_FALSE(encoded.empty());
    const auto decoded = loader->decode_jpeg2k_16bit_from_memory_gpu(encoded, nullptr, true, true);
    ASSERT_EQ(decoded.dtype(), lfs::core::DataType::UInt8);
    ASSERT_EQ(decoded.shape(), (lfs::core::TensorShape({height, width})));
    const auto actual = decoded.cpu().to_vector();
    ASSERT_EQ(actual.size(), values.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        const auto expected = static_cast<uint8_t>(std::clamp(values[i] * 255.0f, 0.0f, 255.0f));
        EXPECT_FLOAT_EQ(actual[i], static_cast<float>(expected)) << "mask sample " << i;
    }
}

TEST(NvCodecImageLoaderJpeg2k8Bit, RejectsNonLegacyStagingStream) {
    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    ASSERT_GT(device_count, 0);

    auto input = lfs::core::Tensor::zeros(
        {size_t{8}, size_t{8}}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
    lfs::io::NvCodecImageLoader::Options options;
    options.decoder_pool_size = 1;
    std::unique_ptr<lfs::io::NvCodecImageLoader> loader;
    try {
        loader = std::make_unique<lfs::io::NvCodecImageLoader>(options);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "nvImageCodec unavailable: " << e.what();
    }

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    EXPECT_THROW(
        (void)loader->encode_grayscale_to_jpeg2k(input, stream, true, true),
        std::runtime_error);
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(NvCodecImageLoaderJpeg, CanonicalJpegMeetsBicyclePsnrGate) {
    const auto path = fs::path(PROJECT_ROOT_PATH) / "data/bicycle/images_4/_DSC8739.JPG";
    if (!fs::is_regular_file(path)) {
        GTEST_SKIP() << "bicycle dataset is absent: " << path;
    }

    lfs::io::NvCodecImageLoader::Options options;
    options.decoder_pool_size = 1;
    std::unique_ptr<lfs::io::NvCodecImageLoader> loader;
    try {
        loader = std::make_unique<lfs::io::NvCodecImageLoader>(options);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "nvImageCodec unavailable: " << e.what();
    }

    const auto source = loader->load_image_gpu(path, 16, 0, nullptr, lfs::io::DecodeFormat::RGB, false);
    const auto source_values = source.cpu().to_vector();
    const size_t height = source.shape()[1];
    const size_t width = source.shape()[2];
    std::vector<float> grayscale(source_values.size());
    for (size_t i = 0; i < height * width; ++i) {
        const float value = (source_values[i] + source_values[height * width + i] +
                             source_values[2 * height * width + i]) /
                            3.0f;
        grayscale[i] = value;
        grayscale[height * width + i] = value;
        grayscale[2 * height * width + i] = value;
    }
    const auto canonical_source = lfs::core::Tensor::from_blob(
                                      grayscale.data(), lfs::core::TensorShape({size_t{3}, height, width}),
                                      lfs::core::Device::CPU, lfs::core::DataType::Float32)
                                      .to(lfs::core::Device::CUDA);
    constexpr int canonical_quality = 100;
    static_assert(canonical_quality >= 95);
    const auto encoded = loader->encode_to_jpeg(canonical_source, canonical_quality, nullptr);
    ASSERT_FALSE(encoded.empty());
    const auto decoded = loader->load_image_from_memory_gpu(
        encoded, 1, 0, nullptr, lfs::io::DecodeFormat::RGB, false);
    const auto decoded_values = decoded.cpu().to_vector();
    const auto reference_values = canonical_source.cpu().to_vector();
    ASSERT_EQ(reference_values.size(), decoded_values.size());
    double mse = 0.0;
    for (size_t i = 0; i < reference_values.size(); ++i) {
        const double error = static_cast<double>(reference_values[i]) - decoded_values[i];
        mse += error * error;
    }
    mse /= static_cast<double>(reference_values.size());
    ASSERT_GT(mse, 0.0);
    const double psnr = 10.0 * std::log10(1.0 / mse);
    EXPECT_GE(psnr, 45.0);
}

TEST(NvCodecImageLoaderJpeg, BatchedDecodeMatchesReferenceWithinTolerance) {
    const auto path = fs::path(PROJECT_ROOT_PATH) / "data/bicycle/images_4/_DSC8739.JPG";
    if (!fs::is_regular_file(path)) {
        GTEST_SKIP() << "bicycle dataset is absent: " << path;
    }

    lfs::io::NvCodecImageLoader::Options options;
    options.decoder_pool_size = 1;
    std::unique_ptr<lfs::io::NvCodecImageLoader> loader;
    try {
        loader = std::make_unique<lfs::io::NvCodecImageLoader>(options);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "nvImageCodec unavailable: " << e.what();
    }

    std::ifstream input(path, std::ios::binary);
    const std::vector<uint8_t> bytes(std::istreambuf_iterator<char>(input), {});
    ASSERT_FALSE(bytes.empty());
    const auto reference = loader->load_image_from_memory_gpu(
        bytes, 1, 0, nullptr, lfs::io::DecodeFormat::RGB, false);
    const auto batch = loader->decode_jpeg_batch_from_spans(
        {{bytes.data(), bytes.size()}}, nullptr, false, true);
    ASSERT_EQ(batch.size(), 1u);
    const auto ref_values = reference.cpu().to_vector();
    const auto batch_values = batch.front().cpu().to_vector();
    ASSERT_EQ(ref_values.size(), batch_values.size());
    for (size_t i = 0; i < ref_values.size(); ++i) {
        EXPECT_NEAR(ref_values[i], batch_values[i], 1.0 / 255.0)
            << "decoded sample " << i;
    }
}
