/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/image_codecs.hpp"
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <gtest/gtest.h>
#include <iterator>
#include <jpeglib.h>
#include <png.h>
#include <stdexcept>
#include <vector>
#include <zlib.h>

namespace {
    namespace codec = lfs::core::image_codecs;

    struct PngMetadataReader {
        png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
        png_infop info = png ? png_create_info_struct(png) : nullptr;
        ~PngMetadataReader() {
            if (png)
                png_destroy_read_struct(&png, &info, nullptr);
        }
    };
    struct PngMetadataInput {
        const std::vector<std::uint8_t>& bytes;
        std::size_t offset = 0;
    };
    void read_metadata_bytes(png_structp png, png_bytep output, png_size_t size) {
        auto* input = static_cast<PngMetadataInput*>(png_get_io_ptr(png));
        if (size > input->bytes.size() - input->offset)
            png_error(png, "Truncated PNG metadata");
        std::memcpy(output, input->bytes.data() + input->offset, size);
        input->offset += size;
    }
    bool read_metadata_info(png_structp png, png_infop info) {
        if (setjmp(png_jmpbuf(png)))
            return false;
        png_read_info(png, info);
        return true;
    }
    std::string read_png_comment(const std::filesystem::path& path) {
        std::ifstream file(path, std::ios::binary);
        std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(file), {}};
        PngMetadataReader reader;
        if (!reader.png || !reader.info)
            throw std::runtime_error("Could not allocate PNG metadata reader");
        PngMetadataInput input{bytes};
        png_set_read_fn(reader.png, &input, read_metadata_bytes);
        if (!read_metadata_info(reader.png, reader.info))
            throw std::runtime_error("Could not read PNG metadata");
        png_textp text = nullptr;
        const int count = png_get_text(reader.png, reader.info, &text, nullptr);
        for (int i = 0; i < count; ++i) {
            if (std::strcmp(text[i].key, "Comment") == 0)
                return text[i].text;
        }
        return {};
    }
    TEST(ProductionDependencies, HeadersMatchRuntimeBackend) {
        EXPECT_STREQ(zlibVersion(), ZLIB_VERSION);
    }
    TEST(ProductionDependencies, StockZlibStreamsAndChecksumsRemainCompatible) {
        // Independently generated with Python's stock zlib.compress(b"hello", 6).
        std::vector<std::uint8_t> encoded{0x78, 0x9c, 0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x07, 0x00, 0x06, 0x2c, 0x02, 0x15};
        std::vector<std::uint8_t> decoded(5);
        uLongf size = static_cast<uLongf>(decoded.size());
        ASSERT_EQ(uncompress(decoded.data(), &size, encoded.data(), static_cast<uLong>(encoded.size())), Z_OK);
        EXPECT_EQ(decoded, (std::vector<std::uint8_t>{'h', 'e', 'l', 'l', 'o'}));
        encoded.back() ^= 1;
        size = static_cast<uLongf>(decoded.size());
        EXPECT_EQ(uncompress(decoded.data(), &size, encoded.data(), static_cast<uLong>(encoded.size())), Z_DATA_ERROR);
        EXPECT_EQ(crc32(0, reinterpret_cast<const Bytef*>("123456789"), 9), 0xcbf43926u);
        EXPECT_EQ(adler32(1, reinterpret_cast<const Bytef*>("Wikipedia"), 9), 0x11e60398u);
    }
    struct TemporaryDirectory {
        std::filesystem::path path = std::filesystem::temp_directory_path() /
                                     ("lfs-codec-evaluation-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        TemporaryDirectory() {
            if (!std::filesystem::create_directory(path))
                throw std::runtime_error("temporary directory already exists");
        }
        ~TemporaryDirectory() {
            std::error_code error;
            std::filesystem::remove_all(path, error);
        }
    };
    TEST(ProductionImageOutput, DispatchPreservesLegacyBytesAndComments) {
        TemporaryDirectory temporary;
        const auto bytes = [](const std::filesystem::path& path) {
            std::ifstream file(path, std::ios::binary);
            return std::vector<std::uint8_t>{std::istreambuf_iterator<char>(file), {}};
        };
        for (const std::string extension : {".PNG", ".JpEg", ".TIFF", ".jpg", ".tif"}) {
            for (const int channels : {1, 2, 3, 4}) {
                for (const bool full_chroma : {false, true}) {
                    SCOPED_TRACE(::testing::Message() << extension << " channels=" << channels << " full_chroma=" << full_chroma);
                    constexpr int width = 19, height = 17;
                    std::vector<std::uint8_t> pixels(width * height * channels);
                    for (std::size_t i = 0; i < pixels.size(); ++i)
                        pixels[i] = static_cast<std::uint8_t>(i * 17);
                    const auto actual = temporary.path / std::filesystem::path(u8"export 日本語").concat(extension);
                    const auto reference = temporary.path / ("reference" + extension);
                    std::string error;
                    const bool jpeg = extension == ".JpEg" || extension == ".jpg";
                    if (jpeg && channels == 2) {
                        EXPECT_FALSE(codec::write_image_u8(actual, pixels.data(), width, height, channels, 95, "provenance", error, full_chroma));
                        EXPECT_FALSE(std::filesystem::exists(actual));
                        continue;
                    }
                    if (jpeg) {
                        // The legacy Studio policy drops alpha without compositing.
                        std::vector<std::uint8_t> rgb(width * height * 3);
                        if (channels == 4) {
                            for (std::size_t pixel = 0; pixel < width * height; ++pixel)
                                for (std::size_t c = 0; c < 3; ++c)
                                    rgb[pixel * 3 + c] = pixels[pixel * 4 + c];
                        }
                        ASSERT_TRUE(codec::write_jpeg(reference, channels == 4 ? rgb.data() : pixels.data(), width, height,
                                                      channels == 4 ? 3 : channels, 95, "provenance", error, full_chroma))
                            << error;
                    } else if (extension == ".PNG") {
                        ASSERT_TRUE(codec::write_png(reference, pixels.data(), width, height, channels, 8, 6, "provenance", error)) << error;
                    } else {
                        ASSERT_TRUE(codec::write_tiff(reference, pixels.data(), width, height, channels, error)) << error;
                    }
                    ASSERT_TRUE(codec::write_image_u8(actual, pixels.data(), width, height, channels, 95, "provenance", error, full_chroma)) << error;
                    EXPECT_EQ(bytes(actual), bytes(reference));
                    if (extension == ".PNG")
                        EXPECT_EQ(read_png_comment(actual), "provenance");
                    std::filesystem::remove(actual);
                    std::filesystem::remove(reference);
                }
            }
        }
    }
    TEST(ProductionImageOutput, AtomicCollisionCancellationAndEncodingFailuresPreserveTarget) {
        TemporaryDirectory temporary;
        const auto bytes = [](const std::filesystem::path& path) {
            std::ifstream file(path, std::ios::binary);
            return std::vector<std::uint8_t>{std::istreambuf_iterator<char>(file), {}};
        };
        for (const std::string extension : {".png", ".jpg", ".tiff"}) {
            SCOPED_TRACE(extension);
            const auto path = temporary.path / ("preserve" + extension);
            std::vector<std::uint8_t> pixels(19 * 17 * 4, 128);
            lfs::core::AtomicFileOptions options{.overwrite = false, .durable = false, .create_directories = false};
            ASSERT_TRUE(codec::write_image_u8(path, pixels.data(), 19, 17, 4, 95, "stamp", options));
            const auto original = bytes(path);
            std::fill(pixels.begin(), pixels.end(), 42);
            const auto collision = codec::write_image_u8(path, pixels.data(), 19, 17, 4, 95, {}, options);
            ASSERT_FALSE(collision);
            EXPECT_EQ(collision.error().code(), lfs::ErrorCode::AlreadyExists);
            EXPECT_EQ(bytes(path), original);
            options.overwrite = true;
            int checks = 0;
            options.cancelled = [&] { return ++checks == 2; };
            const auto cancelled = codec::write_image_u8(path, pixels.data(), 19, 17, 4, 95, {}, options);
            ASSERT_FALSE(cancelled);
            EXPECT_EQ(cancelled.error().code(), lfs::ErrorCode::Cancelled);
            EXPECT_EQ(bytes(path), original);
            options.cancelled = {};
            const auto invalid = codec::write_image_u8(path, nullptr, 19, 17, 4, 95, {}, options);
            ASSERT_FALSE(invalid);
            EXPECT_EQ(invalid.error().code(), lfs::ErrorCode::InvalidArgument);
            EXPECT_EQ(bytes(path), original);
            ASSERT_TRUE(codec::write_image_u8(path, pixels.data(), 19, 17, 4, 95, {}, options));
            EXPECT_NE(bytes(path), original);
            EXPECT_EQ(std::distance(std::filesystem::directory_iterator(temporary.path), std::filesystem::directory_iterator{}), 1);
            std::filesystem::remove(path);
        }
    }
    TEST(ProductionImageOutput, AtomicParallelPngPreservesEncodingAndPixels) {
        TemporaryDirectory temporary;
        constexpr int width = 1024, height = 1025, channels = 4;
        std::vector<std::uint8_t> pixels(std::size_t(width) * height * channels);
        for (std::size_t i = 0; i < pixels.size(); ++i)
            pixels[i] = static_cast<std::uint8_t>(i * 17);
        const auto direct = temporary.path / "direct.png", atomic = temporary.path / "atomic.png";
        std::string error;
        ASSERT_TRUE(codec::write_png(direct, pixels.data(), width, height, channels, 8, 6, "stamp", error)) << error;
        ASSERT_TRUE(codec::write_image_u8(atomic, pixels.data(), width, height, channels, 95, "stamp", error)) << error;
        std::ifstream direct_file(direct, std::ios::binary), atomic_file(atomic, std::ios::binary);
        EXPECT_EQ((std::vector<std::uint8_t>{std::istreambuf_iterator<char>(direct_file), {}}),
                  (std::vector<std::uint8_t>{std::istreambuf_iterator<char>(atomic_file), {}}));
        codec::Image decoded;
        ASSERT_TRUE(codec::decode(atomic, decoded, error)) << error;
        EXPECT_EQ(decoded.data, pixels);
    }
    TEST(ProductionImageOutput, InvalidJpegAndUnsupportedFormatsDoNotTruncateExistingFiles) {
        TemporaryDirectory temporary;
        const std::array<std::uint8_t, 4> pixels{10, 20, 30, 0};
        for (const std::string extension : {".jpg", ".exr", ".unknown"}) {
            const auto path = temporary.path / ("existing" + extension);
            {
                std::ofstream file(path, std::ios::binary);
                file << "keep";
            }
            std::string error;
            EXPECT_FALSE(codec::write_image_u8(path, pixels.data(), JPEG_MAX_DIMENSION + 1, 1, 4, 95, {}, error));
            EXPECT_FALSE(error.empty());
            std::ifstream file(path, std::ios::binary);
            EXPECT_EQ((std::string{std::istreambuf_iterator<char>(file), {}}), "keep");
        }
    }
    TEST(ProductionPng, AllChannelsAndDepthsRoundTripWithCommentsAndUnicodePaths) {
        TemporaryDirectory temporary;
        for (int channels : {1, 2, 3, 4}) {
            for (int depth : {8, 16}) {
                for (int level : {0, 1, 6, 8, 9}) {
                    SCOPED_TRACE(::testing::Message() << channels << " channels depth=" << depth << " level=" << level);
                    constexpr int width = 19, height = 17;
                    const auto samples = std::size_t(width) * height * channels;
                    std::vector<std::uint8_t> bytes(samples * (depth / 8));
                    if (depth == 8) {
                        for (std::size_t i = 0; i < samples; ++i)
                            bytes[i] = static_cast<std::uint8_t>(i * 17);
                    } else {
                        for (std::size_t i = 0; i < samples; ++i) {
                            const auto value = static_cast<std::uint16_t>(i * 257 + 13);
                            std::memcpy(bytes.data() + i * 2, &value, 2);
                        }
                    }
                    const auto path = temporary.path / std::filesystem::path(u8"immagine-測試.png");
                    std::string error;
                    ASSERT_TRUE(codec::write_png(path, bytes.data(), width, height, channels, depth, level, "codec regression", error)) << error;
                    EXPECT_EQ(read_png_comment(path), "codec regression");
                    codec::Probe probe;
                    ASSERT_TRUE(codec::probe(path, probe, error)) << error;
                    EXPECT_EQ(probe.width, width);
                    EXPECT_EQ(probe.height, height);
                    EXPECT_EQ(probe.channels, channels);
                    codec::Image decoded;
                    ASSERT_TRUE(codec::decode(path, decoded, error)) << error;
                    EXPECT_EQ(decoded.channels, channels);
                    EXPECT_EQ(decoded.sample_type, depth == 8 ? codec::SampleType::UInt8 : codec::SampleType::UInt16);
                    EXPECT_EQ(decoded.data, bytes);
                }
            }
        }
    }
    TEST(ProductionPng, ConcurrentReadsAndCorruptedPayload) {
        TemporaryDirectory temporary;
        const auto path = temporary.path / "rgba.png";
        std::vector<std::uint8_t> bytes(std::size_t{257} * 129 * 4);
        for (std::size_t i = 0; i < bytes.size(); ++i)
            bytes[i] = static_cast<std::uint8_t>(i * 17);
        std::string error;
        ASSERT_TRUE(codec::write_png(path, bytes.data(), 257, 129, 4, 8, 6, std::nullopt, error)) << error;
        std::vector<std::future<codec::Image>> jobs;
        jobs.reserve(8);
        for (int i = 0; i < 8; ++i) {
            jobs.emplace_back(std::async(std::launch::async, [&] {
                codec::Image image;
                std::string detail;
                if (!codec::decode(path, image, detail))
                    throw std::runtime_error(detail);
                return image;
            }));
        }
        for (auto& job : jobs)
            EXPECT_EQ(job.get().data, bytes);
        std::ifstream input(path, std::ios::binary);
        std::vector<std::uint8_t> encoded{std::istreambuf_iterator<char>(input), {}};
        encoded[encoded.size() / 2] ^= 0xff;
        codec::Image image;
        EXPECT_FALSE(codec::decode_memory(encoded.data(), encoded.size(), image, error));
        EXPECT_FALSE(error.empty());
        EXPECT_FALSE(codec::decode_memory(encoded.data(), 8, image, error));
    }
    TEST(ProductionJpeg, WriterReaderAndUnicodePaths) {
        TemporaryDirectory temporary;
        for (int channels : {1, 3}) {
            const int width = 19, height = 17;
            std::vector<std::uint8_t> bytes(std::size_t(width) * height * channels, 128);
            for (int quality : {1, 90, 95, 100}) {
                const auto path = temporary.path / std::filesystem::path(u8"frame-測試.jpg");
                std::string error;
                ASSERT_TRUE(codec::write_jpeg(path, bytes.data(), width, height, channels, quality, "evaluation", error)) << error;
                codec::Image decoded;
                ASSERT_TRUE(codec::decode(path, decoded, error)) << error;
                EXPECT_EQ(decoded.width, width);
                EXPECT_EQ(decoded.height, height);
                EXPECT_EQ(decoded.channels, channels);
                EXPECT_EQ(decoded.data, bytes);
            }
        }
    }
    TEST(ProductionPng, CorruptedDirectTargetsReportErrorsWithoutLeaking) {
        TemporaryDirectory temporary;
        const auto path = temporary.path / "rgb.png";
        std::vector<std::uint8_t> pixels(std::size_t{257} * 129 * 3, 128);
        std::string error;
        ASSERT_TRUE(codec::write_png(path, pixels.data(), 257, 129, 3, 8, 6, std::nullopt, error)) << error;
        std::ifstream input(path, std::ios::binary);
        std::vector<std::uint8_t> encoded{std::istreambuf_iterator<char>(input), {}};
        encoded[encoded.size() / 2] ^= 0xff;
        {
            std::ofstream output(path, std::ios::binary);
            output.write(reinterpret_cast<const char*>(encoded.data()), static_cast<std::streamsize>(encoded.size()));
        }
        for (bool memory : {false, true}) {
            std::vector<std::uint8_t> buffer;
            codec::DecodeTarget target{3, codec::SampleType::UInt8};
            target.allocate = [](std::size_t size, void* user) -> void* {
                auto& bytes = *static_cast<std::vector<std::uint8_t>*>(user);
                bytes.resize(size);
                return bytes.data();
            };
            target.user = &buffer;
            codec::Probe probe;
            const bool success = memory ? codec::decode_memory_to_buffer(encoded.data(), encoded.size(), target, probe, error)
                                        : codec::decode_to_buffer(path, target, probe, error);
            EXPECT_FALSE(success);
            EXPECT_FALSE(error.empty());
        }
    }
    TEST(ProductionJpeg, InvalidInputAndUnwritablePathReportErrors) {
        TemporaryDirectory temporary;
        std::vector<std::uint8_t> rgb(std::size_t{19} * 17 * 3, 128);
        std::string error;
        EXPECT_FALSE(codec::write_jpeg(temporary.path / "invalid.jpg", nullptr, 19, 17, 3, 90, std::nullopt, error));
        EXPECT_FALSE(error.empty());
        EXPECT_FALSE(codec::write_jpeg(temporary.path / "invalid.jpg", rgb.data(), -1, 17, 3, 90, std::nullopt, error));
        EXPECT_FALSE(codec::write_jpeg(temporary.path / "invalid.jpg", rgb.data(), 19, 17, 4, 90, std::nullopt, error));
        EXPECT_FALSE(codec::write_jpeg(temporary.path / "invalid.jpg", rgb.data(), JPEG_MAX_DIMENSION + 1, 17, 3, 90, std::nullopt, error));
        EXPECT_FALSE(codec::write_jpeg(temporary.path / "missing" / "frame.jpg", rgb.data(), 19, 17, 3, 90, std::nullopt, error));
        EXPECT_FALSE(error.empty());
    }
    TEST(ProductionJpeg, ExplicitFullChromaPreservesSamplingAndOddDimensions) {
        TemporaryDirectory temporary;
        for (int quality : {1, 90, 91, 95, 100}) {
            for (bool full_chroma : {false, true}) {
                std::vector<std::uint8_t> rgb(std::size_t{19} * 17 * 3);
                for (std::size_t i = 0; i < rgb.size(); ++i)
                    rgb[i] = static_cast<std::uint8_t>(i * 17);
                const auto path = temporary.path / "sampling.jpg";
                std::string error;
                ASSERT_TRUE(codec::write_jpeg(path, rgb.data(), 19, 17, 3, quality, std::nullopt, error, full_chroma)) << error;
                std::ifstream file(path, std::ios::binary);
                std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>(file), {}};
                jpeg_decompress_struct info{};
                jpeg_error_mgr jpeg_error{};
                info.err = jpeg_std_error(&jpeg_error);
                jpeg_create_decompress(&info);
                jpeg_mem_src(&info, bytes.data(), static_cast<unsigned long>(bytes.size()));
                EXPECT_EQ(jpeg_read_header(&info, TRUE), JPEG_HEADER_OK);
                EXPECT_EQ(info.image_width, 19u);
                EXPECT_EQ(info.image_height, 17u);
                EXPECT_EQ(info.comp_info[0].h_samp_factor, full_chroma ? 1 : 2);
                EXPECT_EQ(info.comp_info[0].v_samp_factor, full_chroma ? 1 : 2);
                jpeg_destroy_decompress(&info);
            }
        }
    }
    TEST(ProductionJpeg, ConcurrentEncodersHaveIndependentState) {
        TemporaryDirectory temporary;
        std::vector<std::uint8_t> rgb(std::size_t{19} * 17 * 3, 128);
        std::vector<std::future<bool>> jobs;
        jobs.reserve(8);
        for (int i = 0; i < 8; ++i) {
            jobs.emplace_back(std::async(std::launch::async, [&, i] {
                const auto path = temporary.path / (std::to_string(i) + ".jpg");
                std::string error;
                codec::Image decoded;
                return codec::write_jpeg(path, rgb.data(), 19, 17, 3, 95, std::nullopt, error, true) &&
                       codec::decode(path, decoded, error) && decoded.data == rgb;
            }));
        }
        for (auto& job : jobs)
            EXPECT_TRUE(job.get());
    }
#ifndef _WIN32
    TEST(ProductionPng, OutputFailuresReportErrorsWithoutLeaking) {
        if (!std::filesystem::exists("/dev/full"))
            GTEST_SKIP() << "/dev/full is unavailable";
        for (int dimension : {19, 257}) {
            std::vector<std::uint8_t> rgb(static_cast<std::size_t>(dimension) * dimension * 3);
            std::uint32_t state = 2710;
            for (auto& byte : rgb) {
                state = state * 1664525u + 1013904223u;
                byte = static_cast<std::uint8_t>(state >> 24);
            }
            std::string error;
            EXPECT_FALSE(codec::write_png("/dev/full", rgb.data(), dimension, dimension, 3, 8, 6, std::nullopt, error));
            EXPECT_FALSE(error.empty());
        }
    }
    TEST(ProductionJpeg, BufferedOutputFailuresReportErrors) {
        if (!std::filesystem::exists("/dev/full"))
            GTEST_SKIP() << "/dev/full is unavailable";
        std::vector<std::uint8_t> rgb(std::size_t{19} * 17 * 3, 128);
        std::string error;
        EXPECT_FALSE(codec::write_jpeg("/dev/full", rgb.data(), 19, 17, 3, 95, std::nullopt, error, true));
        EXPECT_FALSE(error.empty());
    }
#endif
} // namespace
