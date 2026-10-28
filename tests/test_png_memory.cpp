/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/image_codecs.hpp"
#include "core/native_fs.hpp"

#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <future>
#include <gtest/gtest.h>
#include <limits>
#include <stb_image.h>
#include <zlib.h>

namespace {
    namespace codec = lfs::core::image_codecs;

    struct TemporaryPng {
        std::filesystem::path path = std::filesystem::temp_directory_path() /
                                     ("png-memory-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".png");
        ~TemporaryPng() {
            std::error_code error;
            lfs::core::native_fs::remove(path, error);
        }
    };

    std::vector<uint8_t> pixels(int width, int height, int channels, int bits) {
        std::vector<uint8_t> result(size_t(width) * height * channels * (bits / 8));
        uint32_t random = 42;
        for (size_t i = 0; i < result.size(); i += bits / 8) {
            random = random * 1664525u + 1013904223u;
            if (bits == 8)
                result[i] = uint8_t(random >> 24);
            else {
                constexpr std::array<uint16_t, 8> boundaries{0, 1, 255, 256, 257, 32767, 32768, 65535};
                const uint16_t value = i < 16 ? boundaries[i / 2] : uint16_t(random >> 8);
                std::memcpy(result.data() + i, &value, 2);
            }
        }
        return result;
    }

    void check_decode(const std::vector<uint8_t>& encoded, const std::vector<uint8_t>& expected,
                      int width, int height, int channels, int bits) {
        codec::Image image;
        std::string error;
        ASSERT_TRUE(codec::decode_memory(encoded.data(), encoded.size(), image, error)) << error;
        EXPECT_EQ(image.width, width);
        EXPECT_EQ(image.height, height);
        EXPECT_EQ(image.channels, channels);
        EXPECT_EQ(image.sample_type, bits == 8 ? codec::SampleType::UInt8 : codec::SampleType::UInt16);
        EXPECT_EQ(image.data, expected);
        ASSERT_GT(encoded.size(), 26);
        EXPECT_EQ(encoded[24], bits);
        int w = 0, h = 0, c = 0;
        void* independent = bits == 8
                                ? static_cast<void*>(stbi_load_from_memory(encoded.data(), int(encoded.size()), &w, &h, &c, 0))
                                : static_cast<void*>(stbi_load_16_from_memory(encoded.data(), int(encoded.size()), &w, &h, &c, 0));
        ASSERT_NE(independent, nullptr) << stbi_failure_reason();
        EXPECT_EQ(w, width);
        EXPECT_EQ(h, height);
        EXPECT_EQ(c, channels);
        if (w == width && h == height && c == channels)
            EXPECT_EQ(std::memcmp(independent, expected.data(), expected.size()), 0);
        stbi_image_free(independent);
    }

    TEST(PngMemory, PreservesNativeSamplesAlphaStrideAndFileBytes) {
        TemporaryPng temporary;
        for (int bits : {8, 16})
            for (int channels : {1, 2, 3, 4})
                for (int level : {-1, 0, 1, 6, 9, 10}) {
                    SCOPED_TRACE(::testing::Message() << bits << " bits, " << channels << " channels, level " << level);
                    constexpr int width = 19, height = 17;
                    const auto packed = pixels(width, height, channels, bits);
                    const size_t row = size_t(width) * channels * (bits / 8), stride = row + 7;
                    std::vector<uint8_t> storage(1 + stride * (height - 1) + row, 0xcd);
                    for (int y = 0; y < height; ++y)
                        std::memcpy(storage.data() + 1 + y * stride, packed.data() + y * row, row);
                    const auto original = storage;
                    codec::PngEncodeOptions options;
                    options.compression_level = level;
                    options.comment = "native samples, straight alpha";
                    std::vector<uint8_t> encoded{42};
                    ASSERT_TRUE(codec::encode_png_memory(std::span<const uint8_t>(storage).subspan(1), width, height, channels, bits, stride, encoded, options));
                    EXPECT_EQ(storage, original);
                    std::string error;
                    ASSERT_TRUE(codec::write_png(temporary.path, packed.data(), width, height, channels, bits, level, options.comment, error)) << error;
                    std::ifstream file(temporary.path, std::ios::binary);
                    EXPECT_EQ(encoded, (std::vector<uint8_t>{std::istreambuf_iterator<char>(file), {}}));
                    codec::Image from_file;
                    ASSERT_TRUE(codec::decode(temporary.path, from_file, error)) << error;
                    EXPECT_EQ(from_file.data, packed);
                    check_decode(encoded, packed, width, height, channels, bits);
                }
    }

    TEST(PngMemory, LimitsValidationAliasingAndReusePreservePreviousOutput) {
        for (int bits : {8, 16}) {
            auto source = pixels(31, 29, 4, bits);
            const auto original = source;
            const size_t stride = size_t(31) * 4 * (bits / 8);
            std::vector<uint8_t> reference;
            ASSERT_TRUE(codec::encode_png_memory(source, 31, 29, 4, bits, stride, reference));
            for (size_t limit : {size_t{1}, size_t{33}, size_t{4095}, size_t{4096}, reference.size() - 1, reference.size(), reference.size() + 1}) {
                codec::PngEncodeOptions options;
                options.max_output_bytes = limit;
                std::vector<uint8_t> output{7, 8, 9};
                const auto result = codec::encode_png_memory(source, 31, 29, 4, bits, stride, output, options);
                ASSERT_EQ(bool(result), limit >= reference.size());
                if (result)
                    EXPECT_EQ(output, reference);
                else {
                    EXPECT_EQ(result.error().code(), lfs::ErrorCode::ResourceExhausted);
                    EXPECT_EQ(output, (std::vector<uint8_t>{7, 8, 9}));
                }
            }
            codec::PngEncodeOptions options;
            options.max_output_bytes = 1;
            EXPECT_FALSE(codec::encode_png_memory(source, 31, 29, 4, bits, stride, source, options));
            EXPECT_EQ(source, original);
            ASSERT_TRUE(codec::encode_png_memory(source, 31, 29, 4, bits, stride, source));
            EXPECT_EQ(source, reference);
        }
        const auto source = pixels(3, 2, 3, 16);
        std::vector<uint8_t> output{7, 8, 9};
        const auto invalid = [&](std::span<const uint8_t> bytes, int w, int h, int c, int b, size_t stride, codec::PngEncodeOptions options = {}) {
            const auto result = codec::encode_png_memory(bytes, w, h, c, b, stride, output, options);
            ASSERT_FALSE(result);
            EXPECT_EQ(result.error().code(), lfs::ErrorCode::InvalidArgument);
            const auto detail = result.error().detail();
            for (const auto& field : {
                     "width=" + std::to_string(w) + ",",
                     "height=" + std::to_string(h) + ",",
                     "channels=" + std::to_string(c) + ",",
                     "depth=" + std::to_string(b) + ",",
                     "stride=" + std::to_string(stride) + ",",
                     "span_bytes=" + std::to_string(bytes.size()) + ",",
                     "limit=" + std::to_string(options.max_output_bytes) + ")"})
                EXPECT_NE(detail.find(field), std::string_view::npos) << detail;
            EXPECT_EQ(output, (std::vector<uint8_t>{7, 8, 9}));
        };
        invalid({}, 3, 2, 3, 16, 18);
        invalid(source, 0, 2, 3, 16, 18);
        invalid(source, 3, -1, 3, 16, 18);
        invalid(source, 3, 2, 0, 16, 18);
        invalid(source, 3, 2, 5, 16, 18);
        invalid(source, 3, 2, 3, 32, 18);
        invalid(source, 3, 2, 3, 16, 0);
        invalid(source, 3, 2, 3, 16, 17);
        invalid(std::span(source).first(source.size() - 1), 3, 2, 3, 16, 18);
        invalid(source, 3, 2, 3, 16, std::numeric_limits<size_t>::max());
        codec::PngEncodeOptions options;
        options.max_output_bytes = 0;
        invalid(source, 3, 2, 3, 16, 18, options);
        options.max_output_bytes = std::numeric_limits<size_t>::max();
        invalid(source, 3, 2, 3, 16, 18, options);
        options = {};
        options.comment = std::string("a\0b", 3);
        invalid(source, 3, 2, 3, 16, 18, options);
        ASSERT_TRUE(codec::encode_png_memory(std::span(source).first(18), 3, 1, 3, 16, std::numeric_limits<size_t>::max(), output));
    }

    TEST(PngMemory, ParallelStripesMatchFileBytesAndKeepInputUnmodified) {
        TemporaryPng temporary;
        constexpr int width = 1025, height = 3073, channels = 4;
        const auto packed = pixels(width, height, channels, 8);
        const size_t row = size_t(width) * channels, stride = row + 3;
        std::vector<uint8_t> padded(stride * (height - 1) + row, 0xcd);
        for (int y = 0; y < height; ++y)
            std::memcpy(padded.data() + y * stride, packed.data() + y * row, row);
        const auto original = padded;
        for (int level : {0, 6}) {
            codec::PngEncodeOptions options;
            options.compression_level = level;
            options.comment = "parallel stripes";
            std::vector<uint8_t> encoded;
            ASSERT_TRUE(codec::encode_png_memory(padded, width, height, channels, 8, stride, encoded, options));
            EXPECT_EQ(padded, original);
            std::string error;
            ASSERT_TRUE(codec::write_png(temporary.path, packed.data(), width, height, channels, 8, level, options.comment, error)) << error;
            std::ifstream file(temporary.path, std::ios::binary);
            EXPECT_EQ(encoded, (std::vector<uint8_t>{std::istreambuf_iterator<char>(file), {}}));
            check_decode(encoded, packed, width, height, channels, 8);
            for (size_t limit : {size_t{1}, encoded.size() - 1, encoded.size()}) {
                options.max_output_bytes = limit;
                std::vector<uint8_t> output{7};
                const auto result = codec::encode_png_memory(padded, width, height, channels, 8, stride, output, options);
                ASSERT_EQ(bool(result), limit >= encoded.size());
                if (result)
                    EXPECT_EQ(output, encoded);
                else {
                    EXPECT_EQ(result.error().code(), lfs::ErrorCode::ResourceExhausted);
                    EXPECT_EQ(output, (std::vector<uint8_t>{7}));
                }
            }
        }
    }

    TEST(PngMemory, EmbeddedXmpStreamsThroughSerialAndParallelWithoutChangingImageBytes) {
        const std::string packet =
            "<?xpacket begin=\"\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>"
            "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"><rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">"
            "<rdf:Description rdf:about=\"\" xmlns:t=\"urn:lfs:test\"><t:value>owned &amp; static</t:value>"
            "</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end=\"r\"?>";
        const auto read = [](const std::filesystem::path& path) {
            std::ifstream input(path, std::ios::binary);
            return std::vector<uint8_t>{std::istreambuf_iterator<char>(input), {}};
        };
        for (const bool parallel : {false, true}) {
            SCOPED_TRACE(parallel);
            TemporaryPng temporary;
            const int width = parallel ? 1025 : 19, height = parallel ? 3073 : 17, channels = parallel ? 3 : 4;
            const auto source = pixels(width, height, channels, 8);
            codec::PngEncodeOptions encoding;
            encoding.comment = "keep the existing comment";
            std::vector<uint8_t> plain;
            ASSERT_TRUE(codec::encode_png_memory(source, width, height, channels, 8, size_t(width) * channels, plain, encoding));
            lfs::core::AtomicFileOptions options{.overwrite = true, .durable = false, .create_directories = false};
            const std::array parts{std::string_view(packet).substr(0, 13), std::string_view{}, std::string_view(packet).substr(13)};
            ASSERT_TRUE(codec::write_image_u8(temporary.path, source.data(), width, height, channels, 90, encoding.comment, options, false, parts));
            const auto encoded = read(temporary.path);
            ASSERT_GE(encoded.size(), 8);
            std::vector<uint8_t> stripped(encoded.begin(), encoded.begin() + 8);
            size_t chunks = 0;
            const auto u32 = [&](size_t at) { return (uint32_t(encoded[at]) << 24) | (uint32_t(encoded[at + 1]) << 16) | (uint32_t(encoded[at + 2]) << 8) | encoded[at + 3]; };
            for (size_t offset = 8; offset < encoded.size();) {
                ASSERT_GE(encoded.size() - offset, 12);
                const auto size = u32(offset);
                ASSERT_LE(size, encoded.size() - offset - 12);
                EXPECT_EQ(u32(offset + 8 + size), crc32_z(0, encoded.data() + offset + 4, size_t(size) + 4));
                if (std::memcmp(encoded.data() + offset + 4, "iTXt", 4) == 0) {
                    ++chunks;
                    constexpr char prefix[] = "XML:com.adobe.xmp\0\0\0\0";
                    ASSERT_EQ(size, sizeof(prefix) + packet.size());
                    EXPECT_EQ(std::memcmp(encoded.data() + offset + 8, prefix, sizeof(prefix)), 0);
                    EXPECT_EQ(std::string_view(reinterpret_cast<const char*>(encoded.data() + offset + 8 + sizeof(prefix)), packet.size()), packet);
                } else {
                    stripped.insert(stripped.end(), encoded.begin() + offset, encoded.begin() + offset + 12 + size);
                }
                offset += 12 + size;
            }
            EXPECT_EQ(chunks, 1);
            EXPECT_EQ(stripped, plain);
            check_decode(encoded, source, width, height, channels, 8);
            const std::string bad_packet("invalid\0XMP", 11);
            const std::array bad_parts{std::string_view(bad_packet)};
            const auto invalid = codec::write_image_u8(temporary.path, source.data(), width, height, channels, 90, encoding.comment, options, false, bad_parts);
            ASSERT_FALSE(invalid);
            EXPECT_EQ(invalid.error().code(), lfs::ErrorCode::InvalidArgument);
            EXPECT_EQ(read(temporary.path), encoded);
            options.overwrite = false;
            const auto collision = codec::write_image_u8(temporary.path, source.data(), width, height, channels, 90, encoding.comment, options, false, parts);
            ASSERT_FALSE(collision);
            EXPECT_EQ(collision.error().code(), lfs::ErrorCode::AlreadyExists);
            EXPECT_EQ(read(temporary.path), encoded);
            options.overwrite = true;
            // Cancel after encoding, before the atomic service publishes bytes.
            options.cancelled = [calls = 0]() mutable { return ++calls > 1; };
            const auto cancelled = codec::write_image_u8(temporary.path, source.data(), width, height, channels, 90, encoding.comment, options, false, parts);
            ASSERT_FALSE(cancelled);
            EXPECT_EQ(cancelled.error().code(), lfs::ErrorCode::Cancelled);
            EXPECT_EQ(read(temporary.path), encoded);
            if (!parallel) {
                // A valid packet larger than libpng's default text allocation
                // budget must still leave pixel decoding usable.
                auto large_packet = packet;
                large_packet.insert(large_packet.find("owned"), 9 * 1024 * 1024, 'a');
                options.cancelled = {};
                const std::array large_parts{std::string_view(large_packet)};
                ASSERT_TRUE(codec::write_image_u8(temporary.path, source.data(), width, height, channels, 90, encoding.comment, options, false, large_parts));
                check_decode(read(temporary.path), source, width, height, channels, 8);
            }
        }
    }

    TEST(PngMemory, EmbeddedXmpRejectsOtherFormatsAndOversizedPacketsBeforePublication) {
        TemporaryPng temporary;
        const auto source = pixels(1, 1, 3, 8);
        const std::vector<uint8_t> sentinel{1, 2, 3, 4};
        lfs::core::AtomicFileOptions options{.overwrite = true, .durable = false, .create_directories = false};
        const auto read = [&] {
            std::ifstream input(temporary.path, std::ios::binary);
            return std::vector<uint8_t>{std::istreambuf_iterator<char>(input), {}};
        };
        const auto rejected = [&](std::span<const std::string_view> parts) {
            const auto result = codec::write_image_u8(temporary.path, source.data(), 1, 1, 3, 90, {}, options, false, parts);
            ASSERT_FALSE(result);
            EXPECT_EQ(result.error().code(), lfs::ErrorCode::InvalidArgument);
        };
        temporary.path.replace_extension(".jpg");
        const std::array packet{std::string_view("<xmp/>")};
        rejected(packet);
        EXPECT_FALSE(std::filesystem::exists(temporary.path));
        {
            std::ofstream output(temporary.path, std::ios::binary);
            output.write(reinterpret_cast<const char*>(sentinel.data()), sentinel.size());
            ASSERT_TRUE(output);
        }
        rejected(packet);
        EXPECT_EQ(read(), sentinel);
        std::filesystem::remove(temporary.path);
        temporary.path.replace_extension(".png");
        // Reuse valid backing storage rather than constructing an out-of-bounds
        // string_view or allocating a multi-gigabyte packet. The combined size
        // exceeds the PNG 31-bit chunk limit before any image is published.
        const std::string fragment(1024 * 1024, 'a');
        const std::vector<std::string_view> oversized(2048, fragment);
        rejected(oversized);
        EXPECT_FALSE(std::filesystem::exists(temporary.path));
        {
            std::ofstream output(temporary.path, std::ios::binary);
            output.write(reinterpret_cast<const char*>(sentinel.data()), sentinel.size());
            ASSERT_TRUE(output);
        }
        rejected(oversized);
        EXPECT_EQ(read(), sentinel);
    }

    TEST(PngMemory, ConcurrentEncodersOwnIndependentState) {
        std::vector<std::future<bool>> jobs;
        for (int i = 0; i < 8; ++i)
            jobs.emplace_back(std::async(std::launch::async, [i] {
                const int bits = i & 1 ? 16 : 8, channels = 1 + i % 4;
                const auto source = pixels(19, 17, channels, bits);
                std::vector<uint8_t> encoded;
                codec::Image image;
                std::string error;
                return codec::encode_png_memory(source, 19, 17, channels, bits, size_t(19) * channels * (bits / 8), encoded) &&
                       codec::decode_memory(encoded.data(), encoded.size(), image, error) && image.data == source;
            }));
        for (auto& job : jobs)
            EXPECT_TRUE(job.get());
    }

    TEST(PngMemory, DeterministicLayoutsRoundTripAndRespectExactLimits) {
        uint32_t random = 0x504e47;
        const auto next = [&] {
            random = random * 1664525u + 1013904223u;
            return random >> 16;
        };
        for (int iteration = 0; iteration < 256; ++iteration) {
            SCOPED_TRACE(iteration);
            const int width = 1 + next() % 31, height = 1 + next() % 29;
            const int channels = 1 + next() % 4, bits = next() & 1 ? 8 : 16;
            const auto source = pixels(width, height, channels, bits);
            const auto row = size_t(width) * channels * (bits / 8), stride = row + next() % 13;
            std::vector<uint8_t> storage(1 + stride * (height - 1) + row, 0xcd);
            for (int y = 0; y < height; ++y)
                std::memcpy(storage.data() + 1 + y * stride, source.data() + y * row, row);
            const auto original = storage;
            codec::PngEncodeOptions options;
            options.compression_level = int(next() % 12) - 1;
            if (iteration % 3 == 0)
                options.comment = std::string(next() % 129, char(1 + next() % 255));
            const auto input = std::span<const uint8_t>(storage).subspan(1);
            std::vector<uint8_t> reference;
            ASSERT_TRUE(codec::encode_png_memory(input, width, height, channels, bits, stride, reference, options));
            check_decode(reference, source, width, height, channels, bits);
            options.max_output_bytes = reference.size() - 1;
            std::vector<uint8_t> output{7, 8, 9};
            const auto limited = codec::encode_png_memory(input, width, height, channels, bits, stride, output, options);
            ASSERT_FALSE(limited);
            EXPECT_EQ(limited.error().code(), lfs::ErrorCode::ResourceExhausted);
            EXPECT_EQ(output, (std::vector<uint8_t>{7, 8, 9}));
            options.max_output_bytes = reference.size();
            ASSERT_TRUE(codec::encode_png_memory(input, width, height, channels, bits, stride, output, options));
            EXPECT_EQ(output, reference);
            EXPECT_EQ(storage, original);
        }
    }

    TEST(PngMemory, FileAndMemoryRgbTargetsRespectBorrowedBuffersAndAllocators) {
        TemporaryPng temporary;
        for (int bits : {8, 16}) {
            constexpr int width = 19, height = 17;
            const auto source = pixels(width, height, 3, bits);
            std::vector<uint8_t> encoded;
            ASSERT_TRUE(codec::encode_png_memory(source, width, height, 3, bits, size_t(width) * 3 * (bits / 8), encoded));
            std::string error;
            ASSERT_TRUE(codec::write_png(temporary.path, source.data(), width, height, 3, bits, 6, {}, error)) << error;
            for (bool file : {false, true})
                for (bool allocate : {false, true}) {
                    struct Buffer {
                        std::vector<uint8_t> bytes;
                        size_t requested = 0;
                        int calls = 0;
                    } buffer{std::vector<uint8_t>(source.size() + 16, 0xcd)};
                    codec::DecodeTarget target{3, bits == 8 ? codec::SampleType::UInt8 : codec::SampleType::UInt16};
                    if (allocate) {
                        target.user = &buffer;
                        target.allocate = [](size_t count, void* user) -> void* {
                            auto& buffer = *static_cast<Buffer*>(user);
                            ++buffer.calls;
                            buffer.requested = count;
                            return count <= buffer.bytes.size() - 16 ? buffer.bytes.data() + 8 : nullptr;
                        };
                    } else
                        target.data = buffer.bytes.data() + 8;
                    codec::Probe info;
                    ASSERT_TRUE(file ? codec::decode_to_buffer(temporary.path, target, info, error)
                                     : codec::decode_memory_to_buffer(encoded.data(), encoded.size(), target, info, error))
                        << error;
                    // The old file UINT16 shortcut replaced the caller's buffer
                    // with an STB allocation and never invoked its allocator.
                    EXPECT_EQ(target.data, buffer.bytes.data() + 8);
                    EXPECT_EQ(buffer.calls, allocate ? 1 : 0);
                    if (allocate)
                        EXPECT_EQ(buffer.requested, source.size());
                    EXPECT_EQ(std::memcmp(buffer.bytes.data() + 8, source.data(), source.size()), 0);
                    for (size_t i = 0; i < 8; ++i) {
                        EXPECT_EQ(buffer.bytes[i], 0xcd);
                        EXPECT_EQ(buffer.bytes[source.size() + 8 + i], 0xcd);
                    }
                    EXPECT_EQ(info.width, width);
                    EXPECT_EQ(info.height, height);
                    EXPECT_EQ(info.channels, 3);
                }
        }
    }

    TEST(PngMemory, DirectRgbTargetsKeepExistingGrayAndAlphaPreparationPolicy) {
        TemporaryPng temporary;
        for (int bits : {8, 16})
            for (int channels : {1, 2, 4}) {
                constexpr int width = 3, height = 2;
                const size_t sample = bits / 8;
                const auto source = pixels(width, height, channels, bits);
                std::vector<uint8_t> encoded;
                ASSERT_TRUE(codec::encode_png_memory(source, width, height, channels, bits, size_t(width) * channels * sample, encoded));
                std::string error;
                ASSERT_TRUE(codec::write_png(temporary.path, source.data(), width, height, channels, bits, 6, {}, error)) << error;
                std::vector<uint8_t> expected(size_t(width) * height * 3 * sample);
                if (channels != 2)
                    for (size_t pixel = 0; pixel < size_t(width) * height; ++pixel)
                        for (size_t c = 0; c < 3; ++c)
                            std::memcpy(expected.data() + (pixel * 3 + c) * sample,
                                        source.data() + (pixel * channels + (channels == 1 ? 0 : c)) * sample, sample);
                for (bool file : {false, true}) {
                    std::vector<uint8_t> output(expected.size() + 2, 0xcd);
                    const auto original = output;
                    codec::DecodeTarget target{3, bits == 8 ? codec::SampleType::UInt8 : codec::SampleType::UInt16, output.data() + 1};
                    codec::Probe info;
                    const bool success = file ? codec::decode_to_buffer(temporary.path, target, info, error)
                                              : codec::decode_memory_to_buffer(encoded.data(), encoded.size(), target, info, error);
                    // Gray-alpha remains unsupported in the RGB-only direct
                    // contract; owning decode still preserves both channels.
                    ASSERT_EQ(success, channels != 2);
                    if (success)
                        EXPECT_EQ(std::memcmp(output.data() + 1, expected.data(), expected.size()), 0);
                    else
                        EXPECT_EQ(output, original);
                    EXPECT_EQ(output.front(), 0xcd);
                    EXPECT_EQ(output.back(), 0xcd);
                    EXPECT_EQ(target.data, output.data() + 1);
                }
            }
    }

    TEST(PngMemory, LargeImagesPreserveSamplesAndRecoverFromOutputLimits) {
        for (const auto layout : {std::array{3840, 2160, 4, 16}, std::array{7680, 4320, 1, 16}, std::array{3840, 2160, 3, 8}}) {
            const auto [width, height, channels, bits] = layout;
            SCOPED_TRACE(::testing::Message() << width << 'x' << height << ", " << channels << " channels, " << bits << " bits");
            const auto source = pixels(width, height, channels, bits);
            const auto stride = size_t(width) * channels * (bits / 8);
            std::vector<uint8_t> encoded;
            ASSERT_TRUE(codec::encode_png_memory(source, width, height, channels, bits, stride, encoded));
            check_decode(encoded, source, width, height, channels, bits);
            codec::PngEncodeOptions options;
            options.max_output_bytes = encoded.size() - 1;
            std::vector<uint8_t> output{7, 8, 9};
            const auto limited = codec::encode_png_memory(source, width, height, channels, bits, stride, output, options);
            ASSERT_FALSE(limited);
            EXPECT_EQ(limited.error().code(), lfs::ErrorCode::ResourceExhausted);
            EXPECT_EQ(output, (std::vector<uint8_t>{7, 8, 9}));
            const auto pixel_bytes = size_t(channels) * (bits / 8);
            ASSERT_TRUE(codec::encode_png_memory(std::span(source).first(pixel_bytes), 1, 1, channels, bits, pixel_bytes, output));
        }
    }
} // namespace
