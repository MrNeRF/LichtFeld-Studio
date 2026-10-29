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
#include <memory>
#include <tiffio.h>

namespace {
    namespace codec = lfs::core::image_codecs;
    struct TemporaryTiff {
        std::filesystem::path path = std::filesystem::temp_directory_path() /
                                     ("tiff-memory-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".tiff");
        ~TemporaryTiff() {
            std::error_code error;
            lfs::core::native_fs::remove(path, error);
        }
    };
    std::vector<uint8_t> read_bytes(const std::filesystem::path& path) {
        std::ifstream file(path, std::ios::binary);
        return {std::istreambuf_iterator<char>(file), {}};
    }
    void save_bytes(const std::filesystem::path& path, std::span<const uint8_t> bytes) {
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        ASSERT_TRUE(file);
    }
    std::vector<uint8_t> samples(int w, int h, int c, int bits) {
        std::vector<uint8_t> result(size_t(w) * h * c * (bits / 8));
        uint32_t random = 42;
        for (size_t i = 0; i < result.size(); i += bits / 8) {
            random = random * 1664525u + 1013904223u;
            if (bits == 8)
                result[i] = uint8_t(random >> 24);
            else {
                constexpr std::array<uint16_t, 8> edge{0, 1, 255, 256, 257, 32767, 32768, 65535};
                const uint16_t value = i < 16 ? edge[i / 2] : uint16_t(random >> 8);
                std::memcpy(result.data() + i, &value, 2);
            }
        }
        if (c == 2 || c == 4) {
            std::memset(result.data() + (c - 1) * (bits / 8), 0, bits / 8);
            if (w * h > 1)
                std::memset(result.data() + (2 * c - 1) * (bits / 8), 0xff, bits / 8);
        }
        return result;
    }
    void check_pixels(std::span<const uint8_t> encoded, const std::vector<uint8_t>& expected,
                      int w, int h, int c, int bits) {
        codec::Image image;
        std::string error;
        ASSERT_TRUE(codec::decode_memory(encoded.data(), encoded.size(), image, error)) << error;
        EXPECT_EQ(image.width, w);
        EXPECT_EQ(image.height, h);
        EXPECT_EQ(image.channels, c);
        EXPECT_EQ(image.sample_type, bits == 8 ? codec::SampleType::UInt8 : codec::SampleType::UInt16);
        EXPECT_EQ(image.data, expected);
        std::vector<uint8_t> output(expected.size() + 2, 0xcd);
        codec::DecodeTarget target{c, image.sample_type, output.data() + 1};
        codec::Probe info;
        ASSERT_TRUE(codec::decode_memory_to_buffer(encoded.data(), encoded.size(), target, info, error)) << error;
        EXPECT_EQ(target.data, output.data() + 1);
        EXPECT_EQ(std::memcmp(output.data() + 1, expected.data(), expected.size()), 0);
        EXPECT_EQ(output.front(), 0xcd);
        EXPECT_EQ(output.back(), 0xcd);
    }

    TEST(TiffMemory, MatchesExistingWritersAndPreservesNativeSamplesAndAlpha) {
        TemporaryTiff temporary;
        for (int bits : {8, 16})
            for (int c : {1, 2, 3, 4})
                for (const auto size : {std::pair{1, 1}, std::pair{19, 17}, std::pair{257, 29}}) {
                    const auto [w, h] = size;
                    SCOPED_TRACE(::testing::Message() << w << 'x' << h << " c=" << c << " bits=" << bits);
                    const auto source = samples(w, h, c, bits);
                    const size_t row = size_t(w) * c * (bits / 8), stride = row + 7;
                    std::vector<uint8_t> padded(1 + stride * (h - 1) + row, 0xcd);
                    for (int y = 0; y < h; ++y)
                        std::memcpy(padded.data() + 1 + y * stride, source.data() + y * row, row);
                    const auto original = padded;
                    std::vector<uint8_t> encoded;
                    ASSERT_TRUE(codec::encode_tiff_memory(std::span<const uint8_t>(padded).subspan(1), w, h, c, bits, stride, encoded));
                    EXPECT_EQ(padded, original);
                    std::string error;
                    if (bits == 8)
                        ASSERT_TRUE(codec::write_tiff(temporary.path, source.data(), w, h, c, error)) << error;
                    else
                        ASSERT_TRUE(codec::write_tiff_u16(temporary.path, source, w, h, c, row, {}, {.overwrite = true, .durable = false}));
                    EXPECT_EQ(encoded, read_bytes(temporary.path));
                    check_pixels(encoded, source, w, h, c, bits);
                    codec::Image image;
                    ASSERT_TRUE(codec::decode(temporary.path, image, error)) << error;
                    EXPECT_EQ(image.data, source);
                    std::unique_ptr<TIFF, decltype(&TIFFClose)> file(TIFFOpen(temporary.path.string().c_str(), "r"), TIFFClose);
                    ASSERT_NE(file, nullptr);
                    uint16_t compression = 0, orientation = 0, alpha_count = 0;
                    uint16_t* alpha = nullptr;
                    ASSERT_TRUE(TIFFGetField(file.get(), TIFFTAG_COMPRESSION, &compression));
                    ASSERT_TRUE(TIFFGetField(file.get(), TIFFTAG_ORIENTATION, &orientation));
                    EXPECT_EQ(compression, COMPRESSION_LZW);
                    EXPECT_EQ(orientation, ORIENTATION_TOPLEFT);
                    if (c == 2 || c == 4) {
                        ASSERT_TRUE(TIFFGetField(file.get(), TIFFTAG_EXTRASAMPLES, &alpha_count, &alpha));
                        ASSERT_EQ(alpha_count, 1);
                        EXPECT_EQ(alpha[0], EXTRASAMPLE_UNASSALPHA);
                    }
                }
    }

    TEST(TiffMemory, ExplicitXmpPreservesSamplesAndRejectsInvalidPacketsBeforePublication) {
        TemporaryTiff temporary;
        const auto source = samples(19, 17, 4, 16);
        const auto original = source;
        const std::string packet = "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"><value>escaped &amp; retained</value></x:xmpmeta>";
        lfs::core::AtomicFileOptions options{.overwrite = true, .durable = false, .create_directories = false};
        ASSERT_TRUE(codec::write_tiff_u16(temporary.path, source, 19, 17, 4, 19 * 8, {}, options, {}, packet));
        const auto encoded = read_bytes(temporary.path);
        {
            std::unique_ptr<TIFF, decltype(&TIFFClose)> file(TIFFOpen(temporary.path.string().c_str(), "r"), TIFFClose);
            ASSERT_TRUE(file);
            uint32_t count = 0;
            void* stored = nullptr;
            ASSERT_TRUE(TIFFGetField(file.get(), TIFFTAG_XMLPACKET, &count, &stored));
            ASSERT_NE(stored, nullptr);
            EXPECT_EQ(std::string_view(static_cast<const char*>(stored), count), packet);
            uint16_t bits = 0, compression = 0, orientation = 0;
            ASSERT_TRUE(TIFFGetField(file.get(), TIFFTAG_BITSPERSAMPLE, &bits));
            ASSERT_TRUE(TIFFGetField(file.get(), TIFFTAG_COMPRESSION, &compression));
            ASSERT_TRUE(TIFFGetField(file.get(), TIFFTAG_ORIENTATION, &orientation));
            EXPECT_EQ(bits, 16);
            EXPECT_EQ(compression, COMPRESSION_LZW);
            EXPECT_EQ(orientation, ORIENTATION_TOPLEFT);
        }
        codec::Image decoded;
        std::string error;
        ASSERT_TRUE(codec::decode_memory(encoded.data(), encoded.size(), decoded, error)) << error;
        EXPECT_EQ(decoded.data, source);
        EXPECT_EQ(source, original);
        for (const auto& invalid : {std::string("bad\0packet", 10), std::string(32 * 1024 * 1024 + 1, 'a')}) {
            const auto refused = codec::write_tiff_u16(temporary.path, source, 19, 17, 4, 19 * 8, {}, options, {}, invalid);
            ASSERT_FALSE(refused);
            EXPECT_EQ(refused.error().code(), lfs::ErrorCode::InvalidArgument);
            EXPECT_EQ(read_bytes(temporary.path), encoded);
        }
        options.overwrite = false;
        const auto collision = codec::write_tiff_u16(temporary.path, source, 19, 17, 4, 19 * 8, {}, options, {}, packet);
        ASSERT_FALSE(collision);
        EXPECT_EQ(collision.error().code(), lfs::ErrorCode::AlreadyExists);
        EXPECT_EQ(read_bytes(temporary.path), encoded);
        options.overwrite = true;
        options.cancelled = [calls = 0]() mutable { return ++calls > 1; };
        const auto cancelled = codec::write_tiff_u16(temporary.path, source, 19, 17, 4, 19 * 8, {}, options, {}, packet);
        ASSERT_FALSE(cancelled);
        EXPECT_EQ(cancelled.error().code(), lfs::ErrorCode::Cancelled);
        EXPECT_EQ(read_bytes(temporary.path), encoded);
        options.cancelled = {};
        // The inclusive packet boundary must work, not just reject limit + 1.
        auto boundary_packet = packet;
        boundary_packet.insert(boundary_packet.find("escaped"), 32 * 1024 * 1024 - packet.size(), 'a');
        ASSERT_TRUE(codec::write_tiff_u16(temporary.path, source, 19, 17, 4, 19 * 8, {}, options, {}, boundary_packet));
        {
            std::unique_ptr<TIFF, decltype(&TIFFClose)> file(TIFFOpen(temporary.path.string().c_str(), "r"), TIFFClose);
            ASSERT_TRUE(file);
            uint32_t count = 0;
            void* stored = nullptr;
            ASSERT_TRUE(TIFFGetField(file.get(), TIFFTAG_XMLPACKET, &count, &stored));
            ASSERT_NE(stored, nullptr);
            ASSERT_EQ(count, boundary_packet.size());
            EXPECT_EQ(std::memcmp(stored, boundary_packet.data(), count), 0);
        }
        ASSERT_TRUE(codec::decode(temporary.path, decoded, error)) << error;
        EXPECT_EQ(decoded.data, source);
        ASSERT_TRUE(codec::write_tiff_u16(temporary.path, source, 19, 17, 4, 19 * 8, {}, options));
        const auto plain = read_bytes(temporary.path);
        ASSERT_TRUE(codec::write_tiff_u16(temporary.path, source, 19, 17, 4, 19 * 8, {}, options, {}, {}));
        EXPECT_EQ(read_bytes(temporary.path), plain);
    }

    TEST(TiffMemory, LimitsSeekRewritesAliasingAndRecoveryPreservePreviousOutput) {
        for (int bits : {8, 16}) {
            const auto source = samples(31, 29, 4, bits);
            const size_t row = size_t(31) * 4 * (bits / 8);
            std::vector<uint8_t> reference;
            ASSERT_TRUE(codec::encode_tiff_memory(source, 31, 29, 4, bits, row, reference));
            for (size_t limit : {size_t{1}, size_t{7}, size_t{8}, size_t{4095}, size_t{4096}, reference.size() - 1, reference.size(), reference.size() + 1}) {
                codec::TiffEncodeOptions options;
                options.max_output_bytes = limit;
                std::vector<uint8_t> output{7, 8, 9};
                const auto result = codec::encode_tiff_memory(source, 31, 29, 4, bits, row, output, options);
                ASSERT_EQ(bool(result), limit >= reference.size());
                if (result)
                    EXPECT_EQ(output, reference);
                else {
                    EXPECT_EQ(result.error().code(), lfs::ErrorCode::ResourceExhausted);
                    EXPECT_EQ(output, (std::vector<uint8_t>{7, 8, 9}));
                }
            }
            auto alias = source;
            codec::TiffEncodeOptions options;
            options.max_output_bytes = 8;
            EXPECT_FALSE(codec::encode_tiff_memory(alias, 31, 29, 4, bits, row, alias, options));
            EXPECT_EQ(alias, source);
            ASSERT_TRUE(codec::encode_tiff_memory(alias, 31, 29, 4, bits, row, alias));
            EXPECT_EQ(alias, reference);
            codec::Image image{1, 1, 1, codec::SampleType::UInt8, alias};
            // The owning decoder must not release an aliased encoded source
            // until libtiff has closed and the decoded image is ready.
            std::string error;
            ASSERT_TRUE(codec::decode_memory(image.data.data(), image.data.size(), image, error)) << error;
            EXPECT_EQ(image.data, source);
        }
    }

    TEST(TiffMemory, InvalidArgumentsIncludeObservedValuesAndNeverReplaceOutput) {
        const auto source = samples(3, 2, 3, 16);
        std::vector<uint8_t> output{7, 8, 9};
        const auto invalid = [&](std::span<const uint8_t> bytes, int w, int h, int c, int bits, size_t stride, codec::TiffEncodeOptions options = {}) {
            const auto result = codec::encode_tiff_memory(bytes, w, h, c, bits, stride, output, options);
            ASSERT_FALSE(result);
            EXPECT_EQ(result.error().code(), lfs::ErrorCode::InvalidArgument);
            for (const auto& field : {"width=" + std::to_string(w) + ",", "height=" + std::to_string(h) + ",",
                                      "channels=" + std::to_string(c) + ",", "depth=" + std::to_string(bits) + ",", "stride=" + std::to_string(stride) + ",",
                                      "span_bytes=" + std::to_string(bytes.size()) + ",", "limit=" + std::to_string(options.max_output_bytes) + ")"})
                EXPECT_NE(result.error().detail().find(field), std::string_view::npos);
            EXPECT_EQ(output, (std::vector<uint8_t>{7, 8, 9}));
        };
        invalid({}, 3, 2, 3, 16, 18);
        invalid(source, 0, 2, 3, 16, 18);
        invalid(source, 3, -1, 3, 16, 18);
        invalid(source, 3, 2, 0, 16, 18);
        invalid(source, 3, 2, 5, 16, 18);
        invalid(source, 3, 2, 3, 32, 18);
        invalid(source, 3, 2, 3, 16, 17);
        invalid(source, 3, 2, 3, 16, 0);
        invalid(std::span(source).first(source.size() - 1), 3, 2, 3, 16, 18);
        invalid(source, 3, 2, 3, 16, std::numeric_limits<size_t>::max());
        codec::TiffEncodeOptions options;
        options.max_output_bytes = 0;
        invalid(source, 3, 2, 3, 16, 18, options);
        options.max_output_bytes = size_t(std::numeric_limits<uint32_t>::max()) + 1;
        invalid(source, 3, 2, 3, 16, 18, options);
        options = {};
        options.description = std::string("a\0b", 3);
        invalid(source, 3, 2, 3, 16, 18, options);
        options.description = std::string(1024 * 1024 + 1, 'x');
        invalid(source, 3, 2, 3, 16, 18, options);
        options = {};
        std::array<uint16_t, 3> table{};
        options.color.transfer_function = table;
        invalid(source, 3, 2, 3, 16, 18, options);
        options = {};
        options.color.chromaticities = std::array<float, 8>{.64f, 0.f, .3f, .6f, .15f, .06f, .3127f, .329f};
        invalid(source, 3, 2, 3, 16, 18, options);
        ASSERT_TRUE(codec::encode_tiff_memory(std::span(source).first(18), 3, 1, 3, 16, std::numeric_limits<size_t>::max(), output));
    }

    TEST(TiffMemory, DescriptionChromaticitiesAndTransferFunctionUseExistingTags) {
        TemporaryTiff temporary;
        for (int bits : {8, 16}) {
            auto source = samples(19, 17, 4, bits);
            codec::TiffEncodeOptions options;
            options.description = "existing TIFF metadata";
            options.color.chromaticities = std::array<float, 8>{.64f, .33f, .3f, .6f, .15f, .06f, .3127f, .329f};
            std::vector<uint16_t> table(size_t{1} << bits);
            for (size_t i = 0; i < table.size(); ++i)
                table[i] = uint16_t(i * (bits == 8 ? 257 : 1));
            options.color.transfer_function = table;
            std::vector<uint8_t> encoded;
            const auto row = size_t(19) * 4 * (bits / 8);
            ASSERT_TRUE(codec::encode_tiff_memory(source, 19, 17, 4, bits, row, encoded, options));
            if (bits == 16) {
                ASSERT_TRUE(codec::write_tiff_u16(temporary.path, source, 19, 17, 4, row, options.description,
                                                  {.overwrite = true, .durable = false}, options.color));
                EXPECT_EQ(encoded, read_bytes(temporary.path));
            }
            save_bytes(temporary.path, encoded);
            std::unique_ptr<TIFF, decltype(&TIFFClose)> file(TIFFOpen(temporary.path.string().c_str(), "r"), TIFFClose);
            ASSERT_NE(file, nullptr);
            char* description = nullptr;
            float* white = nullptr;
            uint16_t* transfer = nullptr;
            ASSERT_TRUE(TIFFGetField(file.get(), TIFFTAG_IMAGEDESCRIPTION, &description));
            EXPECT_EQ(std::string(description), *options.description);
            ASSERT_TRUE(TIFFGetField(file.get(), TIFFTAG_WHITEPOINT, &white));
            EXPECT_FLOAT_EQ(white[0], .3127f);
            // Identical RGB tables are stored as one table; the API still
            // requires three output pointers for a color transfer function.
            uint16_t *green = nullptr, *blue = nullptr;
            ASSERT_TRUE(TIFFGetField(file.get(), TIFFTAG_TRANSFERFUNCTION, &transfer, &green, &blue));
            for (size_t i = 0; i < table.size(); ++i)
                ASSERT_EQ(transfer[i], table[i]);
            check_pixels(encoded, source, 19, 17, 4, bits);
        }
    }

    TEST(TiffMemory, CorruptOrTruncatedInputPreservesOwningImageAndRecovers) {
        const auto source = samples(19, 17, 3, 16);
        std::vector<uint8_t> encoded;
        ASSERT_TRUE(codec::encode_tiff_memory(source, 19, 17, 3, 16, 19 * 3 * 2, encoded));
        for (size_t size : {size_t{4}, size_t{7}, encoded.size() / 2}) {
            codec::Image image{1, 1, 1, codec::SampleType::UInt8, {7, 8, 9}};
            std::string error;
            EXPECT_FALSE(codec::decode_memory(encoded.data(), size, image, error));
            EXPECT_FALSE(error.empty());
            EXPECT_EQ(image.data, (std::vector<uint8_t>{7, 8, 9}));
            check_pixels(encoded, source, 19, 17, 3, 16);
        }
    }

    TEST(TiffMemory, ClassicBigEndianBigTiffTilesPlanesAnd32BitReadMatchFiles) {
        TemporaryTiff temporary;
        constexpr int w = 19, h = 17;
        for (const char* mode : {"wl", "wb", "w8l", "w8b"})
            for (int kind : {8, 16, 32, 33})
                for (int c : {1, 2, 3, 4})
                    for (bool tiled : {false, true})
                        for (bool separate : {false, true}) {
                            SCOPED_TRACE(::testing::Message() << mode << " kind=" << kind << " c=" << c << " tiles=" << tiled << " separate=" << separate);
                            const int bits = kind == 33 ? 32 : kind;
                            const size_t sample = bits / 8;
                            std::vector<uint8_t> source;
                            if (bits < 32)
                                source = samples(w, h, c, bits);
                            else {
                                source.resize(size_t(w) * h * c * 4);
                                constexpr std::array<uint32_t, 8> values{0, 1, 0x80000000, 0xffffffff, 0x3f800000, 0x7fc00001, 0x7f800000, 0xff800000};
                                for (size_t i = 0; i < source.size() / 4; ++i)
                                    std::memcpy(source.data() + i * 4, &values[i % values.size()], 4);
                            }
                            {
                                std::unique_ptr<TIFF, decltype(&TIFFClose)> file(TIFFOpen(temporary.path.string().c_str(), mode), TIFFClose);
                                ASSERT_NE(file, nullptr);
                                ASSERT_TRUE(TIFFSetField(file.get(), TIFFTAG_IMAGEWIDTH, w));
                                ASSERT_TRUE(TIFFSetField(file.get(), TIFFTAG_IMAGELENGTH, h));
                                ASSERT_TRUE(TIFFSetField(file.get(), TIFFTAG_SAMPLESPERPIXEL, c));
                                ASSERT_TRUE(TIFFSetField(file.get(), TIFFTAG_BITSPERSAMPLE, bits));
                                ASSERT_TRUE(TIFFSetField(file.get(), TIFFTAG_SAMPLEFORMAT, kind == 33 ? SAMPLEFORMAT_IEEEFP : SAMPLEFORMAT_UINT));
                                ASSERT_TRUE(TIFFSetField(file.get(), TIFFTAG_PLANARCONFIG, separate ? PLANARCONFIG_SEPARATE : PLANARCONFIG_CONTIG));
                                ASSERT_TRUE(TIFFSetField(file.get(), TIFFTAG_PHOTOMETRIC, c < 3 ? PHOTOMETRIC_MINISBLACK : PHOTOMETRIC_RGB));
                                ASSERT_TRUE(TIFFSetField(file.get(), TIFFTAG_ORIENTATION, ORIENTATION_RIGHTTOP));
                                ASSERT_TRUE(TIFFSetField(file.get(), TIFFTAG_COMPRESSION, COMPRESSION_LZW));
                                if (c == 2 || c == 4) {
                                    uint16_t extra = EXTRASAMPLE_UNASSALPHA;
                                    ASSERT_TRUE(TIFFSetField(file.get(), TIFFTAG_EXTRASAMPLES, 1, &extra));
                                }
                                const int planes = separate ? c : 1;
                                if (tiled) {
                                    ASSERT_TRUE(TIFFSetField(file.get(), TIFFTAG_TILEWIDTH, 16));
                                    ASSERT_TRUE(TIFFSetField(file.get(), TIFFTAG_TILELENGTH, 16));
                                    for (int plane = 0; plane < planes; ++plane)
                                        for (int y = 0; y < h; y += 16)
                                            for (int x = 0; x < w; x += 16) {
                                                std::vector<uint8_t> tile(size_t(16) * 16 * (separate ? 1 : c) * sample, 0);
                                                for (int yy = 0; yy < 16 && y + yy < h; ++yy)
                                                    for (int xx = 0; xx < 16 && x + xx < w; ++xx)
                                                        for (int cc = 0; cc < (separate ? 1 : c); ++cc)
                                                            std::memcpy(tile.data() + ((yy * 16 + xx) * (separate ? 1 : c) + cc) * sample,
                                                                        source.data() + (((y + yy) * w + x + xx) * c + (separate ? plane : cc)) * sample, sample);
                                                ASSERT_GE(TIFFWriteTile(file.get(), tile.data(), x, y, 0, static_cast<uint16_t>(plane)), 0);
                                            }
                                } else {
                                    ASSERT_TRUE(TIFFSetField(file.get(), TIFFTAG_ROWSPERSTRIP, 4));
                                    for (int plane = 0; plane < planes; ++plane)
                                        for (int y = 0; y < h; ++y) {
                                            std::vector<uint8_t> row(size_t(w) * (separate ? 1 : c) * sample);
                                            for (int x = 0; x < w; ++x)
                                                for (int cc = 0; cc < (separate ? 1 : c); ++cc)
                                                    std::memcpy(row.data() + (x * (separate ? 1 : c) + cc) * sample,
                                                                source.data() + ((y * w + x) * c + (separate ? plane : cc)) * sample, sample);
                                            ASSERT_GE(TIFFWriteScanline(file.get(), row.data(), y, static_cast<uint16_t>(plane)), 0);
                                        }
                                }
                                ASSERT_TRUE(TIFFWriteDirectory(file.get()));
                            }
                            const auto encoded = read_bytes(temporary.path);
                            const auto original = encoded;
                            codec::Image file, memory;
                            std::string error;
                            ASSERT_TRUE(codec::decode(temporary.path, file, error)) << error;
                            ASSERT_TRUE(codec::decode_memory(encoded.data(), encoded.size(), memory, error)) << error;
                            EXPECT_EQ(memory.width, w);
                            EXPECT_EQ(memory.height, h);
                            EXPECT_EQ(memory.channels, c);
                            EXPECT_EQ(memory.sample_type, file.sample_type);
                            EXPECT_EQ(memory.data, file.data);
                            auto expected = source;
                            if (kind == 32)
                                for (size_t i = 0; i < expected.size(); i += 4) {
                                    uint32_t integer;
                                    std::memcpy(&integer, source.data() + i, 4);
                                    const float value = float(double(integer) / 4294967295.0);
                                    std::memcpy(expected.data() + i, &value, 4);
                                }
                            EXPECT_EQ(memory.data, expected);
                            EXPECT_EQ(encoded, original);
                            std::vector<uint8_t> output(expected.size() + 2, 0xcd);
                            codec::DecodeTarget target{c, file.sample_type, output.data() + 1};
                            codec::Probe info;
                            const bool direct = codec::decode_memory_to_buffer(encoded.data(), encoded.size(), target, info, error);
                            EXPECT_EQ(direct, !separate && kind != 32);
                            if (direct)
                                EXPECT_EQ(std::memcmp(output.data() + 1, expected.data(), expected.size()), 0);
                            else
                                EXPECT_EQ(output, (std::vector<uint8_t>(expected.size() + 2, 0xcd)));
                            EXPECT_EQ(output.front(), 0xcd);
                            EXPECT_EQ(output.back(), 0xcd);
                        }
    }

    TEST(TiffMemory, RecoverablePaddingEntryMatchesFileAndKeepsFatalDiagnostics) {
        TemporaryTiff temporary;
        for (const auto mode : {"wl", "wb", "w8l", "w8b"})
            for (int bits : {8, 16})
                for (int channels : {1, 3}) {
                    SCOPED_TRACE(::testing::Message() << mode << " bits=" << bits << " channels=" << channels);
                    const auto pixels = samples(1, 1, channels, bits);
                    {
                        std::unique_ptr<TIFF, decltype(&TIFFClose)> file(TIFFOpen(temporary.path.string().c_str(), mode), TIFFClose);
                        ASSERT_NE(file, nullptr);
                        ASSERT_TRUE(TIFFSetField(file.get(), TIFFTAG_IMAGEWIDTH, 1));
                        ASSERT_TRUE(TIFFSetField(file.get(), TIFFTAG_IMAGELENGTH, 1));
                        ASSERT_TRUE(TIFFSetField(file.get(), TIFFTAG_SAMPLESPERPIXEL, channels));
                        ASSERT_TRUE(TIFFSetField(file.get(), TIFFTAG_BITSPERSAMPLE, bits));
                        ASSERT_TRUE(TIFFSetField(file.get(), TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT));
                        ASSERT_TRUE(TIFFSetField(file.get(), TIFFTAG_PHOTOMETRIC, channels == 1 ? PHOTOMETRIC_MINISBLACK : PHOTOMETRIC_RGB));
                        ASSERT_TRUE(TIFFSetField(file.get(), TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG));
                        ASSERT_TRUE(TIFFSetField(file.get(), TIFFTAG_DOCUMENTNAME, "padding entry"));
                        auto row = pixels;
                        ASSERT_GE(TIFFWriteScanline(file.get(), row.data(), 0, 0), 0);
                        ASSERT_TRUE(TIFFWriteDirectory(file.get()));
                    }
                    auto encoded = read_bytes(temporary.path);
                    const bool little = encoded[0] == 'I', big = mode[1] == '8';
                    const auto integer = [&](size_t offset, size_t count) {
                        uint64_t value = 0;
                        for (size_t i = 0; i < count; ++i)
                            value |= uint64_t(encoded.at(offset + i)) << (8 * (little ? i : count - 1 - i));
                        return value;
                    };
                    const size_t directory = static_cast<size_t>(integer(big ? 8 : 4, big ? 8 : 4));
                    const size_t count = static_cast<size_t>(integer(directory, big ? 8 : 2));
                    const size_t entry_size = big ? 20 : 12;
                    size_t padding = 0, sample_format = 0;
                    for (size_t i = 0; i < count; ++i) {
                        const size_t entry = directory + (big ? 8 : 2) + i * entry_size;
                        const auto tag = integer(entry, 2);
                        if (tag == TIFFTAG_DOCUMENTNAME)
                            padding = entry;
                        if (tag == TIFFTAG_SAMPLEFORMAT)
                            sample_format = entry;
                    }
                    ASSERT_NE(padding, 0);
                    ASSERT_NE(sample_format, 0);
                    std::fill_n(encoded.begin() + padding, entry_size, 0);
                    save_bytes(temporary.path, encoded);
                    codec::Image file, memory;
                    std::string file_error, error = "stale diagnostic";
                    ASSERT_TRUE(codec::decode(temporary.path, file, file_error)) << file_error;
                    EXPECT_EQ(file.data, pixels);
                    ASSERT_TRUE(codec::decode_memory(encoded.data(), encoded.size(), memory, error)) << error;
                    EXPECT_EQ(memory.data, file.data);
                    EXPECT_TRUE(error.empty());
                    check_pixels(encoded, pixels, 1, 1, channels, bits);
                    if (channels != 1)
                        continue;
                    // A recoverable tag-0 error precedes a real sample-format
                    // rejection; the latter must be the reported cause.
                    const size_t value = sample_format + (big ? 12 : 8);
                    encoded.at(value) = little ? SAMPLEFORMAT_INT : 0;
                    encoded.at(value + 1) = little ? 0 : SAMPLEFORMAT_INT;
                    save_bytes(temporary.path, encoded);
                    ASSERT_FALSE(codec::decode(temporary.path, file, file_error));
                    EXPECT_EQ(file_error, "Unsupported TIFF sample format");
                    error = "stale diagnostic";
                    const auto previous = memory.data;
                    ASSERT_FALSE(codec::decode_memory(encoded.data(), encoded.size(), memory, error));
                    EXPECT_EQ(error, file_error);
                    EXPECT_EQ(memory.data, previous);
                    std::vector<uint8_t> target_bytes(pixels.size(), 0xcd);
                    codec::DecodeTarget target{channels, bits == 8 ? codec::SampleType::UInt8 : codec::SampleType::UInt16, target_bytes.data()};
                    codec::Probe info;
                    ASSERT_FALSE(codec::decode_to_buffer(temporary.path, target, info, file_error));
                    ASSERT_FALSE(codec::decode_memory_to_buffer(encoded.data(), encoded.size(), target, info, error));
                    EXPECT_EQ(error, file_error);
                    EXPECT_EQ(target_bytes, std::vector<uint8_t>(pixels.size(), 0xcd));
                }
    }

    TEST(TiffMemory, WhiteIsZeroPreservesExistingNativeSamplePolicy) {
        TemporaryTiff temporary;
        for (int bits : {8, 16}) {
            const auto source = samples(19, 17, 1, bits);
            std::string error;
            if (bits == 8)
                ASSERT_TRUE(codec::write_tiff(temporary.path, source.data(), 19, 17, 1, error));
            else
                ASSERT_TRUE(codec::write_tiff_u16(temporary.path, source, 19, 17, 1, 19 * 2,
                                                  {}, {.overwrite = true, .durable = false}));
            {
                std::unique_ptr<TIFF, decltype(&TIFFClose)> file(TIFFOpen(temporary.path.string().c_str(), "r+"), TIFFClose);
                ASSERT_NE(file, nullptr);
                ASSERT_TRUE(TIFFSetField(file.get(), TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISWHITE));
                ASSERT_TRUE(TIFFRewriteDirectory(file.get()));
            }
            auto expected = source;
            for (auto& byte : expected)
                byte = uint8_t(~byte);
            const auto encoded = read_bytes(temporary.path);
            check_pixels(encoded, expected, 19, 17, 1, bits);
            codec::Image file;
            ASSERT_TRUE(codec::decode(temporary.path, file, error)) << error;
            EXPECT_EQ(file.data, expected);
        }
    }

    TEST(TiffMemory, InvalidClassicAndBigTiffOffsetsNeverReplaceOwningOutput) {
        for (bool big : {false, true})
            for (bool little : {false, true}) {
                std::vector<uint8_t> encoded(big ? 16 : 8, 0xff);
                encoded[0] = encoded[1] = little ? 'I' : 'M';
                encoded[2] = little ? (big ? 43 : 42) : 0;
                encoded[3] = little ? 0 : (big ? 43 : 42);
                if (big) {
                    encoded[4] = little ? 8 : 0;
                    encoded[5] = little ? 0 : 8;
                    encoded[6] = encoded[7] = 0;
                }
                codec::Image image;
                image.width = 7;
                image.data = {1, 2, 3};
                std::string error;
                EXPECT_FALSE(codec::decode_memory(encoded.data(), encoded.size(), image, error));
                EXPECT_FALSE(error.empty());
                EXPECT_EQ(image.width, 7);
                EXPECT_EQ(image.data, (std::vector<uint8_t>{1, 2, 3}));
            }
    }

    TEST(TiffMemory, ProviderExifDirectoryBudgetsPreserveMetadataAndRecover) {
        TemporaryTiff temporary;
        const auto pixels = samples(19, 17, 3, 8);
        std::string error;
        ASSERT_TRUE(codec::write_tiff(temporary.path, pixels.data(), 19, 17, 3, error));
        uint64_t offset = 0;
        constexpr char timestamp[] = "2026:10:17 12:34:56";
        {
            std::unique_ptr<TIFF, decltype(&TIFFClose)> file(TIFFOpen(temporary.path.string().c_str(), "r+"), TIFFClose);
            ASSERT_NE(file, nullptr);
            TIFFCreateEXIFDirectory(file.get());
            ASSERT_TRUE(TIFFSetField(file.get(), EXIFTAG_DATETIMEORIGINAL, timestamp));
            ASSERT_TRUE(TIFFWriteCustomDirectory(file.get(), &offset));
            ASSERT_TRUE(TIFFSetDirectory(file.get(), 0));
            ASSERT_TRUE(TIFFSetField(file.get(), TIFFTAG_EXIFIFD, offset));
            ASSERT_TRUE(TIFFRewriteDirectory(file.get()));
        }
        unsigned successes = 0, failures = 0;
        for (tmsize_t limit = 1024; limit <= 65536; limit += 128) {
            std::unique_ptr<TIFFOpenOptions, decltype(&TIFFOpenOptionsFree)> options(TIFFOpenOptionsAlloc(), TIFFOpenOptionsFree);
            ASSERT_NE(options, nullptr);
            TIFFOpenOptionsSetMaxCumulatedMemAlloc(options.get(), limit);
            TIFFOpenOptionsSetErrorHandlerExtR(options.get(), [](TIFF*, void*, const char*, const char*, va_list) { return 1; }, nullptr);
            TIFFOpenOptionsSetWarningHandlerExtR(options.get(), [](TIFF*, void*, const char*, const char*, va_list) { return 1; }, nullptr);
            std::unique_ptr<TIFF, decltype(&TIFFClose)> file(TIFFOpenExt(temporary.path.string().c_str(), "rm", options.get()), TIFFClose);
            if (!file || !TIFFReadEXIFDirectory(file.get(), offset)) {
                ++failures;
                continue;
            }
            char* stored = nullptr;
            if (TIFFGetField(file.get(), EXIFTAG_DATETIMEORIGINAL, &stored) && stored) {
                ++successes;
                EXPECT_STREQ(stored, timestamp);
            } else
                ++failures;
        }
        EXPECT_GT(failures, 0);
        EXPECT_GT(successes, 0);
        check_pixels(read_bytes(temporary.path), pixels, 19, 17, 3, 8);
    }

    TEST(TiffMemory, ProviderFailedFieldTableGrowthPreservesExistingFields) {
        TemporaryTiff temporary;
        unsigned rejected_growth = 0;
        char name[] = "budget field";
        std::array<TIFFFieldInfo, 64> fields{};
        for (size_t i = 0; i < fields.size(); ++i)
            fields[i] = {static_cast<ttag_t>(65000 + i), 1, 1, TIFF_LONG, FIELD_CUSTOM, 1, 0, name};
        for (tmsize_t limit = 1024; limit <= 16384; limit += 32) {
            bool field_failure = false;
            std::unique_ptr<TIFFOpenOptions, decltype(&TIFFOpenOptionsFree)> options(TIFFOpenOptionsAlloc(), TIFFOpenOptionsFree);
            ASSERT_NE(options, nullptr);
            TIFFOpenOptionsSetMaxCumulatedMemAlloc(options.get(), limit);
            TIFFOpenOptionsSetErrorHandlerExtR(options.get(), [](TIFF*, void* context, const char* module, const char*, va_list) {
                if (module && std::strcmp(module, "_TIFFMergeFields") == 0)
                    *static_cast<bool*>(context) = true;
                return 1; }, &field_failure);
            TIFFOpenOptionsSetWarningHandlerExtR(options.get(), [](TIFF*, void*, const char*, const char*, va_list) { return 1; }, nullptr);
            const auto cleanup = [](TIFF* file) {
                // This test never writes pixels or a directory.
                TIFFSetMode(file, 0);
                TIFFClose(file);
            };
            std::unique_ptr<TIFF, decltype(cleanup)> file(TIFFOpenExt(temporary.path.string().c_str(), "w", options.get()), cleanup);
            if (!file)
                continue;
            // Grow the core field table through the public custom-tag API.
            field_failure = false;
            TIFFMergeFieldInfo(file.get(), fields.data(), static_cast<uint32_t>(fields.size()));
            if (field_failure) {
                ++rejected_growth;
                EXPECT_NE(TIFFFieldWithTag(file.get(), TIFFTAG_IMAGEWIDTH), nullptr);
                EXPECT_TRUE(TIFFSetField(file.get(), TIFFTAG_IMAGEWIDTH, 19));
            }
        }
        EXPECT_GT(rejected_growth, 0);
    }

    TEST(TiffMemory, ProviderAllocationBudgetsFailCleanlyAndRecover) {
        TemporaryTiff temporary;
        const auto pixels = samples(19, 17, 3, 16);
        ASSERT_TRUE(codec::write_tiff_u16(temporary.path, pixels, 19, 17, 3, 19 * 3 * 2,
                                          {}, {.overwrite = true, .durable = false}));
        unsigned failures = 0, successes = 0;
        for (tmsize_t limit = 256; limit <= 262144; limit += limit < 32768 ? 64 : 4096) {
            std::unique_ptr<TIFFOpenOptions, decltype(&TIFFOpenOptionsFree)> options(TIFFOpenOptionsAlloc(), TIFFOpenOptionsFree);
            ASSERT_NE(options, nullptr);
            TIFFOpenOptionsSetMaxCumulatedMemAlloc(options.get(), limit);
            TIFFOpenOptionsSetErrorHandlerExtR(options.get(), [](TIFF*, void*, const char*, const char*, va_list) { return 1; }, nullptr);
            TIFFOpenOptionsSetWarningHandlerExtR(options.get(), [](TIFF*, void*, const char*, const char*, va_list) { return 1; }, nullptr);
            std::unique_ptr<TIFF, decltype(&TIFFClose)> file(TIFFOpenExt(temporary.path.string().c_str(), "rm", options.get()), TIFFClose);
            std::array<uint16_t, 19 * 3> row{};
            if (file && TIFFReadScanline(file.get(), row.data(), 0, 0) >= 0) {
                ++successes;
                EXPECT_EQ(std::memcmp(row.data(), pixels.data(), sizeof(row)), 0);
            } else
                ++failures;
        }
        EXPECT_GT(failures, 0);
        EXPECT_GT(successes, 0);
        codec::Image image;
        std::string error;
        ASSERT_TRUE(codec::decode(temporary.path, image, error)) << error;
        EXPECT_EQ(image.data, pixels);
    }

    TEST(TiffMemory, CustomAllocatorFailureAndConcurrentCallsRemainIndependent) {
        auto worker = [](int seed) {
            for (int bits : {8, 16})
                for (int channels : {1, 2, 3, 4}) {
                    const int w = 19 + seed, h = 17;
                    const auto source = samples(w, h, channels, bits);
                    std::vector<uint8_t> encoded;
                    ASSERT_TRUE(codec::encode_tiff_memory(source, w, h, channels, bits,
                                                          size_t(w) * channels * (bits / 8), encoded));
                    struct Allocation {
                        std::vector<uint8_t> bytes;
                        size_t calls = 0;
                    } allocation;
                    codec::DecodeTarget target{channels, bits == 8 ? codec::SampleType::UInt8 : codec::SampleType::UInt16};
                    target.allocate = [](size_t bytes, void* user) -> void* {
                        auto& state = *static_cast<Allocation*>(user);
                        ++state.calls;
                        state.bytes.resize(bytes + 2, 0xcd);
                        return state.bytes.data() + 1;
                    };
                    target.user = &allocation;
                    codec::Probe info;
                    std::string error;
                    ASSERT_TRUE(codec::decode_memory_to_buffer(encoded.data(), encoded.size(), target, info, error)) << error;
                    EXPECT_EQ(allocation.calls, 1);
                    ASSERT_EQ(allocation.bytes.size(), source.size() + 2);
                    EXPECT_EQ(std::memcmp(allocation.bytes.data() + 1, source.data(), source.size()), 0);
                    EXPECT_EQ(allocation.bytes.front(), 0xcd);
                    EXPECT_EQ(allocation.bytes.back(), 0xcd);
                    target.data = nullptr;
                    target.allocate = [](size_t, void*) -> void* { return nullptr; };
                    EXPECT_FALSE(codec::decode_memory_to_buffer(encoded.data(), encoded.size(), target, info, error));
                    EXPECT_EQ(target.data, nullptr);
                    EXPECT_FALSE(error.empty());
                    check_pixels(encoded, source, w, h, channels, bits);
                }
        };
        std::vector<std::future<void>> workers;
        for (int i = 0; i < 4; ++i)
            workers.push_back(std::async(std::launch::async, worker, i));
        for (auto& future : workers)
            future.get();
    }

    TEST(TiffMemory, LargeImagesPreserveSamplesAndRecoverFromOutputLimits) {
        for (const auto size : {std::pair{3840, 2160}, std::pair{7680, 4320}})
            for (int bits : {8, 16}) {
                const auto [w, h] = size;
                SCOPED_TRACE(::testing::Message() << w << 'x' << h << " bits=" << bits);
                // Grayscale exercises both large extents and native 16-bit rows
                // without requiring a full RGBA 8K allocation in ordinary CI.
                const auto source = samples(w, h, 1, bits);
                std::vector<uint8_t> encoded;
                ASSERT_TRUE(codec::encode_tiff_memory(source, w, h, 1, bits, size_t(w) * (bits / 8), encoded));
                EXPECT_LE(encoded.capacity() - encoded.size(), encoded.size() / 2);
                check_pixels(encoded, source, w, h, 1, bits);
                const std::vector<uint8_t> previous{9, 8, 7};
                std::vector<uint8_t> failed = previous;
                codec::TiffEncodeOptions options;
                options.max_output_bytes = encoded.size() - 1;
                EXPECT_FALSE(codec::encode_tiff_memory(source, w, h, 1, bits, size_t(w) * (bits / 8), failed, options));
                EXPECT_EQ(failed, previous);
                options.max_output_bytes = encoded.size();
                ASSERT_TRUE(codec::encode_tiff_memory(source, w, h, 1, bits, size_t(w) * (bits / 8), failed, options));
                EXPECT_EQ(failed, encoded);
            }
    }

} // namespace
