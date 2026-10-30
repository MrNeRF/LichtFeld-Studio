/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/image_codecs.hpp"
#include "core/native_fs.hpp"
#include "long_path_test_support.hpp"
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <gtest/gtest.h>
#include <iterator>
#include <jpeglib.h>
#include <limits>
#include <memory>
#include <png.h>
#include <stdexcept>
#include <tiffio.h>
#include <vector>
#include <zlib.h>

namespace {
    namespace codec = lfs::core::image_codecs;

    TEST(ProductionCodecs, XmpPacketBudgetAndNulValidation) {
        EXPECT_TRUE(codec::valid_xmp_packet({}));
        EXPECT_TRUE(codec::valid_xmp_packet("<x/>"));
        std::string packet(codec::max_xmp_packet_bytes, 'a');
        EXPECT_TRUE(codec::valid_xmp_packet(packet));
        packet.push_back('a');
        EXPECT_FALSE(codec::valid_xmp_packet(packet));
        packet.resize(codec::max_xmp_packet_bytes);
        packet.front() = '\0';
        EXPECT_FALSE(codec::valid_xmp_packet(packet));
        packet.front() = 'a';
        packet.back() = '\0';
        EXPECT_FALSE(codec::valid_xmp_packet(packet));
    }

    TEST(ProductionTiff, TransferTablesAreSharedAndMatchIndependentReference) {
        EXPECT_TRUE(codec::tiff_transfer_function("unknown").empty());
        for (const auto name : {"linear", "srgb", "bt709"}) {
            const auto table = codec::tiff_transfer_function(name);
            ASSERT_EQ(table.size(), 65536);
            EXPECT_EQ(table.data(), codec::tiff_transfer_function(name).data());
            for (size_t i = 0; i < table.size(); ++i) {
                const double value = double(i) / 65535.;
                const double linear = std::string_view(name) == "srgb"
                                          ? (value <= .04045 ? value / 12.92 : std::pow((value + .055) / 1.055, 2.4))
                                      : std::string_view(name) == "bt709"
                                          ? (value < .081 ? value / 4.5 : std::pow((value + .099) / 1.099, 1. / .45))
                                          : value;
                EXPECT_LE(std::abs(int(table[i]) - int(std::floor(linear * 65535. + .5))), 1) << name << " entry " << i;
            }
        }
    }

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
            lfs::core::native_fs::remove_all(path, error);
        }
    };
    TEST(ProductionTiff, WritesExactUint16SamplesAndPreservesBorrowedRows) {
        TemporaryDirectory temporary;
        for (const int channels : {1, 2, 3, 4}) {
            const int width = 3, height = 2;
            const auto stride = static_cast<size_t>(width * channels * 2 + 5);
            // Deliberately unaligned with odd padding; the codec owns its
            // writable, aligned scanline and must never change the source.
            std::vector<uint8_t> storage(1 + stride * height, 0xa5);
            const std::array<uint16_t, 8> values{0, 1, 255, 256, 257, 32768, 65534, 65535};
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width * channels; ++x) {
                    const auto value = values[(y * width * channels + x) % values.size()];
                    std::memcpy(storage.data() + 1 + y * stride + x * 2, &value, 2);
                }
            const auto before = storage;
            const auto path = temporary.path / (std::to_string(channels) + ".tiff");
            std::vector<uint16_t> transfer(65536);
            for (size_t i = 0; i < transfer.size(); ++i)
                transfer[i] = static_cast<uint16_t>(i);
            codec::TiffColorTags tags;
            if (channels >= 3) {
                tags.chromaticities = std::array<float, 8>{.64f, .33f, .30f, .60f, .15f, .06f, .3127f, .3290f};
                tags.transfer_function = transfer;
            }
            const auto result = codec::write_tiff_u16(path, std::span(storage).subspan(1), width, height, channels,
                                                      stride, "precision fixture", {.overwrite = false}, tags);
            ASSERT_TRUE(result) << result.error().detail();
            EXPECT_EQ(storage, before);
            auto* tiff = TIFFOpen(path.string().c_str(), "r");
            ASSERT_NE(tiff, nullptr);
            uint16_t bits = 0, format = 0;
            ASSERT_EQ(TIFFGetField(tiff, TIFFTAG_BITSPERSAMPLE, &bits), 1);
            TIFFGetFieldDefaulted(tiff, TIFFTAG_SAMPLEFORMAT, &format);
            EXPECT_EQ(bits, 16);
            EXPECT_EQ(format, SAMPLEFORMAT_UINT);
            char* description = nullptr;
            ASSERT_EQ(TIFFGetField(tiff, TIFFTAG_IMAGEDESCRIPTION, &description), 1);
            EXPECT_STREQ(description, "precision fixture");
            if (channels >= 3) {
                float *primaries = nullptr, *white = nullptr;
                ASSERT_EQ(TIFFGetField(tiff, TIFFTAG_PRIMARYCHROMATICITIES, &primaries), 1);
                ASSERT_EQ(TIFFGetField(tiff, TIFFTAG_WHITEPOINT, &white), 1);
                EXPECT_NEAR(primaries[0], .64f, 1e-6f);
                EXPECT_NEAR(white[0], .3127f, 1e-6f);
                uint16_t *red = nullptr, *green = nullptr, *blue = nullptr;
                ASSERT_EQ(TIFFGetField(tiff, TIFFTAG_TRANSFERFUNCTION, &red, &green, &blue), 1);
                for (const auto value : values) {
                    EXPECT_EQ(red[value], value);
                    EXPECT_EQ(green[value], value);
                    EXPECT_EQ(blue[value], value);
                }
            }
            std::vector<uint16_t> row(width * channels);
            for (int y = 0; y < height; ++y) {
                ASSERT_GE(TIFFReadScanline(tiff, row.data(), y), 0);
                for (int x = 0; x < width * channels; ++x)
                    EXPECT_EQ(row[x], values[(y * width * channels + x) % values.size()]);
            }
            TIFFClose(tiff);
        }
    }

    TEST(ProductionTiff, Uint16ValidationAndCancellationPreserveExistingFile) {
        TemporaryDirectory temporary;
        const auto path = temporary.path / "previous.tif";
        const std::array<uint16_t, 12> pixels{};
        const auto bytes = std::span(reinterpret_cast<const uint8_t*>(pixels.data()), sizeof(pixels));
        ASSERT_TRUE(codec::write_tiff_u16(path, bytes, 2, 2, 3, 12, std::nullopt, {}));
        const auto contents = [&] {
            std::ifstream input(path, std::ios::binary);
            return std::vector<uint8_t>(std::istreambuf_iterator<char>(input), {});
        };
        const auto before = contents();
        EXPECT_FALSE(codec::write_tiff_u16(path, bytes.first(5), 2, 2, 3, 12, std::nullopt, {.overwrite = true}));
        EXPECT_EQ(contents(), before);
        // Atomic preflight calls the callback first; cancel during scanlines.
        int calls = 0;
        const auto cancelled = codec::write_tiff_u16(path, bytes, 2, 2, 3, 12, std::nullopt,
                                                     {.overwrite = true, .cancelled = [&] { return ++calls >= 2; }});
        ASSERT_FALSE(cancelled);
        EXPECT_EQ(cancelled.error().code(), lfs::ErrorCode::Cancelled);
        EXPECT_EQ(contents(), before);
        EXPECT_EQ(std::distance(std::filesystem::directory_iterator(temporary.path), std::filesystem::directory_iterator{}), 1);
    }

    TEST(ProductionTiff, OversizedDescriptionsKeepLegacyDecodeRegardlessOfLfsSubstring) {
        TemporaryDirectory temporary;
        const auto path = temporary.path / "description.tif";
        for (const bool own : {false, true}) {
            auto* tiff = TIFFOpen(path.string().c_str(), "w");
            ASSERT_NE(tiff, nullptr);
            std::unique_ptr<TIFF, decltype(&TIFFClose)> owner(tiff, TIFFClose);
            std::string description = own ? "{\"lfsEncoding\":\"tiff16-sdr-v1\",\"padding\":\"" : "external description ";
            description.append(1024 * 1024, 'x');
            TIFFSetField(tiff, TIFFTAG_IMAGEWIDTH, 1);
            TIFFSetField(tiff, TIFFTAG_IMAGELENGTH, 1);
            TIFFSetField(tiff, TIFFTAG_SAMPLESPERPIXEL, 3);
            TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, 16);
            TIFFSetField(tiff, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
            TIFFSetField(tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
            TIFFSetField(tiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
            ASSERT_EQ(TIFFSetField(tiff, TIFFTAG_IMAGEDESCRIPTION, description.c_str()), 1);
            std::array<uint16_t, 3> pixel{1234, 23456, 54321};
            ASSERT_GE(TIFFWriteScanline(tiff, pixel.data(), 0), 0);
            owner.reset();
            codec::TiffMetadata metadata;
            std::string error;
            EXPECT_TRUE(codec::read_tiff_metadata(path, metadata, error));
            EXPECT_FALSE(metadata.description);
            codec::Image raw;
            ASSERT_TRUE(codec::decode(path, raw, error));
            EXPECT_EQ(raw.sample_type, codec::SampleType::UInt16);
        }
    }

    TEST(ProductionTiff, NormalizesUnsignedInteger32WithoutMisreportingSamples) {
        TemporaryDirectory temporary;
        for (const auto extension : {".tiff", ".data"})
            for (const auto mode : {"wl", "wb"})
                for (const auto planar : {PLANARCONFIG_CONTIG, PLANARCONFIG_SEPARATE}) {
                    constexpr int width = 5, height = 2, channels = 3;
                    const auto path = temporary.path / (std::string("integer32-") + mode + std::to_string(planar) + extension);
                    TIFF* tiff = TIFFOpen(path.string().c_str(), mode);
                    ASSERT_NE(tiff, nullptr);
                    TIFFSetField(tiff, TIFFTAG_IMAGEWIDTH, width);
                    TIFFSetField(tiff, TIFFTAG_IMAGELENGTH, height);
                    TIFFSetField(tiff, TIFFTAG_SAMPLESPERPIXEL, channels);
                    TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, 32);
                    TIFFSetField(tiff, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_UINT);
                    TIFFSetField(tiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
                    TIFFSetField(tiff, TIFFTAG_PLANARCONFIG, planar);
                    TIFFSetField(tiff, TIFFTAG_ROWSPERSTRIP, 1);
                    const std::array<uint32_t, 7> values{0u, 1u, 65535u, 65536u, 0x7fffffffu, 0x80000000u, 0xffffffffu};
                    std::array<uint32_t, width * height * channels> samples{};
                    for (size_t i = 0; i < samples.size(); ++i)
                        samples[i] = values[i % values.size()];
                    std::array<uint32_t, width> plane{};
                    std::array<uint32_t, width * channels> row{};
                    for (int y = 0; y < height; ++y) {
                        if (planar == PLANARCONFIG_CONTIG) {
                            // libtiff may byte-swap the caller buffer when writing foreign endianness.
                            std::copy_n(samples.data() + y * width * channels, row.size(), row.data());
                            ASSERT_GE(TIFFWriteScanline(tiff, row.data(), y), 0);
                        } else {
                            for (int c = 0; c < channels; ++c) {
                                for (int x = 0; x < width; ++x)
                                    plane[x] = samples[(y * width + x) * channels + c];
                                ASSERT_GE(TIFFWriteScanline(tiff, plane.data(), y, c), 0);
                            }
                        }
                    }
                    TIFFClose(tiff);
                    codec::Probe info;
                    codec::Image image;
                    std::string error;
                    ASSERT_TRUE(codec::probe(path, info, error)) << error;
                    EXPECT_EQ(info.sample_type, codec::SampleType::Float32);
                    ASSERT_TRUE(codec::decode(path, image, error)) << error;
                    EXPECT_EQ(image.sample_type, codec::SampleType::Float32);
                    ASSERT_EQ(image.data.size(), samples.size() * sizeof(float));
                    for (size_t i = 0; i < samples.size(); ++i) {
                        float actual;
                        std::memcpy(&actual, image.data.data() + i * sizeof(float), sizeof(float));
                        EXPECT_FLOAT_EQ(actual, static_cast<float>(double(samples[i]) / 4294967295.0));
                    }
                    if (std::string_view(extension) == ".tiff") {
                        std::array<float, samples.size()> destination{};
                        destination.fill(.125f);
                        const auto before = destination;
                        codec::DecodeTarget target{channels, codec::SampleType::Float32, destination.data()};
                        EXPECT_FALSE(codec::decode_to_buffer(path, target, info, error));
                        EXPECT_EQ(destination, before);
                    }
                }
    }
    TEST(ProductionTiff, PreservesSupportedSamplesInBothByteOrders) {
        TemporaryDirectory temporary;
        for (const auto mode : {"wl", "wb"}) {
            for (const auto bits : {8, 16, 32}) {
                const auto path = temporary.path / (std::to_string(bits) + mode + ".tiff");
                TIFF* tiff = TIFFOpen(path.string().c_str(), mode);
                ASSERT_NE(tiff, nullptr);
                TIFFSetField(tiff, TIFFTAG_IMAGEWIDTH, 2);
                TIFFSetField(tiff, TIFFTAG_IMAGELENGTH, 1);
                TIFFSetField(tiff, TIFFTAG_SAMPLESPERPIXEL, 1);
                TIFFSetField(tiff, TIFFTAG_BITSPERSAMPLE, bits);
                TIFFSetField(tiff, TIFFTAG_SAMPLEFORMAT, bits == 32 ? SAMPLEFORMAT_IEEEFP : SAMPLEFORMAT_UINT);
                TIFFSetField(tiff, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
                TIFFSetField(tiff, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
                std::array<std::uint8_t, 2> bytes{0, 255};
                std::array<std::uint16_t, 2> shorts{0x0102, 65535};
                std::array<float, 2> floats{-0.25f, 4.0f};
                void* samples = bits == 8 ? static_cast<void*>(bytes.data()) : bits == 16 ? static_cast<void*>(shorts.data())
                                                                                          : static_cast<void*>(floats.data());
                std::vector<std::uint8_t> expected(bits / 8 * 2);
                std::memcpy(expected.data(), samples, expected.size());
                const auto written = TIFFWriteScanline(tiff, samples, 0);
                TIFFClose(tiff);
                ASSERT_EQ(written, 1);
                codec::Image image;
                codec::Probe info;
                std::string error;
                ASSERT_TRUE(codec::decode(path, image, error)) << error;
                ASSERT_TRUE(codec::probe(path, info, error)) << error;
                EXPECT_EQ(image.data, expected);
                const auto type = bits == 8 ? codec::SampleType::UInt8 : bits == 16 ? codec::SampleType::UInt16
                                                                                    : codec::SampleType::Float32;
                EXPECT_EQ(info.sample_type, type);
                EXPECT_EQ(image.sample_type, type);
                std::vector<std::uint8_t> output(expected.size());
                codec::DecodeTarget target{1, type, output.data()};
                ASSERT_TRUE(codec::decode_to_buffer(path, target, info, error)) << error;
                EXPECT_EQ(output, expected);
                const auto renamed = path.string() + ".data";
                std::filesystem::copy_file(path, renamed);
                ASSERT_TRUE(codec::probe(renamed, info, error)) << error;
                EXPECT_EQ(info.sample_type, type);
                ASSERT_TRUE(codec::decode(renamed, image, error)) << error;
                EXPECT_EQ(image.data, expected);
            }
        }
    }
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
    TEST(ProductionImagePaths, PngJpegAndTiffRoundTripLongUnicodePaths) {
        namespace fs = lfs::core::native_fs;
#ifdef LFS_REQUIRE_LONG_PATHS_DISABLED
        ASSERT_EQ(fs::process_long_paths_enabled(), std::optional<bool>(false));
#endif
        TemporaryDirectory temporary;
        auto directory = temporary.path;
        for (int i = 0; i < 12; ++i)
            directory /= fs::path(u8"directory_\u6e2c\u8a66_01234567890123456789");
        fs::create_directories(directory);
        constexpr int width = 19, height = 17, channels = 3;
        std::vector<std::uint8_t> pixels(width * height * channels);
        for (size_t i = 0; i < pixels.size(); ++i)
            pixels[i] = static_cast<std::uint8_t>(i * 17);
        for (const auto* extension : {".png", ".jpg", ".tif"}) {
            SCOPED_TRACE(extension);
            auto path = directory / fs::path(u8"image_\u6e2c\u8a66");
            path += extension;
            std::string error;
            if (std::string(extension) == ".png")
                ASSERT_TRUE(codec::write_png(path, pixels.data(), width, height, channels, 8, 6, std::nullopt, error)) << error;
            else if (std::string(extension) == ".jpg")
                ASSERT_TRUE(codec::write_jpeg(path, pixels.data(), width, height, channels, 95, std::nullopt, error)) << error;
            else
                ASSERT_TRUE(codec::write_tiff(path, pixels.data(), width, height, channels, error)) << error;
            codec::Image image;
            ASSERT_TRUE(codec::decode(path, image, error)) << error;
            EXPECT_EQ(image.width, width);
            EXPECT_EQ(image.height, height);
            EXPECT_EQ(image.channels, channels);
            if (std::string(extension) != ".jpg")
                EXPECT_EQ(image.data, pixels);
        }
    }
    TEST(ProductionImagePaths, TiffErrorsNameTheCallersLongPath) {
        namespace fs = lfs::core::native_fs;
        TemporaryDirectory temporary;
        auto directory = temporary.path;
        for (int i = 0; i < 12; ++i)
            directory /= fs::path(u8"directory_\u6e2c\u8a66_01234567890123456789");
        fs::create_directories(directory);
        const auto path = directory / "corrupt.tif";
        {
            std::ofstream file(fs::native_path(path), std::ios::binary);
            file << "II*" << '\0' << "not a tiff directory";
        }
        codec::Image image;
        std::string error;
        ASSERT_FALSE(codec::decode(path, image, error));
        EXPECT_EQ(error.find(R"(\\?\)"), std::string::npos) << error;
    }
    // Fails if a TIFF that libtiff cannot open reports a generic codec failure instead of the reason.
    TEST(ProductionImagePaths, TiffOpenErrorsCarryLibtiffsReason) {
        namespace fs = lfs::core::native_fs;
        TemporaryDirectory temporary;
        const auto path = temporary.path / "corrupt.tif";
        {
            std::ofstream file(fs::native_path(path), std::ios::binary);
            file << "II*" << '\0' << "not a tiff directory";
        }
        codec::Image image;
        std::string error;
        ASSERT_FALSE(codec::decode(path, image, error));
        EXPECT_TRUE(error.starts_with("TIFF open failed: ")) << error;
        EXPECT_NE(error.find("TIFFReadDirectory: "), std::string::npos) << error;
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
    TEST(ProductionJpeg, MemoryOutputMatchesFilesWithPaddedRowsAndBinaryComments) {
        TemporaryDirectory temporary;
        for (int channels : {1, 3}) {
            for (const auto dimensions : {std::pair{1, 1}, std::pair{19, 17}, std::pair{257, 241}}) {
                const auto [width, height] = dimensions;
                const size_t row_bytes = size_t(width) * channels;
                const size_t stride = row_bytes + 11;
                std::vector<uint8_t> packed(row_bytes * height);
                std::vector<uint8_t> padded(stride * (height - 1) + row_bytes, 0xcd);
                uint32_t random = 314159;
                for (int y = 0; y < height; ++y)
                    for (size_t x = 0; x < row_bytes; ++x) {
                        random = random * 1664525u + 1013904223u;
                        padded[size_t(y) * stride + x] = packed[size_t(y) * row_bytes + x] = uint8_t(random >> 24);
                    }
                const auto original = padded;
                for (int quality : {0, 1, 95, 100, 101}) {
                    for (bool full_chroma : {false, true}) {
                        codec::JpegEncodeOptions options;
                        options.quality = quality;
                        options.full_chroma = full_chroma;
                        options.comment = quality == 100 ? std::string(65540, 'x') : std::string("memory\0jpeg", 11);
                        const auto path = temporary.path / "reference.jpg";
                        std::string error;
                        ASSERT_TRUE(codec::write_jpeg(path, packed.data(), width, height, channels, quality, options.comment, error, full_chroma)) << error;
                        std::ifstream file(path, std::ios::binary);
                        const std::vector<uint8_t> expected{std::istreambuf_iterator<char>(file), {}};
                        std::vector<uint8_t> encoded{42};
                        ASSERT_TRUE(codec::encode_jpeg_memory(padded, width, height, channels, stride, encoded, options));
                        EXPECT_EQ(encoded, expected);
                        EXPECT_EQ(padded, original);
                        jpeg_decompress_struct info{};
                        jpeg_error_mgr jpeg_error{};
                        info.err = jpeg_std_error(&jpeg_error);
                        jpeg_create_decompress(&info);
                        jpeg_mem_src(&info, encoded.data(), static_cast<unsigned long>(encoded.size()));
                        jpeg_save_markers(&info, JPEG_COM, 65535);
                        ASSERT_EQ(jpeg_read_header(&info, TRUE), JPEG_HEADER_OK);
                        EXPECT_EQ(info.image_width, unsigned(width));
                        EXPECT_EQ(info.image_height, unsigned(height));
                        EXPECT_EQ(info.num_components, channels);
                        EXPECT_EQ(info.comp_info[0].h_samp_factor, channels == 1 || full_chroma ? 1 : 2);
                        EXPECT_EQ(info.comp_info[0].v_samp_factor, channels == 1 || full_chroma ? 1 : 2);
                        ASSERT_NE(info.marker_list, nullptr);
                        EXPECT_EQ(info.marker_list->data_length, std::min<size_t>(options.comment->size(), 65533));
                        EXPECT_EQ(std::memcmp(info.marker_list->data, options.comment->data(), info.marker_list->data_length), 0);
                        jpeg_destroy_decompress(&info);
                    }
                }
            }
        }
    }
    TEST(ProductionJpeg, MemoryByteLimitKeepsPreviousOutputAndEncoderCanBeReused) {
        for (const int dimension : {1, 129}) {
            std::vector<uint8_t> pixels(size_t(dimension) * dimension * 3);
            uint32_t random = 1234;
            for (auto& value : pixels) {
                random = random * 1664525u + 1013904223u;
                value = uint8_t(random >> 24);
            }
            codec::JpegEncodeOptions options;
            options.full_chroma = true;
            std::vector<uint8_t> expected;
            ASSERT_TRUE(codec::encode_jpeg_memory(pixels, dimension, dimension, 3, size_t(dimension) * 3, expected, options));
            if (dimension > 1)
                ASSERT_GT(expected.size(), 4096);
            options.max_output_bytes = expected.size();
            std::vector<uint8_t> output;
            ASSERT_TRUE(codec::encode_jpeg_memory(pixels, dimension, dimension, 3, size_t(dimension) * 3, output, options));
            EXPECT_EQ(output, expected);
            for (size_t limit : {size_t{1}, expected.size() - 1}) {
                options.max_output_bytes = limit;
                const auto result = codec::encode_jpeg_memory(pixels, dimension, dimension, 3, size_t(dimension) * 3, output, options);
                ASSERT_FALSE(result);
                EXPECT_EQ(result.error().code(), lfs::ErrorCode::ResourceExhausted);
                EXPECT_EQ(output, expected);
            }
            options.max_output_bytes = expected.size();
            ASSERT_TRUE(codec::encode_jpeg_memory(pixels, dimension, dimension, 3, size_t(dimension) * 3, output, options));
            EXPECT_EQ(output, expected);
        }
    }
    TEST(ProductionJpeg, MemoryDeterministicStressPreservesRowsAndLimitBoundaries) {
        TemporaryDirectory temporary;
        uint32_t random = 0x4a504547;
        const auto next = [&] {
            random = random * 1664525u + 1013904223u;
            return random;
        };
        for (int iteration = 0; iteration < 512; ++iteration) {
            SCOPED_TRACE(iteration);
            const int width = 1 + next() % 97, height = 1 + next() % 83;
            const int channels = next() & 0x100 ? 1 : 3;
            const size_t row_bytes = size_t(width) * channels;
            const size_t stride = row_bytes + next() % 33;
            // Offset input and omit final-row padding to expose accidental
            // alignment assumptions and reads beyond the declared sample span.
            std::vector<uint8_t> storage(1 + stride * (height - 1) + row_bytes, 0xcd);
            std::vector<uint8_t> packed(row_bytes * height);
            for (int y = 0; y < height; ++y)
                for (size_t x = 0; x < row_bytes; ++x)
                    storage[1 + size_t(y) * stride + x] = packed[size_t(y) * row_bytes + x] = uint8_t(next() >> 24);
            const auto original = storage;
            const auto input = std::span<const uint8_t>(storage).subspan(1);
            codec::JpegEncodeOptions options;
            options.quality = int(next() % 105) - 2;
            options.full_chroma = next() & 0x100;
            if (iteration % 3 == 0)
                options.comment = std::string(next() % 129, char(next() >> 24));
            std::vector<uint8_t> reference;
            ASSERT_TRUE(codec::encode_jpeg_memory(input, width, height, channels, stride, reference, options));
            const auto path = temporary.path / "stress.jpg";
            std::string error;
            ASSERT_TRUE(codec::write_jpeg(path, packed.data(), width, height, channels, options.quality, options.comment, error, options.full_chroma)) << error;
            std::ifstream file(path, std::ios::binary);
            EXPECT_EQ(reference, (std::vector<uint8_t>{std::istreambuf_iterator<char>(file), {}}));
            for (const size_t limit : {size_t{1}, size_t{4095}, size_t{4096}, size_t{4097}, reference.size() - 1, reference.size(), reference.size() + 1}) {
                options.max_output_bytes = limit;
                std::vector<uint8_t> output{7, 8, 9};
                const auto result = codec::encode_jpeg_memory(input, width, height, channels, stride, output, options);
                ASSERT_EQ(bool(result), limit >= reference.size());
                if (result)
                    EXPECT_EQ(output, reference);
                else {
                    EXPECT_EQ(result.error().code(), lfs::ErrorCode::ResourceExhausted);
                    EXPECT_EQ(output, (std::vector<uint8_t>{7, 8, 9}));
                }
            }
            EXPECT_EQ(storage, original);
            options.max_output_bytes = reference.size();
            if (height == 1) {
                std::vector<uint8_t> output;
                ASSERT_TRUE(codec::encode_jpeg_memory(input, width, height, channels, std::numeric_limits<size_t>::max(), output, options));
                EXPECT_EQ(output, reference);
            }
            auto aliased = packed;
            options.max_output_bytes = 1;
            EXPECT_FALSE(codec::encode_jpeg_memory(aliased, width, height, channels, row_bytes, aliased, options));
            EXPECT_EQ(aliased, packed);
        }
    }
    TEST(ProductionJpeg, MemoryLargeImagesPreserveFileEncodingAndRecoverFromLimits) {
        TemporaryDirectory temporary;
        for (const auto dimensions : {std::pair{3840, 2160}, std::pair{7680, 4320}}) {
            const auto [width, height] = dimensions;
            SCOPED_TRACE(::testing::Message() << width << 'x' << height);
            std::vector<uint8_t> pixels(size_t(width) * height * 3);
            uint32_t random = 42;
            for (auto& value : pixels) {
                random = random * 1664525u + 1013904223u;
                value = uint8_t(random >> 24);
            }
            codec::JpegEncodeOptions options;
            options.quality = 100;
            options.full_chroma = true;
            options.max_output_bytes = 256 * 1024 * 1024;
            std::vector<uint8_t> encoded;
            ASSERT_TRUE(codec::encode_jpeg_memory(pixels, width, height, 3, size_t(width) * 3, encoded, options));
            const auto path = temporary.path / "large.jpg";
            std::string error;
            ASSERT_TRUE(codec::write_jpeg(path, pixels.data(), width, height, 3, options.quality, {}, error, options.full_chroma)) << error;
            std::ifstream file(path, std::ios::binary);
            std::array<char, 4096> block;
            size_t offset = 0;
            while (file.read(block.data(), block.size()) || file.gcount()) {
                const auto count = size_t(file.gcount());
                ASSERT_LE(count, encoded.size() - offset);
                ASSERT_EQ(std::memcmp(encoded.data() + offset, block.data(), count), 0);
                offset += count;
            }
            ASSERT_TRUE(file.eof());
            ASSERT_EQ(offset, encoded.size());
            codec::Image decoded;
            ASSERT_TRUE(codec::decode_memory(encoded.data(), encoded.size(), decoded, error)) << error;
            EXPECT_EQ(decoded.width, width);
            EXPECT_EQ(decoded.height, height);
            EXPECT_EQ(decoded.channels, 3);
            EXPECT_EQ(decoded.data.size(), pixels.size());
            decoded = {};
            if (width == 7680)
                ASSERT_GT(encoded.size(), codec::JpegEncodeOptions{}.max_output_bytes);
            std::vector<size_t> limits{4096, encoded.size() - 1};
            if (width == 7680)
                limits.push_back(codec::JpegEncodeOptions{}.max_output_bytes);
            for (const auto limit : limits) {
                options.max_output_bytes = limit;
                std::vector<uint8_t> output{7, 8, 9};
                const auto result = codec::encode_jpeg_memory(pixels, width, height, 3, size_t(width) * 3, output, options);
                ASSERT_FALSE(result);
                EXPECT_EQ(result.error().code(), lfs::ErrorCode::ResourceExhausted);
                EXPECT_EQ(output, (std::vector<uint8_t>{7, 8, 9}));
            }
            // A failed large encode must not poison the next independent call.
            std::vector<uint8_t> output;
            ASSERT_TRUE(codec::encode_jpeg_memory(std::span<const uint8_t>(pixels).first(3), 1, 1, 3, 3, output));
        }
    }
    TEST(ProductionJpeg, MemoryInputValidationRejectsTruncationOverflowAndUnsupportedChannels) {
        const std::vector<uint8_t> pixels(19 * 17 * 3, 128);
        std::vector<uint8_t> output{7, 8, 9};
        const auto invalid = [&](std::span<const uint8_t> bytes, int width, int height, int channels, size_t stride, codec::JpegEncodeOptions options = {}) {
            const auto result = codec::encode_jpeg_memory(bytes, width, height, channels, stride, output, options);
            ASSERT_FALSE(result);
            EXPECT_EQ(result.error().code(), lfs::ErrorCode::InvalidArgument);
            EXPECT_EQ(output, (std::vector<uint8_t>{7, 8, 9}));
        };
        invalid({}, 19, 17, 3, 57);
        invalid(pixels, 0, 17, 3, 57);
        invalid(pixels, 19, -1, 3, 57);
        invalid(pixels, JPEG_MAX_DIMENSION + 1, 17, 3, 57);
        invalid(pixels, 19, 17, 2, 57);
        invalid(pixels, 19, 17, 4, 76);
        invalid(pixels, 19, 17, 3, 0);
        invalid(pixels, 19, 17, 3, 56);
        invalid(std::span(pixels).first(pixels.size() - 1), 19, 17, 3, 57);
        invalid(pixels, 19, 2, 3, std::numeric_limits<size_t>::max());
        codec::JpegEncodeOptions options;
        options.max_output_bytes = 0;
        invalid(pixels, 19, 17, 3, 57, options);
        options.max_output_bytes = std::numeric_limits<size_t>::max();
        invalid(pixels, 19, 17, 3, 57, options);
    }
    TEST(ProductionJpeg, MemoryAliasingAndConcurrentCallsKeepOwnedIndependentResults) {
        std::vector<uint8_t> aliased(19 * 17 * 3, 128);
        ASSERT_TRUE(codec::encode_jpeg_memory(aliased, 19, 17, 3, 57, aliased));
        codec::Image decoded;
        std::string error;
        ASSERT_TRUE(codec::decode_memory(aliased.data(), aliased.size(), decoded, error)) << error;
        EXPECT_EQ(decoded.data, (std::vector<uint8_t>(19 * 17 * 3, 128)));
        std::vector<std::future<bool>> jobs;
        for (int i = 0; i < 8; ++i)
            jobs.emplace_back(std::async(std::launch::async, [i] {
                std::vector<uint8_t> pixels(19 * 17 * 3, uint8_t(32 + i));
                std::vector<uint8_t> encoded;
                codec::JpegEncodeOptions options;
                options.full_chroma = true;
                if (!codec::encode_jpeg_memory(pixels, 19, 17, 3, 57, encoded, options))
                    return false;
                codec::Image decoded;
                std::string error;
                return codec::decode_memory(encoded.data(), encoded.size(), decoded, error) && decoded.data == pixels;
            }));
        for (auto& job : jobs)
            EXPECT_TRUE(job.get());
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
        EXPECT_NE(error.find("/dev/full"), std::string::npos) << error;
        EXPECT_FALSE(error.empty());
    }
#endif

    // Evaluation images and FLIP maps are written through these codecs. This executable has no long-path
    // manifest, so on Windows a writer or reader that hands the OS the plain path fails beyond 259 units.
    void round_trip_images_around_the_extended_path_switch(const std::filesystem::path& root) {
        using namespace lfs::test::long_paths;
        constexpr int width = 7, height = 5, channels = 3;
        std::vector<std::uint8_t> pixels(std::size_t(width) * height * channels);
        for (std::size_t i = 0; i < pixels.size(); ++i)
            pixels[i] = static_cast<std::uint8_t>(i * 29);
        for (const auto length : kLengthsAroundExtendedPathSwitch) {
            for (const std::string extension : {".png", ".jpg", ".tif"}) {
                SCOPED_TRACE(::testing::Message() << length << extension);
                const auto path = with_native_length(root / unicode_folder() / std::to_string(length), length,
                                                     lfs::core::utf8_to_path("bild 🙂" + extension));
                ASSERT_EQ(path.native().size(), length);
                std::error_code created;
                lfs::core::native_fs::create_directories(path.parent_path(), created);
                ASSERT_FALSE(created) << created.message();
                std::string error;
                const bool written =
                    extension == ".png"   ? codec::write_png(path, pixels.data(), width, height, channels, 8, 6, std::nullopt, error)
                    : extension == ".jpg" ? codec::write_jpeg(path, pixels.data(), width, height, channels, 95, std::nullopt, error)
                                          : codec::write_tiff(path, pixels.data(), width, height, channels, error);
                ASSERT_TRUE(written) << error;
                codec::Probe probe;
                ASSERT_TRUE(codec::probe(path, probe, error)) << error;
                EXPECT_EQ(probe.width, width);
                EXPECT_EQ(probe.height, height);
                codec::Image decoded;
                ASSERT_TRUE(codec::decode(path, decoded, error)) << error;
                EXPECT_EQ(decoded.width, width);
                if (extension != ".jpg")
                    EXPECT_EQ(decoded.data, pixels);
            }
        }
    }

    TEST(ProductionCodecs, WritersAndReadersWorkOnBothSidesOfTheExtendedPathSwitch) {
        lfs::test::long_paths::LongPathDirectory temporary;
        round_trip_images_around_the_extended_path_switch(temporary.path);
    }

    // Fails on Windows if atomic publishing hands the OS a plain path: its temporary name is longer
    // than the target, so a write fails from about 230 units on.
    TEST(ProductionCodecs, AtomicWritersWorkOnBothSidesOfTheExtendedPathSwitch) {
        using namespace lfs::test::long_paths;
        LongPathDirectory temporary;
        constexpr int width = 7, height = 5, channels = 3;
        std::vector<std::uint8_t> pixels(std::size_t(width) * height * channels);
        for (std::size_t i = 0; i < pixels.size(); ++i)
            pixels[i] = static_cast<std::uint8_t>(i * 29);
        std::vector<std::uint8_t> samples(pixels.size() * 2);
        for (std::size_t i = 0; i < samples.size(); ++i)
            samples[i] = static_cast<std::uint8_t>(i * 13);
        for (const auto length : kLengthsAroundExtendedPathSwitch) {
            for (const std::string extension : {".png", ".tif"}) {
                SCOPED_TRACE(::testing::Message() << length << extension);
                const auto path = with_native_length(temporary.path / unicode_folder() / std::to_string(length), length,
                                                     lfs::core::utf8_to_path("bild 🙂" + extension));
                ASSERT_EQ(path.native().size(), length);
                const auto written =
                    extension == ".png"
                        ? codec::write_image_u8(path, pixels.data(), width, height, channels, 95, std::nullopt, {})
                        : codec::write_tiff_u16(path, samples, width, height, channels,
                                                std::size_t(width) * channels * 2, std::nullopt, {});
                ASSERT_TRUE(written) << written.error().detail();
                codec::Image decoded;
                std::string error;
                ASSERT_TRUE(codec::decode(path, decoded, error)) << error;
                EXPECT_EQ(decoded.data, extension == ".png" ? pixels : samples);
            }
        }
    }

#ifdef _WIN32
    TEST(ProductionCodecs, WritersAndReadersWorkThroughANetworkShare) {
        const auto share = lfs::test::long_paths::network_share();
        if (!share) {
            if (lfs::test::long_paths::network_share_required()) {
                FAIL() << "LFS_TEST_UNC_ROOT and LFS_TEST_UNC_LOCAL must name a reachable network share";
            }
            GTEST_SKIP() << "No network share provisioned (LFS_TEST_UNC_ROOT, LFS_TEST_UNC_LOCAL)";
        }
        lfs::test::long_paths::LongPathDirectory run(share->local);
        round_trip_images_around_the_extended_path_switch(share->network / run.path.filename());
    }
#endif
} // namespace
