/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/image_avif.hpp"
#include "core/image_heif.hpp"
#include "core/source_metadata.hpp"
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <gtest/gtest.h>
#include <libheif/heif.h>
#include <stdexcept>

namespace {
    namespace c = lfs::core::image_codecs;
    namespace fs = std::filesystem;
    class AvifEncode : public testing::Test {
    protected:
        fs::path directory;
        void SetUp() override {
            directory = fs::temp_directory_path() / ("avif-output-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
            fs::create_directories(directory);
            ASSERT_TRUE(c::avif_encoder_available());
        }
        void TearDown() override {
            std::error_code error;
            fs::remove_all(directory, error);
        }
        static std::vector<uint8_t> bytes(const fs::path& path) {
            std::ifstream stream(path, std::ios::binary);
            return {(std::istreambuf_iterator<char>(stream)), {}};
        }
        static c::Image independent(std::span<const uint8_t> data, int bits, int channels) {
            const auto require = [](heif_error e) { if (e.code != heif_error_Ok) throw std::runtime_error(e.message); };
            std::unique_ptr<heif_context, decltype(&heif_context_free)> context(heif_context_alloc(), heif_context_free);
            require(heif_context_read_from_memory_without_copy(context.get(), data.data(), data.size(), nullptr));
            heif_image_handle* raw_handle = nullptr;
            require(heif_context_get_primary_image_handle(context.get(), &raw_handle));
            std::unique_ptr<heif_image_handle, decltype(&heif_image_handle_release)> handle(raw_handle, heif_image_handle_release);
            const bool alpha = channels == 2 || channels == 4;
            const bool gray = channels <= 2;
            const auto chroma = gray ? heif_chroma_monochrome : bits == 8 ? (alpha ? heif_chroma_interleaved_RGBA : heif_chroma_interleaved_RGB)
                                                                          : (alpha ? heif_chroma_interleaved_RRGGBBAA_LE : heif_chroma_interleaved_RRGGBB_LE);
            heif_image* raw_image = nullptr;
            require(heif_decode_image(handle.get(), &raw_image, gray ? heif_colorspace_monochrome : heif_colorspace_RGB, chroma, nullptr));
            std::unique_ptr<heif_image, decltype(&heif_image_release)> image(raw_image, heif_image_release);
            c::Image result{heif_image_handle_get_width(handle.get()), heif_image_handle_get_height(handle.get()), channels, bits == 8 ? c::SampleType::UInt8 : c::SampleType::UInt16, {}};
            const size_t sample = bits == 8 ? 1 : 2;
            result.data.resize(size_t(result.width) * result.height * channels * sample);
            for (int channel = 0; channel < (gray ? channels : 1); ++channel) {
                size_t stride = 0;
                const auto* plane = heif_image_get_plane_readonly2(image.get(), gray ? (channel == 0 ? heif_channel_Y : heif_channel_Alpha) : heif_channel_interleaved, &stride);
                if (!plane)
                    throw std::runtime_error("missing independent output plane");
                for (int y = 0; y < result.height; ++y)
                    for (int x = 0; x < result.width; ++x)
                        if (gray)
                            std::memcpy(result.data.data() + (size_t(y) * result.width * channels + size_t(x) * channels + channel) * sample, plane + size_t(y) * stride + size_t(x) * sample, sample);
                        else
                            std::memcpy(result.data.data() + (size_t(y) * result.width + x) * channels * sample, plane + size_t(y) * stride + size_t(x) * channels * sample, channels * sample);
            }
            return result;
        }
    };
    std::string primary_xmp(std::span<const uint8_t> data) {
        std::unique_ptr<heif_context, decltype(&heif_context_free)> context(heif_context_alloc(), heif_context_free);
        if (!context || heif_context_read_from_memory_without_copy(context.get(), data.data(), data.size(), nullptr).code != heif_error_Ok)
            throw std::runtime_error("cannot read encoded AVIF");
        heif_image_handle* raw = nullptr;
        if (heif_context_get_primary_image_handle(context.get(), &raw).code != heif_error_Ok)
            throw std::runtime_error("missing AVIF primary image");
        std::unique_ptr<heif_image_handle, decltype(&heif_image_handle_release)> image(raw, heif_image_handle_release);
        heif_item_id id = 0;
        if (heif_image_handle_get_number_of_metadata_blocks(image.get(), "mime") != 1 ||
            heif_image_handle_get_list_of_metadata_block_IDs(image.get(), "mime", &id, 1) != 1)
            throw std::runtime_error("missing unique primary XMP MIME item");
        const char* content = heif_image_handle_get_metadata_content_type(image.get(), id);
        if (!content || std::string_view(content) != "application/rdf+xml")
            throw std::runtime_error("unexpected primary metadata MIME type");
        std::string packet(heif_image_handle_get_metadata_size(image.get(), id), '\0');
        if (heif_image_handle_get_metadata(image.get(), id, packet.data()).code != heif_error_Ok)
            throw std::runtime_error("cannot read primary XMP packet");
        return packet;
    }

    TEST_F(AvifEncode, ExplicitXmpPreservesSamplesAndEmptyPacketDefaults) {
        const std::string packet = "<r:RDF xmlns:r=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\"><r:Description r:about=\"\"/></r:RDF>";
        for (int depth : {8, 10, 12})
            for (int channels : {1, 2, 3, 4})
                for (bool lossless : {false, true}) {
                    SCOPED_TRACE(testing::Message() << depth << "/" << channels << "/" << lossless);
                    c::AvifEncodeOptions options;
                    options.bit_depth = depth;
                    options.lossless = lossless;
                    const size_t sample = depth == 8 ? 1 : 2;
                    std::vector<uint8_t> pixels(6 * channels * sample);
                    for (size_t i = 0; i < 6 * channels; ++i) {
                        const uint16_t value = uint16_t((i * 71 + depth) % (size_t{1} << depth));
                        if (sample == 1)
                            pixels[i] = uint8_t(value);
                        else
                            std::memcpy(pixels.data() + i * sample, &value, sample);
                    }
                    const auto original = pixels;
                    std::vector<uint8_t> plain, empty, embedded;
                    ASSERT_TRUE(c::encode_avif_memory(pixels, 3, 2, channels, 3 * channels * sample, plain, options));
                    ASSERT_TRUE(c::encode_avif_memory(pixels, 3, 2, channels, 3 * channels * sample, empty, options, {}));
                    EXPECT_EQ(empty, plain);
                    ASSERT_TRUE(c::encode_avif_memory(pixels, 3, 2, channels, 3 * channels * sample, embedded, options, packet));
                    EXPECT_EQ(primary_xmp(embedded), packet);
                    EXPECT_EQ(independent(embedded, depth, channels).data, independent(plain, depth, channels).data);
                    if (lossless)
                        EXPECT_EQ(independent(embedded, depth, channels).data, pixels);
                    EXPECT_EQ(pixels, original);
                    const auto path = directory / (std::to_string(depth) + "-" + std::to_string(channels) + "-" + std::to_string(lossless) + ".avif");
                    ASSERT_TRUE(c::write_avif(path, pixels, 3, 2, channels, 3 * channels * sample, options, {}, packet));
                    EXPECT_EQ(bytes(path), embedded);
                }
    }

    TEST_F(AvifEncode, ConcurrentXmpPacketsStayWithTheirOwnImage) {
        std::vector<std::future<std::pair<std::vector<uint8_t>, std::string>>> calls;
        for (int index = 0; index < 4; ++index)
            calls.push_back(std::async(std::launch::async, [index] {
                const std::array<uint8_t, 3> pixels{uint8_t(index * 41), 73, 191};
                std::string packet = "<r:RDF xmlns:r=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\"><r:Description r:about=\"\" xmlns:t=\"urn:lfs:test\"><t:Source>" +
                                     std::to_string(index) + std::string(65536, char('a' + index)) + "</t:Source></r:Description></r:RDF>";
                const auto expected = packet;
                std::vector<uint8_t> output;
                if (!c::encode_avif_memory(pixels, 1, 1, 3, 3, output, {}, packet))
                    throw std::runtime_error("concurrent XMP encoding failed");
                packet.assign("caller buffer reused");
                return std::pair{std::move(output), expected};
            }));
        for (size_t index = 0; index < calls.size(); ++index) {
            const auto [output, expected] = calls[index].get();
            EXPECT_EQ(primary_xmp(output), expected);
            EXPECT_EQ(independent(output, 8, 3).data, (std::vector<uint8_t>{uint8_t(index * 41), 73, 191}));
        }
    }

    TEST_F(AvifEncode, ExplicitXmpBoundaryRefusalCancellationAndReuseAreAtomic) {
        const std::array<uint8_t, 18> pixels{3, 17, 251, 9, 64, 128, 0, 255, 7, 33, 42, 81, 120, 4, 199, 23, 88, 161};
        std::string packet = "<r:RDF xmlns:r=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\"><r:Description r:about=\"\" xmlns:t=\"urn:lfs:test\"><t:value>";
        const std::string_view suffix = "</t:value></r:Description></r:RDF>";
        packet.append(32 * 1024 * 1024 - packet.size() - suffix.size(), 'a');
        packet += suffix;
        c::AvifEncodeOptions options;
        size_t polls = 0;
        options.cancelled = [&] { ++polls; return false; };
        std::vector<uint8_t> encoded;
        ASSERT_TRUE(c::encode_avif_memory(pixels, 3, 2, 3, 9, encoded, options, packet));
        ASSERT_GT(polls, 1u);
        EXPECT_EQ(primary_xmp(encoded), packet);
        EXPECT_EQ(independent(encoded, 8, 3).data, (std::vector<uint8_t>(pixels.begin(), pixels.end())));
        options.cancelled = {};
        const auto path = directory / "boundary.avif";
        ASSERT_TRUE(c::write_avif(path, pixels, 3, 2, 3, 9, options, {}, packet));
        const auto original = encoded;
        EXPECT_EQ(bytes(path), original);
        EXPECT_EQ(c::read_source_metadata(path).xmp.status, c::MetadataStatus::Limited);
        for (const auto& bad : {packet + "a", std::string("x\0mp", 4)}) {
            const auto refused = c::encode_avif_memory(pixels, 3, 2, 3, 9, encoded, options, bad);
            ASSERT_FALSE(refused);
            EXPECT_EQ(refused.error().code(), lfs::ErrorCode::InvalidArgument);
            EXPECT_EQ(encoded, original);
            const auto file = c::write_avif(path, pixels, 3, 2, 3, 9, options, {.overwrite = true}, bad);
            ASSERT_FALSE(file);
            EXPECT_EQ(file.error().code(), lfs::ErrorCode::InvalidArgument);
            EXPECT_EQ(bytes(path), original);
        }
        for (int failure = 0; failure < 3; ++failure) {
            auto limited = options;
            if (failure == 0)
                limited.max_output_bytes = 64;
            if (failure == 1)
                limited.cancelled = [] { return true; };
            if (failure == 2)
                limited.cancelled = [count = size_t{0}, polls]() mutable { return ++count >= polls; };
            const auto memory = c::encode_avif_memory(pixels, 3, 2, 3, 9, encoded, limited, packet);
            ASSERT_FALSE(memory);
            EXPECT_EQ(memory.error().code(), failure == 0 ? lfs::ErrorCode::ResourceExhausted : lfs::ErrorCode::Cancelled);
            EXPECT_EQ(encoded, original);
            // Reset the stateful cancellation callback for the independent file call.
            if (failure == 2)
                limited.cancelled = [count = size_t{0}, polls]() mutable { return ++count >= polls; };
            const auto file = c::write_avif(path, pixels, 3, 2, 3, 9, limited, {.overwrite = true}, packet);
            ASSERT_FALSE(file);
            EXPECT_EQ(file.error().code(), failure == 0 ? lfs::ErrorCode::ResourceExhausted : lfs::ErrorCode::Cancelled);
            EXPECT_EQ(bytes(path), original);
            for (const auto& entry : fs::directory_iterator(directory))
                EXPECT_EQ(entry.path(), path);
        }
        const std::string_view small = "<r:RDF xmlns:r=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\"><r:Description r:about=\"\"/></r:RDF>";
        ASSERT_TRUE(c::write_avif(path, pixels, 3, 2, 3, 9, options, {.overwrite = true}, small));
        EXPECT_EQ(primary_xmp(bytes(path)), small);
        std::vector<uint8_t> alias(pixels.begin(), pixels.end());
        ASSERT_TRUE(c::encode_avif_memory(alias, 3, 2, 3, 9, alias, options, small));
        EXPECT_EQ(independent(alias, 8, 3).data, (std::vector<uint8_t>(pixels.begin(), pixels.end())));
    }

    TEST_F(AvifEncode, ProviderPolicyKeepsHevcUnavailable) {
        EXPECT_FALSE(c::avif_encoder_version().empty());
        EXPECT_FALSE(heif_have_encoder_for_format(heif_compression_HEVC));
        EXPECT_FALSE(heif_have_decoder_for_format(heif_compression_HEVC));
        std::array<const heif_decoder_descriptor*, 8> decoders{};
        const int count = heif_get_decoder_descriptors(heif_compression_AV1, decoders.data(), int(decoders.size()));
        ASSERT_EQ(count, 1);
        EXPECT_STREQ(heif_decoder_descriptor_get_id_name(decoders[0]), "dav1d");
    }
    TEST_F(AvifEncode, GenericWriterRejectsMetadataAndKeepsStraightAlpha) {
        const std::array<uint8_t, 8> input{31, 71, 127, 0, 5, 9, 11, 255};
        const auto path = directory / "image.AVIF";
        ASSERT_TRUE(c::write_image_u8(path, input.data(), 2, 1, 4, 95, std::nullopt, {}));
        const auto before = bytes(path);
        c::Image decoded;
        std::string error;
        ASSERT_TRUE(c::decode_heif(path, decoded, error)) << error;
        EXPECT_EQ(decoded.data, (std::vector<uint8_t>(input.begin(), input.end())));
        EXPECT_FALSE(c::write_image_u8(path, input.data(), 2, 1, 4, 95, std::string("preserve me"), {}));
        EXPECT_EQ(bytes(path), before);
    }
    TEST_F(AvifEncode, LosslessLayoutsPrecisionsPaddingAndIndependentDecoder) {
        constexpr int width = 7, height = 5;
        for (int bits : {8, 10, 12})
            for (int channels : {1, 2, 3, 4}) {
                SCOPED_TRACE(testing::Message() << bits << "/" << channels);
                const size_t sample = bits == 8 ? 1 : 2, row = width * channels * sample, stride = row + 5;
                std::vector<uint8_t> padded(1 + height * stride, 0xcd), packed(height * row);
                for (int y = 0; y < height; ++y)
                    for (int x = 0; x < width; ++x)
                        for (int channel = 0; channel < channels; ++channel) {
                            const uint16_t value = uint16_t((x * 17 + y * 43 + channel * 97) % (1 << bits));
                            auto* target = packed.data() + size_t(y) * row + (x * channels + channel) * sample;
                            if (sample == 1)
                                *target = uint8_t(value);
                            else
                                std::memcpy(target, &value, sample);
                            std::memcpy(padded.data() + 1 + size_t(y) * stride + (x * channels + channel) * sample, target, sample);
                        }
                const auto before = padded;
                c::AvifEncodeOptions options{.bit_depth = bits};
                std::vector<uint8_t> encoded;
                ASSERT_TRUE(c::encode_avif_memory(std::span(padded).subspan(1), width, height, channels, stride, encoded, options));
                EXPECT_EQ(padded, before);
                EXPECT_EQ(independent(encoded, bits, channels).data, packed);
                const auto path = directory / ("image-" + std::to_string(bits) + "-" + std::to_string(channels) + ".avif");
                ASSERT_TRUE(c::write_avif(path, std::span(padded).subspan(1), width, height, channels, stride, options));
                EXPECT_EQ(bytes(path), encoded);
                c::Image app;
                std::string error;
                ASSERT_TRUE(c::decode_heif(encoded, app, error)) << error;
                EXPECT_EQ(app.width, width);
                EXPECT_EQ(app.height, height);
                const int app_channels = channels <= 2 ? channels + 2 : channels;
                ASSERT_EQ(app.channels, app_channels);
                for (size_t pixel = 0; pixel < size_t(width) * height; ++pixel)
                    for (int channel = 0; channel < app_channels; ++channel) {
                        const int source_channel = channels <= 2 ? (channel == 3 ? 1 : 0) : channel;
                        uint16_t original = packed[(pixel * channels + source_channel) * sample];
                        uint16_t decoded = app.data[(pixel * app_channels + channel) * sample];
                        if (sample == 2) {
                            std::memcpy(&original, packed.data() + (pixel * channels + source_channel) * sample, 2);
                            std::memcpy(&decoded, app.data.data() + (pixel * app_channels + channel) * sample, 2);
                            original = uint16_t((uint32_t(original) * 65535 + ((1u << bits) - 1) / 2) / ((1u << bits) - 1));
                        }
                        EXPECT_EQ(decoded, original);
                    }
            }
    }
    TEST_F(AvifEncode, LimitsInvalidViewsCancellationAliasingAndRecovery) {
        const std::array<uint8_t, 12> input{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        std::vector<uint8_t> output{123};
        const auto rejected = [&](c::AvifEncodeOptions options, size_t stride = 6) {
            EXPECT_FALSE(c::encode_avif_memory(input, 2, 2, 3, stride, output, options));
            EXPECT_EQ(output, (std::vector<uint8_t>{123}));
        };
        rejected({.quality = 101});
        rejected({.speed = 10});
        rejected({.bit_depth = 16});
        rejected({.pixels = 3});
        rejected({.max_output_bytes = 1});
        rejected({.threads = 0});
        rejected({}, 5);
        rejected({.cancelled = [] { return true; }});
        rejected({.cancelled = []() -> bool { throw std::runtime_error("callback sentinel"); }});
        rejected({.cancelled = []() -> bool { throw 1; }});
        ASSERT_TRUE(c::encode_avif_memory(input, 2, 2, 3, 6, output));
        auto alias = std::vector<uint8_t>(input.begin(), input.end());
        ASSERT_TRUE(c::encode_avif_memory(alias, 2, 2, 3, 6, alias));
        EXPECT_EQ(alias, output);
        const auto expected = output;
        ASSERT_TRUE(c::encode_avif_memory(input, 2, 2, 3, 6, output, {.max_output_bytes = expected.size()}));
        EXPECT_FALSE(c::encode_avif_memory(input, 2, 2, 3, 6, output, {.max_output_bytes = expected.size() - 1}));
        EXPECT_EQ(output, expected);
    }
    TEST_F(AvifEncode, AtomicPublicationFailureKeepsPreviousDestination) {
        const std::array<uint8_t, 3> input{1, 2, 3};
        const auto path = directory / "previous.avif";
        ASSERT_TRUE(c::write_avif(path, input, 1, 1, 3, 3));
        const auto before = bytes(path);
        EXPECT_FALSE(c::write_avif(path, input, 1, 1, 3, 3, {}, {.overwrite = false}));
        EXPECT_FALSE(c::write_avif(path, input, 1, 1, 3, 3, {.max_output_bytes = 1}, {.overwrite = true}));
        EXPECT_FALSE(c::write_avif(path, input, 1, 1, 3, 3, {}, {.overwrite = true, .cancelled = [] { return true; }}));
        EXPECT_EQ(bytes(path), before);
        EXPECT_EQ(std::distance(fs::directory_iterator(directory), fs::directory_iterator{}), 1);
    }
    TEST_F(AvifEncode, AllocationAndCallbackErrorsKeepTypedDiagnosticsAndDestination) {
        const std::array<uint8_t, 3> input{1, 2, 3};
        std::vector<uint8_t> output{123};
        auto allocation = c::encode_avif_memory(input, 1, 1, 3, 3, output, {.cancelled = []() -> bool { throw std::bad_alloc(); }});
        ASSERT_FALSE(allocation);
        EXPECT_EQ(allocation.error().code(), lfs::ErrorCode::ResourceExhausted);
        auto exception = c::encode_avif_memory(input, 1, 1, 3, 3, output, {.cancelled = []() -> bool { throw std::runtime_error("AVIF detailed sentinel"); }});
        ASSERT_FALSE(exception);
        EXPECT_EQ(exception.error().code(), lfs::ErrorCode::Internal);
        EXPECT_NE(exception.error().detail().find("AVIF detailed sentinel"), std::string_view::npos);
        EXPECT_EQ(output, (std::vector<uint8_t>{123}));
        struct CopyFailure {
            std::shared_ptr<bool> fail;
            explicit CopyFailure(std::shared_ptr<bool> value) : fail(std::move(value)) {}
            CopyFailure(const CopyFailure& other) : fail(other.fail) {
                if (*fail)
                    throw std::bad_alloc();
            }
            bool operator()() const { return false; }
        };
        const auto path = directory / "allocation.avif";
        ASSERT_TRUE(c::write_avif(path, input, 1, 1, 3, 3));
        const auto before = bytes(path);
        auto fail = std::make_shared<bool>(false);
        c::AvifEncodeOptions options;
        options.cancelled = CopyFailure(fail);
        *fail = true;
        auto publication = c::write_avif(path, input, 1, 1, 3, 3, options);
        ASSERT_FALSE(publication);
        EXPECT_EQ(publication.error().code(), lfs::ErrorCode::ResourceExhausted);
        EXPECT_EQ(bytes(path), before);
        EXPECT_EQ(std::distance(fs::directory_iterator(directory), fs::directory_iterator{}), 1);
        *fail = false;
        ASSERT_TRUE(c::write_avif(path, input, 1, 1, 3, 3, options));
        EXPECT_EQ(bytes(path), before);
    }
    TEST_F(AvifEncode, LosslessPreservesEvery12BitCodeAndHiddenColor) {
        std::vector<uint16_t> input(4096 * 4);
        for (size_t i = 0; i < 4096; ++i) {
            input[i * 4] = uint16_t(i);
            input[i * 4 + 1] = uint16_t(4095 - i);
            input[i * 4 + 2] = uint16_t((i * 43) % 4096);
            input[i * 4 + 3] = i % 2 ? 4095 : 0;
        }
        const auto data = std::span(reinterpret_cast<const uint8_t*>(input.data()), input.size() * 2);
        std::vector<uint8_t> encoded;
        ASSERT_TRUE(c::encode_avif_memory(data, 4096, 1, 4, data.size(), encoded, {.bit_depth = 12}));
        const auto decoded = independent(encoded, 12, 4);
        ASSERT_EQ(decoded.data.size(), data.size());
        EXPECT_EQ(std::memcmp(decoded.data.data(), data.data(), data.size()), 0);
        input[0] = 4096;
        const auto before = encoded;
        EXPECT_FALSE(c::encode_avif_memory(data, 4096, 1, 4, data.size(), encoded, {.bit_depth = 12}));
        EXPECT_EQ(encoded, before);
    }
    TEST_F(AvifEncode, LossyUsesDeclaredYcbcr444WhileLosslessKeepsIdentity) {
        for (int bits : {8, 10, 12})
            for (auto primaries : {c::AvifPrimaries::Bt709, c::AvifPrimaries::Bt2020})
                for (bool lossless : {false, true}) {
                    SCOPED_TRACE(testing::Message() << bits << "/" << int(primaries) << "/" << lossless);
                    constexpr int width = 24, height = 16;
                    const size_t sample = bits == 8 ? 1 : 2;
                    std::vector<uint8_t> input(width * height * 3 * sample);
                    for (size_t i = 0; i < width * height * 3; ++i) {
                        const uint16_t code = uint16_t((i * 11 + i / 3 * 5) % (1u << bits));
                        if (sample == 1)
                            input[i] = uint8_t(code);
                        else
                            std::memcpy(input.data() + i * sample, &code, sample);
                    }
                    std::vector<uint8_t> encoded;
                    ASSERT_TRUE(c::encode_avif_memory(input, width, height, 3, width * 3 * sample, encoded,
                                                      {.lossless = lossless, .quality = 90, .bit_depth = bits, .primaries = primaries}));
                    std::unique_ptr<heif_context, decltype(&heif_context_free)> context(heif_context_alloc(), heif_context_free);
                    ASSERT_EQ(heif_context_read_from_memory_without_copy(context.get(), encoded.data(), encoded.size(), nullptr).code, heif_error_Ok);
                    heif_image_handle* raw_handle = nullptr;
                    ASSERT_EQ(heif_context_get_primary_image_handle(context.get(), &raw_handle).code, heif_error_Ok);
                    std::unique_ptr<heif_image_handle, decltype(&heif_image_handle_release)> handle(raw_handle, heif_image_handle_release);
                    heif_color_profile_nclx* raw_color = nullptr;
                    ASSERT_EQ(heif_image_handle_get_nclx_color_profile(handle.get(), &raw_color).code, heif_error_Ok);
                    std::unique_ptr<heif_color_profile_nclx, decltype(&heif_nclx_color_profile_free)> color(raw_color, heif_nclx_color_profile_free);
                    EXPECT_EQ(int(color->matrix_coefficients), lossless ? 0 : primaries == c::AvifPrimaries::Bt709 ? 1
                                                                                                                   : 9);
                    EXPECT_EQ(int(color->color_primaries), primaries == c::AvifPrimaries::Bt709 ? 1 : 9);
                    EXPECT_EQ(color->transfer_characteristics, 13);
                    EXPECT_EQ(color->full_range_flag, 1);
                    heif_image* raw_image = nullptr;
                    ASSERT_EQ(heif_decode_image(handle.get(), &raw_image, heif_colorspace_undefined, heif_chroma_undefined, nullptr).code, heif_error_Ok);
                    std::unique_ptr<heif_image, decltype(&heif_image_release)> image(raw_image, heif_image_release);
                    EXPECT_EQ(heif_image_get_chroma_format(image.get()), heif_chroma_444);
                    const auto decoded = independent(encoded, bits, 3);
                    if (lossless)
                        EXPECT_EQ(decoded.data, input);
                    else {
                        double error = 0;
                        for (size_t i = 0; i < width * height * 3; ++i) {
                            uint16_t a = input[i * sample], b = decoded.data[i * sample];
                            if (sample == 2) {
                                std::memcpy(&a, input.data() + i * sample, sample);
                                std::memcpy(&b, decoded.data.data() + i * sample, sample);
                            }
                            error += std::abs(int(a) - int(b));
                        }
                        EXPECT_LT(error / (width * height * 3), ((1u << bits) - 1) * .04);
                    }
                }
    }
    TEST_F(AvifEncode, LossyAlphaIsExactAndParallelCallsRecover) {
        constexpr int width = 32, height = 24;
        std::vector<uint8_t> input(width * height * 4);
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) {
                const size_t i = (size_t(y) * width + x) * 4;
                input[i] = uint8_t(x * 7);
                input[i + 1] = uint8_t(y * 9);
                input[i + 2] = uint8_t((x + y) * 4);
                input[i + 3] = uint8_t(x * y);
            }
        const auto encode_one = [&] {
            std::vector<uint8_t> bytes;
            if (!c::encode_avif_memory(input, width, height, 4, width * 4, bytes, {.lossless = false, .quality = 70}))
                throw std::runtime_error("parallel AVIF encode failed");
            return bytes;
        };
        const auto encoded = encode_one();
        const auto decoded = independent(encoded, 8, 4);
        double error = 0;
        for (size_t i = 0; i < input.size(); i += 4) {
            EXPECT_EQ(input[i + 3], decoded.data[i + 3]);
            for (size_t channel = 0; channel < 3; ++channel)
                error += std::abs(int(input[i + channel]) - int(decoded.data[i + channel]));
        }
        EXPECT_LT(error / (width * height * 3), 20.0); // Fixture-specific, not a quality guarantee.
        std::vector<std::future<std::vector<uint8_t>>> calls;
        for (int i = 0; i < 4; ++i)
            calls.push_back(std::async(std::launch::async, encode_one));
        for (auto& call : calls)
            EXPECT_EQ(call.get(), encoded);
    }
    TEST_F(AvifEncode, CancellationAfterEncodingDoesNotPublishAndRecovers) {
        const std::array<uint8_t, 12> input{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        const auto path = directory / "previous.avif";
        ASSERT_TRUE(c::write_avif(path, input, 2, 2, 3, 6));
        const auto before = bytes(path);
        bool saw_output = false;
        const auto cancelled = [&] {
            for (const auto& entry : fs::directory_iterator(directory))
                if (entry.path() != path && entry.is_regular_file() && entry.file_size() > 0)
                    saw_output = true;
            return saw_output;
        };
        EXPECT_FALSE(c::write_avif(path, input, 2, 2, 3, 6, {}, {.cancelled = cancelled}));
        EXPECT_TRUE(saw_output);
        EXPECT_EQ(bytes(path), before);
        EXPECT_EQ(std::distance(fs::directory_iterator(directory), fs::directory_iterator{}), 1);
        ASSERT_TRUE(c::write_avif(path, input, 2, 2, 3, 6));
    }
} // namespace
