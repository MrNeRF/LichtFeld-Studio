/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/color_transfer.h"
#include "core/image_jxl.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <gtest/gtest.h>
#include <jxl/cms.h>
#include <jxl/decode.h>
#include <jxl/encode.h>
#include <lcms2.h>
#include <limits>
#include <memory>
#include <stdexcept>

namespace {
    namespace codec = lfs::core::image_codecs;
    // Precision/conformance tests explicitly request native decoded samples.
    // Application/generic dispatch defaults to canonical sRGB instead.
    template <class Source>
    bool native_decode(Source source, codec::Image& image, std::string& error, bool display = false, codec::JxlDecodeLimits limits = {}) {
        limits.color_policy = codec::JxlColorPolicy::Native;
        return codec::decode_jxl(source, image, error, display, limits);
    }
    template <class Source>
    bool native_target(Source source, codec::DecodeTarget& target, codec::Probe& info, std::string& error, bool display = false, codec::JxlDecodeLimits limits = {}) {
        limits.color_policy = codec::JxlColorPolicy::Native;
        return codec::decode_jxl_to_buffer(source, target, info, error, display, limits);
    }
    struct Encoding {
        int width = 3, height = 2, channels = 3, bits = 8;
        bool floating = false, container = false, animation = false, premultiplied = false, lossless = true;
        bool codestream_range = false, depth = false;
        JxlTransferFunction transfer = JXL_TRANSFER_FUNCTION_UNKNOWN;
        double gamma = 0.;
        float intensity = 255.f;
        JxlOrientation orientation = JXL_ORIENT_IDENTITY;
        std::span<const uint8_t> metadata;
        std::span<const uint8_t> icc;
    };
    void check(JxlEncoderStatus status) {
        if (status != JXL_ENC_SUCCESS)
            throw std::runtime_error("Independent JXL fixture encoder failed");
    }
    std::vector<uint8_t> encode(const void* pixels, size_t bytes, Encoding options = {}) {
        std::unique_ptr<JxlEncoder, decltype(&JxlEncoderDestroy)> encoder(JxlEncoderCreate(nullptr), JxlEncoderDestroy);
        if (!encoder)
            throw std::bad_alloc();
        check(JxlEncoderUseContainer(encoder.get(), options.container));
        JxlBasicInfo basic;
        JxlEncoderInitBasicInfo(&basic);
        basic.xsize = options.width;
        basic.ysize = options.height;
        basic.bits_per_sample = options.bits;
        basic.exponent_bits_per_sample = options.floating ? 8 : 0;
        basic.uses_original_profile = options.lossless;
        basic.num_color_channels = options.channels <= 2 ? 1 : 3;
        const bool alpha = options.channels % 2 == 0;
        basic.num_extra_channels = alpha || options.depth ? 1 : 0;
        basic.alpha_bits = alpha ? options.bits : 0;
        basic.alpha_exponent_bits = alpha && options.floating ? 8 : 0;
        basic.alpha_premultiplied = options.premultiplied;
        basic.orientation = options.orientation;
        basic.have_animation = options.animation;
        basic.intensity_target = options.intensity;
        check(JxlEncoderSetBasicInfo(encoder.get(), &basic));
        if (options.depth) {
            JxlExtraChannelInfo depth;
            JxlEncoderInitExtraChannelInfo(JXL_CHANNEL_DEPTH, &depth);
            check(JxlEncoderSetExtraChannelInfo(encoder.get(), 0, &depth));
        }
        JxlColorEncoding color;
        if (options.floating)
            JxlColorEncodingSetToLinearSRGB(&color, basic.num_color_channels == 1);
        else
            JxlColorEncodingSetToSRGB(&color, basic.num_color_channels == 1);
        if (options.transfer != JXL_TRANSFER_FUNCTION_UNKNOWN)
            color.transfer_function = options.transfer;
        if (options.gamma) {
            color.transfer_function = JXL_TRANSFER_FUNCTION_GAMMA;
            color.gamma = 1. / options.gamma;
        }
        if (options.icc.empty())
            check(JxlEncoderSetColorEncoding(encoder.get(), &color));
        else
            check(JxlEncoderSetICCProfile(encoder.get(), options.icc.data(), options.icc.size()));
        if (!options.metadata.empty()) {
            check(JxlEncoderUseBoxes(encoder.get()));
            const JxlBoxType type{'x', 'm', 'l', ' '};
            check(JxlEncoderAddBox(encoder.get(), type, options.metadata.data(), options.metadata.size(), JXL_FALSE));
        }
        auto* frame = JxlEncoderFrameSettingsCreate(encoder.get(), nullptr);
        if (!frame)
            throw std::bad_alloc();
        check(JxlEncoderSetFrameLossless(frame, options.lossless));
        check(JxlEncoderFrameSettingsSetOption(frame, JXL_ENC_FRAME_SETTING_EFFORT, 1));
        if (options.codestream_range) {
            const JxlBitDepth depth{JXL_BIT_DEPTH_FROM_CODESTREAM, static_cast<uint32_t>(options.bits), 0};
            check(JxlEncoderSetFrameBitDepth(frame, &depth));
        }
        if (options.animation) {
            JxlFrameHeader header;
            JxlEncoderInitFrameHeader(&header);
            header.duration = 1;
            check(JxlEncoderSetFrameHeader(frame, &header));
        }
        const JxlPixelFormat format{static_cast<uint32_t>(options.channels), options.floating ? JXL_TYPE_FLOAT : options.bits > 8 ? JXL_TYPE_UINT16
                                                                                                                                  : JXL_TYPE_UINT8,
                                    JXL_NATIVE_ENDIAN, 0};
        check(JxlEncoderAddImageFrame(frame, &format, pixels, bytes));
        if (options.depth) {
            const std::vector<uint8_t> depth(options.width * options.height, 123);
            const JxlPixelFormat depth_format{1, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
            check(JxlEncoderSetExtraChannelBuffer(frame, &depth_format, depth.data(), depth.size(), 0));
        }
        if (options.animation)
            check(JxlEncoderAddImageFrame(frame, &format, pixels, bytes));
        JxlEncoderCloseInput(encoder.get());
        std::vector<uint8_t> result(1024);
        size_t used = 0;
        for (;;) {
            auto* next = result.data() + used;
            size_t available = result.size() - used;
            const auto status = JxlEncoderProcessOutput(encoder.get(), &next, &available);
            used = result.size() - available;
            if (status == JXL_ENC_SUCCESS)
                break;
            if (status != JXL_ENC_NEED_MORE_OUTPUT)
                throw std::runtime_error("JXL fixture output failed");
            result.resize(result.size() * 2);
        }
        result.resize(used);
        return result;
    }

    class JxlCodec : public ::testing::Test {
    protected:
        std::filesystem::path directory;
        void SetUp() override {
            directory = std::filesystem::temp_directory_path() / ("lfs-jxl-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
            ASSERT_TRUE(std::filesystem::create_directory(directory));
        }
        void TearDown() override {
            std::error_code error;
            std::filesystem::remove_all(directory, error);
        }
        std::filesystem::path write(std::span<const uint8_t> bytes) {
            const auto path = directory / std::filesystem::path(u8"immagine_画像.JXL");
            std::ofstream file(path, std::ios::binary);
            file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
            if (!file)
                throw std::runtime_error("Cannot write JXL fixture");
            return path;
        }
    };

    TEST_F(JxlCodec, LosslessIntegerFileMemoryProbeAndGenericDispatchAgree) {
        for (int channels : {1, 2, 3, 4})
            for (int bits : {8, 16})
                for (bool container : {false, true}) {
                    SCOPED_TRACE(std::to_string(channels) + "/" + std::to_string(bits) + "/" + std::to_string(container));
                    std::vector<uint8_t> samples(6 * channels * (bits / 8));
                    for (size_t i = 0; i < samples.size(); ++i)
                        samples[i] = static_cast<uint8_t>(i * 73 + 19);
                    const auto encoded = encode(samples.data(), samples.size(), {.channels = channels, .bits = bits, .container = container});
                    ASSERT_TRUE(codec::is_jxl(encoded));
                    const auto path = write(encoded);
                    codec::Probe probe;
                    std::string error;
                    ASSERT_TRUE(codec::probe(path, probe, error)) << error;
                    EXPECT_EQ(probe.width, 3);
                    EXPECT_EQ(probe.height, 2);
                    EXPECT_EQ(probe.channels, channels);
                    EXPECT_EQ(probe.sample_type, bits == 8 ? codec::SampleType::UInt8 : codec::SampleType::UInt16);
                    codec::Image file, memory;
                    ASSERT_TRUE(codec::decode(path, file, error)) << error;
                    ASSERT_TRUE(codec::decode_memory(encoded.data(), encoded.size(), memory, error)) << error;
                    EXPECT_EQ(file.data, samples);
                    EXPECT_EQ(memory.data, samples);
                    EXPECT_EQ(file.channels, channels);
                }
    }

    TEST_F(JxlCodec, FloatValuesRemainOutsideIntegerRange) {
        const std::array<float, 18> samples{-0.5f, 0.f, 2.f, 0.25f, 1.f, 4.f, 0.5f, 0.75f, 1.5f, -1.f, 3.f, 0.f, 0.5f, 0.25f, 0.125f, 1.f, 2.f, 3.f};
        const auto encoded = encode(samples.data(), sizeof(samples), {.bits = 32, .floating = true});
        codec::Image image;
        std::string error;
        ASSERT_TRUE(native_decode(encoded, image, error)) << error;
        ASSERT_EQ(image.sample_type, codec::SampleType::Float32);
        ASSERT_EQ(image.data.size(), sizeof(samples));
        EXPECT_EQ(std::memcmp(image.data.data(), samples.data(), sizeof(samples)), 0);
    }

    TEST_F(JxlCodec, LimitsAndTruncationPreserveOutputAndRecover) {
        const std::array<uint8_t, 18> samples{0, 255, 10, 11, 12, 13, 14, 15, 16, 255, 0, 127, 128, 200, 20, 30, 40, 50};
        const auto encoded = encode(samples.data(), samples.size());
        const auto path = write(encoded);
        for (int mode = 0; mode < 6; ++mode) {
            codec::JxlDecodeLimits limits;
            if (mode == 0)
                limits.pixels = 5;
            if (mode == 1)
                limits.output_bytes = 17;
            if (mode == 2)
                limits.decoder_bytes = 0;
            if (mode == 3)
                limits.icc_bytes = 0;
            if (mode == 4)
                limits.buffered_input_bytes = 0;
            if (mode == 5)
                limits.encoded_bytes = encoded.size() - 1;
            codec::Image image{1, 1, 1, codec::SampleType::UInt8, {123}};
            std::string error = "old error";
            EXPECT_FALSE(native_decode(path, image, error, false, limits));
            EXPECT_NE(error, "old error");
            EXPECT_FALSE(error.empty());
            EXPECT_EQ(image.data, (std::vector<uint8_t>{123}));
            ASSERT_TRUE(native_decode(path, image, error)) << error;
            EXPECT_TRUE(error.empty());
            EXPECT_EQ(image.data, (std::vector<uint8_t>(samples.begin(), samples.end())));
        }
        for (const size_t length : {size_t{0}, size_t{1}, encoded.size() / 2}) {
            codec::Image image{1, 1, 1, codec::SampleType::UInt8, {123}};
            std::string error;
            EXPECT_FALSE(native_decode(std::span(encoded).first(length), image, error));
            EXPECT_EQ(image.data, (std::vector<uint8_t>{123}));
        }
    }

    TEST_F(JxlCodec, OwningOutputSurvivesTrailingInputAfterPixelDecode) {
        for (const int bits : {8, 16, 32}) {
            const std::array<uint8_t, 18> u8{10, 20, 30};
            const std::array<uint16_t, 18> u16{1000, 2000, 3000};
            const std::array<float, 18> f32{.1f, .2f, .3f};
            const void* samples = bits == 32 ? static_cast<const void*>(f32.data()) : bits == 16 ? static_cast<const void*>(u16.data())
                                                                                                 : u8.data();
            const size_t bytes = bits == 32 ? sizeof(f32) : bits == 16 ? sizeof(u16)
                                                                       : sizeof(u8);
            const auto valid = encode(samples, bytes, {.bits = bits, .floating = bits == 32});
            auto invalid = valid;
            invalid.push_back(0xff);
            codec::Image image{7, 8, 1, codec::SampleType::UInt8, {123}};
            std::string error;
            for (const bool file : {false, true}) {
                EXPECT_FALSE(file ? native_decode(write(invalid), image, error) : native_decode(invalid, image, error));
                EXPECT_NE(error.find("trailing input"), std::string::npos) << error;
                EXPECT_EQ(image.width, 7);
                EXPECT_EQ(image.height, 8);
                EXPECT_EQ(image.channels, 1);
                EXPECT_EQ(image.sample_type, codec::SampleType::UInt8);
                EXPECT_EQ(image.data, (std::vector<uint8_t>{123}));
            }
            ASSERT_TRUE(native_decode(valid, image, error)) << error;
            EXPECT_EQ(image.data.size(), bytes);
            EXPECT_EQ(std::memcmp(image.data.data(), samples, bytes), 0);
        }
    }

    TEST_F(JxlCodec, TargetIsAllocatedOnlyAfterSuccessfulDecode) {
        const std::array<uint8_t, 6> gray{0, 128, 255, 30, 60, 90};
        const auto encoded = encode(gray.data(), gray.size(), {.channels = 1});
        struct Allocation {
            unsigned calls = 0;
            std::vector<float> values;
        } allocation;
        codec::DecodeTarget target{3, codec::SampleType::Float32, nullptr,
                                   [](size_t bytes, void* opaque) -> void* {
                                       auto& allocation = *static_cast<Allocation*>(opaque);
                                       ++allocation.calls;
                                       allocation.values.resize(bytes / sizeof(float));
                                       return allocation.values.data();
                                   },
                                   &allocation};
        codec::Probe probe{9, 9, 9};
        std::string error;
        EXPECT_FALSE(native_target(std::span(encoded).first(encoded.size() / 2), target, probe, error));
        EXPECT_EQ(allocation.calls, 0u);
        EXPECT_EQ(probe.width, 9);
        ASSERT_TRUE(native_target(encoded, target, probe, error)) << error;
        ASSERT_EQ(allocation.calls, 1u);
        ASSERT_EQ(allocation.values.size(), 18u);
        for (size_t i = 0; i < gray.size(); ++i)
            for (size_t channel = 0; channel < 3; ++channel)
                EXPECT_FLOAT_EQ(allocation.values[i * 3 + channel], float(gray[i]) / 255.f);
    }

    TEST_F(JxlCodec, ContainerMetadataStreamsAndAnimationIsExplicitlyRejected) {
        const std::array<uint8_t, 18> pixels{};
        const std::vector<uint8_t> metadata(256 * 1024, 'x');
        const auto encoded = encode(pixels.data(), pixels.size(), {.container = true, .metadata = metadata});
        codec::Image image;
        std::string error;
        ASSERT_TRUE(native_decode(write(encoded), image, error)) << error;
        EXPECT_EQ(image.data, (std::vector<uint8_t>(pixels.begin(), pixels.end())));
        const auto animated = encode(pixels.data(), pixels.size(), {.container = true, .animation = true});
        const auto previous = image.data;
        EXPECT_FALSE(native_decode(animated, image, error));
        EXPECT_NE(error.find("animation"), std::string::npos);
        EXPECT_EQ(image.data, previous);
    }

    TEST_F(JxlCodec, AllOrientationsAreAppliedOnceAndProbeMatchesPixels) {
        const std::array<uint8_t, 18> source{1, 2, 3, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150};
        for (int orientation = 1; orientation <= 8; ++orientation) {
            SCOPED_TRACE(orientation);
            const auto encoded = encode(source.data(), source.size(), {.orientation = static_cast<JxlOrientation>(orientation)});
            codec::Image image;
            codec::Probe probe;
            std::string error;
            ASSERT_TRUE(native_decode(encoded, image, error)) << error;
            ASSERT_TRUE(codec::probe_jxl(write(encoded), probe, error)) << error;
            EXPECT_EQ(probe.width, image.width);
            EXPECT_EQ(probe.height, image.height);
            ASSERT_EQ(image.width, orientation >= 5 ? 2 : 3);
            ASSERT_EQ(image.height, orientation >= 5 ? 3 : 2);
            for (int y = 0; y < image.height; ++y)
                for (int x = 0; x < image.width; ++x) {
                    int sx = x, sy = y;
                    switch (orientation) {
                    case 2: sx = 2 - x; break;
                    case 3:
                        sx = 2 - x;
                        sy = 1 - y;
                        break;
                    case 4: sy = 1 - y; break;
                    case 5:
                        sx = y;
                        sy = x;
                        break;
                    case 6:
                        sx = y;
                        sy = 1 - x;
                        break;
                    case 7:
                        sx = 2 - y;
                        sy = 1 - x;
                        break;
                    case 8:
                        sx = 2 - y;
                        sy = x;
                        break;
                    default: break;
                    }
                    for (int channel = 0; channel < 3; ++channel)
                        EXPECT_EQ(image.data[(y * image.width + x) * 3 + channel], source[(sy * 3 + sx) * 3 + channel]);
                }
        }
    }

    TEST_F(JxlCodec, EveryUInt16SampleSurvivesLosslessDecoding) {
        std::vector<uint16_t> values(65536);
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = static_cast<uint16_t>(i);
        const auto encoded = encode(values.data(), values.size() * sizeof(uint16_t), {.width = 256, .height = 256, .channels = 1, .bits = 16});
        codec::Image image;
        std::string error;
        ASSERT_TRUE(native_decode(encoded, image, error)) << error;
        ASSERT_EQ(image.data.size(), values.size() * sizeof(uint16_t));
        EXPECT_EQ(std::memcmp(image.data.data(), values.data(), image.data.size()), 0);
    }

    TEST_F(JxlCodec, DisplayColorConversionIsSeparateFromRawFloatValues) {
        const std::array<float, 6> linear{0.f, 0.003f, 0.18f, 0.5f, 0.75f, 1.f};
        const auto encoded = encode(linear.data(), sizeof(linear), {.channels = 1, .bits = 32, .floating = true});
        codec::Image raw, display;
        std::string error;
        ASSERT_TRUE(native_decode(encoded, raw, error)) << error;
        ASSERT_TRUE(native_decode(encoded, display, error, true)) << error;
        for (size_t i = 0; i < linear.size(); ++i) {
            float a, b;
            std::memcpy(&a, raw.data.data() + i * sizeof(float), sizeof(float));
            std::memcpy(&b, display.data.data() + i * sizeof(float), sizeof(float));
            EXPECT_FLOAT_EQ(a, linear[i]);
            const double expected = linear[i] <= .0031308 ? 12.92 * linear[i] : 1.055 * std::pow(linear[i], 1. / 2.4) - .055;
            EXPECT_NEAR(b, expected, 3e-4);
        }
    }

    TEST_F(JxlCodec, LossyCodestreamAndConcurrentReadersHaveIndependentOutputs) {
        const std::array<uint8_t, 18> source{20, 40, 60, 30, 50, 70, 40, 60, 80, 50, 70, 90, 60, 80, 100, 70, 90, 110};
        const auto encoded = encode(source.data(), source.size(), {.lossless = false});
        const auto path = write(encoded);
        codec::Image expected;
        std::string error;
        ASSERT_TRUE(native_decode(encoded, expected, error)) << error;
        std::vector<std::future<std::vector<uint8_t>>> readers;
        for (int i = 0; i < 8; ++i)
            readers.push_back(std::async(std::launch::async, [&] {
                codec::Image image;
                std::string diagnostic;
                if (!native_decode(path, image, diagnostic))
                    throw std::runtime_error(diagnostic);
                return image.data;
            }));
        for (auto& reader : readers)
            EXPECT_EQ(reader.get(), expected.data);
    }

    TEST_F(JxlCodec, DecoderAllocationBudgetsFailCleanlyAndRecover) {
        const std::array<uint8_t, 18> source{};
        const auto encoded = encode(source.data(), source.size());
        unsigned failures = 0;
        for (size_t budget = 32; budget <= 1024 * 1024; budget *= 2) {
            codec::JxlDecodeLimits limits;
            limits.decoder_bytes = budget;
            codec::Image image{1, 1, 1, codec::SampleType::UInt8, {123}};
            std::string error;
            if (!native_decode(encoded, image, error, false, limits)) {
                ++failures;
                EXPECT_NE(error.find("allocation"), std::string::npos) << error;
                EXPECT_EQ(image.data, (std::vector<uint8_t>{123}));
            }
            ASSERT_TRUE(native_decode(encoded, image, error)) << error;
        }
        EXPECT_GT(failures, 0u);
    }

    TEST_F(JxlCodec, BorrowedInputAndFailedTargetsDoNotChangeOwnedResults) {
        const std::array<uint8_t, 18> samples{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
        auto encoded = encode(samples.data(), samples.size());
        const auto valid = encoded;
        codec::Image image;
        std::string error;
        ASSERT_TRUE(native_decode(encoded, image, error)) << error;
        std::fill(encoded.begin(), encoded.end(), 0xff);
        EXPECT_EQ(image.data, (std::vector<uint8_t>(samples.begin(), samples.end())));
        std::array<uint8_t, 18> output;
        output.fill(123);
        codec::DecodeTarget target{3, codec::SampleType::UInt8, output.data()};
        codec::Probe probe{9, 9, 9};
        EXPECT_FALSE(native_target(encoded, target, probe, error));
        EXPECT_EQ(probe.width, 9);
        for (const auto value : output)
            EXPECT_EQ(value, 123);
        ASSERT_TRUE(native_target(valid, target, probe, error)) << error;
        EXPECT_EQ(output, samples);
        unsigned calls = 0;
        target.data = nullptr;
        target.user = &calls;
        target.allocate = [](size_t, void* user) -> void* { ++*static_cast<unsigned*>(user); return nullptr; };
        EXPECT_FALSE(native_target(valid, target, probe, error));
        EXPECT_EQ(calls, 1u);
        EXPECT_EQ(target.data, nullptr);
    }

    // Independent provider reader: do not validate the writer solely through
    // the application decoder, which could hide matching policy mistakes.
    std::vector<uint8_t> direct_decode(std::span<const uint8_t> bytes, int channels, codec::SampleType type, bool srgb = false, bool original = false) {
        std::unique_ptr<JxlDecoder, decltype(&JxlDecoderDestroy)> decoder(JxlDecoderCreate(nullptr), JxlDecoderDestroy);
        if (!decoder || JxlDecoderSubscribeEvents(decoder.get(), JXL_DEC_FULL_IMAGE | (srgb || original ? JXL_DEC_COLOR_ENCODING : 0)) != JXL_DEC_SUCCESS ||
            JxlDecoderSetInput(decoder.get(), bytes.data(), bytes.size()) != JXL_DEC_SUCCESS)
            throw std::runtime_error("Independent JXL decoder initialization failed");
        if ((srgb || original) && (!JxlGetDefaultCms() || JxlDecoderSetCms(decoder.get(), *JxlGetDefaultCms()) != JXL_DEC_SUCCESS))
            throw std::runtime_error("Independent JXL CMS initialization failed");
        JxlDecoderCloseInput(decoder.get());
        const JxlPixelFormat format{uint32_t(channels), type == codec::SampleType::Float32 ? JXL_TYPE_FLOAT : type == codec::SampleType::UInt16 ? JXL_TYPE_UINT16
                                                                                                                                                : JXL_TYPE_UINT8,
                                    JXL_NATIVE_ENDIAN, 0};
        std::vector<uint8_t> result;
        for (;;) {
            const auto status = JxlDecoderProcessInput(decoder.get());
            if (status == JXL_DEC_COLOR_ENCODING && (srgb || original)) {
                JxlColorEncoding color;
                if (original) {
                    if (JxlDecoderGetColorAsEncodedProfile(decoder.get(), JXL_COLOR_PROFILE_TARGET_ORIGINAL, &color) != JXL_DEC_SUCCESS)
                        throw std::runtime_error("Independent JXL original profile failed");
                } else {
                    JxlColorEncodingSetToSRGB(&color, channels <= 2);
                }
                if (JxlDecoderSetOutputColorProfile(decoder.get(), &color, nullptr, 0) != JXL_DEC_SUCCESS)
                    throw std::runtime_error("Independent JXL sRGB output failed");
            } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
                size_t size = 0;
                if (JxlDecoderImageOutBufferSize(decoder.get(), &format, &size) != JXL_DEC_SUCCESS)
                    throw std::runtime_error("Independent JXL size failed");
                result.resize(size);
                if (JxlDecoderSetImageOutBuffer(decoder.get(), &format, result.data(), size) != JXL_DEC_SUCCESS)
                    throw std::runtime_error("Independent JXL output failed");
            } else if (status == JXL_DEC_SUCCESS) {
                return result;
            } else if (status != JXL_DEC_FULL_IMAGE) {
                throw std::runtime_error("Independent JXL decode failed");
            }
        }
    }

    // Independent box framing, including extended sizes and partial codestreams.
    std::pair<std::vector<uint8_t>, std::vector<std::string>> exported_boxes(std::span<const uint8_t> input) {
        if (input.size() >= 2 && input[0] == 0xff && input[1] == 0x0a)
            return {{input.begin(), input.end()}, {}};
        std::vector<uint8_t> stream;
        std::vector<std::string> xml;
        size_t offset = 0;
        const auto number = [&](size_t start, size_t count) {
            uint64_t value = 0;
            for (size_t i = 0; i < count; ++i)
                value = (value << 8) | input[start + i];
            return value;
        };
        while (offset < input.size()) {
            if (input.size() - offset < 8)
                throw std::runtime_error("truncated box");
            uint64_t size = number(offset, 4);
            size_t header = 8;
            const std::string_view kind(reinterpret_cast<const char*>(input.data() + offset + 4), 4);
            if (size == 1) {
                if (input.size() - offset < 16)
                    throw std::runtime_error("truncated extended box");
                size = number(offset + 8, 8);
                header = 16;
            } else if (!size) {
                size = input.size() - offset;
            }
            if (size < header || size > input.size() - offset)
                throw std::runtime_error("invalid box extent");
            auto payload = input.subspan(offset + header, size_t(size) - header);
            if (kind == "xml ")
                xml.emplace_back(reinterpret_cast<const char*>(payload.data()), payload.size());
            if (kind == "jxlp") {
                if (payload.size() < 4)
                    throw std::runtime_error("truncated codestream index");
                payload = payload.subspan(4);
            }
            if (kind == "jxlc" || kind == "jxlp")
                stream.insert(stream.end(), payload.begin(), payload.end());
            offset += size_t(size);
        }
        if (stream.empty())
            throw std::runtime_error("missing codestream");
        return {std::move(stream), std::move(xml)};
    }

    TEST_F(JxlCodec, ExplicitXmpPreservesCodestreamSamplesAndDefaultOutput) {
        const std::string packet = "<r:RDF xmlns:r=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\"><r:Description r:about=\"\"/></r:RDF>";
        for (auto type : {codec::SampleType::UInt8, codec::SampleType::UInt16, codec::SampleType::Float32})
            for (bool lossless : {true, false})
                for (bool premultiplied : {false, true}) {
                    if (premultiplied && type != codec::SampleType::Float32)
                        continue;
                    const size_t sample = type == codec::SampleType::UInt8 ? 1 : type == codec::SampleType::UInt16 ? 2
                                                                                                                   : 4;
                    std::vector<uint8_t> pixels(3 * 2 * 4 * sample);
                    for (size_t i = 0; i < 24; ++i) {
                        if (sample == 4) {
                            const float value = i % 4 == 3 ? .5f : float(i % 7) / 16;
                            std::memcpy(pixels.data() + i * sample, &value, sample);
                        } else if (sample == 2) {
                            const uint16_t value = uint16_t(i * 1703);
                            std::memcpy(pixels.data() + i * sample, &value, sample);
                        } else {
                            pixels[i] = uint8_t(i * 7);
                        }
                    }
                    const auto original = pixels;
                    codec::JxlEncodeOptions options;
                    options.lossless = lossless;
                    options.color = sample == 4 ? codec::JxlEncodingColor::LinearBT2020 : codec::JxlEncodingColor::SRGB;
                    options.premultiplied_alpha = premultiplied;
                    std::vector<uint8_t> plain, empty, embedded;
                    ASSERT_TRUE(codec::encode_jxl_memory(pixels, 3, 2, 4, type, 12 * sample, plain, options));
                    ASSERT_TRUE(codec::encode_jxl_memory(pixels, 3, 2, 4, type, 12 * sample, empty, options, {}));
                    EXPECT_EQ(plain, empty);
                    ASSERT_TRUE(codec::encode_jxl_memory(pixels, 3, 2, 4, type, 12 * sample, embedded, options, packet));
                    const auto [stream, packets] = exported_boxes(embedded);
                    EXPECT_EQ(stream, exported_boxes(plain).first);
                    EXPECT_EQ(packets, (std::vector<std::string>{packet}));
                    EXPECT_EQ(direct_decode(embedded, 4, type), direct_decode(plain, 4, type));
                    EXPECT_EQ(pixels, original);
                    const auto path = write({});
                    ASSERT_TRUE(codec::write_jxl(path, pixels, 3, 2, 4, type, 12 * sample, options, {}, packet));
                    std::ifstream file(path, std::ios::binary);
                    EXPECT_EQ((std::vector<uint8_t>((std::istreambuf_iterator<char>(file)), {})), embedded);
                }
    }

    TEST_F(JxlCodec, ExplicitXmpLimitsFailuresAndReusePreserveDestination) {
        const std::array<uint8_t, 18> pixels{};
        codec::JxlEncodeOptions options;
        std::string packet = "<r:RDF xmlns:r=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\"><r:Description r:about=\"\" xmlns:t=\"urn:lfs:test\"><t:value>";
        packet.append(32 * 1024 * 1024 - packet.size() - std::string_view("</t:value></r:Description></r:RDF>").size(), 'a');
        packet += "</t:value></r:Description></r:RDF>";
        std::vector<uint8_t> encoded;
        ASSERT_TRUE(codec::encode_jxl_memory(pixels, 3, 2, 3, codec::SampleType::UInt8, 9, encoded, options, packet));
        EXPECT_EQ(exported_boxes(encoded).second, (std::vector<std::string>{packet}));
        EXPECT_EQ(direct_decode(encoded, 3, codec::SampleType::UInt8), (std::vector<uint8_t>(18)));
        const auto path = write(encoded);
        const auto limited_read = codec::read_source_metadata(path);
        EXPECT_EQ(limited_read.xmp.status, codec::MetadataStatus::Limited);
        EXPECT_TRUE(limited_read.xmp.blocks.empty());
        const auto original = encoded;
        for (const auto& bad : {packet + "a", std::string("x\0mp", 4)}) {
            auto failed = codec::encode_jxl_memory(pixels, 3, 2, 3, codec::SampleType::UInt8, 9, encoded, options, bad);
            ASSERT_FALSE(failed);
            EXPECT_EQ(failed.error().code(), lfs::ErrorCode::InvalidArgument);
            EXPECT_EQ(encoded, original);
            EXPECT_FALSE(codec::write_jxl(path, pixels, 3, 2, 3, codec::SampleType::UInt8, 9, options, {.overwrite = true}, bad));
        }
        for (int failure = 0; failure < 3; ++failure) {
            auto limited = options;
            if (failure == 0)
                limited.encoder_bytes = packet.size() / 2;
            if (failure == 1)
                limited.max_output_bytes = 64;
            if (failure == 2)
                limited.cancelled = [] { return true; };
            auto failed = codec::encode_jxl_memory(pixels, 3, 2, 3, codec::SampleType::UInt8, 9, encoded, limited, packet);
            ASSERT_FALSE(failed);
            EXPECT_EQ(failed.error().code(), failure == 2 ? lfs::ErrorCode::Cancelled : lfs::ErrorCode::ResourceExhausted);
            EXPECT_EQ(encoded, original);
            EXPECT_FALSE(codec::write_jxl(path, pixels, 3, 2, 3, codec::SampleType::UInt8, 9, limited, {.overwrite = true}, packet));
        }
        std::ifstream file(path, std::ios::binary);
        EXPECT_EQ((std::vector<uint8_t>((std::istreambuf_iterator<char>(file)), {})), original);
        file.close();
        ASSERT_TRUE(codec::write_jxl(path, pixels, 3, 2, 3, codec::SampleType::UInt8, 9, options, {.overwrite = true}, "<x/>"));
        const auto observed = codec::read_source_metadata(path);
        ASSERT_EQ(observed.xmp.blocks.size(), 1u);
        EXPECT_EQ(observed.xmp.blocks[0].envelope, codec::XmpEnvelope::JxlXml);
        EXPECT_EQ(observed.xmp.blocks[0].bytes, (std::vector<uint8_t>{'<', 'x', '/', '>'}));
        std::vector<uint8_t> alias(pixels.begin(), pixels.end());
        ASSERT_TRUE(codec::encode_jxl_memory(alias, 3, 2, 3, codec::SampleType::UInt8, 9, alias, options, "<x/>"));
        EXPECT_EQ(direct_decode(alias, 3, codec::SampleType::UInt8), (std::vector<uint8_t>(18)));
        for (const auto& entry : std::filesystem::directory_iterator(directory))
            EXPECT_EQ(entry.path(), path);
    }

    TEST_F(JxlCodec, EncoderLosslessFileMemoryAndIndependentReaderAgreeForEveryLayout) {
        for (int effort : {1, 2, 3, 4, 5, 9})
            for (const auto type : {codec::SampleType::UInt8, codec::SampleType::UInt16, codec::SampleType::Float32})
                for (bool padded : {false, true})
                    for (int channels = 1; channels <= 4; ++channels) {
                        SCOPED_TRACE(::testing::Message() << "effort=" << effort << ", type=" << int(type) << ", channels=" << channels << ", padded=" << padded);
                        const size_t sample = type == codec::SampleType::Float32 ? 4 : type == codec::SampleType::UInt16 ? 2
                                                                                                                         : 1;
                        const size_t row = 3 * channels * sample, stride = row + (padded ? 3 : 0);
                        std::vector<uint8_t> pixels(stride + row + 1, 0xa5), expected(2 * row);
                        for (int y = 0; y < 2; ++y)
                            for (int x = 0; x < 3 * channels; ++x) {
                                const size_t offset = y * row + x * sample;
                                if (sample == 4) {
                                    const float value = (channels % 2 == 0 && x % channels == channels - 1) ? .5f : (x == 0 ? -0.0f : float(x - 2) / 4);
                                    std::memcpy(expected.data() + offset, &value, sample);
                                } else if (sample == 2) {
                                    const uint16_t value = uint16_t(x * 5003 + y * 97);
                                    std::memcpy(expected.data() + offset, &value, sample);
                                } else {
                                    expected[offset] = uint8_t(x * 17 + y * 3);
                                }
                            }
                        if (channels % 2 == 0)
                            std::fill_n(expected.data() + (channels - 1) * sample, sample, uint8_t{0});
                        for (int y = 0; y < 2; ++y)
                            std::memcpy(pixels.data() + 1 + y * stride, expected.data() + y * row, row);
                        const auto original = pixels;
                        codec::JxlEncodeOptions options;
                        options.effort = effort;
                        options.color = sample == 4 ? codec::JxlEncodingColor::LinearSRGB : codec::JxlEncodingColor::SRGB;
                        std::vector<uint8_t> encoded{0x12};
                        const auto status = codec::encode_jxl_memory(std::span(pixels).subspan(1), 3, 2, channels, type, stride, encoded, options);
                        if (type == codec::SampleType::Float32 && effort < 4) {
                            ASSERT_FALSE(status);
                            EXPECT_EQ(status.error().code(), lfs::ErrorCode::InvalidArgument);
                            EXPECT_NE(status.error().detail().find("[4,9]"), std::string_view::npos);
                            EXPECT_EQ(encoded, (std::vector<uint8_t>{0x12}));
                            const std::array<uint8_t, 1> sentinel{0x12};
                            const auto path = write(sentinel);
                            EXPECT_FALSE(codec::write_jxl(path, std::span(pixels).subspan(1), 3, 2, channels, type, stride, options));
                            std::ifstream previous(path, std::ios::binary);
                            EXPECT_EQ((std::vector<uint8_t>((std::istreambuf_iterator<char>(previous)), {})), (std::vector<uint8_t>{0x12}));
                            continue;
                        }
                        ASSERT_TRUE(status) << (status ? "" : status.error().detail());
                        EXPECT_EQ(direct_decode(encoded, channels, type), expected);
                        codec::Image decoded;
                        std::string error;
                        ASSERT_TRUE(native_decode(encoded, decoded, error)) << error;
                        EXPECT_EQ(decoded.data, expected);
                        const auto path = write({});
                        ASSERT_TRUE(codec::write_jxl(path, std::span(pixels).subspan(1), 3, 2, channels, type, stride, options));
                        std::ifstream file(path, std::ios::binary);
                        const std::vector<uint8_t> stored((std::istreambuf_iterator<char>(file)), {});
                        EXPECT_EQ(stored, encoded);
                        EXPECT_EQ(pixels, original);
                    }
    }

    TEST_F(JxlCodec, EncoderColorDeclarationsMatchTheSamplesWithoutImplicitConversion) {
        for (auto color : {codec::JxlEncodingColor::SRGB, codec::JxlEncodingColor::LinearSRGB, codec::JxlEncodingColor::LinearBT2020, codec::JxlEncodingColor::SRGBBT2020}) {
            const std::array<float, 3> input{.125f, .5f, 2.f};
            codec::JxlEncodeOptions options;
            options.color = color;
            std::vector<uint8_t> encoded;
            ASSERT_TRUE(codec::encode_jxl_memory({reinterpret_cast<const uint8_t*>(input.data()), sizeof(input)}, 1, 1, 3, codec::SampleType::Float32, sizeof(input), encoded, options));
            const auto samples = direct_decode(encoded, 3, codec::SampleType::Float32);
            ASSERT_EQ(samples.size(), sizeof(input));
            EXPECT_EQ(std::memcmp(samples.data(), input.data(), samples.size()), 0);
            std::unique_ptr<JxlDecoder, decltype(&JxlDecoderDestroy)> decoder(JxlDecoderCreate(nullptr), JxlDecoderDestroy);
            ASSERT_EQ(JxlDecoderSubscribeEvents(decoder.get(), JXL_DEC_COLOR_ENCODING), JXL_DEC_SUCCESS);
            ASSERT_EQ(JxlDecoderSetInput(decoder.get(), encoded.data(), encoded.size()), JXL_DEC_SUCCESS);
            JxlDecoderCloseInput(decoder.get());
            ASSERT_EQ(JxlDecoderProcessInput(decoder.get()), JXL_DEC_COLOR_ENCODING);
            JxlColorEncoding profile{};
            ASSERT_EQ(JxlDecoderGetColorAsEncodedProfile(decoder.get(), JXL_COLOR_PROFILE_TARGET_ORIGINAL, &profile), JXL_DEC_SUCCESS);
            EXPECT_EQ(profile.transfer_function, (color == codec::JxlEncodingColor::SRGB || color == codec::JxlEncodingColor::SRGBBT2020) ? JXL_TRANSFER_FUNCTION_SRGB : JXL_TRANSFER_FUNCTION_LINEAR);
            EXPECT_EQ(profile.primaries, (color == codec::JxlEncodingColor::LinearBT2020 || color == codec::JxlEncodingColor::SRGBBT2020) ? JXL_PRIMARIES_2100 : JXL_PRIMARIES_SRGB);
        }
    }

    TEST_F(JxlCodec, EncoderBt2020DisplayMatchesIndependentColorTransform) {
        constexpr int width = 16, height = 16;
        std::vector<float> input(width * height * 3), expected(input.size());
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) {
                const auto offset = (y * width + x) * 3;
                input[offset] = .25f + x * .005f;
                input[offset + 1] = .3f + y * .005f;
                input[offset + 2] = .35f + (x + y) * .002f;
            }
        const cmsCIExyY white{.3127, .329, 1.};
        const cmsCIExyYTRIPLE primaries{{.708, .292, 1.}, {.17, .797, 1.}, {.131, .046, 1.}};
        std::unique_ptr<void, decltype(&cmsCloseProfile)> srgb(cmsCreate_sRGBProfile(), cmsCloseProfile);
        ASSERT_TRUE(srgb);
        for (auto color : {codec::JxlEncodingColor::LinearBT2020, codec::JxlEncodingColor::SRGBBT2020}) {
            std::unique_ptr<cmsToneCurve, decltype(&cmsFreeToneCurve)> curve(
                color == codec::JxlEncodingColor::LinearBT2020 ? cmsBuildGamma(nullptr, 1.) : cmsDupToneCurve(static_cast<cmsToneCurve*>(cmsReadTag(srgb.get(), cmsSigRedTRCTag))), cmsFreeToneCurve);
            ASSERT_TRUE(curve);
            cmsToneCurve* curves[]{curve.get(), curve.get(), curve.get()};
            std::unique_ptr<void, decltype(&cmsCloseProfile)> original(cmsCreateRGBProfile(&white, &primaries, curves), cmsCloseProfile);
            ASSERT_TRUE(original);
            std::unique_ptr<void, decltype(&cmsDeleteTransform)> transform(cmsCreateTransform(original.get(), TYPE_RGB_FLT, srgb.get(), TYPE_RGB_FLT, INTENT_RELATIVE_COLORIMETRIC, cmsFLAGS_NOOPTIMIZE), cmsDeleteTransform);
            ASSERT_TRUE(transform);
            cmsDoTransform(transform.get(), input.data(), expected.data(), width * height);
            for (bool lossless : {false, true}) {
                SCOPED_TRACE(::testing::Message() << "color=" << int(color) << " lossless=" << lossless);
                codec::JxlEncodeOptions options;
                options.color = color;
                options.lossless = lossless;
                options.distance = .25f;
                std::vector<uint8_t> encoded;
                ASSERT_TRUE(codec::encode_jxl_memory({reinterpret_cast<const uint8_t*>(input.data()), input.size() * sizeof(float)}, width, height, 3, codec::SampleType::Float32, width * 3 * sizeof(float), encoded, options));
                std::vector<float> actual(input.size());
                codec::DecodeTarget target{3, codec::SampleType::Float32, actual.data()};
                codec::Probe info;
                std::string error;
                ASSERT_TRUE(codec::decode_jxl_to_buffer(encoded, target, info, error)) << error;
                const auto independent = direct_decode(encoded, 3, codec::SampleType::Float32, true);
                ASSERT_EQ(independent.size(), actual.size() * sizeof(float));
                const auto native_bytes = direct_decode(encoded, 3, codec::SampleType::Float32, false, true);
                ASSERT_EQ(native_bytes.size(), actual.size() * sizeof(float));
                std::vector<float> native(actual.size()), transformed(actual.size());
                std::memcpy(native.data(), native_bytes.data(), native_bytes.size());
                cmsDoTransform(transform.get(), native.data(), transformed.data(), width * height);
                for (size_t i = 0; i < actual.size(); ++i) {
                    float reference;
                    std::memcpy(&reference, independent.data() + i * sizeof(float), sizeof(float));
                    EXPECT_NEAR(actual[i], reference, .000001f);
                    EXPECT_NEAR(actual[i], transformed[i], .0005f);
                    // Retain the source-color bound for the linear fixture. Lossy
                    // distortion is separate from the decoded-sample CMS check above.
                    if (lossless || color == codec::JxlEncodingColor::LinearBT2020)
                        EXPECT_NEAR(actual[i], expected[i], lossless ? .0005f : .01f);
                }
            }
        }
    }

    TEST_F(JxlCodec, EncoderLimitsCancellationAndAliasingPreserveOutputAndRecover) {
        const std::vector<uint8_t> input(3 * 64 * 32, 123);
        for (int mode = 0; mode < 7; ++mode) {
            codec::JxlEncodeOptions options;
            if (mode == 0)
                options.pixels = 1;
            if (mode == 1)
                options.encoder_bytes = 32;
            if (mode == 2)
                options.max_output_bytes = 1;
            if (mode == 3)
                options.effort = 0;
            if (mode == 4)
                options.distance = std::numeric_limits<float>::quiet_NaN();
            if (mode == 5)
                options.cancelled = [] { return true; };
            unsigned calls = 0;
            if (mode == 6)
                options.cancelled = [&] { return ++calls > 1; };
            std::vector<uint8_t> result{9, 8, 7};
            EXPECT_FALSE(codec::encode_jxl_memory(input, 64, 32, 3, codec::SampleType::UInt8, 192, result, options));
            EXPECT_EQ(result, (std::vector<uint8_t>{9, 8, 7}));
            ASSERT_TRUE(codec::encode_jxl_memory(input, 64, 32, 3, codec::SampleType::UInt8, 192, result));
            EXPECT_EQ(direct_decode(result, 3, codec::SampleType::UInt8), input);
        }
        auto alias = input;
        ASSERT_TRUE(codec::encode_jxl_memory(alias, 64, 32, 3, codec::SampleType::UInt8, 192, alias));
        EXPECT_EQ(direct_decode(alias, 3, codec::SampleType::UInt8), input);
    }

    TEST_F(JxlCodec, EncoderLossyPreservesAlphaAndConcurrentCallsAreIndependent) {
        constexpr int width = 128, height = 64;
        std::vector<uint8_t> input(width * height * 4);
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) {
                const auto i = (y * width + x) * 4;
                input[i] = uint8_t(x * 2);
                input[i + 1] = uint8_t(y * 4);
                input[i + 2] = 90;
                input[i + 3] = uint8_t(x + y);
            }
        codec::JxlEncodeOptions options;
        options.lossless = false;
        options.distance = .5f;
        std::vector<uint8_t> encoded;
        const auto status = codec::encode_jxl_memory(input, width, height, 4, codec::SampleType::UInt8, width * 4, encoded, options);
        ASSERT_TRUE(status) << (status ? "" : status.error().detail());
        codec::Image decoded;
        std::string error;
        ASSERT_TRUE(codec::decode_jxl(encoded, decoded, error)) << error;
        ASSERT_EQ(decoded.data.size(), input.size());
        double weighted_error = 0, weight = 0;
        for (size_t i = 0; i < input.size(); ++i)
            if (i % 4 == 3)
                EXPECT_EQ(decoded.data[i], input[i]);
            else {
                const double alpha = input[(i / 4) * 4 + 3] / 255.;
                const double delta = int(decoded.data[i]) - int(input[i]);
                weighted_error += alpha * alpha * delta * delta;
                weight += alpha * alpha;
            }
        // Perceptual distance is not a per-channel maximum error promise;
        // invisible RGB has no visual weight. Check this fixture's weighted MSE.
        EXPECT_LT(weighted_error / weight, 16.);
        std::vector<std::future<std::vector<uint8_t>>> calls;
        for (int i = 0; i < 4; ++i)
            calls.push_back(std::async(std::launch::async, [&] {
                std::vector<uint8_t> result;
                if (!codec::encode_jxl_memory(input, width, height, 4, codec::SampleType::UInt8, width * 4, result, options))
                    throw std::runtime_error("Concurrent JXL encode failed");
                return result;
            }));
        for (auto& call : calls)
            EXPECT_EQ(call.get(), encoded);
    }

    TEST_F(JxlCodec, EncoderLossyHighDistanceKeepsEveryAlphaSample) {
        constexpr int width = 128, height = 96;
        for (auto type : {codec::SampleType::UInt8, codec::SampleType::UInt16, codec::SampleType::Float32})
            for (int channels : {2, 4})
                for (float distance : {9.9f, 10.f, 25.f}) {
                    SCOPED_TRACE(::testing::Message() << int(type) << "/" << channels << "/" << distance);
                    const size_t sample = type == codec::SampleType::UInt8 ? 1 : type == codec::SampleType::UInt16 ? 2
                                                                                                                   : 4;
                    std::vector<uint8_t> input(width * height * channels * sample);
                    for (size_t i = 0; i < width * height * channels; ++i) {
                        const uint32_t code = uint32_t((i * 97 + i / channels * 31) % 256);
                        if (sample == 1)
                            input[i] = uint8_t(code);
                        else if (sample == 2) {
                            const uint16_t value = uint16_t(code * 257);
                            std::memcpy(input.data() + i * sample, &value, sample);
                        } else {
                            const float value = float(code) / 255.f;
                            std::memcpy(input.data() + i * sample, &value, sample);
                        }
                    }
                    codec::JxlEncodeOptions options;
                    options.lossless = false;
                    options.distance = distance;
                    options.color = sample == 4 ? codec::JxlEncodingColor::LinearSRGB : codec::JxlEncodingColor::SRGB;
                    std::vector<uint8_t> encoded;
                    ASSERT_TRUE(codec::encode_jxl_memory(input, width, height, channels, type, width * channels * sample, encoded, options));
                    const auto decoded = direct_decode(encoded, channels, type);
                    ASSERT_EQ(decoded.size(), input.size());
                    codec::Image app;
                    std::string error;
                    ASSERT_TRUE(native_decode(encoded, app, error)) << error;
                    ASSERT_EQ(app.data.size(), input.size());
                    for (size_t pixel = 0; pixel < width * height; ++pixel) {
                        const auto offset = (pixel * channels + channels - 1) * sample;
                        ASSERT_EQ(std::memcmp(decoded.data() + offset, input.data() + offset, sample), 0) << pixel;
                        ASSERT_EQ(std::memcmp(app.data.data() + offset, input.data() + offset, sample), 0) << pixel;
                    }
                }
    }

    TEST_F(JxlCodec, EncoderFloatAlphaAndNonfiniteInputHaveExplicitPolicies) {
        const std::array<float, 8> input{.25f, -.125f, 1.f, .5f, 0.f, 0.f, 0.f, 0.f};
        const auto bytes = std::span(reinterpret_cast<const uint8_t*>(input.data()), sizeof(input));
        codec::JxlEncodeOptions options;
        options.color = codec::JxlEncodingColor::LinearSRGB;
        options.premultiplied_alpha = true;
        std::vector<uint8_t> encoded;
        ASSERT_TRUE(codec::encode_jxl_memory(bytes, 2, 1, 4, codec::SampleType::Float32, sizeof(input), encoded, options));
        const auto output = direct_decode(encoded, 4, codec::SampleType::Float32);
        const std::array<float, 8> expected{.5f, -.25f, 2.f, .5f, 0.f, 0.f, 0.f, 0.f};
        ASSERT_EQ(output.size(), sizeof(expected));
        EXPECT_EQ(std::memcmp(output.data(), expected.data(), output.size()), 0);
        for (const auto value : {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
            auto bad = input;
            bad[0] = value;
            const auto previous = encoded;
            EXPECT_FALSE(codec::encode_jxl_memory({reinterpret_cast<const uint8_t*>(bad.data()), sizeof(bad)}, 2, 1, 4, codec::SampleType::Float32, sizeof(bad), encoded, options));
            EXPECT_EQ(encoded, previous);
        }
    }

    TEST_F(JxlCodec, EncoderParallelLosslessCallsShareWorkerBudgetWithoutChangingBytes) {
        constexpr int width = 2048, height = 1024;
        std::vector<uint8_t> input(width * height * 3);
        for (size_t i = 0; i < input.size(); ++i)
            input[i] = uint8_t((i / 127) ^ (i / (width * 3)));
        codec::JxlEncodeOptions options;
        options.effort = 1;
        std::vector<uint8_t> baseline;
        ASSERT_TRUE(codec::encode_jxl_memory(input, width, height, 3, codec::SampleType::UInt8, width * 3, baseline, options));
        std::vector<std::future<std::vector<uint8_t>>> calls;
        for (int i = 0; i < 4; ++i)
            calls.push_back(std::async(std::launch::async, [&] {
                std::vector<uint8_t> output;
                if (!codec::encode_jxl_memory(input, width, height, 3, codec::SampleType::UInt8, width * 3, output, options))
                    throw std::runtime_error("Parallel JXL encoding failed");
                return output;
            }));
        for (auto& call : calls)
            EXPECT_EQ(call.get(), baseline);
        EXPECT_EQ(direct_decode(baseline, 3, codec::SampleType::UInt8), input);
    }

    TEST_F(JxlCodec, EncoderFloat32ExtremesAndSignedZeroSurviveLossless) {
        const std::array<float, 12> input{0.f, -0.f, std::numeric_limits<float>::denorm_min(),
                                          -std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::min(), -std::numeric_limits<float>::min(),
                                          std::numeric_limits<float>::max(), std::numeric_limits<float>::lowest(), .125f, -.5f, 1.f, 2.f};
        codec::JxlEncodeOptions options;
        options.color = codec::JxlEncodingColor::LinearSRGB;
        std::vector<uint8_t> output;
        const auto status = codec::encode_jxl_memory({reinterpret_cast<const uint8_t*>(input.data()), sizeof(input)}, 4, 1, 3, codec::SampleType::Float32, sizeof(input), output, options);
        ASSERT_TRUE(status) << (status ? "" : status.error().detail());
        const auto decoded = direct_decode(output, 3, codec::SampleType::Float32);
        ASSERT_EQ(decoded.size(), sizeof(input));
        EXPECT_EQ(std::memcmp(decoded.data(), input.data(), decoded.size()), 0);
    }

    TEST_F(JxlCodec, EncoderRejectsInvalidViewsAndEnforcesExactOutputBudget) {
        const std::array<uint8_t, 18> input{};
        std::vector<uint8_t> encoded;
        ASSERT_TRUE(codec::encode_jxl_memory(input, 3, 2, 3, codec::SampleType::UInt8, 9, encoded));
        const auto valid = encoded;
        codec::JxlEncodeOptions options;
        options.max_output_bytes = valid.size();
        ASSERT_TRUE(codec::encode_jxl_memory(input, 3, 2, 3, codec::SampleType::UInt8, 9, encoded, options));
        EXPECT_EQ(encoded, valid);
        options.max_output_bytes--;
        EXPECT_FALSE(codec::encode_jxl_memory(input, 3, 2, 3, codec::SampleType::UInt8, 9, encoded, options));
        EXPECT_EQ(encoded, valid);
        for (int mode = 0; mode < 7; ++mode) {
            int width = 3, height = 2, channels = 3;
            size_t stride = 9;
            options = {};
            if (mode == 0)
                width = 0;
            if (mode == 1)
                channels = 5;
            if (mode == 2)
                stride = 8;
            if (mode == 3)
                stride = SIZE_MAX;
            if (mode == 4) {
                width = INT_MAX;
                height = INT_MAX;
            }
            if (mode == 5)
                options.color = static_cast<codec::JxlEncodingColor>(99);
            if (mode == 6)
                options.premultiplied_alpha = true;
            EXPECT_FALSE(codec::encode_jxl_memory(input, width, height, channels, codec::SampleType::UInt8, stride, encoded, options));
            EXPECT_EQ(encoded, valid);
        }
        for (bool standard : {false, true}) {
            options = {};
            options.cancelled = [standard]() -> bool { if(standard) throw std::runtime_error("callback sentinel"); throw 7; };
            EXPECT_FALSE(codec::encode_jxl_memory(input, 3, 2, 3, codec::SampleType::UInt8, 9, encoded, options));
            EXPECT_EQ(encoded, valid);
        }
    }

    TEST_F(JxlCodec, EncoderPartialOutputFailurePreservesExistingFile) {
        const std::array<uint8_t, 3> old{9, 8, 7};
        const auto path = write(old);
        std::vector<uint8_t> pixels(1024 * 1024 * 3);
        uint32_t random = 0x127645;
        for (auto& value : pixels) {
            random ^= random << 13;
            random ^= random >> 17;
            random ^= random << 5;
            value = uint8_t(random);
        }
        codec::JxlEncodeOptions options;
        options.effort = 1;
        options.max_output_bytes = 65537;
        EXPECT_FALSE(codec::write_jxl(path, pixels, 1024, 1024, 3, codec::SampleType::UInt8, 3072, options));
        std::ifstream file(path, std::ios::binary);
        EXPECT_EQ((std::vector<uint8_t>((std::istreambuf_iterator<char>(file)), {})), (std::vector<uint8_t>(old.begin(), old.end())));
    }

    TEST_F(JxlCodec, EncoderLargeImagesPreserveSamplesAndRejectPixelBudget) {
        for (const auto dimensions : {std::pair{3840, 2160}, std::pair{7680, 4320}}) {
            const auto [width, height] = dimensions;
            std::vector<uint8_t> pixels(size_t(width) * height);
            for (size_t i = 0; i < pixels.size(); ++i)
                pixels[i] = uint8_t((i / 31) ^ (i / width));
            codec::JxlEncodeOptions options;
            options.effort = 1;
            options.max_threads = 4;
            std::vector<uint8_t> encoded;
            const auto status = codec::encode_jxl_memory(pixels, width, height, 1, codec::SampleType::UInt8, width, encoded, options);
            ASSERT_TRUE(status) << (status ? "" : status.error().detail());
            EXPECT_EQ(direct_decode(encoded, 1, codec::SampleType::UInt8), pixels);
            const auto previous = encoded;
            options.pixels = pixels.size() - 1;
            EXPECT_FALSE(codec::encode_jxl_memory(pixels, width, height, 1, codec::SampleType::UInt8, width, encoded, options));
            EXPECT_EQ(encoded, previous);
        }
    }

    TEST_F(JxlCodec, GenericDisplayByteWriterUsesTheSameAtomicJxlEncoder) {
        const std::array<uint8_t, 24> pixels{1, 2, 3, 0, 4, 5, 6, 255, 7, 8, 9, 128, 11, 12, 13, 64, 14, 15, 16, 255, 17, 18, 19, 0};
        const auto path = directory / std::filesystem::path(u8"generic_画像.JXL");
        const lfs::core::AtomicFileOptions publication{.overwrite = false, .durable = false, .create_directories = false};
        const auto written = codec::write_image_u8(path, pixels.data(), 3, 2, 4, 1, std::nullopt, publication);
        ASSERT_TRUE(written) << (written ? "" : written.error().detail());
        codec::Image decoded;
        std::string error;
        ASSERT_TRUE(codec::decode(path, decoded, error)) << error;
        EXPECT_EQ(decoded.data, (std::vector<uint8_t>(pixels.begin(), pixels.end())));
        const auto rejected = codec::write_image_u8(path, pixels.data(), 3, 2, 4, 95, std::string("metadata"), publication);
        EXPECT_FALSE(rejected);
        EXPECT_EQ(rejected.error().code(), lfs::ErrorCode::Unsupported);
        ASSERT_TRUE(codec::decode(path, decoded, error)) << error;
        EXPECT_EQ(decoded.data, (std::vector<uint8_t>(pixels.begin(), pixels.end())));
    }

    TEST_F(JxlCodec, EncoderAtomicFailureAndNoReplacePreserveExistingFile) {
        const std::array<uint8_t, 18> input{0, 1, 2, 3};
        const std::array<uint8_t, 3> old{9, 8, 7};
        const auto path = write(old);
        codec::JxlEncodeOptions options;
        options.max_output_bytes = 1;
        EXPECT_FALSE(codec::write_jxl(path, input, 3, 2, 3, codec::SampleType::UInt8, 9, options));
        options = {};
        EXPECT_FALSE(codec::write_jxl(path, input, 3, 2, 3, codec::SampleType::UInt8, 9, options, {.overwrite = false}));
        options.cancelled = [] { return true; };
        EXPECT_FALSE(codec::write_jxl(path, input, 3, 2, 3, codec::SampleType::UInt8, 9, options));
        std::ifstream file(path, std::ios::binary);
        EXPECT_EQ((std::vector<uint8_t>((std::istreambuf_iterator<char>(file)), {})), (std::vector<uint8_t>(old.begin(), old.end())));
    }

    TEST_F(JxlCodec, FixedFixtureDecodesWithoutInvokingTheEncoder) {
        const auto path = std::filesystem::path(LFS_JXL_FIXTURES) / "rgb8.jxl";
        codec::Image image;
        std::string error;
        ASSERT_TRUE(native_decode(path, image, error)) << error;
        ASSERT_EQ(image.data.size(), 12u);
        for (size_t i = 0; i < 4; ++i) {
            EXPECT_EQ(image.data[i * 3], 255);
            EXPECT_EQ(image.data[i * 3 + 1], 0);
            EXPECT_EQ(image.data[i * 3 + 2], 0);
        }
    }

    TEST_F(JxlCodec, TenAndTwelveBitSamplesUseTheFullUInt16Range) {
        for (const int bits : {10, 12}) {
            const uint32_t maximum = (1u << bits) - 1;
            const std::array<uint16_t, 6> samples{0, 1, uint16_t(maximum / 2), uint16_t(maximum / 2 + 1), uint16_t(maximum - 1), uint16_t(maximum)};
            const auto encoded = encode(samples.data(), sizeof(samples), {.channels = 1, .bits = bits, .codestream_range = true});
            codec::Image image;
            std::string error;
            ASSERT_TRUE(native_decode(encoded, image, error)) << error;
            ASSERT_EQ(image.sample_type, codec::SampleType::UInt16);
            for (size_t i = 0; i < samples.size(); ++i) {
                uint16_t actual;
                std::memcpy(&actual, image.data.data() + i * sizeof(actual), sizeof(actual));
                // libjxl normalizes through Float32, which can differ by one
                // UInt16 unit at a rounding boundary. Source codes stay exact.
                EXPECT_LE(std::abs(int(actual) - int((uint32_t(samples[i]) * 65535 + maximum / 2) / maximum)), 1);
                EXPECT_EQ((uint32_t(actual) * maximum + 32767) / 65535, samples[i]);
            }
        }
    }

    TEST_F(JxlCodec, AssociatedAlphaIsReturnedStraight) {
        std::array<float, 24> samples;
        for (size_t pixel = 0; pixel < 6; ++pixel) {
            samples[pixel * 4] = .25f;
            samples[pixel * 4 + 1] = .125f;
            samples[pixel * 4 + 2] = 0.f;
            samples[pixel * 4 + 3] = .5f;
        }
        const auto encoded = encode(samples.data(), sizeof(samples), {.channels = 4, .bits = 32, .floating = true, .premultiplied = true});
        codec::Image image;
        std::string error;
        ASSERT_TRUE(native_decode(encoded, image, error)) << error;
        for (size_t pixel = 0; pixel < 6; ++pixel)
            for (size_t channel = 0; channel < 4; ++channel) {
                float actual;
                std::memcpy(&actual, image.data.data() + (pixel * 4 + channel) * sizeof(float), sizeof(float));
                EXPECT_FLOAT_EQ(actual, channel == 0 || channel == 3 ? .5f : channel == 1 ? .25f
                                                                                          : 0.f);
            }
    }

    TEST_F(JxlCodec, HdrRawCodesRemainAvailableWithoutImplicitDisplayToneMapping) {
        const std::array<uint16_t, 18> samples{0, 1000, 65535, 65535, 32768, 0, 1, 2, 3, 500, 600, 700, 800, 900, 1000, 10000, 20000, 30000};
        for (const auto transfer : {JXL_TRANSFER_FUNCTION_PQ, JXL_TRANSFER_FUNCTION_HLG}) {
            const auto encoded = encode(samples.data(), sizeof(samples), {.bits = 16, .transfer = transfer, .intensity = 1000.f});
            codec::Image image;
            std::string error;
            ASSERT_TRUE(native_decode(encoded, image, error)) << error;
            ASSERT_EQ(image.data.size(), sizeof(samples));
            EXPECT_EQ(std::memcmp(image.data.data(), samples.data(), sizeof(samples)), 0);
            const auto previous = image.data;
            EXPECT_FALSE(native_decode(encoded, image, error, true));
            EXPECT_NE(error.find("tone-mapping"), std::string::npos);
            EXPECT_EQ(image.data, previous);
        }
    }

    TEST_F(JxlCodec, IccColorEncodingIsPreservedRawAndConvertedForDisplay) {
        const cmsCIExyY white{.3127, .329, 1.};
        const cmsCIExyYTRIPLE primaries{{.64, .33, 1.}, {.30, .60, 1.}, {.15, .06, 1.}};
        std::unique_ptr<cmsToneCurve, decltype(&cmsFreeToneCurve)> curve(cmsBuildGamma(nullptr, 2.2), cmsFreeToneCurve);
        ASSERT_TRUE(curve);
        cmsToneCurve* curves[]{curve.get(), curve.get(), curve.get()};
        std::unique_ptr<void, decltype(&cmsCloseProfile)> profile(cmsCreateRGBProfile(&white, &primaries, curves), cmsCloseProfile);
        ASSERT_TRUE(profile);
        cmsUInt32Number size = 0;
        ASSERT_TRUE(cmsSaveProfileToMem(profile.get(), nullptr, &size));
        std::vector<uint8_t> icc(size);
        ASSERT_TRUE(cmsSaveProfileToMem(profile.get(), icc.data(), &size));
        const std::array<uint8_t, 18> samples{0, 64, 128, 16, 32, 48, 128, 192, 255, 20, 40, 60, 10, 30, 50, 100, 150, 200};
        const auto encoded = encode(samples.data(), samples.size(), {.icc = icc});
        codec::Image raw, display;
        std::string error;
        ASSERT_TRUE(native_decode(encoded, raw, error)) << error;
        EXPECT_EQ(raw.data, (std::vector<uint8_t>(samples.begin(), samples.end())));
        ASSERT_TRUE(native_decode(encoded, display, error, true)) << error;
        for (size_t i = 0; i < samples.size(); ++i) {
            const double linear = std::pow(samples[i] / 255., 2.2);
            const double srgb = linear <= .0031308 ? linear * 12.92 : 1.055 * std::pow(linear, 1. / 2.4) - .055;
            EXPECT_NEAR(display.data[i], std::floor(srgb * 255. + .5), 1.);
        }
    }

    TEST_F(JxlCodec, AuxiliaryChannelsAreNotSilentlyDropped) {
        const std::array<uint8_t, 18> samples{};
        const auto encoded = encode(samples.data(), samples.size(), {.depth = true});
        codec::Image image{1, 1, 1, codec::SampleType::UInt8, {123}};
        std::string error;
        EXPECT_FALSE(native_decode(encoded, image, error));
        EXPECT_NE(error.find("auxiliary"), std::string::npos);
        EXPECT_EQ(image.data, (std::vector<uint8_t>{123}));
    }

    TEST_F(JxlCodec, BoundedMutationsPreserveFailureOutputAndDoNotPoisonLaterCalls) {
        const std::array<uint8_t, 18> samples{};
        const auto encoded = encode(samples.data(), samples.size(), {.container = true});
        codec::JxlDecodeLimits limits;
        limits.pixels = 4096;
        limits.output_bytes = 64 * 1024;
        limits.decoder_bytes = 8 * 1024 * 1024;
        for (size_t offset = 0; offset < encoded.size(); ++offset) {
            auto changed = encoded;
            changed[offset] ^= 0x80;
            codec::Image image{1, 1, 1, codec::SampleType::UInt8, {123}};
            std::string error;
            if (!native_decode(changed, image, error, false, limits)) {
                EXPECT_FALSE(error.empty());
                EXPECT_EQ(image.data, (std::vector<uint8_t>{123}));
            } else {
                EXPECT_LE(size_t(image.width) * image.height, limits.pixels);
                EXPECT_LE(image.data.size(), limits.output_bytes);
            }
        }
        codec::Image recovered;
        std::string error;
        ASSERT_TRUE(native_decode(encoded, recovered, error)) << error;
        EXPECT_EQ(recovered.data, (std::vector<uint8_t>(samples.begin(), samples.end())));
    }
    TEST_F(JxlCodec, OfficialStaticProfilesPatchesAndCoalescedLayersDecode) {
        for (const auto name : {"grayscale.jxl", "patches-lossless.jxl", "sunset-logo.jxl"}) {
            SCOPED_TRACE(name);
            const auto path = std::filesystem::path(LFS_JXL_FIXTURES) / "conformance" / name;
            std::ifstream input(path, std::ios::binary);
            const std::vector<uint8_t> encoded((std::istreambuf_iterator<char>(input)), {});
            codec::Image file, memory;
            std::string error;
            ASSERT_TRUE(native_decode(path, file, error)) << error;
            ASSERT_TRUE(native_decode(encoded, memory, error)) << error;
            EXPECT_EQ(file.width, memory.width);
            EXPECT_EQ(file.height, memory.height);
            EXPECT_EQ(file.data, memory.data);
            ASSERT_TRUE(native_decode(path, file, error, true)) << error;
            ASSERT_TRUE(native_decode(encoded, memory, error, true)) << error;
            EXPECT_EQ(file.data, memory.data);
        }
    }

    TEST_F(JxlCodec, FloatTargetAvoidsIntermediateIntegerClippingAndQuantization) {
        for (const auto name : {"grayscale.jxl", "sunset-logo.jxl"}) {
            SCOPED_TRACE(name);
            const auto path = std::filesystem::path(LFS_JXL_FIXTURES) / "conformance" / name;
            std::vector<float> file, memory;
            const auto allocate = [](size_t bytes, void* user) -> void* {
                auto& storage = *static_cast<std::vector<float>*>(user);
                storage.resize(bytes / sizeof(float));
                return storage.data();
            };
            codec::DecodeTarget file_target{0, codec::SampleType::Float32, nullptr, allocate, &file};
            codec::DecodeTarget memory_target{0, codec::SampleType::Float32, nullptr, allocate, &memory};
            codec::Probe info, memory_info;
            std::string error;
            ASSERT_TRUE(native_target(path, file_target, info, error)) << error;
            std::ifstream input(path, std::ios::binary);
            const std::vector<uint8_t> encoded((std::istreambuf_iterator<char>(input)), {});
            ASSERT_TRUE(native_target(encoded, memory_target, memory_info, error)) << error;
            EXPECT_EQ(file, memory);
            codec::JxlDecodeLimits limits;
            limits.output_bytes = file.size() * sizeof(float) - 1;
            unsigned allocations = 0;
            codec::DecodeTarget limited{0, codec::SampleType::Float32, nullptr,
                                        [](size_t, void* user) -> void* { ++*static_cast<unsigned*>(user); return nullptr; }, &allocations};
            const auto previous = memory_info;
            EXPECT_FALSE(native_target(encoded, limited, memory_info, error, false, limits));
            EXPECT_NE(error.find("output byte limit"), std::string::npos);
            EXPECT_EQ(allocations, 0u);
            EXPECT_EQ(limited.data, nullptr);
            EXPECT_EQ(memory_info.width, previous.width);
            EXPECT_EQ(memory_info.sample_type, previous.sample_type);
            EXPECT_GT(*std::max_element(file.begin(), file.end()), 1.f);
            if (std::string_view(name) == "grayscale.jxl") {
                ASSERT_EQ(info.width, 200);
                ASSERT_EQ(info.height, 200);
                ASSERT_EQ(info.channels, 1);
                // Independently published corpus Float32 reference values.
                EXPECT_NEAR(file[50 * 200 + 50], .08087914437055588f, .004f);
                EXPECT_NEAR(file[100 * 200 + 100], .3699168562889099f, .004f);
                EXPECT_NEAR(file[199 * 200 + 199], 1.001685380935669f, .004f);
            } else {
                EXPECT_LT(*std::min_element(file.begin(), file.end()), -1.f);
            }
        }
    }

    std::vector<uint8_t> review_icc(int hdr, double gamma = 1.8) {
        std::vector<float> values(4096);
        for (size_t i = 0; i < values.size(); ++i) {
            const float x = float(i) / float(values.size() - 1);
            values[i] = hdr == 16 ? lfs::core::color::pqToLinear(x) : hdr == 18 ? lfs::core::color::hlgToSceneLinear(x)
                                                                                : std::pow(x, gamma);
        }
        std::unique_ptr<cmsToneCurve, decltype(&cmsFreeToneCurve)> curve(cmsBuildTabulatedToneCurveFloat(nullptr, static_cast<cmsUInt32Number>(values.size()), values.data()), cmsFreeToneCurve);
        if (!curve)
            throw std::bad_alloc();
        cmsToneCurve* curves[]{curve.get(), curve.get(), curve.get()};
        cmsCIExyY white{.3127, .3290, 1.};
        cmsCIExyYTRIPLE primaries{{.708, .292, 1.}, {.170, .797, 1.}, {.131, .046, 1.}};
        if (!hdr)
            primaries = {{.64, .33, 1.}, {.30, .60, 1.}, {.15, .06, 1.}};
        std::unique_ptr<void, decltype(&cmsCloseProfile)> profile(cmsCreateRGBProfile(&white, &primaries, curves), cmsCloseProfile);
        if (!profile || cmsIsTag(profile.get(), cmsSigcicpTag))
            throw std::runtime_error("Expected ICC-only profile without cicp");
        cmsUInt32Number bytes = 0;
        if (!cmsSaveProfileToMem(profile.get(), nullptr, &bytes))
            throw std::runtime_error("ICC serialization failed");
        std::vector<uint8_t> result(bytes);
        if (!cmsSaveProfileToMem(profile.get(), result.data(), &bytes))
            throw std::runtime_error("ICC serialization failed");
        return result;
    }

    TEST_F(JxlCodec, IccOnlyPqAndHlgRequireExplicitNativePolicy) {
        const std::array<uint8_t, 18> samples{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
        for (const int transfer : {16, 18}) {
            const auto icc = review_icc(transfer);
            const auto encoded = encode(samples.data(), samples.size(), {.icc = icc});
            const auto path = write(encoded);
            codec::Image image{1, 1, 1, codec::SampleType::UInt8, {123}};
            std::string error;
            EXPECT_FALSE(codec::decode_jxl(encoded, image, error));
            EXPECT_NE(error.find("HDR"), std::string::npos);
            EXPECT_EQ(image.data, (std::vector<uint8_t>{123}));
            EXPECT_FALSE(codec::decode(path, image, error, true));
            EXPECT_NE(error.find("HDR"), std::string::npos);
            ASSERT_TRUE(native_decode(encoded, image, error)) << error;
            EXPECT_EQ(image.data, (std::vector<uint8_t>(samples.begin(), samples.end())));
            EXPECT_FALSE(native_decode(encoded, image, error, true));
        }
    }

    TEST_F(JxlCodec, CanonicalSrgbIsIndependentOfPrecisionAndOriginalProfileRepresentation) {
        const std::array<uint8_t, 18> samples{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
        const auto icc = review_icc(0);
        for (const bool lossless : {true, false})
            for (const bool embedded : {true, false}) {
                const auto encoded = encode(samples.data(), samples.size(), {.lossless = lossless, .gamma = embedded ? 0. : 1.8, .icc = embedded ? std::span<const uint8_t>(icc) : std::span<const uint8_t>()});
                const auto path = write(encoded);
                codec::Image bytes;
                std::string error;
                ASSERT_TRUE(codec::decode(path, bytes, error)) << error;
                std::array<float, 18> floats{};
                codec::DecodeTarget target{3, codec::SampleType::Float32, floats.data()};
                codec::Probe info;
                ASSERT_TRUE(codec::decode_memory_to_buffer(encoded.data(), encoded.size(), target, info, error)) << error;
                const float expected = lfs::core::color::linearToSrgb(std::pow(128.f / 255.f, 1.8f));
                EXPECT_NEAR(floats[0], expected, .003f);
                EXPECT_NEAR(bytes.data[0] / 255.f, floats[0], 1.1f / 255.f);
                codec::Image display;
                ASSERT_TRUE(codec::decode(path, display, error, true)) << error;
                EXPECT_EQ(bytes.data, display.data);
                std::array<uint16_t, 18> wide{};
                codec::DecodeTarget wide_target{3, codec::SampleType::UInt16, wide.data()};
                ASSERT_TRUE(codec::decode_to_buffer(path, wide_target, info, error, true)) << error;
                EXPECT_NEAR(wide[0] / 65535.f, floats[0], 2.f / 65535.f);
            }
    }

    TEST_F(JxlCodec, MissingEmptyAndTruncatedFilesHaveDistinctDiagnostics) {
        codec::Image image;
        std::string error;
        EXPECT_FALSE(codec::decode_jxl(directory / "missing.jxl", image, error));
        EXPECT_NE(error.find("cannot open"), std::string::npos);
        const auto empty = write({});
        EXPECT_FALSE(codec::decode_jxl(empty, image, error));
        EXPECT_NE(error.find("empty source"), std::string::npos);
        const std::array<uint8_t, 2> truncated{0xff, 0x0a};
        EXPECT_FALSE(codec::decode_jxl(write(truncated), image, error));
        EXPECT_EQ(error.find("cannot open"), std::string::npos);
    }

    TEST_F(JxlCodec, ParallelAndSerialDecodingAgreeAndAllocationLimitsRecover) {
        std::vector<uint8_t> samples(2048 * 1024 * 3);
        for (size_t i = 0; i < samples.size(); ++i)
            samples[i] = static_cast<uint8_t>(i * 73 + i / 127);
        const auto encoded = encode(samples.data(), samples.size(), {.width = 2048, .height = 1024, .lossless = false});
        codec::Image serial, parallel;
        std::string error;
        codec::JxlDecodeLimits limits;
        limits.max_threads = 1;
        ASSERT_TRUE(codec::decode_jxl(encoded, serial, error, false, limits)) << error;
        limits.max_threads = 4;
        ASSERT_TRUE(codec::decode_jxl(encoded, parallel, error, false, limits)) << error;
        EXPECT_EQ(serial.data, parallel.data);
        limits.decoder_bytes = 32;
        EXPECT_FALSE(codec::decode_jxl(encoded, parallel, error, false, limits));
        EXPECT_EQ(serial.data, parallel.data);
        limits.decoder_bytes = size_t{1} << 30;
        ASSERT_TRUE(codec::decode_jxl(encoded, parallel, error, false, limits)) << error;
        EXPECT_EQ(serial.data, parallel.data);
        std::vector<std::future<codec::Image>> readers;
        for (int i = 0; i < 4; ++i)
            readers.push_back(std::async(std::launch::async, [&] {
                codec::Image result;
                std::string diagnostic;
                if (!codec::decode_jxl(encoded, result, diagnostic, false, limits))
                    throw std::runtime_error(diagnostic);
                return result;
            }));
        for (auto& reader : readers)
            EXPECT_EQ(reader.get().data, serial.data);
    }

    TEST_F(JxlCodec, LargeImagesPreserveSamplesAndRecoverFromLimits) {
        for (const auto dimensions : {std::pair{3840, 2160}, std::pair{7680, 4320}}) {
            const auto [width, height] = dimensions;
            std::vector<uint8_t> samples(size_t(width) * height);
            for (size_t i = 0; i < samples.size(); ++i)
                samples[i] = static_cast<uint8_t>((i / width + i % width) % 251);
            const auto encoded = encode(samples.data(), samples.size(), {.width = width, .height = height, .channels = 1});
            codec::Image image{1, 1, 1, codec::SampleType::UInt8, {123}};
            codec::JxlDecodeLimits limits;
            limits.output_bytes = samples.size() - 1;
            std::string error;
            EXPECT_FALSE(native_decode(encoded, image, error, false, limits));
            EXPECT_EQ(image.data, (std::vector<uint8_t>{123}));
            ASSERT_TRUE(native_decode(encoded, image, error)) << error;
            EXPECT_EQ(image.width, width);
            EXPECT_EQ(image.height, height);
            EXPECT_EQ(image.data, samples);
        }
    }
} // namespace
