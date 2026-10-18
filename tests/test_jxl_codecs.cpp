/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/image_jxl.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <gtest/gtest.h>
#include <jxl/encode.h>
#include <lcms2.h>
#include <memory>
#include <stdexcept>

namespace {
    namespace codec = lfs::core::image_codecs;
    struct Encoding {
        int width = 3, height = 2, channels = 3, bits = 8;
        bool floating = false, container = false, animation = false, premultiplied = false, lossless = true;
        bool codestream_range = false, depth = false;
        JxlTransferFunction transfer = JXL_TRANSFER_FUNCTION_UNKNOWN;
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
        ASSERT_TRUE(codec::decode_jxl(encoded, image, error)) << error;
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
            EXPECT_FALSE(codec::decode_jxl(path, image, error, false, limits));
            EXPECT_NE(error, "old error");
            EXPECT_FALSE(error.empty());
            EXPECT_EQ(image.data, (std::vector<uint8_t>{123}));
            ASSERT_TRUE(codec::decode_jxl(path, image, error)) << error;
            EXPECT_TRUE(error.empty());
            EXPECT_EQ(image.data, (std::vector<uint8_t>(samples.begin(), samples.end())));
        }
        for (const size_t length : {size_t{0}, size_t{1}, encoded.size() / 2}) {
            codec::Image image{1, 1, 1, codec::SampleType::UInt8, {123}};
            std::string error;
            EXPECT_FALSE(codec::decode_jxl(std::span(encoded).first(length), image, error));
            EXPECT_EQ(image.data, (std::vector<uint8_t>{123}));
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
        EXPECT_FALSE(codec::decode_jxl_to_buffer(std::span(encoded).first(encoded.size() / 2), target, probe, error));
        EXPECT_EQ(allocation.calls, 0u);
        EXPECT_EQ(probe.width, 9);
        ASSERT_TRUE(codec::decode_jxl_to_buffer(encoded, target, probe, error)) << error;
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
        ASSERT_TRUE(codec::decode_jxl(write(encoded), image, error)) << error;
        EXPECT_EQ(image.data, (std::vector<uint8_t>(pixels.begin(), pixels.end())));
        const auto animated = encode(pixels.data(), pixels.size(), {.container = true, .animation = true});
        const auto previous = image.data;
        EXPECT_FALSE(codec::decode_jxl(animated, image, error));
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
            ASSERT_TRUE(codec::decode_jxl(encoded, image, error)) << error;
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
        ASSERT_TRUE(codec::decode_jxl(encoded, image, error)) << error;
        ASSERT_EQ(image.data.size(), values.size() * sizeof(uint16_t));
        EXPECT_EQ(std::memcmp(image.data.data(), values.data(), image.data.size()), 0);
    }

    TEST_F(JxlCodec, DisplayColorConversionIsSeparateFromRawFloatValues) {
        const std::array<float, 6> linear{0.f, 0.003f, 0.18f, 0.5f, 0.75f, 1.f};
        const auto encoded = encode(linear.data(), sizeof(linear), {.channels = 1, .bits = 32, .floating = true});
        codec::Image raw, display;
        std::string error;
        ASSERT_TRUE(codec::decode_jxl(encoded, raw, error)) << error;
        ASSERT_TRUE(codec::decode_jxl(encoded, display, error, true)) << error;
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
        ASSERT_TRUE(codec::decode_jxl(encoded, expected, error)) << error;
        std::vector<std::future<std::vector<uint8_t>>> readers;
        for (int i = 0; i < 8; ++i)
            readers.push_back(std::async(std::launch::async, [&] {
                codec::Image image;
                std::string diagnostic;
                if (!codec::decode_jxl(path, image, diagnostic))
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
            if (!codec::decode_jxl(encoded, image, error, false, limits)) {
                ++failures;
                EXPECT_NE(error.find("allocation"), std::string::npos) << error;
                EXPECT_EQ(image.data, (std::vector<uint8_t>{123}));
            }
            ASSERT_TRUE(codec::decode_jxl(encoded, image, error)) << error;
        }
        EXPECT_GT(failures, 0u);
    }

    TEST_F(JxlCodec, BorrowedInputAndFailedTargetsDoNotChangeOwnedResults) {
        const std::array<uint8_t, 18> samples{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
        auto encoded = encode(samples.data(), samples.size());
        const auto valid = encoded;
        codec::Image image;
        std::string error;
        ASSERT_TRUE(codec::decode_jxl(encoded, image, error)) << error;
        std::fill(encoded.begin(), encoded.end(), 0xff);
        EXPECT_EQ(image.data, (std::vector<uint8_t>(samples.begin(), samples.end())));
        std::array<uint8_t, 18> output;
        output.fill(123);
        codec::DecodeTarget target{3, codec::SampleType::UInt8, output.data()};
        codec::Probe probe{9, 9, 9};
        EXPECT_FALSE(codec::decode_jxl_to_buffer(encoded, target, probe, error));
        EXPECT_EQ(probe.width, 9);
        for (const auto value : output)
            EXPECT_EQ(value, 123);
        ASSERT_TRUE(codec::decode_jxl_to_buffer(valid, target, probe, error)) << error;
        EXPECT_EQ(output, samples);
        unsigned calls = 0;
        target.data = nullptr;
        target.user = &calls;
        target.allocate = [](size_t, void* user) -> void* { ++*static_cast<unsigned*>(user); return nullptr; };
        EXPECT_FALSE(codec::decode_jxl_to_buffer(valid, target, probe, error));
        EXPECT_EQ(calls, 1u);
        EXPECT_EQ(target.data, nullptr);
    }

    TEST_F(JxlCodec, FixedFixtureDecodesWithoutInvokingTheEncoder) {
        const auto path = std::filesystem::path(LFS_JXL_FIXTURES) / "rgb8.jxl";
        codec::Image image;
        std::string error;
        ASSERT_TRUE(codec::decode_jxl(path, image, error)) << error;
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
            ASSERT_TRUE(codec::decode_jxl(encoded, image, error)) << error;
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
        ASSERT_TRUE(codec::decode_jxl(encoded, image, error)) << error;
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
            ASSERT_TRUE(codec::decode_jxl(encoded, image, error)) << error;
            ASSERT_EQ(image.data.size(), sizeof(samples));
            EXPECT_EQ(std::memcmp(image.data.data(), samples.data(), sizeof(samples)), 0);
            const auto previous = image.data;
            EXPECT_FALSE(codec::decode_jxl(encoded, image, error, true));
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
        ASSERT_TRUE(codec::decode_jxl(encoded, raw, error)) << error;
        EXPECT_EQ(raw.data, (std::vector<uint8_t>(samples.begin(), samples.end())));
        ASSERT_TRUE(codec::decode_jxl(encoded, display, error, true)) << error;
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
        EXPECT_FALSE(codec::decode_jxl(encoded, image, error));
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
            if (!codec::decode_jxl(changed, image, error, false, limits)) {
                EXPECT_FALSE(error.empty());
                EXPECT_EQ(image.data, (std::vector<uint8_t>{123}));
            } else {
                EXPECT_LE(size_t(image.width) * image.height, limits.pixels);
                EXPECT_LE(image.data.size(), limits.output_bytes);
            }
        }
        codec::Image recovered;
        std::string error;
        ASSERT_TRUE(codec::decode_jxl(encoded, recovered, error)) << error;
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
            ASSERT_TRUE(codec::decode_jxl(path, file, error)) << error;
            ASSERT_TRUE(codec::decode_jxl(encoded, memory, error)) << error;
            EXPECT_EQ(file.width, memory.width);
            EXPECT_EQ(file.height, memory.height);
            EXPECT_EQ(file.data, memory.data);
            ASSERT_TRUE(codec::decode_jxl(path, file, error, true)) << error;
            ASSERT_TRUE(codec::decode_jxl(encoded, memory, error, true)) << error;
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
            ASSERT_TRUE(codec::decode_jxl_to_buffer(path, file_target, info, error)) << error;
            std::ifstream input(path, std::ios::binary);
            const std::vector<uint8_t> encoded((std::istreambuf_iterator<char>(input)), {});
            ASSERT_TRUE(codec::decode_jxl_to_buffer(encoded, memory_target, memory_info, error)) << error;
            EXPECT_EQ(file, memory);
            codec::JxlDecodeLimits limits;
            limits.output_bytes = file.size() * sizeof(float) - 1;
            unsigned allocations = 0;
            codec::DecodeTarget limited{0, codec::SampleType::Float32, nullptr,
                                        [](size_t, void* user) -> void* { ++*static_cast<unsigned*>(user); return nullptr; }, &allocations};
            const auto previous = memory_info;
            EXPECT_FALSE(codec::decode_jxl_to_buffer(encoded, limited, memory_info, error, false, limits));
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
            EXPECT_FALSE(codec::decode_jxl(encoded, image, error, false, limits));
            EXPECT_EQ(image.data, (std::vector<uint8_t>{123}));
            ASSERT_TRUE(codec::decode_jxl(encoded, image, error)) << error;
            EXPECT_EQ(image.width, width);
            EXPECT_EQ(image.height, height);
            EXPECT_EQ(image.data, samples);
        }
    }
} // namespace
