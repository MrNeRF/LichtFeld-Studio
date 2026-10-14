/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/image_codecs.hpp"
#include "core/image_heif.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <future>
#include <gtest/gtest.h>
#include <iterator>
#include <libheif/heif.h>
#include <libheif/heif_properties.h>
#ifdef _WIN32
#include <objbase.h>
#endif
namespace {
    namespace codec = lfs::core::image_codecs;
    const std::filesystem::path fixtures = LFS_HEIF_FIXTURES;
    TEST(ProductionHeif, DistributionUsesDav1dWithoutHevcOrEncoders) {
        EXPECT_TRUE(heif_have_decoder_for_format(heif_compression_AV1));
        EXPECT_FALSE(heif_have_decoder_for_format(heif_compression_HEVC));
        EXPECT_FALSE(heif_have_encoder_for_format(heif_compression_AV1));
        EXPECT_FALSE(heif_have_encoder_for_format(heif_compression_HEVC));
        const heif_decoder_descriptor* decoder = nullptr;
        ASSERT_EQ(heif_get_decoder_descriptors(heif_compression_AV1, &decoder, 1), 1);
        EXPECT_STREQ(heif_decoder_descriptor_get_id_name(decoder), "dav1d");
    }
    TEST(ProductionHeif, SystemProviderPreservesQualifiedSamplesAndMemoryTargets) {
        const auto path = fixtures / "system-p3icc.heic";
        codec::Image image;
        std::string error;
        if (!codec::decode(path, image, error)) {
            EXPECT_TRUE(image.data.empty());
            EXPECT_NE(error.find("HEVC"), std::string::npos) << error;
#ifdef __APPLE__
            FAIL() << "Apple system provider failed its qualified RGB8 input: " << error;
#elif defined(_WIN32)
            ASSERT_TRUE(error.find("provider open failed") != std::string::npos || error.find("provider factory failed") != std::string::npos) << error;
            GTEST_SKIP() << "Microsoft HEIF/HEVC extension unavailable: " << error;
#else
            GTEST_SKIP() << "No configured system HEIC provider: " << error;
#endif
        }
        const auto check = [&](const std::filesystem::path& input, const char* reference, int bits) {
            codec::Image raw, memory, display;
            codec::Probe info;
            ASSERT_TRUE(codec::probe(input, info, error)) << error;
            ASSERT_TRUE(codec::decode(input, raw, error)) << error;
            std::ifstream source(input, std::ios::binary), ref(fixtures / reference, std::ios::binary);
            const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(source)), {});
            const std::vector<uint8_t> expected((std::istreambuf_iterator<char>(ref)), {});
            ASSERT_EQ(raw.data.size(), expected.size());
            if (bits == 8)
                EXPECT_EQ(raw.data, expected);
            else {
                const uint32_t maximum = (1u << bits) - 1;
                for (size_t i = 0; i < expected.size(); i += 2) {
                    uint16_t code, actual;
                    std::memcpy(&code, expected.data() + i, 2);
                    std::memcpy(&actual, raw.data.data() + i, 2);
                    code >>= 16 - bits;
                    EXPECT_EQ(actual, (uint32_t(code) * 65535 + maximum / 2) / maximum) << i;
                }
            }
            EXPECT_EQ(raw.width, info.width);
            EXPECT_EQ(raw.height, info.height);
            EXPECT_EQ(raw.sample_type, info.sample_type);
            ASSERT_TRUE(codec::decode_heif(bytes, memory, error)) << error;
            EXPECT_EQ(memory.data, raw.data);
            ASSERT_TRUE(codec::decode_heif(input, display, error, true)) << error;
            codec::DecodeTarget target;
            target.channels = 3;
            target.sample_type = raw.sample_type;
            std::vector<uint8_t> output(raw.data.size());
            target.data = output.data();
            ASSERT_TRUE(codec::decode_heif_to_buffer(bytes, target, info, error)) << error;
            EXPECT_EQ(output, raw.data);
        };
#ifdef __APPLE__
        check(fixtures / "rgb8.heic", "system-grid.reference", 8);
#endif
        for (const auto name : {"rotated", "mirrored", "multi", "p3icc"}) {
            SCOPED_TRACE(name);
            check(fixtures / (std::string("system-") + name + ".heic"), (std::string("system-") + name + ".reference").c_str(), 8);
        }
#ifdef __APPLE__
        check(fixtures / "system-rgb10.heic", "system-rgb10.reference", 10);
#endif
        std::array<std::future<bool>, 4> workers;
        for (auto& worker : workers)
            worker = std::async(std::launch::async, [&] {
                codec::Image concurrent;
                std::string failure;
                return codec::decode(path, concurrent, failure) && concurrent.data == image.data;
            });
        for (auto& worker : workers)
            EXPECT_TRUE(worker.get());
#ifdef _WIN32
        EXPECT_TRUE(std::async(std::launch::async, [&] {
                        const auto status = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
                        if (FAILED(status))
                            return false;
                        struct Apartment {
                            ~Apartment() { CoUninitialize(); }
                        } apartment;
                        codec::Image sta;
                        std::string failure;
                        return codec::decode(path, sta, failure) && sta.data == image.data;
                    }).get());
#endif
        codec::Image invalid;
        EXPECT_FALSE(codec::decode_heif(fixtures / "system-badicc.heic", invalid, error, true));
        EXPECT_NE(error.find("profile"), std::string::npos) << error;
        codec::DecodeTarget failed;
        failed.channels = 3;
        failed.sample_type = codec::SampleType::UInt8;
        failed.allocate = [](size_t, void*) -> void* { return nullptr; };
        codec::Probe info;
        EXPECT_FALSE(codec::decode_heif_to_buffer(path, failed, info, error));
        EXPECT_EQ(failed.data, nullptr);
        EXPECT_NE(error.find("allocate"), std::string::npos) << error;
    }
    TEST(ProductionHeif, SystemProviderNeverInventsPrecisionOrAlpha) {
        for (const auto name : {"system-rgb12.heic", "system-rgba8.heic"
#ifndef __APPLE__
                                ,
                                "system-rgb10.heic", "rgb8.heic"
#endif
             }) {
            SCOPED_TRACE(name);
            codec::Image image;
            codec::Probe info;
            std::string error;
            EXPECT_FALSE(codec::probe(fixtures / name, info, error));
            EXPECT_EQ(info.width, 0);
            EXPECT_FALSE(codec::decode_heif(fixtures / name, image, error));
            EXPECT_TRUE(image.data.empty());
            EXPECT_NE(error.find("HEVC"), std::string::npos) << error;
            std::array<uint8_t, 16> unchanged;
            unchanged.fill(0xa5);
            codec::DecodeTarget target;
            target.channels = 3;
            target.sample_type = codec::SampleType::UInt8;
            target.data = unchanged.data();
            EXPECT_FALSE(codec::decode_heif_to_buffer(fixtures / name, target, info, error));
            EXPECT_TRUE(std::all_of(unchanged.begin(), unchanged.end(), [](auto value) { return value == 0xa5; }));
        }
    }
    TEST(ProductionHeif, SystemProviderUsesContainerRotationInsteadOfExifTwice) {
        codec::Image available;
        std::string error;
        if (!codec::decode(fixtures / "system-p3icc.heic", available, error)) {
#ifdef __APPLE__
            FAIL() << "Apple system provider failed its qualified RGB8 input: " << error;
#elif defined(_WIN32)
            ASSERT_TRUE(error.find("provider open failed") != std::string::npos || error.find("provider factory failed") != std::string::npos) << error;
#endif
            GTEST_SKIP() << "OS provider unavailable: " << error;
        }
        std::ifstream input(fixtures / "system-rotated.heic", std::ios::binary);
        const std::vector<uint8_t> original((std::istreambuf_iterator<char>(input)), {});
        std::ifstream reference(fixtures / "system-p3icc.reference", std::ios::binary);
        const std::vector<uint8_t> rgb((std::istreambuf_iterator<char>(reference)), {});
        ASSERT_EQ(rgb.size(), 32u * 24 * 3);
        const std::array<uint8_t, 4> marker{'i', 'r', 'o', 't'};
        const auto property = std::search(original.begin(), original.end(), marker.begin(), marker.end());
        ASSERT_NE(property, original.end());
        const auto offset = static_cast<size_t>(property - original.begin());
        ASSERT_LT(offset + 4, original.size());
        for (uint8_t angle = 0; angle < 4; ++angle) {
            SCOPED_TRACE(angle);
            auto bytes = original;
            bytes[offset + 4] = angle; // Leave EXIF orientation 6 unchanged deliberately.
            codec::Image image;
            ASSERT_TRUE(codec::decode_heif(bytes, image, error)) << error;
            const int width = angle % 2 ? 24 : 32, height = angle % 2 ? 32 : 24;
            ASSERT_EQ(image.width, width);
            ASSERT_EQ(image.height, height);
            std::vector<uint8_t> expected(rgb.size());
            for (int y = 0; y < 24; ++y)
                for (int x = 0; x < 32; ++x) {
                    const int dx = angle == 0 ? x : angle == 1 ? y
                                                : angle == 2   ? 31 - x
                                                               : 23 - y;
                    const int dy = angle == 0 ? y : angle == 1 ? 31 - x
                                                : angle == 2   ? 23 - y
                                                               : x;
                    std::memcpy(expected.data() + (dy * width + dx) * 3, rgb.data() + (y * 32 + x) * 3, 3);
                }
            EXPECT_EQ(image.data, expected);
        }
    }
    TEST(ProductionHeif, PrimaryRgbAlphaRotationAndPrecisionMatchIndependentDecode) {
        for (const auto name : {"rgb8", "rgba8", "rotated", "mirrored", "rgb10", "rgb12", "pq", "yuv420", "bt709", "linear", "untagged", "hlg", "rgba12"}) {
            SCOPED_TRACE(name);
            codec::Probe info;
            codec::Image image;
            std::string error;
            const auto path = fixtures / (std::string(name) + ".avif");
            ASSERT_TRUE(codec::probe(path, info, error)) << error;
            ASSERT_TRUE(codec::decode(path, image, error)) << error;
            EXPECT_EQ(image.width, info.width);
            EXPECT_EQ(image.height, info.height);
            EXPECT_EQ(image.channels, (std::string_view(name) == "rgba8" || std::string_view(name) == "rgba12") ? 4 : 3);
            EXPECT_EQ(image.width, std::string_view(name) == "rotated" ? 24 : std::string_view(name) == "rgb8" ? 128
                                                                                                               : 32);
            EXPECT_EQ(image.height, std::string_view(name) == "rotated" ? 32 : std::string_view(name) == "rgb8" ? 96
                                                                                                                : 24);
            std::ifstream file(fixtures / (std::string(name) + ".reference"), std::ios::binary);
            std::vector<uint8_t> expected((std::istreambuf_iterator<char>(file)), {});
            ASSERT_EQ(expected.size(), image.data.size());
            if (std::string_view(name) == "rgb12" || std::string_view(name) == "rgb10" || std::string_view(name) == "pq" || std::string_view(name) == "hlg" || std::string_view(name) == "rgba12") {
                ASSERT_EQ(image.sample_type, codec::SampleType::UInt16);
                bool precision = false;
                for (size_t i = 0; i < expected.size(); i += 2) {
                    const int bits = std::string_view(name) == "rgb10" ? 10 : 12;
                    const auto sample = (uint16_t(expected[i]) | (uint16_t(expected[i + 1]) << 8)) >> (16 - bits);
                    const uint32_t maximum = (1u << bits) - 1;
                    const auto reference = (uint32_t(sample) * 65535 + maximum / 2) / maximum;
                    uint16_t actual;
                    std::memcpy(&actual, image.data.data() + i, 2);
                    ASSERT_LE(std::abs(int(actual) - int(reference)), 1) << i;
                    precision |= actual % 257 != 0;
                }
                EXPECT_TRUE(precision);
                uint16_t first, second;
                std::memcpy(&first, image.data.data(), 2);
                std::memcpy(&second, image.data.data() + 2, 2);
                EXPECT_NE(first, second);
                EXPECT_EQ((uint32_t(first) + 128) / 257, (uint32_t(second) + 128) / 257) << "Fixture must contain distinct native codes inside one RGB8 bucket";
            } else {
                for (size_t i = 0; i < expected.size(); ++i)
                    ASSERT_LE(std::abs(int(expected[i]) - image.data[i]), 1) << i;
            }
        }
    }
    TEST(ProductionHeif, DisplayP3MatchesIndependentMatrixAndHdrIsNotSilentlySdr) {
        codec::Image raw, display;
        std::string error;
        for (const auto name : {"p3.avif", "p3icc.avif"}) {
            ASSERT_TRUE(codec::decode_heif(fixtures / name, raw, error)) << error;
            ASSERT_TRUE(codec::decode_heif(fixtures / name, display, error, true)) << error;
            EXPECT_NE(raw.data, display.data);
            const auto linear = [](double v) { return v <= .04045 ? v / 12.92 : std::pow((v + .055) / 1.055, 2.4); };
            const auto encoded = [](double v) {v=std::clamp(v,0.,1.);return v<=.0031308?v*12.92:1.055*std::pow(v,1./2.4)-.055; };
            for (size_t i = 0; i < raw.data.size(); i += 3) {
                const double r = linear(raw.data[i] / 255.), g = linear(raw.data[i + 1] / 255.), b = linear(raw.data[i + 2] / 255.);
                const std::array<double, 3> reference{encoded(1.224940176 * r - .224940176 * g), encoded(-.042056955 * r + 1.042056955 * g), encoded(-.019637555 * r - .078636046 * g + 1.098273601 * b)};
                for (size_t c = 0; c < 3; ++c)
                    EXPECT_NEAR(display.data[i + c], std::lround(reference[c] * 255), 2) << i << ":" << c;
            }
        }
        ASSERT_TRUE(codec::decode_heif(fixtures / "pq.avif", raw, error)) << error;
        EXPECT_FALSE(codec::decode_heif(fixtures / "pq.avif", display, error, true));
        EXPECT_NE(error.find("SDR"), std::string::npos);
        ASSERT_TRUE(codec::decode_heif(fixtures / "hlg.avif", raw, error));
        EXPECT_FALSE(codec::decode_heif(fixtures / "hlg.avif", display, error, true));
    }
    TEST(ProductionHeif, IccProfilesAndPrimaryItemAreExplicit) {
        codec::Image image, raw;
        std::string error;
        ASSERT_TRUE(codec::decode_heif(fixtures / "icc.avif", image, error, true)) << error;
        ASSERT_TRUE(codec::decode_heif(fixtures / "icc.avif", raw, error)) << error;
        ASSERT_EQ(image.data.size(), raw.data.size());
        for (size_t i = 0; i < image.data.size(); ++i)
            EXPECT_NEAR(image.data[i], raw.data[i], 1);
        ASSERT_TRUE(codec::decode_heif(fixtures / "badicc.avif", raw, error)) << error;
        EXPECT_FALSE(codec::decode_heif(fixtures / "badicc.avif", image, error, true));
        EXPECT_NE(error.find("profile"), std::string::npos);
        codec::Probe info;
        ASSERT_TRUE(codec::probe(fixtures / "multi.avif", info, error)) << error;
        EXPECT_EQ(info.width, 16);
        EXPECT_EQ(info.height, 16);
        ASSERT_TRUE(codec::decode(fixtures / "multi.avif", image, error)) << error;
        EXPECT_EQ(image.width, 16);
        EXPECT_EQ(image.height, 16);
        ASSERT_GE(image.data.size(), 3);
        EXPECT_NEAR(image.data[0], 200, 1);
        EXPECT_NEAR(image.data[1], 80, 1);
        EXPECT_NEAR(image.data[2], 30, 1);
    }
    TEST(ProductionHeif, IccPrecedesHdrNclxAndInvalidIccDoesNotBypassValidation) {
        for (const auto name : {"p3icc.avif", "badicc.avif", "system-p3icc.heic", "system-badicc.heic"}) {
            SCOPED_TRACE(name);
            std::ifstream file(fixtures / name, std::ios::binary);
            const std::vector<uint8_t> original((std::istreambuf_iterator<char>(file)), {});
            codec::Image reference_raw, reference_display;
            std::string error;
            const bool decoded = codec::decode_heif(original, reference_raw, error);
            if (!decoded && std::string_view(name).ends_with(".heic")) {
                EXPECT_NE(error.find("HEVC"), std::string::npos) << error;
                continue; // An absent OS provider does not skip the AV1 precedence tests.
            }
            ASSERT_TRUE(decoded) << error;
            const bool valid = std::string_view(name).find("p3icc") != std::string_view::npos;
            if (valid)
                ASSERT_TRUE(codec::decode_heif(original, reference_display, error, true)) << error;
            for (const auto transfer : {13, 16, 18}) {
                SCOPED_TRACE(transfer);
                // Modify only the explicit NCLX transfer field, preserving ICC,
                // compressed pixels, item associations and container geometry.
                auto bytes = original;
                const std::array<uint8_t, 8> marker{'c', 'o', 'l', 'r', 'n', 'c', 'l', 'x'};
                const auto tag = std::search(bytes.begin(), bytes.end(), marker.begin(), marker.end());
                ASSERT_NE(tag, bytes.end());
                const auto offset = static_cast<size_t>(tag - bytes.begin());
                ASSERT_GE(bytes.size() - offset, 15u);
                bytes[offset + 10] = 0;
                bytes[offset + 11] = static_cast<uint8_t>(transfer);
                // Verify both declarations survived serialization before testing precedence.
                std::unique_ptr<heif_context, decltype(&heif_context_free)> parsed(heif_context_alloc(), heif_context_free);
                ASSERT_NE(parsed, nullptr);
                const auto read_status = heif_context_read_from_memory_without_copy(parsed.get(), bytes.data(), bytes.size(), nullptr);
                ASSERT_EQ(read_status.code, heif_error_Ok) << read_status.message;
                heif_image_handle* handle = nullptr;
                ASSERT_EQ(heif_context_get_primary_image_handle(parsed.get(), &handle).code, heif_error_Ok);
                std::unique_ptr<heif_image_handle, decltype(&heif_image_handle_release)> owner(handle, heif_image_handle_release);
                EXPECT_GT(heif_image_handle_get_raw_color_profile_size(handle), 0u);
                heif_color_profile_nclx* declaration = nullptr;
                ASSERT_EQ(heif_image_handle_get_nclx_color_profile(handle, &declaration).code, heif_error_Ok);
                std::unique_ptr<heif_color_profile_nclx, decltype(&heif_nclx_color_profile_free)> declaration_owner(declaration, heif_nclx_color_profile_free);
                EXPECT_EQ(declaration->transfer_characteristics, transfer);
                codec::Image raw, display;
#ifdef __APPLE__
                if (valid && std::string_view(name).ends_with(".heic") && transfer != 13) {
                    // Shared ICC policy wins, but the OS decoder cannot produce
                    // pixels for this conflict. Keep its capability error explicit.
                    EXPECT_FALSE(codec::decode_heif(bytes, raw, error));
                    EXPECT_NE(error.find("HEVC Apple system provider unsupported"), std::string::npos) << error;
                    EXPECT_FALSE(codec::decode_heif(bytes, display, error, true));
                    EXPECT_NE(error.find("ICC policy was validated"), std::string::npos) << error;
                    EXPECT_TRUE(raw.data.empty());
                    EXPECT_TRUE(display.data.empty());
                    continue;
                }
#endif
                const bool raw_supported = codec::decode_heif(bytes, raw, error);
                if (!raw_supported && !valid && std::string_view(name).ends_with(".heic")) {
                    // Invalid ICC may let the OS select an unqualified HDR layout.
                    // Raw decoding must reject it, not quantize it back to RGB8.
                    EXPECT_FALSE(error.empty());
                    EXPECT_TRUE(raw.data.empty());
                } else {
                    ASSERT_TRUE(raw_supported) << error;
                    EXPECT_EQ(raw.data, reference_raw.data);
                }
                if (valid) {
                    ASSERT_TRUE(codec::decode_heif(bytes, display, error, true)) << error;
                    EXPECT_EQ(display.data, reference_display.data);
                } else {
                    EXPECT_FALSE(codec::decode_heif(bytes, display, error, true));
                    EXPECT_NE(error.find("profile"), std::string::npos) << error;
                    EXPECT_TRUE(display.data.empty());
                }
            }
        }
        for (const auto name : {"pq.avif", "hlg.avif"}) {
            codec::Image display;
            std::string error;
            EXPECT_FALSE(codec::decode_heif(fixtures / name, display, error, true));
            EXPECT_NE(error.find("SDR"), std::string::npos);
        }
    }
    TEST(ProductionHeif, CachedDisplayTransformsPreserveProfileLayoutAndThreadIsolation) {
        const std::array names{"p3icc.avif", "icc.avif", "rgb12.avif", "rgba12.avif", "linear.avif", "bt709.avif", "rgba8.avif"};
        std::array<codec::Image, names.size()> references;
        for (size_t i = 0; i < names.size(); ++i) {
            std::string error;
            ASSERT_TRUE(codec::decode_heif(fixtures / names[i], references[i], error, true)) << error;
        }
        std::vector<std::future<bool>> reads;
        for (size_t worker = 0; worker < 4; ++worker)
            reads.push_back(std::async(std::launch::async, [&, worker] {
                for (size_t round = 0; round < 3; ++round)
                    for (size_t n = 0; n < names.size(); ++n) {
                        const size_t i = (n + worker) % names.size();
                        for (int repeat = 0; repeat < 2; ++repeat) {
                            codec::Image image;
                            std::string error;
                            if (!codec::decode_heif(fixtures / names[i], image, error, true) ||
                                image.sample_type != references[i].sample_type || image.channels != references[i].channels ||
                                image.data != references[i].data)
                                return false;
                        }
                    }
                return true;
            }));
        for (auto& read : reads)
            EXPECT_TRUE(read.get());
    }
    TEST(ProductionHeif, SdrTransfersMatchScalarReferenceAndUntaggedDefaultsToSrgb) {
        for (const auto name : {"bt709", "linear", "untagged"}) {
            SCOPED_TRACE(name);
            codec::Image raw, display;
            std::string error;
            const auto path = fixtures / (std::string(name) + ".avif");
            ASSERT_TRUE(codec::decode_heif(path, raw, error)) << error;
            ASSERT_TRUE(codec::decode_heif(path, display, error, true)) << error;
            ASSERT_EQ(raw.data.size(), display.data.size());
            for (size_t i = 0; i < raw.data.size(); ++i) {
                const double v = raw.data[i] / 255.;
                const double linear = std::string_view(name) == "linear" ? v : v < .081 ? v / 4.5
                                                                                        : std::pow((v + .099) / 1.099, 1. / .45);
                const double encoded = std::string_view(name) == "untagged" ? v : linear <= .0031308 ? 12.92 * linear
                                                                                                     : 1.055 * std::pow(linear, 1. / 2.4) - .055;
                EXPECT_NEAR(display.data[i], std::lround(encoded * 255), 1) << i;
            }
        }
    }
    TEST(ProductionHeif, PremultipliedDisplayAlphaAndConcurrentReads) {
        codec::Image raw, display;
        std::string error;
        ASSERT_TRUE(codec::decode_heif(fixtures / "premultiplied.avif", raw, error)) << error;
        ASSERT_TRUE(codec::decode_heif(fixtures / "premultiplied.avif", display, error, true)) << error;
        ASSERT_EQ(raw.channels, 4);
        ASSERT_EQ(raw.data.size(), display.data.size());
        for (size_t i = 0; i < raw.data.size(); i += 4) {
            const int alpha = raw.data[i + 3];
            EXPECT_EQ(display.data[i + 3], alpha);
            for (size_t c = 0; c < 3; ++c) {
                const int expected = alpha ? std::min(255, (int(raw.data[i + c]) * 255 + alpha / 2) / alpha) : 0;
                EXPECT_NEAR(display.data[i + c], expected, 1);
            }
        }
        std::vector<std::future<bool>> reads;
        for (int i = 0; i < 8; ++i)
            reads.push_back(std::async(std::launch::async, [] {
                codec::Image image;
                std::string diagnostic;
                return codec::decode_heif(fixtures / "p3.avif", image, diagnostic, true) && image.width == 32 && image.height == 24;
            }));
        for (auto& read : reads)
            EXPECT_TRUE(read.get());
    }
    TEST(ProductionHeif, MemoryAndDirectTargetsMatchFileWithoutLosingPrecision) {
        for (const auto name : {"rgb12", "rgba12", "p3", "yuv420", "rotated"}) {
            SCOPED_TRACE(name);
            std::ifstream file(fixtures / (std::string(name) + ".avif"), std::ios::binary);
            std::vector<uint8_t> encoded((std::istreambuf_iterator<char>(file)), {});
            codec::Image raw, memory;
            std::string error;
            ASSERT_TRUE(codec::decode(fixtures / (std::string(name) + ".avif"), raw, error)) << error;
            ASSERT_TRUE(codec::decode_memory(encoded.data(), encoded.size(), memory, error)) << error;
            EXPECT_EQ(raw.data, memory.data);
            for (const auto type : {codec::SampleType::UInt8, codec::SampleType::UInt16, codec::SampleType::Float32}) {
                codec::DecodeTarget target{raw.channels, type};
                codec::Probe info;
                ASSERT_TRUE(codec::decode_memory_to_buffer(encoded.data(), encoded.size(), target, info, error)) << error;
                std::unique_ptr<void, decltype(&std::free)> owner(target.data, std::free);
                ASSERT_EQ(info.width, raw.width);
                ASSERT_EQ(info.height, raw.height);
                ASSERT_EQ(info.channels, raw.channels);
                for (size_t i = 0; i < size_t(raw.width) * raw.height * raw.channels; ++i) {
                    uint16_t sample = raw.data[i] * 257u;
                    if (raw.sample_type == codec::SampleType::UInt16)
                        std::memcpy(&sample, raw.data.data() + i * 2, 2);
                    if (type == codec::SampleType::UInt16)
                        EXPECT_EQ(static_cast<uint16_t*>(target.data)[i], sample);
                    else if (type == codec::SampleType::UInt8)
                        EXPECT_EQ(static_cast<uint8_t*>(target.data)[i], (uint32_t(sample) + 128) / 257);
                    else
                        EXPECT_FLOAT_EQ(static_cast<float*>(target.data)[i], sample / 65535.f);
                }
            }
            codec::DecodeTarget bad{2, codec::SampleType::UInt8};
            codec::Probe info;
            EXPECT_FALSE(codec::decode_memory_to_buffer(encoded.data(), encoded.size(), bad, info, error));
            EXPECT_EQ(bad.data, nullptr);
            codec::DecodeTarget allocation{3, codec::SampleType::UInt8, nullptr, [](size_t, void*) -> void* { return nullptr; }, nullptr};
            EXPECT_FALSE(codec::decode_memory_to_buffer(encoded.data(), encoded.size(), allocation, info, error));
            EXPECT_EQ(allocation.data, nullptr);
        }
    }
    TEST(ProductionHeif, ExtensionAliasesUnicodeAndTruncation) {
        const auto root = std::filesystem::temp_directory_path() / ("lfs-heif-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directory(root);
        struct Cleanup {
            std::filesystem::path path;
            ~Cleanup() {
                std::error_code e;
                std::filesystem::remove_all(path, e);
            }
        } cleanup{root};
        const auto path = root / std::filesystem::path(u8"é-日本語.HEIF");
        std::filesystem::copy_file(fixtures / "rgb8.avif", path);
        codec::Image image;
        std::string error;
        ASSERT_TRUE(codec::decode(path, image, error)) << error;
        const auto renamed = root / "renamed.avif";
        std::filesystem::copy_file(fixtures / "rgb8.heic", renamed);
        codec::Image original;
        const bool supported = codec::decode(fixtures / "rgb8.heic", original, error);
        EXPECT_EQ(codec::decode(renamed, image, error), supported);
        if (supported)
            EXPECT_EQ(image.data, original.data);
        else
            EXPECT_NE(error.find("HEVC"), std::string::npos) << error;
        const auto native_alias = root / std::filesystem::path(u8"é-日本語.heic");
        std::filesystem::copy_file(fixtures / "system-p3icc.heic", native_alias);
        const bool native_supported = codec::decode(fixtures / "system-p3icc.heic", original, error);
        EXPECT_EQ(codec::decode(native_alias, image, error), native_supported);
        if (native_supported)
            EXPECT_EQ(image.data, original.data);
        std::filesystem::resize_file(path, 32);
        codec::Probe info;
        EXPECT_FALSE(codec::probe(path, info, error));
        EXPECT_FALSE(error.empty());
        EXPECT_FALSE(codec::decode(path, image, error));
        EXPECT_FALSE(error.empty());
        EXPECT_TRUE(image.data.empty());
    }
} // namespace
