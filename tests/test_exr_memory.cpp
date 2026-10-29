/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/image_codecs.hpp"
#include "core/image_exr.hpp"
#include "core/source_metadata.hpp"
#include <Imath/half.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfChromaticitiesAttribute.h>
#include <OpenEXR/ImfFrameBuffer.h>
#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfIO.h>
#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImfStringAttribute.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

namespace {
    namespace codec = lfs::core::image_codecs;
    namespace exr = OPENEXR_IMF_NAMESPACE;
    class Input : public exr::IStream {
        std::span<const uint8_t> bytes_;
        uint64_t position_ = 0;

    public:
        explicit Input(std::span<const uint8_t> bytes) : exr::IStream("reference"), bytes_(bytes) {}
        bool read(char* data, int size) override {
            if (size < 0 || position_ > bytes_.size() || uint64_t(size) > bytes_.size() - position_)
                throw std::runtime_error("reference read outside input");
            std::memcpy(data, bytes_.data() + position_, size);
            position_ += size;
            return position_ < bytes_.size();
        }
        uint64_t tellg() override { return position_; }
        void seekg(uint64_t offset) override { position_ = offset; }
    };

    struct Pixels {
        int width = 19, height = 35, channels = 4;
        size_t stride;
        std::vector<uint8_t> bytes;
        Pixels(int c = 4, size_t padding = 0, size_t offset = 0)
            : channels(c), stride(size_t(width) * c * sizeof(float) + padding), bytes(offset + stride * (height - 1) + size_t(width) * c * sizeof(float), 0xcc), offset_(offset) {
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x)
                    for (int c = 0; c < channels; ++c) {
                        const float value = c == 3 ? float(x % 5) / 4.f : float((x + y * 7 + c) % 61) / 8.f - 2.f;
                        std::memcpy(bytes.data() + offset_ + y * stride + (x * channels + c) * sizeof(float), &value, sizeof(float));
                    }
        }
        codec::ExrEncodeView view() const { return {width, height, channels, stride, std::span(bytes).subspan(offset_)}; }
        size_t offset_;
    };

    std::vector<float> reference(std::span<const uint8_t> bytes, int width, int height) {
        Input stream(bytes);
        exr::InputFile file(stream);
        std::vector<float> pixels(size_t(width) * height * 4);
        exr::FrameBuffer frame;
        const std::array<const char*, 4> names{"R", "G", "B", "A"};
        for (int c = 0; c < 4; ++c)
            frame.insert(names[c], exr::Slice(exr::FLOAT, reinterpret_cast<char*>(pixels.data() + c),
                                              4 * sizeof(float), width * 4 * sizeof(float), 1, 1, c == 3 ? 1.0 : 0.0));
        file.setFrameBuffer(frame);
        file.readPixels(0, height - 1);
        return pixels;
    }

    TEST(ExrMemory, ExplicitXmpPreservesPixelsAndKeepsGenericTextBudgets) {
        const std::string packet =
            "<?xpacket begin=\"\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>"
            "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"><rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">"
            "<rdf:Description rdf:about=\"\" xmlns:t=\"urn:lfs:test\"><t:value>escaped &amp; retained</t:value>"
            "</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end=\"r\"?>";
        Pixels input;
        const auto original = input.bytes;
        std::vector<uint8_t> encoded;
        const codec::ExrWriteStream stream{"XMP test", &encoded,
                                           [](void* user, const void* bytes, uint64_t size, uint64_t offset) noexcept -> int64_t {
                                               auto& out = *static_cast<std::vector<uint8_t>*>(user);
                                               if (offset > SIZE_MAX || size > SIZE_MAX - offset)
                                                   return -1;
                                               try {
                                                   if (out.size() < offset + size)
                                                       out.resize(size_t(offset + size));
                                                   std::memcpy(out.data() + size_t(offset), bytes, size_t(size));
                                                   return int64_t(size);
                                               } catch (...) { return -1; }
                                           }};
        for (const auto precision : {codec::ExrSampleType::Half, codec::ExrSampleType::Float})
            for (const auto compression : {codec::ExrCompression::None, codec::ExrCompression::Zip})
                for (const bool premultiply : {false, true}) {
                    codec::ExrEncodeOptions options;
                    options.precision = precision;
                    options.compression = compression;
                    options.premultiply_alpha = premultiply;
                    std::vector<uint8_t> plain;
                    ASSERT_TRUE(codec::encode_exr_memory(input.view(), plain, options));
                    encoded.clear();
                    ASSERT_TRUE(codec::encode_exr(input.view(), stream, options, packet));
                    EXPECT_EQ(reference(encoded, input.width, input.height), reference(plain, input.width, input.height));
                    const auto decoded = reference(encoded, input.width, input.height);
                    for (int y = 0; y < input.height; ++y)
                        for (int x = 0; x < input.width; ++x) {
                            std::array<float, 4> values;
                            std::memcpy(values.data(), input.bytes.data() + y * input.stride + x * 4 * sizeof(float), sizeof(values));
                            for (int c = 0; c < 4; ++c) {
                                float expected = premultiply && c < 3 ? values[c] * values[3] : values[c];
                                if (precision == codec::ExrSampleType::Half)
                                    expected = float(Imath::half(expected));
                                EXPECT_EQ(decoded[(y * input.width + x) * 4 + c], expected);
                            }
                        }
                    Input reader(encoded);
                    exr::InputFile file(reader);
                    EXPECT_EQ(file.header().typedAttribute<exr::StringAttribute>("xmp").value(), packet);
                    EXPECT_EQ(input.bytes, original);
                }
        codec::ExrEncodeOptions options;
        const auto previous = encoded;
        for (const auto& bad : {std::string("x\0mp", 4), std::string(32 * 1024 * 1024 + 1, 'a')}) {
            const auto refused = codec::encode_exr(input.view(), stream, options, bad);
            ASSERT_FALSE(refused);
            EXPECT_EQ(refused.error().code(), lfs::ErrorCode::InvalidArgument);
            EXPECT_EQ(encoded, previous);
        }
        const std::array<std::pair<std::string, std::string>, 1> duplicate{{{"xmp", "legacy"}}};
        options.text = duplicate;
        EXPECT_FALSE(codec::encode_exr(input.view(), stream, options, packet));
        EXPECT_EQ(encoded, previous);
        encoded.clear();
        ASSERT_TRUE(codec::encode_exr(input.view(), stream, options));
        const std::array<std::pair<std::string, std::string>, 1> too_large{{{"comments", std::string(1024 * 1024 + 4097, 'a')}}};
        options.text = too_large;
        const auto legacy = encoded;
        EXPECT_FALSE(codec::encode_exr(input.view(), stream, options, packet));
        EXPECT_EQ(encoded, legacy);
        options.text = {};
        auto boundary = packet;
        boundary.insert(boundary.find("escaped"), 32 * 1024 * 1024 - packet.size(), 'a');
        encoded.clear();
        ASSERT_TRUE(codec::encode_exr(input.view(), stream, options, boundary));
        Input reader(encoded);
        exr::InputFile file(reader);
        EXPECT_EQ(file.header().typedAttribute<exr::StringAttribute>("xmp").value(), boundary);
        std::vector<uint8_t> plain;
        ASSERT_TRUE(codec::encode_exr_memory(input.view(), plain, options));
        EXPECT_EQ(reference(encoded, input.width, input.height), reference(plain, input.width, input.height));
        EXPECT_EQ(input.bytes, original);
    }

    TEST(ExrMemory, MetadataHeaderCollectionKeepsOpaqueBytesAndBoundsMalformedInput) {
        struct Temporary {
            std::filesystem::path path = std::filesystem::temp_directory_path() / ("exr-metadata-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".exr");
            ~Temporary() {
                std::error_code error;
                std::filesystem::remove(path, error);
            }
        } temporary;
        // Synthetic metadata headers, not decodable images: collection ends at
        // the header terminator and never attempts to interpret pixel chunks.
        const std::vector<uint8_t> prefix{0x76, 0x2f, 0x31, 0x01, 2, 0, 0, 0};
        const std::string opaque("<raw>\0\xff</raw>", 13);
        const auto attr = [](std::vector<uint8_t>& data, std::string_view key, std::string_view type, std::string_view value) {
            for (const auto text : {key, type}) {
                data.insert(data.end(), text.begin(), text.end());
                data.push_back(0);
            }
            for (int i = 0; i < 4; ++i)
                data.push_back(uint8_t(value.size() >> (8 * i)));
            data.insert(data.end(), value.begin(), value.end());
        };
        const auto collect = [&](std::vector<uint8_t> data, codec::XmpLimits limits = {}) {
            data.push_back(0);
            {
                std::ofstream f(temporary.path, std::ios::binary);
                f.write(reinterpret_cast<const char*>(data.data()), data.size());
            }
            return codec::read_source_metadata(temporary.path, {}, limits);
        };
        auto data = prefix;
        attr(data, "xmp", "string", opaque);
        auto result = collect(data);
        ASSERT_EQ(result.xmp.status, codec::MetadataStatus::Collected);
        ASSERT_EQ(result.xmp.blocks.size(), 1);
        EXPECT_EQ(result.xmp.scope, "exr-single-header-string-xmp");
        EXPECT_EQ(result.xmp.blocks[0].envelope, codec::XmpEnvelope::ExrString);
        EXPECT_EQ(result.xmp.blocks[0].bytes, (std::vector<uint8_t>(opaque.begin(), opaque.end())));
        EXPECT_EQ(result.exif.status, codec::MetadataStatus::Unsupported);
        codec::XmpLimits limits;
        limits.block_bytes = opaque.size() - 1;
        EXPECT_EQ(collect(data, limits).xmp.status, codec::MetadataStatus::Limited);
        limits.block_bytes = opaque.size();
        EXPECT_EQ(collect(data, limits).xmp.status, codec::MetadataStatus::Collected);
        attr(data, "xmp", "string", "second");
        result = collect(data);
        EXPECT_EQ(result.xmp.status, codec::MetadataStatus::Malformed);
        EXPECT_EQ(result.xmp.blocks.size(), 2);
        limits.blocks = 1;
        EXPECT_EQ(collect(data, limits).xmp.status, codec::MetadataStatus::Limited);
        for (size_t size = 4; size < prefix.size(); ++size)
            EXPECT_EQ(collect({prefix.begin(), prefix.begin() + size}).xmp.status, codec::MetadataStatus::Malformed);
        data = prefix;
        data[5] = 0x10; // Multipart is an explicit unsupported scope.
        EXPECT_EQ(collect(data).xmp.status, codec::MetadataStatus::Unsupported);
        data = prefix;
        attr(data, "xmp", "int", "1234");
        EXPECT_EQ(collect(data).xmp.status, codec::MetadataStatus::Unsupported);
        data = prefix;
        attr(data, std::string(32, 'n'), "string", "");
        EXPECT_EQ(collect(data).xmp.status, codec::MetadataStatus::Malformed);
        data = prefix;
        data[5] = 4; // Long names.
        attr(data, std::string(255, 'n'), "string", "");
        EXPECT_EQ(collect(data).xmp.status, codec::MetadataStatus::Absent);
        data = prefix;
        attr(data, "xmp", "string", "payload");
        data.resize(data.size() - 3);
        EXPECT_EQ(collect(data).xmp.status, codec::MetadataStatus::Malformed);
        data = prefix;
        attr(data, "xmp", "string", "");
        data[data.size() - 1] = 0x80; // Negative signed attribute extent.
        EXPECT_EQ(collect(data).xmp.status, codec::MetadataStatus::Malformed);
        data = prefix;
        attr(data, "xmp", "string", "value");
        for (size_t i = 1; i < 4096; ++i)
            attr(data, "field" + std::to_string(i), "string", "");
        EXPECT_EQ(collect(data).xmp.status, codec::MetadataStatus::Collected);
        attr(data, "extra", "string", "");
        EXPECT_EQ(collect(data).xmp.status, codec::MetadataStatus::Limited);
        data = prefix;
        attr(data, "xmp", "string", opaque);
        attr(data, "xmp", "int", "1234");
        limits.block_bytes = opaque.size() - 1;
        EXPECT_EQ(collect(data, limits).xmp.status, codec::MetadataStatus::Limited);
        data = prefix;
        attr(data, "padding", "string", "");
        data.back() = 4; // Declared 64 MiB exceeds the header budget before allocation.
        EXPECT_EQ(collect(data).xmp.status, codec::MetadataStatus::Limited);
        for (const uint8_t flag : {uint8_t(1), uint8_t(8), uint8_t(16)}) {
            data = prefix;
            data[5] = flag;
            EXPECT_EQ(collect(data).xmp.status, codec::MetadataStatus::Unsupported);
        }
        data = prefix;
        data[5] = 2; // Single-header tiled scope needs no pixel decoding.
        attr(data, "xmp", "string", opaque);
        EXPECT_EQ(collect(data).xmp.status, codec::MetadataStatus::Collected);
        data = prefix;
        attr(data, std::string(31, 'n'), "string", "");
        EXPECT_EQ(collect(data).xmp.status, codec::MetadataStatus::Absent);
        data = prefix;
        data[5] = 4;
        attr(data, std::string(256, 'n'), "string", "");
        EXPECT_EQ(collect(data).xmp.status, codec::MetadataStatus::Malformed);
    }

    TEST(ExrMemory, SharedEncoderMatchesIndependentReaderAndStreamBytes) {
        for (int channels : {3, 4})
            for (auto precision : {codec::ExrSampleType::Half, codec::ExrSampleType::Float})
                for (auto compression : {codec::ExrCompression::None, codec::ExrCompression::Zip})
                    for (size_t offset : {size_t(0), size_t(1)}) {
                        Pixels input(channels, offset ? 3 : 0, offset);
                        const auto original = input.bytes;
                        codec::ExrEncodeOptions options;
                        options.precision = precision;
                        options.compression = compression;
                        const std::array<std::pair<std::string, std::string>, 1> text{{{"comments", "memory test"}}};
                        options.text = text;
                        options.chromaticities = {.64f, .33f, .3f, .6f, .15f, .06f, .3127f, .329f};
                        std::vector<uint8_t> encoded;
                        ASSERT_TRUE(codec::encode_exr_memory(input.view(), encoded, options));
                        EXPECT_EQ(input.bytes, original);
                        const auto actual = reference(encoded, input.width, input.height);
                        for (int y = 0; y < input.height; ++y)
                            for (int x = 0; x < input.width; ++x)
                                for (int c = 0; c < 4; ++c) {
                                    float expected = 1.f;
                                    if (c < channels)
                                        std::memcpy(&expected, input.view().pixels.data() + y * input.stride + (x * channels + c) * sizeof(float), sizeof(float));
                                    if (precision == codec::ExrSampleType::Half)
                                        expected = float(Imath::half(expected));
                                    ASSERT_EQ(actual[(y * input.width + x) * 4 + c], expected);
                                }
                        Input stream(encoded);
                        exr::InputFile header(stream);
                        EXPECT_EQ(header.header().typedAttribute<exr::StringAttribute>("comments").value(), "memory test");
                        const auto& chroma = header.header().typedAttribute<exr::ChromaticitiesAttribute>("chromaticities").value();
                        EXPECT_EQ(chroma.red.x, options.chromaticities[0]);
                        EXPECT_EQ(chroma.red.y, options.chromaticities[1]);
                        EXPECT_EQ(chroma.green.x, options.chromaticities[2]);
                        EXPECT_EQ(chroma.green.y, options.chromaticities[3]);
                        EXPECT_EQ(chroma.blue.x, options.chromaticities[4]);
                        EXPECT_EQ(chroma.blue.y, options.chromaticities[5]);
                        EXPECT_EQ(chroma.white.x, options.chromaticities[6]);
                        EXPECT_EQ(chroma.white.y, options.chromaticities[7]);
                        EXPECT_EQ(header.header().channels().findChannel("R")->type,
                                  precision == codec::ExrSampleType::Half ? exr::HALF : exr::FLOAT);
                        EXPECT_EQ(header.header().compression(), compression == codec::ExrCompression::Zip ? exr::ZIP_COMPRESSION : exr::NO_COMPRESSION);

                        std::vector<uint8_t> file_stream;
                        const codec::ExrWriteStream destination{"reference stream", &file_stream,
                                                                [](void* context, const void* data, uint64_t size, uint64_t offset) noexcept -> int64_t {
                                                                    try {
                                                                        auto& bytes = *static_cast<std::vector<uint8_t>*>(context);
                                                                        bytes.resize(std::max<size_t>(bytes.size(), offset + size));
                                                                        std::memcpy(bytes.data() + offset, data, size);
                                                                        return int64_t(size);
                                                                    } catch (...) {
                                                                        return -1;
                                                                    }
                                                                }};
                        ASSERT_TRUE(codec::encode_exr(input.view(), destination, options));
                        EXPECT_EQ(encoded, file_stream);
                        codec::Image decoded;
                        std::string error = "stale";
                        ASSERT_TRUE(codec::decode_memory(encoded.data(), encoded.size(), decoded, error)) << error;
                        EXPECT_TRUE(error.empty());
                        ASSERT_EQ(decoded.data.size(), actual.size() * sizeof(float));
                        EXPECT_EQ(std::memcmp(decoded.data.data(), actual.data(), decoded.data.size()), 0);
                    }
    }

    TEST(ExrMemory, ExtentLimitCancellationAndAliasingPreserveOutput) {
        Pixels input;
        std::vector<uint8_t> encoded;
        ASSERT_TRUE(codec::encode_exr_memory(input.view(), encoded));
        const auto size = encoded.size();
        const std::vector<uint8_t> sentinel{3, 2, 1};
        for (size_t limit : {size_t(1), size_t(256), size - 1}) {
            auto output = sentinel;
            const auto status = codec::encode_exr_memory(input.view(), output, {}, limit);
            ASSERT_FALSE(status);
            EXPECT_EQ(status.error().code(), lfs::ErrorCode::ResourceExhausted);
            EXPECT_EQ(output, sentinel);
            ASSERT_TRUE(codec::encode_exr_memory(input.view(), output, {}, size));
            EXPECT_EQ(output, encoded);
        }
        auto output = sentinel;
        codec::ExrEncodeOptions options;
        // Before the context and before each of this fixture's three ZIP chunks.
        for (unsigned checkpoint = 1; checkpoint <= 4; ++checkpoint) {
            unsigned calls = 0;
            options.cancelled = [&] { return ++calls >= checkpoint; };
            const auto cancelled = codec::encode_exr_memory(input.view(), output, options);
            ASSERT_FALSE(cancelled);
            EXPECT_EQ(cancelled.error().code(), lfs::ErrorCode::Cancelled);
            EXPECT_EQ(output, sentinel);
            EXPECT_EQ(calls, checkpoint);
            options.cancelled = {};
        }
        auto view = input.view();
        ASSERT_TRUE(codec::encode_exr_memory(view, input.bytes));
        EXPECT_EQ(input.bytes, encoded);
        codec::Image image;
        image.data = encoded;
        std::string error;
        ASSERT_TRUE(codec::decode_memory(image.data.data(), image.data.size(), image, error));
        EXPECT_EQ(image.width, input.width);
    }

    TEST(ExrMemory, ConsumerDecodedByteLimitPreservesPreviousOutputAndRecovers) {
        Pixels input;
        std::vector<uint8_t> encoded;
        ASSERT_TRUE(codec::encode_exr_memory(input.view(), encoded));
        const size_t required = size_t(input.width) * input.height * 4 * sizeof(float);
        for (size_t limit : {size_t(0), required - 1}) {
            codec::Image output{7, 8, 3, codec::SampleType::UInt8, {1, 2, 3}};
            std::string error;
            EXPECT_FALSE(codec::decode_exr_memory(encoded, output, error, 1, limit));
            EXPECT_TRUE(error.starts_with("EXR: decoded image exceeds byte limit"));
            EXPECT_NE(error.find("bytes=" + std::to_string(required)), std::string::npos);
            EXPECT_NE(error.find("limit=" + std::to_string(limit)), std::string::npos);
            EXPECT_EQ(output.width, 7);
            EXPECT_EQ(output.data, (std::vector<uint8_t>{1, 2, 3}));
            ASSERT_TRUE(codec::decode_exr_memory(encoded, output, error, 1, required)) << error;
            EXPECT_TRUE(error.empty());
            EXPECT_EQ(output.data.size(), required);
        }
        codec::Image original_api, bounded;
        std::string error;
        ASSERT_TRUE(codec::decode_exr_memory(encoded, original_api, error));
        ASSERT_TRUE(codec::decode_exr_memory(encoded, bounded, error, 4, required));
        EXPECT_EQ(bounded.data, original_api.data);
        encoded.pop_back(); // Header is intact; the last pixel payload is not.
        EXPECT_FALSE(codec::decode_exr_memory(encoded, bounded, error, 1, 0));
        EXPECT_TRUE(error.starts_with("EXR: decoded image exceeds byte limit"));
        EXPECT_EQ(bounded.data, original_api.data);
    }

    TEST(ExrMemory, InvalidInputOptionsAndSamplesKeepPreviousOutput) {
        Pixels input;
        const std::vector<uint8_t> sentinel{1, 2, 3};
        auto check = [&](codec::ExrEncodeView view, codec::ExrEncodeOptions options = {}, size_t limit = 1024 * 1024) {
            auto output = sentinel;
            const auto status = codec::encode_exr_memory(view, output, options, limit);
            EXPECT_FALSE(status);
            if (!status)
                EXPECT_EQ(status.error().code(), lfs::ErrorCode::InvalidArgument);
            EXPECT_EQ(output, sentinel);
        };
        auto view = input.view();
        view.width = 0;
        check(view);
        view = input.view();
        view.row_stride--;
        check(view);
        view = input.view();
        view.pixels = view.pixels.first(view.pixels.size() - 1);
        check(view);
        check(input.view(), {}, 0);
        check(input.view(), {}, std::numeric_limits<size_t>::max());
        codec::ExrEncodeOptions options;
        options.precision = static_cast<codec::ExrSampleType>(7);
        check(input.view(), options);
        options = {};
        options.compression = static_cast<codec::ExrCompression>(7);
        check(input.view(), options);
        for (float value : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), 65505.f}) {
            std::memcpy(input.bytes.data(), &value, sizeof(float));
            check(input.view());
        }
    }

    TEST(ExrMemory, MalformedReadsAndDirectTargetsFailWithoutPartialResults) {
        Pixels input;
        std::vector<uint8_t> encoded;
        ASSERT_TRUE(codec::encode_exr_memory(input.view(), encoded));
        for (size_t size : {size_t(4), size_t(8), size_t(32), encoded.size() / 2, encoded.size() - 1}) {
            codec::Image result{7, 8, 3, codec::SampleType::UInt8, {1, 2, 3}};
            std::string error;
            EXPECT_FALSE(codec::decode_memory(encoded.data(), size, result, error));
            EXPECT_EQ(result.data, (std::vector<uint8_t>{1, 2, 3}));
            EXPECT_EQ(result.width, 7);
            EXPECT_FALSE(error.empty());
        }
        for (int channels : {3, 4}) {
            std::vector<uint8_t> bytes(size_t(input.width) * input.height * channels * sizeof(float) + 2, 0x5a);
            codec::DecodeTarget target{channels, codec::SampleType::Float32, bytes.data() + 1};
            codec::Probe info;
            std::string error;
            ASSERT_TRUE(codec::decode_memory_to_buffer(encoded.data(), encoded.size(), target, info, error)) << error;
            EXPECT_EQ(info.channels, channels);
            EXPECT_EQ(bytes.front(), 0x5a);
            EXPECT_EQ(bytes.back(), 0x5a);
            const auto expected = reference(encoded, input.width, input.height);
            for (size_t i = 0; i < expected.size() / 4; ++i)
                EXPECT_EQ(std::memcmp(bytes.data() + 1 + i * channels * sizeof(float), expected.data() + i * 4, channels * sizeof(float)), 0);
            const auto original = bytes;
            EXPECT_FALSE(codec::decode_memory_to_buffer(encoded.data(), encoded.size() - 1, target, info, error));
            EXPECT_EQ(bytes, original);
            target.sample_type = codec::SampleType::UInt8;
            EXPECT_FALSE(codec::decode_memory_to_buffer(encoded.data(), encoded.size(), target, info, error));
            EXPECT_EQ(bytes, original);
        }
        unsigned allocations = 0;
        codec::DecodeTarget target{4, codec::SampleType::Float32, nullptr,
                                   [](size_t, void* context) -> void* { ++*static_cast<unsigned*>(context); return nullptr; }, &allocations};
        codec::Probe info{3, 4, 1, codec::SampleType::UInt8};
        std::string error;
        EXPECT_FALSE(codec::decode_memory_to_buffer(encoded.data(), encoded.size(), target, info, error));
        EXPECT_EQ(allocations, 1u);
        EXPECT_EQ(info.width, 3);
        struct Allocation {
            std::vector<uint8_t> bytes;
            unsigned calls = 0;
        } storage;
        target.channels = 0;
        target.user = &storage;
        target.allocate = [](size_t size, void* user) -> void* {
            auto& state = *static_cast<Allocation*>(user);
            ++state.calls;
            state.bytes.resize(size + 2, 0x5a);
            return state.bytes.data() + 1;
        };
        EXPECT_FALSE(codec::decode_memory_to_buffer(encoded.data(), encoded.size() - 1, target, info, error));
        EXPECT_EQ(storage.calls, 0u);
        ASSERT_TRUE(codec::decode_memory_to_buffer(encoded.data(), encoded.size(), target, info, error));
        EXPECT_EQ(storage.calls, 1u);
        EXPECT_EQ(target.data, storage.bytes.data() + 1);
        EXPECT_EQ(info.channels, 4);
        EXPECT_EQ(storage.bytes.front(), 0x5a);
        EXPECT_EQ(storage.bytes.back(), 0x5a);
        const auto expected = reference(encoded, input.width, input.height);
        EXPECT_EQ(std::memcmp(storage.bytes.data() + 1, expected.data(), expected.size() * sizeof(float)), 0);
        codec::DecodeTarget automatic{4, codec::SampleType::Float32};
        ASSERT_TRUE(codec::decode_memory_to_buffer(encoded.data(), encoded.size(), automatic, info, error));
        EXPECT_NE(automatic.data, nullptr);
        EXPECT_EQ(std::memcmp(automatic.data, expected.data(), expected.size() * sizeof(float)), 0);
        std::free(automatic.data);
    }

    TEST(ExrMemory, PremultiplicationFloatRangeAndMetadataFollowExistingPolicy) {
        Pixels input;
        auto original = input.bytes;
        codec::ExrEncodeOptions options;
        options.precision = codec::ExrSampleType::Float;
        options.premultiply_alpha = true;
        float large = 1e20f;
        std::memcpy(input.bytes.data(), &large, sizeof(float));
        original = input.bytes;
        std::vector<uint8_t> output;
        ASSERT_TRUE(codec::encode_exr_memory(input.view(), output, options));
        EXPECT_EQ(input.bytes, original);
        const auto decoded = reference(output, input.width, input.height);
        for (int y = 0; y < input.height; ++y)
            for (int x = 0; x < input.width; ++x) {
                std::array<float, 4> values;
                std::memcpy(values.data(), input.bytes.data() + y * input.stride + x * 4 * sizeof(float), sizeof(values));
                for (int c = 0; c < 4; ++c)
                    EXPECT_EQ(decoded[(y * input.width + x) * 4 + c], c < 3 ? values[c] * values[3] : values[c]);
            }
        const auto preserved = output;
        const std::array<std::pair<std::string, std::string>, 2> duplicate{{{"comments", "a"}, {"comments", "b"}}};
        options.text = duplicate;
        EXPECT_FALSE(codec::encode_exr_memory(input.view(), output, options));
        EXPECT_EQ(output, preserved);
        const std::array<std::pair<std::string, std::string>, 1> reserved{{{"channels", "invalid"}}};
        options.text = reserved;
        EXPECT_FALSE(codec::encode_exr_memory(input.view(), output, options));
        EXPECT_EQ(output, preserved);
        options.text = {};
        options.chromaticities[0] = std::numeric_limits<float>::quiet_NaN();
        EXPECT_FALSE(codec::encode_exr_memory(input.view(), output, options));
        EXPECT_EQ(output, preserved);
    }

    TEST(ExrMemory, FloatPreservesHdrSubnormalsAndSignedZeroWithoutQuantization) {
        const std::array<float, 9> values{std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
                                          std::numeric_limits<float>::min(), std::numeric_limits<float>::denorm_min(), -0.f,
                                          std::nextafter(1.f, 2.f), 1e-20f, 1e20f, -1e20f};
        const codec::ExrEncodeView view{3, 1, 3, sizeof(values), {reinterpret_cast<const uint8_t*>(values.data()), sizeof(values)}};
        for (auto compression : {codec::ExrCompression::None, codec::ExrCompression::Zip}) {
            codec::ExrEncodeOptions options;
            options.precision = codec::ExrSampleType::Float;
            options.compression = compression;
            std::vector<uint8_t> encoded;
            ASSERT_TRUE(codec::encode_exr_memory(view, encoded, options));
            const auto expected = reference(encoded, 3, 1);
            codec::Image decoded;
            std::string error;
            ASSERT_TRUE(codec::decode_exr_memory(encoded, decoded, error));
            EXPECT_EQ(std::memcmp(expected.data(), decoded.data.data(), decoded.data.size()), 0);
            for (size_t pixel = 0; pixel < 3; ++pixel)
                EXPECT_EQ(std::memcmp(values.data() + pixel * 3, expected.data() + pixel * 4, 3 * sizeof(float)), 0);
        }
    }

    TEST(ExrMemory, LargeImagesPreserveSamplesAndRecoverFromLimits) {
        for (const auto dimensions : {std::pair{3840, 2160}, std::pair{7680, 4320}}) {
            const auto [width, height] = dimensions;
            const std::vector<float> pixels(size_t(width) * height * 3, .5f);
            const codec::ExrEncodeView view{width, height, 3, size_t(width) * 3 * sizeof(float), {reinterpret_cast<const uint8_t*>(pixels.data()), pixels.size() * sizeof(float)}};
            std::vector<uint8_t> output{1, 2, 3};
            const auto failure = codec::encode_exr_memory(view, output, {}, 64);
            ASSERT_FALSE(failure);
            EXPECT_EQ(failure.error().code(), lfs::ErrorCode::ResourceExhausted);
            EXPECT_EQ(output, (std::vector<uint8_t>{1, 2, 3}));
            ASSERT_TRUE(codec::encode_exr_memory(view, output));
            codec::Image decoded;
            std::string error;
            ASSERT_TRUE(codec::decode_exr_memory(output, decoded, error)) << error;
            ASSERT_EQ(decoded.data.size(), size_t(width) * height * 4 * sizeof(float));
            const auto* actual = reinterpret_cast<const float*>(decoded.data.data());
            for (size_t i = 0; i < size_t(width) * height; ++i) {
                if (actual[i * 4] != .5f || actual[i * 4 + 1] != .5f || actual[i * 4 + 2] != .5f || actual[i * 4 + 3] != 1.f) {
                    FAIL() << "Unexpected EXR sample at pixel " << i;
                }
            }
        }
    }

    TEST(ExrMemory, NestedCancellationCallbacksKeepDiagnosticsIndependent) {
        Pixels input;
        std::vector<uint8_t> expected;
        ASSERT_TRUE(codec::encode_exr_memory(input.view(), expected));
        bool nested = false;
        codec::ExrEncodeOptions options;
        options.cancelled = [&] {
            if (!nested) {
                nested = true;
                std::vector<uint8_t> rejected{7};
                EXPECT_FALSE(codec::encode_exr_memory(input.view(), rejected, {}, 1));
                EXPECT_EQ(rejected, (std::vector<uint8_t>{7}));
                codec::Image image;
                std::string error = "stale";
                EXPECT_TRUE(codec::decode_exr_memory(expected, image, error));
                EXPECT_TRUE(error.empty());
            }
            return false;
        };
        std::vector<uint8_t> output;
        ASSERT_TRUE(codec::encode_exr_memory(input.view(), output, options));
        EXPECT_TRUE(nested);
        EXPECT_EQ(output, expected);
    }

    TEST(ExrMemory, ConcurrentEncodesAndParallelReadsRemainIndependent) {
        Pixels input;
        std::vector<uint8_t> encoded;
        ASSERT_TRUE(codec::encode_exr_memory(input.view(), encoded));
        std::vector<std::future<void>> tasks;
        for (int i = 0; i < 8; ++i)
            tasks.emplace_back(std::async(std::launch::async, [&] {
                std::vector<uint8_t> bytes;
                ASSERT_TRUE(codec::encode_exr_memory(input.view(), bytes));
                EXPECT_EQ(bytes, encoded);
                codec::Image a, b;
                std::string error;
                ASSERT_TRUE(codec::decode_exr_memory(bytes, a, error, 1)) << error;
                ASSERT_TRUE(codec::decode_exr_memory(bytes, b, error, 4)) << error;
                EXPECT_EQ(a.data, b.data);
            }));
        for (auto& task : tasks)
            task.get();
    }
} // namespace
