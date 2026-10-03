/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/image_codecs.hpp"
#include <Imath/half.h>
#include <OpenEXR/ImfChannelList.h>
#include <OpenEXR/ImfDeepScanLineOutputFile.h>
#include <OpenEXR/ImfFrameBuffer.h>
#include <OpenEXR/ImfHeader.h>
#include <OpenEXR/ImfIO.h>
#include <OpenEXR/ImfInputFile.h>
#include <OpenEXR/ImfMultiPartOutputFile.h>
#include <OpenEXR/ImfOutputFile.h>
#include <OpenEXR/ImfOutputPart.h>
#include <OpenEXR/ImfPartType.h>
#include <OpenEXR/ImfTiledInputFile.h>
#include <OpenEXR/ImfTiledOutputFile.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <gtest/gtest.h>
#include <iterator>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {
    namespace exr = OPENEXR_IMF_NAMESPACE;
    namespace codec = lfs::core::image_codecs;

    // Use filesystem-aware streams so fixture creation and the reference reader
    // support the same Unicode paths as the production OpenEXRCore reader.
    class OutputStream : public exr::OStream {
        std::ofstream file_;

    public:
        explicit OutputStream(const std::filesystem::path& p) : exr::OStream("fixture"), file_(p, std::ios::binary) {}
        void write(const char c[], int n) override {
            if (!file_.write(c, n))
                throw std::runtime_error("fixture write failed");
        }
        uint64_t tellp() override { return static_cast<uint64_t>(file_.tellp()); }
        void seekp(uint64_t p) override { file_.seekp(static_cast<std::streamoff>(p)); }
    };
    class InputStream : public exr::IStream {
        std::ifstream file_;

    public:
        explicit InputStream(const std::filesystem::path& p) : exr::IStream("fixture"), file_(p, std::ios::binary) {}
        bool read(char c[], int n) override {
            if (!file_.read(c, n))
                throw std::runtime_error("fixture read failed");
            return file_.rdbuf()->sgetc() != std::char_traits<char>::eof();
        }
        uint64_t tellg() override { return static_cast<uint64_t>(file_.tellg()); }
        void seekg(uint64_t p) override {
            file_.clear();
            file_.seekg(static_cast<std::streamoff>(p));
        }
    };

    struct Fixture {
        static constexpr int width = 19, height = 291;
        std::filesystem::path path;
        Fixture() {
            static std::atomic_uint64_t sequence{0};
            path = std::filesystem::temp_directory_path() /
                   ("lfs_exr_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "_" +
                    std::to_string(sequence.fetch_add(1)) + ".exr");
        }
        ~Fixture() {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }

        void write(exr::Compression compression, bool tiled, exr::PixelType type,
                   bool grayscale = false, bool alpha = true,
                   Imath::V2i origin = {0, 0}, exr::LineOrder order = exr::INCREASING_Y,
                   exr::LevelMode levels = exr::ONE_LEVEL) {
            exr::Header header(width, height);
            header.dataWindow() = {origin, origin + Imath::V2i(width - 1, height - 1)};
            header.compression() = compression;
            header.lineOrder() = order;
            const std::vector<std::string> names = grayscale ? std::vector<std::string>{"Y"}
                                                   : alpha   ? std::vector<std::string>{"R", "G", "B", "A", "Z"}
                                                             : std::vector<std::string>{"R", "G", "B", "Z"};
            std::vector<std::vector<float>> floats(names.size(), std::vector<float>(width * height));
            std::vector<std::vector<Imath::half>> halves(names.size(), std::vector<Imath::half>(width * height));
            std::vector<std::vector<uint32_t>> ints(names.size(), std::vector<uint32_t>(width * height));
            exr::FrameBuffer fb;
            for (size_t c = 0; c < names.size(); ++c) {
                for (int i = 0; i < width * height; ++i) {
                    const float value = names[c] == "A" ? 0.25f + float(i % 4) / 4 : type == exr::UINT ? float(i % 1000 + c * 100)
                                                                                                       : float(i % 31) / 8 - 0.5f + float(c);
                    floats[c][i] = value;
                    halves[c][i] = value;
                    ints[c][i] = static_cast<uint32_t>(std::max(value, 0.0f));
                }
                header.channels().insert(names[c], exr::Channel(type));
                const size_t bytes = type == exr::HALF ? sizeof(Imath::half) : sizeof(float);
                char* data = type == exr::HALF ? reinterpret_cast<char*>(halves[c].data()) : type == exr::FLOAT ? reinterpret_cast<char*>(floats[c].data())
                                                                                                                : reinterpret_cast<char*>(ints[c].data());
                fb.insert(names[c], exr::Slice::Make(type, data, header.dataWindow(), bytes, width * bytes));
            }
            OutputStream stream(path);
            if (tiled) {
                header.setTileDescription(exr::TileDescription(7, 11, levels));
                exr::TiledOutputFile output(stream, header);
                output.setFrameBuffer(fb);
                for (int ly = 0; ly < output.numYLevels(); ++ly)
                    for (int lx = 0; lx < output.numXLevels(); ++lx)
                        if (output.isValidLevel(lx, ly))
                            output.writeTiles(0, output.numXTiles(lx) - 1, 0, output.numYTiles(ly) - 1, lx, ly);
            } else {
                exr::OutputFile output(stream, header);
                output.setFrameBuffer(fb);
                output.writePixels(height);
            }
        }

        std::vector<float> reference(bool tiled, bool grayscale = false) const {
            std::vector<float> pixels(width * height * 4, 0.0f);
            auto buffer = [&](const exr::Header& h) {
                exr::FrameBuffer fb;
                if (grayscale) {
                    fb.insert("Y", exr::Slice::Make(exr::FLOAT, pixels.data(), h.dataWindow(), 4 * sizeof(float), width * 4 * sizeof(float)));
                } else {
                    const char* names[] = {"R", "G", "B", "A"};
                    for (int c = 0; c < 4; ++c)
                        fb.insert(names[c], exr::Slice::Make(exr::FLOAT, pixels.data() + c, h.dataWindow(), 4 * sizeof(float), width * 4 * sizeof(float), 1, 1, c == 3 ? 1.0 : 0.0));
                }
                return fb;
            };
            InputStream stream(path);
            if (tiled) {
                exr::TiledInputFile input(stream);
                input.setFrameBuffer(buffer(input.header()));
                input.readTiles(0, input.numXTiles() - 1, 0, input.numYTiles() - 1);
            } else {
                exr::InputFile input(stream);
                input.setFrameBuffer(buffer(input.header()));
                input.readPixels(input.header().dataWindow().min.y, input.header().dataWindow().max.y);
            }
            if (grayscale) {
                for (size_t i = 0; i < pixels.size(); i += 4) {
                    pixels[i + 1] = pixels[i + 2] = pixels[i];
                    pixels[i + 3] = 1.0f;
                }
            }
            return pixels;
        }
    };

    void expect_reference(const Fixture& fixture, bool tiled, bool grayscale = false) {
        codec::Image image;
        std::string error;
        ASSERT_TRUE(codec::decode(fixture.path, image, error)) << error;
        EXPECT_EQ(image.width, Fixture::width);
        EXPECT_EQ(image.height, Fixture::height);
        EXPECT_EQ(image.channels, 4);
        EXPECT_EQ(image.sample_type, codec::SampleType::Float32);
        const auto expected = fixture.reference(tiled, grayscale);
        ASSERT_EQ(image.data.size(), expected.size() * sizeof(float));
        const auto* actual = reinterpret_cast<const float*>(image.data.data());
        for (size_t i = 0; i < expected.size(); ++i)
            ASSERT_FLOAT_EQ(actual[i], expected[i]) << "sample " << i;
        codec::Probe info;
        ASSERT_TRUE(codec::probe(fixture.path, info, error)) << error;
        EXPECT_EQ(info.width, image.width);
        EXPECT_EQ(info.height, image.height);
        EXPECT_EQ(info.channels, image.channels);
        EXPECT_EQ(info.sample_type, image.sample_type);
    }

    using Parameters = std::tuple<exr::Compression, bool, exr::PixelType>;
    class ExrCodecs : public testing::TestWithParam<Parameters> {};
    TEST_P(ExrCodecs, RgbaAndAuxiliaryChannelMatchReference) {
        const auto [compression, tiled, type] = GetParam();
        Fixture f;
        f.write(compression, tiled, type);
        expect_reference(f, tiled);
    }
    TEST_P(ExrCodecs, RgbWithoutAlphaAndOffsetWindowMatchReference) {
        const auto [compression, tiled, type] = GetParam();
        Fixture f;
        f.write(compression, tiled, type, false, false, {-7, 13});
        expect_reference(f, tiled);
    }
    TEST_P(ExrCodecs, GrayscaleReplicatesToRgb) {
        const auto [compression, tiled, type] = GetParam();
        Fixture f;
        f.write(compression, tiled, type, true);
        expect_reference(f, tiled, true);
    }
    INSTANTIATE_TEST_SUITE_P(AllSupportedCompression, ExrCodecs,
                             testing::Combine(testing::Values(exr::NO_COMPRESSION, exr::RLE_COMPRESSION, exr::ZIPS_COMPRESSION,
                                                              exr::ZIP_COMPRESSION, exr::PIZ_COMPRESSION, exr::PXR24_COMPRESSION,
                                                              exr::B44_COMPRESSION, exr::B44A_COMPRESSION, exr::DWAA_COMPRESSION, exr::DWAB_COMPRESSION),
                                              testing::Bool(), testing::Values(exr::HALF, exr::FLOAT, exr::UINT)));

    TEST(ExrCodecRegression, DecreasingScanlines) {
        Fixture f;
        f.write(exr::DWAB_COMPRESSION, false, exr::HALF, false, true, {9, -17}, exr::DECREASING_Y);
        expect_reference(f, false);
    }
    TEST(ExrCodecRegression, MipmapAndRipmapUseFullResolutionLevel) {
        for (const auto mode : {exr::MIPMAP_LEVELS, exr::RIPMAP_LEVELS}) {
            Fixture f;
            f.write(exr::PIZ_COMPRESSION, true, exr::HALF, false, true, {-7, 13}, exr::INCREASING_Y, mode);
            expect_reference(f, true);
        }
    }
    TEST(ExrCodecRegression, MultipartUsesFirstPart) {
        Fixture f;
        exr::Header headers[] = {exr::Header(Fixture::width, Fixture::height), exr::Header(Fixture::width, Fixture::height)};
        for (int part = 0; part < 2; ++part) {
            headers[part].setName("part" + std::to_string(part));
            headers[part].setType(exr::SCANLINEIMAGE);
            for (const char* name : {"R", "G", "B"})
                headers[part].channels().insert(name, exr::Channel(exr::FLOAT));
        }
        {
            OutputStream stream(f.path);
            exr::MultiPartOutputFile file(stream, headers, 2);
            for (int part = 0; part < 2; ++part) {
                std::vector<float> values(Fixture::width * Fixture::height, part + 0.5f);
                exr::FrameBuffer fb;
                for (const char* name : {"R", "G", "B"})
                    fb.insert(name, exr::Slice::Make(exr::FLOAT, values.data(), headers[part].dataWindow()));
                exr::OutputPart output(file, part);
                output.setFrameBuffer(fb);
                output.writePixels(Fixture::height);
            }
        }
        expect_reference(f, false);
    }
    TEST(ExrCodecRegression, DeepAndMissingRgbAreRejected) {
        for (const bool deep : {false, true}) {
            Fixture f;
            exr::Header header(2, 1);
            header.compression() = exr::ZIPS_COMPRESSION;
            header.channels().insert("R", exr::Channel(exr::FLOAT));
            header.channels().insert("G", exr::Channel(exr::FLOAT));
            {
                OutputStream stream(f.path);
                if (deep) {
                    header.setType(exr::DEEPSCANLINE);
                    exr::DeepScanLineOutputFile output(stream, header);
                } else {
                    exr::OutputFile output(stream, header);
                }
            }
            codec::Probe info;
            codec::Image image;
            std::string error;
            EXPECT_FALSE(codec::probe(f.path, info, error));
            EXPECT_FALSE(error.empty());
            EXPECT_FALSE(codec::decode(f.path, image, error));
        }
    }
    TEST(ExrCodecRegression, UnicodeAndUppercaseExtension) {
        Fixture f;
        const auto name = std::filesystem::path(u8"照明_è_").native() + f.path.filename().native();
        f.path = std::filesystem::temp_directory_path() / name;
        f.path.replace_extension(".EXR");
        f.write(exr::DWAA_COMPRESSION, false, exr::HALF);
        expect_reference(f, false);
    }
    TEST(ExrCodecRegression, ProbeDoesNotReadPixels) {
        Fixture f;
        f.write(exr::ZIP_COMPRESSION, false, exr::FLOAT);
        // Keep only the magic/version and complete header. Neither chunk offsets
        // nor pixel data remain: dimensions must still be available.
        std::ifstream in(f.path, std::ios::binary);
        std::vector<char> bytes((std::istreambuf_iterator<char>(in)), {});
        in.close();
        size_t pos = 8;
        while (pos < bytes.size() && bytes[pos] != 0) {
            while (bytes.at(pos++) != 0) {}
            while (bytes.at(pos++) != 0) {}
            uint32_t size = 0;
            for (int i = 0; i < 4; ++i)
                size |= uint32_t(static_cast<unsigned char>(bytes.at(pos++))) << (8 * i);
            pos += size;
        }
        std::filesystem::resize_file(f.path, pos + 1);
        codec::Probe info;
        std::string error;
        ASSERT_TRUE(codec::probe(f.path, info, error)) << error;
        EXPECT_EQ(info.width, Fixture::width);
        EXPECT_EQ(info.height, Fixture::height);
        codec::Image image;
        EXPECT_FALSE(codec::decode(f.path, image, error));
        EXPECT_FALSE(error.empty());
    }
    TEST(ExrCodecRegression, InvalidAndMissingFilesFailCleanly) {
        Fixture f;
        codec::Image image;
        codec::Probe info;
        std::string error;
        EXPECT_FALSE(codec::decode(f.path, image, error));
        EXPECT_FALSE(error.empty());
        {
            std::ofstream out(f.path, std::ios::binary);
            out << "not an EXR";
        }
        EXPECT_FALSE(codec::probe(f.path, info, error));
        EXPECT_FALSE(codec::decode(f.path, image, error));
    }
    TEST(ExrCodecRegression, ConcurrentReadersAreIndependent) {
        Fixture f;
        f.write(exr::DWAB_COMPRESSION, false, exr::HALF);
        std::vector<std::future<std::vector<uint8_t>>> jobs;
        for (int i = 0; i < 8; ++i)
            jobs.push_back(std::async(std::launch::async, [&] {
                codec::Image image;
                std::string error;
                if (!codec::decode(f.path, image, error))
                    throw std::runtime_error(error);
                return image.data;
            }));
        const auto expected = jobs.front().get();
        for (size_t i = 1; i < jobs.size(); ++i)
            EXPECT_EQ(jobs[i].get(), expected);
    }
} // namespace
