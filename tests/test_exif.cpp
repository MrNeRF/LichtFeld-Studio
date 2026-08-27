/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/exif.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {

    enum class Endian { Little,
                        Big };

    constexpr std::uint16_t kTypeShort = 3;
    constexpr std::uint16_t kTypeLong = 4;
    constexpr std::uint16_t kTypeRational = 5;
    constexpr std::uint16_t kTypeSRational = 10;
    constexpr std::uint16_t kTagExifIfd = 0x8769;
    constexpr std::uint16_t kTagExposureTime = 0x829A;
    constexpr std::uint16_t kTagFNumber = 0x829D;
    constexpr std::uint16_t kTagIso = 0x8827;
    constexpr std::uint16_t kTagShutterSpeedValue = 0x9201;
    constexpr std::uint16_t kTagApertureValue = 0x9202;
    constexpr std::uint16_t kTagIsoSpeed = 0x8833;

    void put_u16(std::vector<std::uint8_t>& out, const std::uint16_t value, const Endian endian) {
        if (endian == Endian::Little) {
            out.push_back(static_cast<std::uint8_t>(value));
            out.push_back(static_cast<std::uint8_t>(value >> 8));
        } else {
            out.push_back(static_cast<std::uint8_t>(value >> 8));
            out.push_back(static_cast<std::uint8_t>(value));
        }
    }

    void put_u32(std::vector<std::uint8_t>& out, const std::uint32_t value, const Endian endian) {
        if (endian == Endian::Little) {
            out.push_back(static_cast<std::uint8_t>(value));
            out.push_back(static_cast<std::uint8_t>(value >> 8));
            out.push_back(static_cast<std::uint8_t>(value >> 16));
            out.push_back(static_cast<std::uint8_t>(value >> 24));
        } else {
            out.push_back(static_cast<std::uint8_t>(value >> 24));
            out.push_back(static_cast<std::uint8_t>(value >> 16));
            out.push_back(static_cast<std::uint8_t>(value >> 8));
            out.push_back(static_cast<std::uint8_t>(value));
        }
    }

    struct ExifTag {
        std::uint16_t tag = 0;
        std::uint16_t type = 0;
        std::uint32_t a = 0;
        std::uint32_t b = 0;
    };

    [[nodiscard]] bool is_rational(const std::uint16_t type) {
        return type == kTypeRational || type == kTypeSRational;
    }

    std::vector<std::uint8_t> build_tiff(const Endian endian, const std::vector<ExifTag>& tags) {
        const std::uint32_t ifd0_offset = 8;
        const std::uint32_t exif_ifd_offset = 26;
        const std::uint32_t extra_offset =
            exif_ifd_offset + 2 + 12 * static_cast<std::uint32_t>(tags.size()) + 4;

        std::vector<std::uint8_t> tiff;
        if (endian == Endian::Little) {
            tiff.push_back('I');
            tiff.push_back('I');
        } else {
            tiff.push_back('M');
            tiff.push_back('M');
        }
        put_u16(tiff, 42, endian);
        put_u32(tiff, ifd0_offset, endian);

        put_u16(tiff, 1, endian);
        put_u16(tiff, kTagExifIfd, endian);
        put_u16(tiff, kTypeLong, endian);
        put_u32(tiff, 1, endian);
        put_u32(tiff, exif_ifd_offset, endian);
        put_u32(tiff, 0, endian);

        put_u16(tiff, static_cast<std::uint16_t>(tags.size()), endian);
        std::uint32_t extra_cursor = extra_offset;
        std::vector<std::uint8_t> extra;
        for (const auto& tag : tags) {
            put_u16(tiff, tag.tag, endian);
            put_u16(tiff, tag.type, endian);
            put_u32(tiff, 1, endian);
            if (is_rational(tag.type)) {
                put_u32(tiff, extra_cursor, endian);
                put_u32(extra, tag.a, endian);
                put_u32(extra, tag.b, endian);
                extra_cursor += 8;
            } else if (tag.type == kTypeShort) {
                put_u16(tiff, static_cast<std::uint16_t>(tag.a), endian);
                put_u16(tiff, 0, endian);
            } else {
                put_u32(tiff, tag.a, endian);
            }
        }
        put_u32(tiff, 0, endian);
        tiff.insert(tiff.end(), extra.begin(), extra.end());
        return tiff;
    }

    std::vector<std::uint8_t> wrap_jpeg(const std::vector<std::uint8_t>& tiff) {
        std::vector<std::uint8_t> jpeg;
        jpeg.reserve(12 + tiff.size());
        jpeg.push_back(0xFF);
        jpeg.push_back(0xD8);
        jpeg.push_back(0xFF);
        jpeg.push_back(0xE1);
        const std::uint16_t length = static_cast<std::uint16_t>(2 + 6 + tiff.size());
        jpeg.push_back(static_cast<std::uint8_t>(length >> 8));
        jpeg.push_back(static_cast<std::uint8_t>(length));
        jpeg.push_back('E');
        jpeg.push_back('x');
        jpeg.push_back('i');
        jpeg.push_back('f');
        jpeg.push_back(0);
        jpeg.push_back(0);
        jpeg.insert(jpeg.end(), tiff.begin(), tiff.end());
        jpeg.push_back(0xFF);
        jpeg.push_back(0xD9);
        return jpeg;
    }

    std::filesystem::path write_temp(const std::string& name, const std::vector<std::uint8_t>& bytes) {
        const auto dir = std::filesystem::temp_directory_path() / "lfs_exif_test";
        std::filesystem::create_directories(dir);
        const auto path = dir / name;
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        return path;
    }

    const std::vector<ExifTag> kDirectTags{
        {kTagExposureTime, kTypeRational, 1, 1},
        {kTagFNumber, kTypeRational, 2, 1},
        {kTagIso, kTypeShort, 16, 0},
    };

    const std::vector<ExifTag> kApexTags{
        {kTagShutterSpeedValue, kTypeSRational, 0, 1},
        {kTagApertureValue, kTypeRational, 2, 1},
        {kTagIso, kTypeShort, 16, 0},
    };

} // namespace

TEST(ExifExposureTest, DirectTagsLittleEndian) {
    const auto path = write_temp("direct_le.jpg", wrap_jpeg(build_tiff(Endian::Little, kDirectTags)));
    const auto ev = lfs::core::exif_exposure_ev(path);
    ASSERT_TRUE(ev.has_value());
    EXPECT_DOUBLE_EQ(*ev, 2.0);
}

TEST(ExifExposureTest, DirectTagsBigEndian) {
    const auto path = write_temp("direct_be.jpg", wrap_jpeg(build_tiff(Endian::Big, kDirectTags)));
    const auto ev = lfs::core::exif_exposure_ev(path);
    ASSERT_TRUE(ev.has_value());
    EXPECT_DOUBLE_EQ(*ev, 2.0);
}

TEST(ExifExposureTest, ApexOnlyMatchesDirect) {
    const auto path = write_temp("apex_le.jpg", wrap_jpeg(build_tiff(Endian::Little, kApexTags)));
    const auto ev = lfs::core::exif_exposure_ev(path);
    ASSERT_TRUE(ev.has_value());
    EXPECT_DOUBLE_EQ(*ev, 2.0);
}

TEST(ExifExposureTest, ZeroFNumberTreatedAsMissing) {
    const std::vector<ExifTag> tags{
        {kTagExposureTime, kTypeRational, 1, 1},
        {kTagFNumber, kTypeRational, 0, 1},
        {kTagIso, kTypeShort, 16, 0},
    };
    const auto path = write_temp("fnumber0.jpg", wrap_jpeg(build_tiff(Endian::Little, tags)));
    const auto ev = lfs::core::exif_exposure_ev(path);
    ASSERT_TRUE(ev.has_value());
    EXPECT_DOUBLE_EQ(*ev, 4.0);
}

TEST(ExifExposureTest, NoExposureTagsReturnsNullopt) {
    const auto path = write_temp("empty_exif.jpg", wrap_jpeg(build_tiff(Endian::Little, {})));
    EXPECT_EQ(lfs::core::exif_exposure_ev(path), std::nullopt);
}

TEST(ExifExposureTest, TruncatedAndGarbageOffsetsReturnNullopt) {
    const std::vector<std::uint8_t> truncated{0xFF, 0xD8, 0xFF, 0xE1, 0x00, 0x40, 'E', 'x', 'i', 'f', 0, 0, 'I', 'I'};
    const auto truncated_path = write_temp("truncated.jpg", truncated);
    EXPECT_EQ(lfs::core::exif_exposure_ev(truncated_path), std::nullopt);

    std::vector<std::uint8_t> garbage_tiff{'I', 'I', 42, 0, 0xFF, 0xFF, 0x00, 0x00};
    const auto garbage_path = write_temp("garbage_ifd.jpg", wrap_jpeg(garbage_tiff));
    EXPECT_EQ(lfs::core::exif_exposure_ev(garbage_path), std::nullopt);
}

TEST(ExifExposureTest, PngPathReturnsNullopt) {
    const std::vector<std::uint8_t> png{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 0};
    const auto path = write_temp("not_a_jpeg.png", png);
    EXPECT_EQ(lfs::core::exif_exposure_ev(path), std::nullopt);
}

TEST(ExifExposureTest, ImagesDirectoryFallback) {
    const auto root = std::filesystem::temp_directory_path() / "lfs_exif_fallback";
    const auto resized_dir = root / "images_8";
    const auto full_dir = root / "images";
    std::filesystem::create_directories(resized_dir);
    std::filesystem::create_directories(full_dir);

    const auto stripped = wrap_jpeg(build_tiff(Endian::Little, {}));
    const auto tagged = wrap_jpeg(build_tiff(Endian::Little, kDirectTags));
    const auto resized = resized_dir / "frame.jpg";
    const auto full = full_dir / "frame.jpg";
    {
        std::ofstream out(resized, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(stripped.data()),
                  static_cast<std::streamsize>(stripped.size()));
    }
    {
        std::ofstream out(full, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(tagged.data()),
                  static_cast<std::streamsize>(tagged.size()));
    }

    EXPECT_EQ(lfs::core::exif_exposure_ev(resized), std::nullopt);
    const auto ev = lfs::core::exif_exposure_ev_for_training_image(resized, root);
    ASSERT_TRUE(ev.has_value());
    EXPECT_DOUBLE_EQ(*ev, 2.0);
}

TEST(ExifExposureTest, IsoSpeedFallbackWithoutIsoTag) {
    const std::vector<ExifTag> tags{
        {kTagExposureTime, kTypeRational, 1, 1},
        {kTagFNumber, kTypeRational, 2, 1},
        {kTagIsoSpeed, kTypeLong, 16, 0},
    };
    const auto path = write_temp("iso_speed.jpg", wrap_jpeg(build_tiff(Endian::Little, tags)));
    const auto ev = lfs::core::exif_exposure_ev(path);
    ASSERT_TRUE(ev.has_value());
    EXPECT_DOUBLE_EQ(*ev, 2.0);
}
