/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "io/formats/colmap.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace {
    namespace fs = std::filesystem;
    constexpr size_t READ_CHUNK_BYTES = 64ull * 1024ull * 1024ull;

    template <typename T>
    void append_pod(std::vector<char>& bytes, const T value) {
        const auto* begin = reinterpret_cast<const char*>(&value);
        bytes.insert(bytes.end(), begin, begin + sizeof(T));
    }

    template <typename T>
    T read_pod(const std::vector<char>& bytes, size_t& offset) {
        T value{};
        std::memcpy(&value, bytes.data() + offset, sizeof(T));
        offset += sizeof(T);
        return value;
    }

    void append_valid_image(std::vector<char>& bytes,
                            const uint32_t image_id,
                            const uint32_t camera_id,
                            const std::string_view name) {
        append_pod(bytes, image_id);
        append_pod(bytes, 1.0);
        append_pod(bytes, 0.0);
        append_pod(bytes, 0.0);
        append_pod(bytes, 0.0);
        append_pod(bytes, 0.0);
        append_pod(bytes, 0.0);
        append_pod(bytes, 0.0);
        append_pod(bytes, camera_id);
        bytes.insert(bytes.end(), name.begin(), name.end());
        bytes.push_back('\0');
        append_pod(bytes, uint64_t{0});
    }

    struct ImageReference {
        std::vector<lfs::io::ImageData> images;
        size_t invalid_pose_count = 0;
        std::vector<std::string> invalid_pose_samples;
        size_t invalid_point_count = 0;
        std::vector<std::string> invalid_point_samples;
    };

    ImageReference parse_images_serial(const std::vector<char>& bytes) {
        ImageReference reference;
        size_t offset = 0;
        const auto count = read_pod<uint64_t>(bytes, offset);
        reference.images.reserve(count);
        for (uint64_t index = 0; index < count; ++index) {
            lfs::io::ImageData image;
            image.image_id = read_pod<uint32_t>(bytes, offset);
            for (float& value : image.qvec) {
                value = static_cast<float>(read_pod<double>(bytes, offset));
            }
            for (float& value : image.tvec) {
                value = static_cast<float>(read_pod<double>(bytes, offset));
            }
            image.camera_id = read_pod<uint32_t>(bytes, offset);
            const auto* name_begin = bytes.data() + offset;
            const auto* name_end = static_cast<const char*>(
                std::memchr(name_begin, '\0', bytes.size() - offset));
            image.name.assign(name_begin, name_end);
            offset += static_cast<size_t>(name_end - name_begin) + 1;
            bool valid = !image.name.empty();
            const auto point_count = read_pod<uint64_t>(bytes, offset);
            image.points2D.reserve(point_count);
            for (uint64_t point_index = 0; point_index < point_count; ++point_index) {
                lfs::io::ImagePoint2D point;
                point.x = read_pod<double>(bytes, offset);
                point.y = read_pod<double>(bytes, offset);
                point.point3D_id = read_pod<uint64_t>(bytes, offset);
                if (std::isfinite(point.x) && std::isfinite(point.y)) {
                    image.points2D.push_back(point);
                } else {
                    ++reference.invalid_point_count;
                    if (reference.invalid_point_samples.size() < 8) {
                        reference.invalid_point_samples.push_back(
                            std::format("image_id={},point_index={}", image.image_id, point_index));
                    }
                }
            }
            const double qnorm = std::sqrt(
                static_cast<double>(image.qvec[0]) * image.qvec[0] +
                static_cast<double>(image.qvec[1]) * image.qvec[1] +
                static_cast<double>(image.qvec[2]) * image.qvec[2] +
                static_cast<double>(image.qvec[3]) * image.qvec[3]);
            valid = valid && std::isfinite(qnorm) && std::abs(qnorm - 1.0) <= 1e-4 &&
                    std::ranges::all_of(image.tvec, [](const float value) {
                        return std::isfinite(value);
                    });
            if (!valid) {
                ++reference.invalid_pose_count;
                if (reference.invalid_pose_samples.size() < 8) {
                    reference.invalid_pose_samples.push_back(
                        std::format("image_id={}", image.image_id));
                }
            } else {
                reference.images.push_back(std::move(image));
            }
        }
        return reference;
    }

    std::vector<lfs::io::Point3DData> parse_points_serial(const std::vector<char>& bytes) {
        std::vector<lfs::io::Point3DData> points;
        size_t offset = 0;
        const auto count = read_pod<uint64_t>(bytes, offset);
        points.reserve(count);
        for (uint64_t index = 0; index < count; ++index) {
            lfs::io::Point3DData point;
            point.point3D_id = read_pod<uint64_t>(bytes, offset);
            for (double& value : point.xyz) {
                value = read_pod<double>(bytes, offset);
            }
            for (uint8_t& value : point.color) {
                value = read_pod<uint8_t>(bytes, offset);
            }
            point.error = read_pod<double>(bytes, offset);
            const auto track_count = read_pod<uint64_t>(bytes, offset);
            point.track_count = static_cast<size_t>(track_count);
            point.track.reserve(track_count);
            for (uint64_t track_index = 0; track_index < track_count; ++track_index) {
                point.track.push_back(lfs::io::Point3DTrackElement{
                    .image_id = read_pod<uint32_t>(bytes, offset),
                    .point2D_idx = read_pod<uint32_t>(bytes, offset)});
            }
            if (std::isfinite(point.xyz[0]) && std::isfinite(point.xyz[1]) &&
                std::isfinite(point.xyz[2]) && std::isfinite(point.error)) {
                points.push_back(std::move(point));
            }
        }
        return points;
    }

    void write_bytes(const fs::path& path, const std::vector<char>& bytes) {
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(stream.is_open());
        stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        ASSERT_TRUE(stream.good());
    }

    void expect_image_equal(const lfs::io::ImageData& actual,
                            const lfs::io::ImageData& expected) {
        EXPECT_EQ(actual.image_id, expected.image_id);
        EXPECT_EQ(actual.camera_id, expected.camera_id);
        EXPECT_EQ(actual.name, expected.name);
        EXPECT_EQ(actual.qvec, expected.qvec);
        EXPECT_EQ(actual.tvec, expected.tvec);
        ASSERT_EQ(actual.points2D.size(), expected.points2D.size());
        for (size_t i = 0; i < actual.points2D.size(); ++i) {
            EXPECT_EQ(actual.points2D[i].x, expected.points2D[i].x);
            EXPECT_EQ(actual.points2D[i].y, expected.points2D[i].y);
            EXPECT_EQ(actual.points2D[i].point3D_id, expected.points2D[i].point3D_id);
        }
    }

    void expect_point_equal(const lfs::io::Point3DData& actual,
                            const lfs::io::Point3DData& expected) {
        EXPECT_EQ(actual.point3D_id, expected.point3D_id);
        for (size_t i = 0; i < 3; ++i) {
            EXPECT_EQ(actual.xyz[i], expected.xyz[i]);
            EXPECT_EQ(actual.color[i], expected.color[i]);
        }
        EXPECT_EQ(actual.error, expected.error);
        EXPECT_EQ(actual.track_count, expected.track_count);
        ASSERT_EQ(actual.track.size(), expected.track.size());
        for (size_t i = 0; i < actual.track.size(); ++i) {
            EXPECT_EQ(actual.track[i].image_id, expected.track[i].image_id);
            EXPECT_EQ(actual.track[i].point2D_idx, expected.track[i].point2D_idx);
        }
    }

    std::string diagnostic_message(const size_t count,
                                   const std::vector<std::string>& samples,
                                   const std::string_view noun) {
        std::string joined;
        for (size_t i = 0; i < samples.size(); ++i) {
            if (i != 0) {
                joined += ", ";
            }
            joined += samples[i];
        }
        return std::format("COLMAP import skipped {} {} ({} sampled: {})",
                           count, noun, samples.size(), joined);
    }

    TEST(ColmapBinaryParallelParseTest, MatchesSerialReferenceAndDiagnosticOrder) {
        const fs::path temp_dir = fs::temp_directory_path() / "lfs_colmap_binary_parallel_parse";
        std::error_code ec;
        fs::remove_all(temp_dir, ec);
        ASSERT_TRUE(fs::create_directories(temp_dir));

        std::vector<char> image_bytes;
        constexpr uint64_t image_count = 201;
        append_pod(image_bytes, image_count);
        for (uint64_t image_index = 0; image_index < image_count; ++image_index) {
            const uint32_t image_id = static_cast<uint32_t>(1000 + image_index);
            append_pod(image_bytes, image_id);
            append_pod(image_bytes, image_index == 137 ? 2.0 : 1.0);
            append_pod(image_bytes, 0.01 * static_cast<double>(image_index));
            append_pod(image_bytes, -0.02 * static_cast<double>(image_index));
            append_pod(image_bytes, 0.03 * static_cast<double>(image_index));
            append_pod(image_bytes, 0.1 * static_cast<double>(image_index));
            append_pod(image_bytes, -0.2 * static_cast<double>(image_index));
            append_pod(image_bytes, 0.3 * static_cast<double>(image_index));
            append_pod(image_bytes, uint32_t{7});
            const std::string name = std::format("frame_{:04}.png", image_index);
            image_bytes.insert(image_bytes.end(), name.begin(), name.end());
            image_bytes.push_back('\0');
            const uint64_t point_count = image_index == 0 ? 0 : image_index * 25;
            append_pod(image_bytes, point_count);
            for (uint64_t point_index = 0; point_index < point_count; ++point_index) {
                const bool non_finite = point_index % 997 == 0;
                append_pod(image_bytes, non_finite ? std::numeric_limits<double>::quiet_NaN()
                                                   : static_cast<double>(point_index) + 0.25);
                append_pod(image_bytes, non_finite ? 1.0 : static_cast<double>(point_index) + 0.5);
                append_pod(image_bytes, point_index % 257);
            }
        }

        std::vector<char> point_bytes;
        constexpr uint64_t point_count = 257;
        append_pod(point_bytes, point_count);
        for (uint64_t point_index = 0; point_index < point_count; ++point_index) {
            append_pod(point_bytes, point_index);
            append_pod(point_bytes, static_cast<double>(point_index) + 0.1);
            append_pod(point_bytes, static_cast<double>(point_index) + 0.2);
            append_pod(point_bytes, static_cast<double>(point_index) + 0.3);
            append_pod(point_bytes, static_cast<uint8_t>(point_index));
            append_pod(point_bytes, static_cast<uint8_t>(point_index + 1));
            append_pod(point_bytes, static_cast<uint8_t>(point_index + 2));
            append_pod(point_bytes, 0.001 * static_cast<double>(point_index));
            const uint64_t track_count = point_index % 5;
            append_pod(point_bytes, track_count);
            for (uint64_t track_index = 0; track_index < track_count; ++track_index) {
                append_pod(point_bytes, static_cast<uint32_t>(1000 + track_index));
                append_pod(point_bytes, static_cast<uint32_t>(point_index + track_index));
            }
        }

        const fs::path image_path = temp_dir / "images.bin";
        const fs::path point_path = temp_dir / "points3D.bin";
        write_bytes(image_path, image_bytes);
        write_bytes(point_path, point_bytes);

        const auto image_reference = parse_images_serial(image_bytes);
        const auto point_reference = parse_points_serial(point_bytes);
        const auto images_result = lfs::io::read_colmap_images_binary(image_path);
        ASSERT_TRUE(images_result.has_value()) << images_result.error().format();
        ASSERT_EQ(images_result->value.size(), image_reference.images.size());
        for (size_t i = 0; i < image_reference.images.size(); ++i) {
            expect_image_equal(images_result->value[i], image_reference.images[i]);
        }
        ASSERT_EQ(images_result->warnings.size(), 2u);
        EXPECT_EQ(images_result->warnings[0].message,
                  diagnostic_message(image_reference.invalid_pose_count,
                                     image_reference.invalid_pose_samples,
                                     "image(s) with an invalid pose"));
        EXPECT_EQ(images_result->warnings[1].message,
                  diagnostic_message(image_reference.invalid_point_count,
                                     image_reference.invalid_point_samples,
                                     "point2D observation(s)"));

        const auto points_result = lfs::io::read_colmap_point3D_binary_records(point_path);
        ASSERT_TRUE(points_result.has_value()) << points_result.error().format();
        ASSERT_EQ(points_result->value.size(), point_reference.size());
        ASSERT_TRUE(points_result->warnings.empty());
        for (size_t i = 0; i < point_reference.size(); ++i) {
            expect_point_equal(points_result->value[i], point_reference[i]);
        }

        fs::remove_all(temp_dir, ec);
    }

    TEST(ColmapBinaryParallelParseTest, ReadsFilesBelowAndAcrossParallelReadChunks) {
        const fs::path temp_dir = fs::temp_directory_path() / "lfs_colmap_binary_parallel_read_chunks";
        std::error_code ec;
        fs::remove_all(temp_dir, ec);
        ASSERT_TRUE(fs::create_directories(temp_dir));

        std::vector<char> small_bytes;
        append_pod(small_bytes, uint64_t{1});
        append_valid_image(small_bytes, 1, 7, "small.png");
        ASSERT_LT(small_bytes.size(), READ_CHUNK_BYTES);
        const fs::path small_path = temp_dir / "small_images.bin";
        write_bytes(small_path, small_bytes);
        const auto small_result = lfs::io::read_colmap_images_binary(small_path);
        ASSERT_TRUE(small_result.has_value()) << small_result.error().format();
        ASSERT_EQ(small_result->value.size(), 1u);
        EXPECT_EQ(small_result->value.front().name, "small.png");

        std::string long_name(2 * READ_CHUNK_BYTES + 123, 'x');
        std::vector<char> spanning_bytes;
        append_pod(spanning_bytes, uint64_t{1});
        append_valid_image(spanning_bytes, 2, 7, long_name);
        ASSERT_GT(spanning_bytes.size(), 2 * READ_CHUNK_BYTES);
        const fs::path spanning_path = temp_dir / "spanning_images.bin";
        write_bytes(spanning_path, spanning_bytes);
        const auto spanning_result = lfs::io::read_colmap_images_binary(spanning_path);
        ASSERT_TRUE(spanning_result.has_value()) << spanning_result.error().format();
        ASSERT_EQ(spanning_result->value.size(), 1u);
        EXPECT_EQ(spanning_result->value.front().name.size(), long_name.size());
        EXPECT_EQ(spanning_result->value.front().name, long_name);

        fs::remove_all(temp_dir, ec);
    }
} // namespace
