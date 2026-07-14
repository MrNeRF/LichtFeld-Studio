/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "colmap.hpp"
#include "core/assert.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "io/atomic_output.hpp"
#include "io/filesystem_utils.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <tbb/parallel_for.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef __linux__
#include <cerrno>
#include <fcntl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace lfs::io {

    // Import types from lfs::core for convenience
    using lfs::core::Camera;
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::PointCloud;
    using lfs::core::Tensor;

    namespace fs = std::filesystem;

    namespace {
        constexpr size_t CANCEL_POLL_INTERVAL = 64;
        constexpr size_t IMAGE_METADATA_PROBE_LIMIT = 8192;
        constexpr size_t POINTS3D_PARALLEL_MIN_BYTES = 16ull * 1024ull * 1024ull;
        constexpr size_t POINTS3D_TARGET_CHUNK_BYTES = 8ull * 1024ull * 1024ull;

        class ScopedStagingDirectory {
        public:
            explicit ScopedStagingDirectory(fs::path path)
                : path_(std::move(path)) {}

            ~ScopedStagingDirectory() {
                std::error_code ec;
                fs::remove_all(path_, ec);
            }

            ScopedStagingDirectory(const ScopedStagingDirectory&) = delete;
            ScopedStagingDirectory& operator=(const ScopedStagingDirectory&) = delete;

        private:
            fs::path path_;
        };

        void clone_sparse_directory_for_staging(const fs::path& source, const fs::path& staging) {
            std::error_code ec;
            fs::create_directory(staging, ec);
            if (ec) {
                throw std::runtime_error(std::format(
                    "Failed to create COLMAP staging directory '{}': {}",
                    lfs::core::path_to_utf8(staging), ec.message()));
            }

            if (!fs::exists(source)) {
                return;
            }
            if (!fs::is_directory(source)) {
                throw std::runtime_error("COLMAP output path is not a directory: " +
                                         lfs::core::path_to_utf8(source));
            }

            constexpr auto copy_options = fs::copy_options::recursive |
                                          fs::copy_options::copy_symlinks |
                                          fs::copy_options::overwrite_existing;
            for (const auto& entry : fs::directory_iterator(source)) {
                fs::copy(entry.path(), staging / entry.path().filename(), copy_options, ec);
                if (ec) {
                    throw std::runtime_error(std::format(
                        "Failed to stage COLMAP entry '{}': {}",
                        lfs::core::path_to_utf8(entry.path()), ec.message()));
                }
            }
        }

        void sync_colmap_generation(const fs::path& staging, const bool binary) {
            const std::array<const char*, 3> names = binary
                                                         ? std::array{"cameras.bin", "images.bin", "points3D.bin"}
                                                         : std::array{"cameras.txt", "images.txt", "points3D.txt"};
            for (const char* name : names) {
                if (auto result = detail::sync_file_for_durable_replace(staging / name); !result) {
                    throw std::runtime_error(result.error().format());
                }
            }
            if (auto result = detail::sync_parent_directory(staging / names.front()); !result) {
                throw std::runtime_error(result.error().format());
            }
        }

        void publish_staged_colmap_generation(const fs::path& staging, const fs::path& output) {
            const bool replacing = fs::exists(output);
            std::error_code ec;

            if (!replacing) {
                fs::rename(staging, output, ec);
                if (ec) {
                    throw std::runtime_error(std::format(
                        "Failed to publish COLMAP staging directory '{}': {}",
                        lfs::core::path_to_utf8(staging), ec.message()));
                }
            } else {
#ifdef __linux__
                // renameat2(RENAME_EXCHANGE) is the only Linux namespace operation
                // that swaps two non-empty directories atomically. After it returns,
                // readers see the complete old or complete new reconstruction; the
                // old generation is left at `staging` for post-commit cleanup.
                if (::syscall(SYS_renameat2,
                              AT_FDCWD,
                              staging.c_str(),
                              AT_FDCWD,
                              output.c_str(),
                              RENAME_EXCHANGE) != 0) {
                    throw std::runtime_error(std::format(
                        "Failed to atomically exchange COLMAP generation '{}': {}",
                        lfs::core::path_to_utf8(output), std::strerror(errno)));
                }
#else
                // Standard C++ and Win32 do not expose a non-empty directory
                // exchange primitive. Keep rollback ownership explicit rather than
                // truncating live files; a durable manifest indirection is required
                // to make this fallback power-loss atomic on those platforms.
                const auto backup = make_atomic_temp_output_path(output);
                fs::rename(output, backup, ec);
                if (ec) {
                    throw std::runtime_error(std::format(
                        "Failed to preserve previous COLMAP generation '{}': {}",
                        lfs::core::path_to_utf8(output), ec.message()));
                }
                fs::rename(staging, output, ec);
                if (ec) {
                    std::error_code rollback_ec;
                    fs::rename(backup, output, rollback_ec);
                    throw std::runtime_error(std::format(
                        "Failed to publish COLMAP generation '{}': {}{}",
                        lfs::core::path_to_utf8(output), ec.message(),
                        rollback_ec ? std::format("; rollback failed: {}", rollback_ec.message()) : ""));
                }
                fs::remove_all(backup, ec);
                if (ec) {
                    LOG_WARN("Failed to remove previous COLMAP generation '{}': {}",
                             lfs::core::path_to_utf8(backup), ec.message());
                }
#endif
            }

            if (auto result = detail::sync_parent_directory(output); !result) {
                throw std::runtime_error(result.error().format());
            }

#ifdef __linux__
            if (replacing) {
                fs::remove_all(staging, ec);
                if (ec) {
                    LOG_WARN("Failed to remove previous COLMAP generation '{}': {}",
                             lfs::core::path_to_utf8(staging), ec.message());
                } else if (auto result = detail::sync_parent_directory(output); !result) {
                    LOG_WARN("Failed to durably record COLMAP generation cleanup: {}",
                             result.error().format());
                }
            }
#endif
        }

        [[nodiscard]] bool should_poll_cancel(const size_t index) {
            return (index % CANCEL_POLL_INTERVAL) == 0;
        }

        [[nodiscard]] double elapsed_ms(const std::chrono::high_resolution_clock::time_point start) {
            return std::chrono::duration<double, std::milli>(
                       std::chrono::high_resolution_clock::now() - start)
                .count();
        }

        void skip_ascii_spaces(const char*& cur, const char* end) {
            while (cur < end && (*cur == ' ' || *cur == '\t')) {
                ++cur;
            }
        }

        template <typename T>
        bool parse_next_number(const char*& cur, const char* end, T& value) {
            skip_ascii_spaces(cur, end);
            if (cur >= end) {
                return false;
            }

            const auto parsed = std::from_chars(cur, end, value);
            if (parsed.ec != std::errc{}) {
                return false;
            }

            cur = parsed.ptr;
            return true;
        }

        enum class TrackParseMode {
            None,
            CountOnly,
            Full
        };

        size_t count_remaining_track_pairs(const char* cur, const char* end) {
            size_t pairs = 0;
            while (true) {
                skip_ascii_spaces(cur, end);
                if (cur == end) {
                    return pairs;
                }

                uint32_t image_id = 0;
                uint32_t point2D_idx = 0;
                LFS_ASSERT_MSG(parse_next_number(cur, end, image_id),
                               std::format("COLMAP point track contains a malformed image id "
                                           "(bytes_remaining={}, parsed_pairs={})",
                                           static_cast<size_t>(end - cur), pairs));
                LFS_ASSERT_MSG(parse_next_number(cur, end, point2D_idx),
                               std::format("COLMAP point track has an unmatched image id "
                                           "(image_id={}, bytes_remaining={}, parsed_pairs={})",
                                           image_id, static_cast<size_t>(end - cur), pairs));
                LFS_ASSERT_MSG(image_id != 0,
                               std::format("COLMAP point track image id must be non-zero "
                                           "(image_id={}, point2D_index={}, pair_index={})",
                                           image_id, point2D_idx, pairs));
                (void)point2D_idx;
                ++pairs;
            }
        }

        struct TextChunk {
            const char* begin = nullptr;
            const char* end = nullptr;
        };

        std::vector<TextChunk> split_line_aligned_chunks(std::span<const char> buffer,
                                                         const size_t target_chunk_bytes = POINTS3D_TARGET_CHUNK_BYTES) {
            std::vector<TextChunk> chunks;
            if (buffer.empty()) {
                return chunks;
            }

            const char* base = buffer.data();
            const char* end = base + buffer.size();
            const size_t nominal_chunks = std::max<size_t>(1, buffer.size() / std::max<size_t>(target_chunk_bytes, 1));
            chunks.reserve(nominal_chunks + 1);

            const char* chunk_begin = base;
            while (chunk_begin < end) {
                const char* nominal_end = chunk_begin + std::min<size_t>(target_chunk_bytes, static_cast<size_t>(end - chunk_begin));
                const char* chunk_end = end;
                if (nominal_end < end) {
                    const void* newline = std::memchr(nominal_end, '\n', static_cast<size_t>(end - nominal_end));
                    chunk_end = newline ? static_cast<const char*>(newline) + 1 : end;
                }

                chunks.push_back(TextChunk{.begin = chunk_begin, .end = chunk_end});
                chunk_begin = chunk_end;
            }

            return chunks;
        }

        template <typename LineFn>
        size_t for_each_data_line(std::span<const char> buffer,
                                  const LoadOptions& options,
                                  const char* cancel_msg,
                                  LineFn&& fn) {
            const char* cur = buffer.data();
            const char* end = cur + buffer.size();
            size_t line_count = 0;

            while (cur < end) {
                if (should_poll_cancel(line_count)) {
                    throw_if_load_cancel_requested(options, cancel_msg);
                }

                const char* line_begin = cur;
                const void* newline = std::memchr(cur, '\n', static_cast<size_t>(end - cur));
                const char* line_end = newline ? static_cast<const char*>(newline) : end;
                cur = newline ? line_end + 1 : end;
                ++line_count;

                if (line_end > line_begin && *(line_end - 1) == '\r') {
                    --line_end;
                }
                if (line_begin == line_end || *line_begin == '#') {
                    continue;
                }

                fn(std::string_view(line_begin, static_cast<size_t>(line_end - line_begin)), line_count);
            }

            return line_count;
        }
    } // namespace

    // -----------------------------------------------------------------------------
    //  Quaternion to rotation matrix (torch-free)
    // -----------------------------------------------------------------------------
    inline Tensor qvec2rotmat(const std::vector<float>& q_raw) {
        LFS_ASSERT_MSG(q_raw.size() == 4,
                       std::format("quaternion must have four elements "
                                   "(component_count={})",
                                   q_raw.size()));
        LFS_ASSERT_MSG(std::ranges::all_of(q_raw, [](const float value) {
                           return std::isfinite(value);
                       }),
                       std::format("quaternion components must be finite "
                                   "(q=[{},{},{},{}])",
                                   q_raw[0], q_raw[1], q_raw[2], q_raw[3]));

        // Normalize quaternion
        float len = std::sqrt(q_raw[0] * q_raw[0] + q_raw[1] * q_raw[1] +
                              q_raw[2] * q_raw[2] + q_raw[3] * q_raw[3]);
        LFS_ASSERT_MSG(std::isfinite(len) && len >= 1e-8f,
                       std::format("quaternion must have finite non-zero length "
                                   "(norm={}, q=[{},{},{},{}])",
                                   len, q_raw[0], q_raw[1], q_raw[2], q_raw[3]));

        float w = q_raw[0] / len;
        float x = q_raw[1] / len;
        float y = q_raw[2] / len;
        float z = q_raw[3] / len;

        // Build rotation matrix [3, 3]
        std::vector<float> R_data = {
            1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y - z * w), 2.0f * (x * z + y * w),
            2.0f * (x * y + z * w), 1.0f - 2.0f * (x * x + z * z), 2.0f * (y * z - x * w),
            2.0f * (x * z - y * w), 2.0f * (y * z + x * w), 1.0f - 2.0f * (x * x + y * y)};

        return Tensor::from_vector(R_data, {3, 3}, Device::CPU);
    }

    // -----------------------------------------------------------------------------
    //  Image data structure
    // -----------------------------------------------------------------------------
    constexpr uint64_t INVALID_POINT3D_ID = std::numeric_limits<uint64_t>::max();

    struct ImagePoint2D {
        double x = 0.0;
        double y = 0.0;
        uint64_t point3D_id = INVALID_POINT3D_ID;
    };

    struct ImageData {
        uint32_t image_id = 0;
        uint32_t camera_id = 0;
        std::string name;
        std::vector<float> qvec = {1.0f, 0.0f, 0.0f, 0.0f}; // [w, x, y, z]
        std::vector<float> tvec = {0.0f, 0.0f, 0.0f};
        std::vector<ImagePoint2D> points2D;
    };

    struct Point3DTrackElement {
        uint32_t image_id = 0;
        uint32_t point2D_idx = 0;
    };

    struct Point3DData {
        uint64_t point3D_id = 0;
        double xyz[3] = {0.0, 0.0, 0.0};
        uint8_t color[3] = {255, 255, 255};
        double error = 0.0;
        size_t track_count = 0;
        std::vector<Point3DTrackElement> track;
    };

    namespace {
        struct Point3DTextPointCloudData {
            std::vector<float> positions;
            std::vector<std::uint8_t> colors;
            std::vector<std::uint64_t> point_ids;
            std::size_t point_count = 0;
            std::size_t file_lines = 0;
            std::uintmax_t byte_size = 0;
        };

        bool parse_point3D_header(const std::string_view line,
                                  Point3DData& point,
                                  const char*& cur,
                                  const char*& end) {
            cur = line.data();
            end = cur + line.size();
            int red = 255;
            int green = 255;
            int blue = 255;
            if (!parse_next_number(cur, end, point.point3D_id) ||
                !parse_next_number(cur, end, point.xyz[0]) ||
                !parse_next_number(cur, end, point.xyz[1]) ||
                !parse_next_number(cur, end, point.xyz[2]) ||
                !parse_next_number(cur, end, red) ||
                !parse_next_number(cur, end, green) ||
                !parse_next_number(cur, end, blue) ||
                !parse_next_number(cur, end, point.error)) {
                return false;
            }

            LFS_ASSERT_MSG(point.point3D_id != 0,
                           std::format("COLMAP point id must be non-zero "
                                       "(point3D_id={})",
                                       point.point3D_id));
            LFS_ASSERT_MSG(std::ranges::all_of(point.xyz, [](const double value) { return std::isfinite(value); }),
                           std::format("COLMAP point coordinates must be finite "
                                       "(point3D_id={}, xyz=[{},{},{}])",
                                       point.point3D_id, point.xyz[0], point.xyz[1], point.xyz[2]));
            LFS_ASSERT_MSG(red >= 0 && red <= 255 && green >= 0 && green <= 255 &&
                               blue >= 0 && blue <= 255,
                           std::format("COLMAP point colors must be in [0,255] "
                                       "(point3D_id={}, rgb=[{},{},{}])",
                                       point.point3D_id, red, green, blue));
            LFS_ASSERT_MSG(std::isfinite(point.error) && point.error >= 0.0,
                           std::format("COLMAP point error must be finite and non-negative "
                                       "(point3D_id={}, error={})",
                                       point.point3D_id, point.error));

            point.color[0] = static_cast<uint8_t>(red);
            point.color[1] = static_cast<uint8_t>(green);
            point.color[2] = static_cast<uint8_t>(blue);
            return true;
        }

        void parse_point3D_record_line(const std::string_view line,
                                       const TrackParseMode track_mode,
                                       Point3DData& point,
                                       size_t& total_track_elements) {
            const char* cur = nullptr;
            const char* end = nullptr;
            if (!parse_point3D_header(line, point, cur, end)) {
                LOG_ERROR("Invalid format in points3D.txt: {}", std::string(line));
                throw std::runtime_error("Invalid format in points3D.txt");
            }

            if (track_mode == TrackParseMode::CountOnly) {
                point.track_count = count_remaining_track_pairs(cur, end);
                total_track_elements += point.track_count;
            } else if (track_mode == TrackParseMode::Full) {
                const size_t estimated_pairs = static_cast<size_t>(std::max<std::ptrdiff_t>((end - cur) / 12, 0));
                point.track.reserve(estimated_pairs);
                while (true) {
                    skip_ascii_spaces(cur, end);
                    if (cur == end) {
                        break;
                    }
                    Point3DTrackElement track;
                    LFS_ASSERT_MSG(parse_next_number(cur, end, track.image_id),
                                   std::format("COLMAP point track contains a malformed image id "
                                               "(point3D_id={}, bytes_remaining={}, parsed_pairs={})",
                                               point.point3D_id, static_cast<size_t>(end - cur),
                                               point.track.size()));
                    LFS_ASSERT_MSG(parse_next_number(cur, end, track.point2D_idx),
                                   std::format("COLMAP point track has an unmatched image id "
                                               "(point3D_id={}, image_id={}, bytes_remaining={}, parsed_pairs={})",
                                               point.point3D_id, track.image_id,
                                               static_cast<size_t>(end - cur), point.track.size()));
                    LFS_ASSERT_MSG(track.image_id != 0,
                                   std::format("COLMAP point track image id must be non-zero "
                                               "(point3D_id={}, image_id={}, point2D_index={}, pair_index={})",
                                               point.point3D_id, track.image_id, track.point2D_idx,
                                               point.track.size()));
                    point.track.push_back(track);
                }
                point.track_count = point.track.size();
                total_track_elements += point.track_count;
            }
        }

        void parse_point3D_point_cloud_line(const std::string_view line,
                                            std::vector<float>& positions,
                                            std::vector<uint8_t>& colors,
                                            std::vector<uint64_t>& point_ids,
                                            size_t& point_count) {
            const char* cur = line.data();
            const char* end = cur + line.size();
            uint64_t point_id = 0;
            float x = 0.0f;
            float y = 0.0f;
            float z = 0.0f;
            int red = 255;
            int green = 255;
            int blue = 255;
            double error = 0.0;
            if (!parse_next_number(cur, end, point_id) ||
                !parse_next_number(cur, end, x) ||
                !parse_next_number(cur, end, y) ||
                !parse_next_number(cur, end, z) ||
                !parse_next_number(cur, end, red) ||
                !parse_next_number(cur, end, green) ||
                !parse_next_number(cur, end, blue) ||
                !parse_next_number(cur, end, error)) {
                LOG_ERROR("Invalid format in points3D.txt: {}", std::string(line));
                throw std::runtime_error("Invalid format in points3D.txt");
            }
            LFS_ASSERT_MSG(point_id != 0,
                           std::format("COLMAP point id must be non-zero "
                                       "(point_id={}, parsed_point_index={}, raw_record='{}')",
                                       point_id, point_count, line));
            LFS_ASSERT_MSG(std::isfinite(x) && std::isfinite(y) && std::isfinite(z),
                           std::format("COLMAP point coordinates must be finite "
                                       "(point_id={}, xyz=[{},{},{}], parsed_point_index={})",
                                       point_id, x, y, z, point_count));
            LFS_ASSERT_MSG(red >= 0 && red <= 255 && green >= 0 && green <= 255 &&
                               blue >= 0 && blue <= 255,
                           std::format("COLMAP point colors must be in [0,255] "
                                       "(point_id={}, rgb=[{},{},{}], parsed_point_index={})",
                                       point_id, red, green, blue, point_count));
            LFS_ASSERT_MSG(std::isfinite(error) && error >= 0.0,
                           std::format("COLMAP point error must be finite and non-negative "
                                       "(point_id={}, error={}, parsed_point_index={})",
                                       point_id, error, point_count));
            (void)error;

            // Even the point-cloud-only fast path must validate the complete record;
            // otherwise malformed tracks silently survive when filtering is disabled.
            (void)count_remaining_track_pairs(cur, end);

            positions.push_back(x);
            positions.push_back(y);
            positions.push_back(z);
            colors.push_back(static_cast<uint8_t>(red));
            colors.push_back(static_cast<uint8_t>(green));
            colors.push_back(static_cast<uint8_t>(blue));
            point_ids.push_back(point_id);
            ++point_count;
        }
    } // namespace

    // -----------------------------------------------------------------------------
    //  Camera data structure (intermediate)
    // -----------------------------------------------------------------------------
    struct CameraDataIntermediate {
        uint32_t camera_id = 0;
        int model_id = 0;
        int width = 0;
        int height = 0;
        std::vector<float> params;
    };

    // -----------------------------------------------------------------------------
    //  POD read helpers
    // -----------------------------------------------------------------------------
    template <typename T>
    static T read_binary_pod(const char*& p, const char* end, const std::string_view field) {
        LFS_ASSERT_MSG(p <= end && static_cast<size_t>(end - p) >= sizeof(T),
                       std::format("Truncated COLMAP binary while reading {}", field));
        T value{};
        std::memcpy(&value, p, sizeof(T));
        p += sizeof(T);
        return value;
    }

    static inline uint64_t read_u64(const char*& p, const char* end, const std::string_view field) {
        return read_binary_pod<uint64_t>(p, end, field);
    }
    static inline uint32_t read_u32(const char*& p, const char* end, const std::string_view field) {
        return read_binary_pod<uint32_t>(p, end, field);
    }
    static inline int32_t read_i32(const char*& p, const char* end, const std::string_view field) {
        return read_binary_pod<int32_t>(p, end, field);
    }
    static inline double read_f64(const char*& p, const char* end, const std::string_view field) {
        return read_binary_pod<double>(p, end, field);
    }

    template <typename T>
    static inline void write_pod(std::ostream& stream, const T& value) {
        stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
        if (!stream.good()) {
            throw std::runtime_error("Failed to write COLMAP binary data");
        }
    }

    // -----------------------------------------------------------------------------
    //  COLMAP camera-model map
    // -----------------------------------------------------------------------------
    enum class CAMERA_MODEL {
        SIMPLE_PINHOLE = 0,
        PINHOLE = 1,
        SIMPLE_RADIAL = 2,
        RADIAL = 3,
        OPENCV = 4,
        OPENCV_FISHEYE = 5,
        FULL_OPENCV = 6,
        FOV = 7,
        SIMPLE_RADIAL_FISHEYE = 8,
        RADIAL_FISHEYE = 9,
        THIN_PRISM_FISHEYE = 10,
        UNDEFINED = 11,
        // Equirectangular 360 panorama model (COLMAP model id 17, params
        // [width, height], no intrinsics). Mapped to
        // CameraModelType::EQUIRECTANGULAR, which the rasterizer handles
        // natively. Upstream COLMAP names this model "EQUIRECTANGULAR"
        // (colmap/colmap#4441); the legacy "SPHERICAL" name written by the
        // original equirectangular fork is still accepted on read.
        EQUIRECTANGULAR = 17
    };

    static const std::unordered_map<int, std::pair<CAMERA_MODEL, int32_t>> camera_model_ids = {
        {0, {CAMERA_MODEL::SIMPLE_PINHOLE, 3}},
        {1, {CAMERA_MODEL::PINHOLE, 4}},
        {2, {CAMERA_MODEL::SIMPLE_RADIAL, 4}},
        {3, {CAMERA_MODEL::RADIAL, 5}},
        {4, {CAMERA_MODEL::OPENCV, 8}},
        {5, {CAMERA_MODEL::OPENCV_FISHEYE, 8}},
        {6, {CAMERA_MODEL::FULL_OPENCV, 12}},
        {7, {CAMERA_MODEL::FOV, 5}},
        {8, {CAMERA_MODEL::SIMPLE_RADIAL_FISHEYE, 4}},
        {9, {CAMERA_MODEL::RADIAL_FISHEYE, 5}},
        {10, {CAMERA_MODEL::THIN_PRISM_FISHEYE, 12}},
        {11, {CAMERA_MODEL::UNDEFINED, -1}},
        {17, {CAMERA_MODEL::EQUIRECTANGULAR, 2}}};

    static const std::unordered_map<std::string, CAMERA_MODEL> camera_model_names = {
        {"SIMPLE_PINHOLE", CAMERA_MODEL::SIMPLE_PINHOLE},
        {"PINHOLE", CAMERA_MODEL::PINHOLE},
        {"SIMPLE_RADIAL", CAMERA_MODEL::SIMPLE_RADIAL},
        {"RADIAL", CAMERA_MODEL::RADIAL},
        {"OPENCV", CAMERA_MODEL::OPENCV},
        {"OPENCV_FISHEYE", CAMERA_MODEL::OPENCV_FISHEYE},
        {"FULL_OPENCV", CAMERA_MODEL::FULL_OPENCV},
        {"FOV", CAMERA_MODEL::FOV},
        {"SIMPLE_RADIAL_FISHEYE", CAMERA_MODEL::SIMPLE_RADIAL_FISHEYE},
        {"RADIAL_FISHEYE", CAMERA_MODEL::RADIAL_FISHEYE},
        {"THIN_PRISM_FISHEYE", CAMERA_MODEL::THIN_PRISM_FISHEYE},
        {"EQUIRECTANGULAR", CAMERA_MODEL::EQUIRECTANGULAR},
        // Accept the legacy "SPHERICAL" name (written by the original
        // equirectangular COLMAP fork) as an alias for the same model.
        {"SPHERICAL", CAMERA_MODEL::EQUIRECTANGULAR}};

    // Placeholder focal length for equirectangular/spherical cameras. They carry
    // no real intrinsics; the rasterizer reinterprets K for EQUIRECTANGULAR
    // (focal := full image dimensions, principal point := tile offsets). This
    // value only keeps the FoV/intrinsics bookkeeping non-degenerate and mirrors
    // the convention used by the transforms.json loader.
    static constexpr float EQUIRECTANGULAR_DUMMY_FOCAL = 20.0f;

    static const char* camera_model_name(const int model_id) {
        switch (static_cast<CAMERA_MODEL>(model_id)) {
        case CAMERA_MODEL::SIMPLE_PINHOLE: return "SIMPLE_PINHOLE";
        case CAMERA_MODEL::PINHOLE: return "PINHOLE";
        case CAMERA_MODEL::SIMPLE_RADIAL: return "SIMPLE_RADIAL";
        case CAMERA_MODEL::RADIAL: return "RADIAL";
        case CAMERA_MODEL::OPENCV: return "OPENCV";
        case CAMERA_MODEL::OPENCV_FISHEYE: return "OPENCV_FISHEYE";
        case CAMERA_MODEL::FULL_OPENCV: return "FULL_OPENCV";
        case CAMERA_MODEL::FOV: return "FOV";
        case CAMERA_MODEL::SIMPLE_RADIAL_FISHEYE: return "SIMPLE_RADIAL_FISHEYE";
        case CAMERA_MODEL::RADIAL_FISHEYE: return "RADIAL_FISHEYE";
        case CAMERA_MODEL::THIN_PRISM_FISHEYE: return "THIN_PRISM_FISHEYE";
        // Write the canonical upstream name so exported reconstructions round-
        // trip with current COLMAP (which no longer accepts "SPHERICAL").
        case CAMERA_MODEL::EQUIRECTANGULAR: return "EQUIRECTANGULAR";
        default: return "UNDEFINED";
        }
    }

    constexpr float DISTORTION_ZERO_EPSILON = 1e-8f;

    static bool is_effectively_zero(const float value) {
        return std::abs(value) <= DISTORTION_ZERO_EPSILON;
    }

    static Tensor make_distortion_tensor(std::initializer_list<float> values) {
        const bool all_zero = std::all_of(values.begin(), values.end(), [](const float value) {
            return is_effectively_zero(value);
        });
        if (all_zero) {
            return Tensor::empty({0}, Device::CPU);
        }
        return Tensor::from_vector(std::vector<float>(values), {values.size()}, Device::CPU);
    }

    static std::unexpected<Error> make_ambiguous_image_reference_error(
        const fs::path& images_path,
        const std::string& image_name) {

        const fs::path image_rel_path = lfs::core::utf8_to_path(image_name);
        const std::string basename = lfs::core::path_to_utf8(image_rel_path.filename());

        return make_error(
            ErrorCode::INVALID_DATASET,
            std::format("COLMAP dataset contract violation: image '{}' is ambiguous under '{}': "
                        "multiple files share that basename in subdirectories. "
                        "Preserve the relative image path in COLMAP metadata, for example 'cam_a/{}' instead of '{}'.",
                        image_name,
                        lfs::core::path_to_utf8(images_path),
                        basename,
                        basename),
            images_path);
    }

    static std::unexpected<Error> make_ambiguous_mask_reference_error(
        const fs::path& base_path,
        const std::string& image_name) {

        return make_error(
            ErrorCode::INVALID_DATASET,
            std::format("COLMAP dataset contract violation: mask for image '{}' is ambiguous across the dataset "
                        "mask folders. Keep masks in the same relative subdirectories as the images, for example "
                        "'masks/{}', or rename them uniquely.",
                        image_name,
                        image_name),
            base_path);
    }

    struct BasenameLayoutInfo {
        size_t file_count = 0;
        std::vector<fs::path> sample_relative_paths;
    };

    static std::string format_relative_path_examples(const std::vector<fs::path>& relative_paths) {
        std::string formatted;
        for (size_t i = 0; i < relative_paths.size(); ++i) {
            if (i > 0) {
                formatted += ", ";
            }
            formatted += std::format("'{}'", lfs::core::path_to_utf8(relative_paths[i]));
        }
        return formatted;
    }

    static std::unordered_map<std::string, BasenameLayoutInfo>
    scan_image_basename_layout(const fs::path& images_path,
                               const LoadOptions& options = {}) {
        std::unordered_map<std::string, BasenameLayoutInfo> layout;

        if (!safe_is_directory(images_path)) {
            return layout;
        }

        std::error_code ec;
        size_t scanned_entries = 0;
        for (fs::recursive_directory_iterator it(
                 images_path,
                 fs::directory_options::skip_permission_denied,
                 ec),
             end;
             !ec && it != end;
             it.increment(ec)) {
            if (should_poll_cancel(scanned_entries)) {
                throw_if_load_cancel_requested(options, "COLMAP image layout scan cancelled");
            }
            ++scanned_entries;

            const auto& entry = *it;
            std::error_code file_ec;
            if (!entry.is_regular_file(file_ec) || file_ec || !is_image_file(entry.path())) {
                continue;
            }

            const fs::path relative_path = entry.path().lexically_relative(images_path);
            if (relative_path.empty()) {
                continue;
            }

            const std::string basename_key = detail::normalize_lookup_key(entry.path().filename());
            auto& info = layout[basename_key];
            ++info.file_count;
            if (info.sample_relative_paths.size() < 2) {
                info.sample_relative_paths.push_back(relative_path);
            }
        }

        return layout;
    }

    static std::unexpected<Error> make_nested_image_contract_error(
        const fs::path& images_path,
        const std::string& image_name,
        const BasenameLayoutInfo& basename_layout,
        const size_t metadata_reference_count) {

        const fs::path image_rel_path = lfs::core::utf8_to_path(image_name);
        const std::string basename = lfs::core::path_to_utf8(image_rel_path.filename());
        const std::string examples = format_relative_path_examples(basename_layout.sample_relative_paths);
        const std::string example_path = basename_layout.sample_relative_paths.empty()
                                             ? basename
                                             : lfs::core::path_to_utf8(basename_layout.sample_relative_paths.front());

        std::string message = std::format(
            "COLMAP dataset contract violation: image '{}' is referenced by basename only, but {} files with that "
            "basename exist under '{}': {}. Preserve the relative image path in COLMAP metadata, for example '{}' "
            "instead of '{}'.",
            image_name,
            basename_layout.file_count,
            lfs::core::path_to_utf8(images_path),
            examples,
            example_path,
            basename);

        if (metadata_reference_count != basename_layout.file_count) {
            message += std::format(
                " Metadata contains {} record(s) named '{}' while the dataset contains {} file(s) with that "
                "basename, so the export likely flattened or dropped subdirectory images.",
                metadata_reference_count,
                basename,
                basename_layout.file_count);
        }

        return make_error(ErrorCode::INVALID_DATASET, std::move(message), images_path);
    }

    static Result<void> validate_colmap_dataset_layout_impl(
        const fs::path& base,
        const std::string& images_folder,
        const std::vector<ImageData>& images,
        const LoadOptions& options = {}) {

        LOG_TIMER_DEBUG("COLMAP validate dataset layout");
        const fs::path images_path = base / lfs::core::utf8_to_path(images_folder);
        LOG_INFO("[COLMAP_LOAD] validate_layout images={} images_path='{}'",
                 images.size(),
                 lfs::core::path_to_utf8(images_path));
        if (!safe_is_directory(images_path)) {
            return make_error(ErrorCode::PATH_NOT_FOUND, "Images folder does not exist", images_path);
        }

        const auto basename_layout = scan_image_basename_layout(images_path, options);
        RecursiveFileCache image_cache(images_path, options.cancel_requested);
        MaskDirCache mask_cache(base, options.cancel_requested);
        DepthDirCache depth_cache(base, options.cancel_requested);

        std::unordered_map<std::string, size_t> basename_only_metadata_counts;
        basename_only_metadata_counts.reserve(images.size());

        for (size_t i = 0; i < images.size(); ++i) {
            if (should_poll_cancel(i)) {
                throw_if_load_cancel_requested(options, "COLMAP metadata validation cancelled");
            }
            const auto& image = images[i];
            const fs::path image_rel_path = lfs::core::utf8_to_path(image.name).lexically_normal();
            if (image_rel_path.parent_path().empty()) {
                const std::string basename_key = detail::normalize_lookup_key(image_rel_path.filename());
                ++basename_only_metadata_counts[basename_key];
            }
        }

        for (size_t i = 0; i < images.size(); ++i) {
            if (should_poll_cancel(i)) {
                throw_if_load_cancel_requested(options, "COLMAP dataset validation cancelled");
            }
            const auto& image = images[i];
            const fs::path image_rel_path = lfs::core::utf8_to_path(image.name).lexically_normal();
            const std::string basename_key = detail::normalize_lookup_key(image_rel_path.filename());

            if (image_rel_path.parent_path().empty()) {
                if (auto it = basename_layout.find(basename_key);
                    it != basename_layout.end() && it->second.file_count > 1) {
                    return make_nested_image_contract_error(
                        images_path,
                        image.name,
                        it->second,
                        basename_only_metadata_counts[basename_key]);
                }
            }

            if (auto image_lookup = image_cache.lookup(image_rel_path); image_lookup.found()) {
                if (auto mask_lookup = mask_cache.lookup(image.name); mask_lookup.ambiguous()) {
                    return make_ambiguous_mask_reference_error(base, image.name);
                }
                if (auto depth_lookup = depth_cache.lookup(image.name); depth_lookup.ambiguous()) {
                    return make_error(
                        ErrorCode::INVALID_DATASET,
                        std::format("Depth map for image '{}' is ambiguous across the dataset depth folders. "
                                    "Keep depth maps in the same relative subdirectories as the images or rename them uniquely.",
                                    image.name),
                        base);
                }
            } else if (image_lookup.ambiguous()) {
                return make_ambiguous_image_reference_error(images_path, image.name);
            } else {
                return make_error(
                    ErrorCode::PATH_NOT_FOUND,
                    std::format("Image '{}' was not found under '{}'",
                                image.name,
                                lfs::core::path_to_utf8(images_path)),
                    images_path / image_rel_path);
            }
        }

        return {};
    }

    // -----------------------------------------------------------------------------
    //  Binary-file loader
    // -----------------------------------------------------------------------------
    static std::unique_ptr<std::vector<char>>
    read_binary(const std::filesystem::path& p) {
        LOG_TRACE("Reading binary file: {}", lfs::core::path_to_utf8(p));
        std::ifstream f;
        if (!lfs::core::open_file_for_read(p, std::ios::binary | std::ios::ate, f)) {
            LOG_ERROR("Failed to open binary file: {}", lfs::core::path_to_utf8(p));
            throw std::runtime_error("Failed to open " + lfs::core::path_to_utf8(p));
        }

        const std::streampos end_position = f.tellg();
        LFS_ASSERT_MSG(end_position != std::streampos{-1},
                       std::format("Could not determine binary file size for {}",
                                   lfs::core::path_to_utf8(p)));
        const auto file_size = static_cast<std::streamoff>(end_position);
        LFS_ASSERT_MSG(file_size >= 0,
                       std::format("Binary file has a negative size: {}",
                                   lfs::core::path_to_utf8(p)));
        LFS_ASSERT_MSG(static_cast<uintmax_t>(file_size) <=
                           static_cast<uintmax_t>(std::numeric_limits<size_t>::max()),
                       std::format("binary file size must fit in size_t "
                                   "(path='{}', file_bytes={}, size_t_max={})",
                                   lfs::core::path_to_utf8(p), file_size,
                                   std::numeric_limits<size_t>::max()));
        LFS_ASSERT_MSG(static_cast<uintmax_t>(file_size) <=
                           static_cast<uintmax_t>(std::numeric_limits<std::streamsize>::max()),
                       std::format("binary file size must fit in one stream read "
                                   "(path='{}', file_bytes={}, streamsize_max={})",
                                   lfs::core::path_to_utf8(p), file_size,
                                   std::numeric_limits<std::streamsize>::max()));

        const auto sz = static_cast<std::streamsize>(file_size);
        auto buf = std::make_unique<std::vector<char>>(static_cast<size_t>(file_size));

        f.seekg(0, std::ios::beg);
        LFS_ASSERT_MSG(f.good(),
                       std::format("Could not seek binary file: {}",
                                   lfs::core::path_to_utf8(p)));
        if (sz > 0) {
            f.read(buf->data(), sz);
        }
        if (!f || f.gcount() != sz) {
            LOG_ERROR("Short read on binary file: {}", lfs::core::path_to_utf8(p));
            throw std::runtime_error("Short read on " + lfs::core::path_to_utf8(p));
        }
        LOG_TRACE("Read {} bytes from {}", sz, lfs::core::path_to_utf8(p));
        return buf;
    }

    // -----------------------------------------------------------------------------
    //  Helper to scale camera intrinsics
    // -----------------------------------------------------------------------------
    static void scale_camera_intrinsics(CAMERA_MODEL model, std::vector<float>& params, float factor) {
        switch (model) {
        case CAMERA_MODEL::SIMPLE_PINHOLE:
            params[0] /= factor; // f
            params[1] /= factor; // cx
            params[2] /= factor; // cy
            break;

        case CAMERA_MODEL::PINHOLE:
            params[0] /= factor; // fx
            params[1] /= factor; // fy
            params[2] /= factor; // cx
            params[3] /= factor; // cy
            break;

        case CAMERA_MODEL::SIMPLE_RADIAL:
        case CAMERA_MODEL::RADIAL:
            params[0] /= factor; // f
            params[1] /= factor; // cx
            params[2] /= factor; // cy
            break;

        case CAMERA_MODEL::OPENCV:
        case CAMERA_MODEL::OPENCV_FISHEYE:
        case CAMERA_MODEL::FULL_OPENCV:
            params[0] /= factor; // fx
            params[1] /= factor; // fy
            params[2] /= factor; // cx
            params[3] /= factor; // cy
            break;

        case CAMERA_MODEL::SIMPLE_RADIAL_FISHEYE:
        case CAMERA_MODEL::RADIAL_FISHEYE:
            params[0] /= factor; // f
            params[1] /= factor; // cx
            params[2] /= factor; // cy
            break;

        case CAMERA_MODEL::FOV:
            params[0] /= factor; // fx
            params[1] /= factor; // fy
            params[2] /= factor; // cx
            params[3] /= factor; // cy
            break;

        case CAMERA_MODEL::THIN_PRISM_FISHEYE:
            params[0] /= factor; // fx
            params[1] /= factor; // fy
            params[2] /= factor; // cx
            params[3] /= factor; // cy
            break;

        case CAMERA_MODEL::EQUIRECTANGULAR:
            params[0] /= factor; // width
            params[1] /= factor; // height
            break;

        default:
            LOG_WARN("Unknown camera model for scaling");
            if (params.size() >= 4) {
                params[2] /= factor; // cx
                params[3] /= factor; // cy
            }
            break;
        }
    }

    // -----------------------------------------------------------------------------
    //  Helper to extract scale factor from folder name
    // -----------------------------------------------------------------------------
    static float extract_scale_from_folder(const std::string& folder_name) {
        size_t underscore_pos = folder_name.rfind('_');
        if (underscore_pos != std::string::npos) {
            std::string suffix = folder_name.substr(underscore_pos + 1);
            try {
                float factor = std::stof(suffix);
                if (factor > 0 && factor <= 16) {
                    LOG_DEBUG("Extracted scale factor {} from folder name", factor);
                    return factor;
                }
            } catch (...) {
            }
        }
        return 1.0f;
    }

    // -----------------------------------------------------------------------------
    //  images.bin
    // -----------------------------------------------------------------------------
    std::vector<ImageData> read_images_binary(const std::filesystem::path& file_path,
                                              const LoadOptions& options = {}) {
        LOG_TIMER_TRACE("Read images.bin");
        auto buf_owner = read_binary(file_path);
        const char* cur = buf_owner->data();
        const char* end = cur + buf_owner->size();

        const uint64_t n_images = read_u64(cur, end, "images.bin image count");
        constexpr size_t MIN_IMAGE_RECORD_BYTES = sizeof(uint32_t) + 7 * sizeof(double) +
                                                  sizeof(uint32_t) + 1 + sizeof(uint64_t);
        LFS_ASSERT_MSG(n_images > 0,
                       std::format("images.bin must declare at least one image "
                                   "(image_count={})",
                                   n_images));
        LFS_ASSERT_MSG(n_images <= static_cast<uint64_t>(end - cur) / MIN_IMAGE_RECORD_BYTES,
                       std::format("images.bin image count exceeds the remaining payload "
                                   "(image_count={}, remaining_bytes={}, minimum_record_bytes={})",
                                   n_images, static_cast<size_t>(end - cur), MIN_IMAGE_RECORD_BYTES));
        LOG_DEBUG("Reading {} images from binary file", n_images);
        std::vector<ImageData> images;
        images.reserve(n_images);
        std::unordered_set<uint32_t> image_ids;
        std::unordered_set<std::string> image_names;

        for (uint64_t i = 0; i < n_images; ++i) {
            if (should_poll_cancel(static_cast<size_t>(i))) {
                throw_if_load_cancel_requested(options, "COLMAP image metadata load cancelled");
            }
            ImageData img;
            img.image_id = read_u32(cur, end, "images.bin image id");
            LFS_ASSERT_MSG(img.image_id != 0 && image_ids.insert(img.image_id).second,
                           std::format("images.bin image ids must be non-zero and unique "
                                       "(record_index={}, image_id={}, previously_seen={})",
                                       i, img.image_id, image_ids.contains(img.image_id)));

            // Read quaternion [w, x, y, z]
            for (int k = 0; k < 4; ++k) {
                img.qvec[k] = static_cast<float>(read_f64(cur, end, "images.bin quaternion"));
            }

            // Read translation [x, y, z]
            for (int k = 0; k < 3; ++k) {
                img.tvec[k] = static_cast<float>(read_f64(cur, end, "images.bin translation"));
            }

            img.camera_id = read_u32(cur, end, "images.bin camera id");
            LFS_ASSERT_MSG(img.camera_id != 0,
                           std::format("images.bin camera ids must be non-zero "
                                       "(record_index={}, image_id={}, camera_id={})",
                                       i, img.image_id, img.camera_id));

            const void* terminator = std::memchr(cur, '\0', static_cast<size_t>(end - cur));
            LFS_ASSERT_MSG(terminator != nullptr,
                           std::format("images.bin image name must be null-terminated "
                                       "(record_index={}, image_id={}, remaining_bytes={})",
                                       i, img.image_id, static_cast<size_t>(end - cur)));
            const char* name_end = static_cast<const char*>(terminator);
            img.name.assign(cur, name_end);
            cur = name_end + 1;
            LFS_ASSERT_MSG(!img.name.empty() && image_names.insert(img.name).second,
                           std::format("images.bin image names must be non-empty and unique "
                                       "(record_index={}, image_id={}, name='{}')",
                                       i, img.image_id, img.name));

            const uint64_t npts = read_u64(cur, end, "images.bin point2D count");
            constexpr size_t POINT2D_RECORD_BYTES = 2 * sizeof(double) + sizeof(uint64_t);
            LFS_ASSERT_MSG(npts <= static_cast<uint64_t>(end - cur) / POINT2D_RECORD_BYTES,
                           std::format("images.bin point2D count exceeds the remaining payload "
                                       "(image_id={}, point_count={}, remaining_bytes={}, record_bytes={})",
                                       img.image_id, npts, static_cast<size_t>(end - cur),
                                       POINT2D_RECORD_BYTES));
            img.points2D.reserve(npts);
            for (uint64_t j = 0; j < npts; ++j) {
                ImagePoint2D point;
                point.x = read_f64(cur, end, "images.bin point2D x");
                point.y = read_f64(cur, end, "images.bin point2D y");
                point.point3D_id = read_u64(cur, end, "images.bin point3D id");
                LFS_ASSERT_MSG(std::isfinite(point.x) && std::isfinite(point.y),
                               std::format("images.bin point2D coordinates must be finite "
                                           "(image_id={}, point_index={}, x={}, y={})",
                                           img.image_id, j, point.x, point.y));
                img.points2D.push_back(point);
            }

            const double qnorm = std::sqrt(
                static_cast<double>(img.qvec[0]) * img.qvec[0] +
                static_cast<double>(img.qvec[1]) * img.qvec[1] +
                static_cast<double>(img.qvec[2]) * img.qvec[2] +
                static_cast<double>(img.qvec[3]) * img.qvec[3]);
            LFS_ASSERT_MSG(std::isfinite(qnorm) && std::abs(qnorm - 1.0) <= 1e-4,
                           std::format("images.bin quaternion must be finite and normalized "
                                       "(image_id={}, norm={}, q=[{},{},{},{}])",
                                       img.image_id, qnorm, img.qvec[0], img.qvec[1],
                                       img.qvec[2], img.qvec[3]));
            LFS_ASSERT_MSG(std::ranges::all_of(img.tvec, [](const float value) { return std::isfinite(value); }),
                           std::format("images.bin translation must be finite "
                                       "(image_id={}, t=[{},{},{}], record_index={})",
                                       img.image_id, img.tvec[0], img.tvec[1], img.tvec[2], i));

            images.push_back(std::move(img));
        }

        if (cur != end) {
            LOG_ERROR("images.bin has trailing bytes");
            throw std::runtime_error("images.bin: trailing bytes");
        }
        return images;
    }

    // -----------------------------------------------------------------------------
    //  cameras.bin
    // -----------------------------------------------------------------------------
    std::unordered_map<uint32_t, CameraDataIntermediate>
    read_cameras_binary(const std::filesystem::path& file_path,
                        float scale_factor = 1.0f,
                        const LoadOptions& options = {}) {
        LOG_TIMER_TRACE("Read cameras.bin");
        LFS_ASSERT_MSG(std::isfinite(scale_factor) && scale_factor > 0.0f,
                       std::format("COLMAP camera scale factor must be finite and positive "
                                   "(scale_factor={}, path='{}')",
                                   scale_factor, lfs::core::path_to_utf8(file_path)));
        auto buf_owner = read_binary(file_path);
        const char* cur = buf_owner->data();
        const char* end = cur + buf_owner->size();

        const uint64_t n_cams = read_u64(cur, end, "cameras.bin camera count");
        constexpr size_t MIN_CAMERA_RECORD_BYTES = 2 * sizeof(uint32_t) +
                                                   2 * sizeof(uint64_t);
        LFS_ASSERT_MSG(n_cams > 0,
                       std::format("cameras.bin must declare at least one camera "
                                   "(camera_count={}, path='{}', remaining_bytes={})",
                                   n_cams, lfs::core::path_to_utf8(file_path),
                                   static_cast<size_t>(end - cur)));
        LFS_ASSERT_MSG(n_cams <= static_cast<uint64_t>(end - cur) / MIN_CAMERA_RECORD_BYTES,
                       std::format("cameras.bin camera count must fit the remaining payload "
                                   "(camera_count={}, remaining_bytes={}, minimum_record_bytes={}, "
                                   "path='{}')",
                                   n_cams, static_cast<size_t>(end - cur),
                                   MIN_CAMERA_RECORD_BYTES, lfs::core::path_to_utf8(file_path)));
        LOG_DEBUG("Reading {} cameras from binary file{}", n_cams,
                  scale_factor != 1.0f ? std::format(" with scale factor {}", scale_factor) : "");

        std::unordered_map<uint32_t, CameraDataIntermediate> cams;
        cams.reserve(n_cams);

        for (uint64_t i = 0; i < n_cams; ++i) {
            if (should_poll_cancel(static_cast<size_t>(i))) {
                throw_if_load_cancel_requested(options, "COLMAP camera metadata load cancelled");
            }
            CameraDataIntermediate cam;
            cam.camera_id = read_u32(cur, end, "cameras.bin camera id");
            cam.model_id = read_i32(cur, end, "cameras.bin model id");
            const uint64_t width = read_u64(cur, end, "cameras.bin width");
            const uint64_t height = read_u64(cur, end, "cameras.bin height");
            LFS_ASSERT_MSG(cam.camera_id != 0,
                           std::format("cameras.bin camera ids must be non-zero "
                                       "(camera_id={}, record_index={}, camera_count={}, path='{}')",
                                       cam.camera_id, i, n_cams,
                                       lfs::core::path_to_utf8(file_path)));
            LFS_ASSERT_MSG(width > 0 && height > 0 &&
                               width <= static_cast<uint64_t>(std::numeric_limits<int>::max()) &&
                               height <= static_cast<uint64_t>(std::numeric_limits<int>::max()),
                           std::format("cameras.bin dimensions must be positive and fit in int "
                                       "(camera_id={}, width={}, height={}, int_max={}, record_index={})",
                                       cam.camera_id, width, height,
                                       std::numeric_limits<int>::max(), i));
            cam.width = static_cast<int>(width);
            cam.height = static_cast<int>(height);

            if (scale_factor != 1.0f) {
                cam.width = static_cast<int>(cam.width / scale_factor);
                cam.height = static_cast<int>(cam.height / scale_factor);
            }
            LFS_ASSERT_MSG(cam.width > 0 && cam.height > 0,
                           std::format("scaled COLMAP camera dimensions must remain positive "
                                       "(camera_id={}, scaled_width={}, scaled_height={}, "
                                       "scale_factor={}, source_width={}, source_height={})",
                                       cam.camera_id, cam.width, cam.height, scale_factor,
                                       width, height));

            auto it = camera_model_ids.find(cam.model_id);
            if (it == camera_model_ids.end() || it->second.second < 0) {
                LOG_ERROR("Unsupported camera-model id: {}", cam.model_id);
                throw std::runtime_error("Unsupported camera-model id");
            }

            int32_t param_cnt = it->second.second;
            cam.params.resize(param_cnt);

            for (int j = 0; j < param_cnt; j++) {
                cam.params[j] = static_cast<float>(read_f64(cur, end, "cameras.bin parameter"));
                LFS_ASSERT_MSG(std::isfinite(cam.params[j]),
                               std::format("cameras.bin camera parameters must be finite "
                                           "(camera_id={}, parameter_index={}, value={}, "
                                           "parameter_count={})",
                                           cam.camera_id, j, cam.params[j], param_cnt));
            }

            if (scale_factor != 1.0f) {
                scale_camera_intrinsics(it->second.first, cam.params, scale_factor);
            }

            const auto [_, inserted] = cams.emplace(cam.camera_id, std::move(cam));
            LFS_ASSERT_MSG(inserted,
                           std::format("cameras.bin camera ids must be unique "
                                       "(camera_id={}, record_index={}, parsed_unique_count={})",
                                       cam.camera_id, i, cams.size()));
        }

        if (cur != end) {
            LOG_ERROR("cameras.bin has trailing bytes");
            throw std::runtime_error("cameras.bin: trailing bytes");
        }
        return cams;
    }

    // -----------------------------------------------------------------------------
    //  points3D.bin
    // -----------------------------------------------------------------------------
    std::vector<Point3DData> read_point3D_binary_records(const std::filesystem::path& file_path,
                                                         const LoadOptions& options = {}) {
        LOG_TIMER_TRACE("Read points3D.bin");
        auto buf_owner = read_binary(file_path);
        const char* cur = buf_owner->data();
        const char* end = cur + buf_owner->size();

        const uint64_t N = read_u64(cur, end, "points3D.bin point count");
        constexpr size_t MIN_POINT3D_RECORD_BYTES = sizeof(uint64_t) + 3 * sizeof(double) + 3 +
                                                    sizeof(double) + sizeof(uint64_t);
        LFS_ASSERT_MSG(N <= static_cast<uint64_t>(end - cur) / MIN_POINT3D_RECORD_BYTES,
                       std::format("points3D.bin point count must fit the remaining payload "
                                   "(point_count={}, remaining_bytes={}, minimum_record_bytes={}, "
                                   "path='{}')",
                                   N, static_cast<size_t>(end - cur),
                                   MIN_POINT3D_RECORD_BYTES, lfs::core::path_to_utf8(file_path)));
        LOG_DEBUG("Reading {} 3D points from binary file", N);

        std::vector<Point3DData> points;
        points.reserve(N);
        std::unordered_set<uint64_t> point_ids;

        for (uint64_t i = 0; i < N; ++i) {
            if (should_poll_cancel(static_cast<size_t>(i))) {
                throw_if_load_cancel_requested(options, "COLMAP point cloud load cancelled");
            }
            Point3DData point;
            point.point3D_id = read_u64(cur, end, "points3D.bin point id");
            LFS_ASSERT_MSG(point.point3D_id != 0 && point_ids.insert(point.point3D_id).second,
                           std::format("points3D.bin point ids must be non-zero and unique "
                                       "(point_id={}, record_index={}, point_count={}, "
                                       "unique_count={})",
                                       point.point3D_id, i, N, point_ids.size()));

            point.xyz[0] = read_f64(cur, end, "points3D.bin x");
            point.xyz[1] = read_f64(cur, end, "points3D.bin y");
            point.xyz[2] = read_f64(cur, end, "points3D.bin z");
            LFS_ASSERT_MSG(std::ranges::all_of(point.xyz, [](const double value) { return std::isfinite(value); }),
                           std::format("points3D.bin coordinates must be finite "
                                       "(point_id={}, xyz=[{},{},{}], record_index={})",
                                       point.point3D_id, point.xyz[0], point.xyz[1],
                                       point.xyz[2], i));

            LFS_ASSERT_MSG(static_cast<size_t>(end - cur) >= 3,
                           std::format("points3D.bin color payload requires three remaining bytes "
                                       "(point_id={}, remaining_bytes={}, required_bytes=3, "
                                       "record_index={})",
                                       point.point3D_id, static_cast<size_t>(end - cur), i));
            point.color[0] = static_cast<uint8_t>(*cur++);
            point.color[1] = static_cast<uint8_t>(*cur++);
            point.color[2] = static_cast<uint8_t>(*cur++);

            point.error = read_f64(cur, end, "points3D.bin error");
            LFS_ASSERT_MSG(std::isfinite(point.error) && point.error >= 0.0,
                           std::format("points3D.bin reprojection error must be finite and non-negative "
                                       "(point_id={}, error={}, record_index={})",
                                       point.point3D_id, point.error, i));
            const uint64_t track_len = read_u64(cur, end, "points3D.bin track length");
            LFS_ASSERT_MSG(track_len <= static_cast<uint64_t>(end - cur) /
                                            (2 * sizeof(uint32_t)),
                           std::format("points3D.bin track length must fit the remaining payload "
                                       "(point_id={}, track_length={}, remaining_bytes={}, "
                                       "track_record_bytes={})",
                                       point.point3D_id, track_len,
                                       static_cast<size_t>(end - cur), 2 * sizeof(uint32_t)));
            point.track_count = static_cast<size_t>(track_len);
            point.track.reserve(track_len);
            for (uint64_t j = 0; j < track_len; ++j) {
                Point3DTrackElement track;
                track.image_id = read_u32(cur, end, "points3D.bin track image id");
                track.point2D_idx = read_u32(cur, end, "points3D.bin track point index");
                LFS_ASSERT_MSG(track.image_id != 0,
                               std::format("points3D.bin track image ids must be non-zero "
                                           "(point_id={}, image_id={}, point2D_index={}, "
                                           "track_index={}, track_length={})",
                                           point.point3D_id, track.image_id, track.point2D_idx,
                                           j, track_len));
                point.track.push_back(track);
            }
            points.push_back(std::move(point));
        }

        if (cur != end) {
            LOG_ERROR("points3D.bin has trailing bytes");
            throw std::runtime_error("points3D.bin: trailing bytes");
        }

        return points;
    }

    PointCloud point3D_records_to_point_cloud(const std::vector<Point3DData>& points) {
        const uint64_t N = points.size();
        if (N == 0)
            return PointCloud();

        std::vector<float> positions(N * 3);
        std::vector<uint8_t> colors(N * 3);

        for (uint64_t i = 0; i < N; ++i) {
            positions[i * 3 + 0] = static_cast<float>(points[i].xyz[0]);
            positions[i * 3 + 1] = static_cast<float>(points[i].xyz[1]);
            positions[i * 3 + 2] = static_cast<float>(points[i].xyz[2]);
            colors[i * 3 + 0] = points[i].color[0];
            colors[i * 3 + 1] = points[i].color[1];
            colors[i * 3 + 2] = points[i].color[2];
        }

        Tensor means = Tensor::from_vector(positions, {N, 3}, Device::CUDA);
        Tensor colors_tensor = Tensor::from_blob(colors.data(), {N, 3}, Device::CPU, DataType::UInt8)
                                   .to(Device::CUDA)
                                   .contiguous();

        return PointCloud(std::move(means), std::move(colors_tensor));
    }

    ColmapPointCloudLoadStats point3D_records_to_point_cloud_with_stats(
        std::vector<Point3DData> points,
        const LoadOptions& options) {
        ColmapPointCloudLoadStats result{
            .point_cloud = {},
            .total_points = points.size(),
            .points_after_filtering = points.size(),
            .track_filter_applied = options.min_track_length > 0,
        };

        const int min_track_length = options.min_track_length;
        if (min_track_length > 0) {
            const auto min_track = static_cast<size_t>(min_track_length);
            std::erase_if(points, [min_track](const Point3DData& point) {
                return point.track_count < min_track;
            });
            result.points_after_filtering = points.size();
        }

        result.point_cloud = point3D_records_to_point_cloud(points);
        return result;
    }

    PointCloud read_point3D_binary(const std::filesystem::path& file_path,
                                   const LoadOptions& options = {}) {
        return point3D_records_to_point_cloud(read_point3D_binary_records(file_path, options));
    }

    // -----------------------------------------------------------------------------
    //  Text-file helpers
    // -----------------------------------------------------------------------------
    std::vector<std::string> read_text_file(const std::filesystem::path& file_path,
                                            const LoadOptions& options = {}) {
        LOG_TRACE("Reading text file: {}", lfs::core::path_to_utf8(file_path));
        const auto start = std::chrono::high_resolution_clock::now();
        std::error_code file_size_ec;
        const auto byte_size = fs::file_size(file_path, file_size_ec);
        std::ifstream file;
        if (!lfs::core::open_file_for_read(file_path, file)) {
            LOG_ERROR("Failed to open text file: {}", lfs::core::path_to_utf8(file_path));
            throw std::runtime_error("Failed to open " + lfs::core::path_to_utf8(file_path));
        }

        std::vector<std::string> lines;
        std::string line;
        while (std::getline(file, line)) {
            if (should_poll_cancel(lines.size())) {
                throw_if_load_cancel_requested(options, "COLMAP text metadata load cancelled");
            }
            if (line.starts_with("#"))
                continue;
            if (!line.empty() && line.back() == '\r')
                line.pop_back();
            // Skip empty lines
            if (line.empty())
                continue;
            lines.push_back(line);
        }

        if (lines.empty()) {
            LOG_ERROR("File is empty: {}", lfs::core::path_to_utf8(file_path));
            throw std::runtime_error("File is empty");
        }

        if (lines.back().empty())
            lines.pop_back();

        LOG_TRACE("Read {} lines from text file", lines.size());
        LOG_INFO("[COLMAP_LOAD] read_text_file file='{}' bytes={} data_lines={} elapsed_ms={:.2f}",
                 lfs::core::path_to_utf8(file_path),
                 file_size_ec ? std::string("unknown") : std::format("{}", byte_size),
                 lines.size(),
                 elapsed_ms(start));
        return lines;
    }

    std::vector<std::string> split_string(const std::string& s, char delimiter) {
        std::vector<std::string> tokens;
        size_t start = 0;
        size_t end = s.find(delimiter);

        while (end != std::string::npos) {
            tokens.push_back(s.substr(start, end - start));
            start = end + 1;
            end = s.find(delimiter, start);
        }
        tokens.push_back(s.substr(start));

        return tokens;
    }

    bool parse_image_metadata_line(const std::string& line, ImageData& img) {
        std::istringstream iss(line);
        if (!(iss >> img.image_id >> img.qvec[0] >> img.qvec[1] >> img.qvec[2] >> img.qvec[3] >> img.tvec[0] >> img.tvec[1] >> img.tvec[2] >> img.camera_id)) {
            return false;
        }

        iss >> std::ws;
        if (!std::getline(iss, img.name) || img.name.empty()) {
            return false;
        }

        const auto dot_pos = img.name.rfind('.');
        if (dot_pos == std::string::npos || dot_pos == img.name.size() - 1) {
            return false;
        }
        if (!std::isalpha(static_cast<unsigned char>(img.name[dot_pos + 1]))) {
            return false;
        }

        LFS_ASSERT_MSG(img.image_id != 0,
                       std::format("COLMAP image id must be non-zero "
                                   "(image_id={}, name='{}')",
                                   img.image_id, img.name));
        LFS_ASSERT_MSG(img.camera_id != 0,
                       std::format("COLMAP image camera id must be non-zero "
                                   "(image_id={}, camera_id={}, name='{}')",
                                   img.image_id, img.camera_id, img.name));
        const double qnorm = std::sqrt(
            static_cast<double>(img.qvec[0]) * img.qvec[0] +
            static_cast<double>(img.qvec[1]) * img.qvec[1] +
            static_cast<double>(img.qvec[2]) * img.qvec[2] +
            static_cast<double>(img.qvec[3]) * img.qvec[3]);
        LFS_ASSERT_MSG(std::isfinite(qnorm) && std::abs(qnorm - 1.0) <= 1e-4,
                       std::format("COLMAP image quaternion must be finite and normalized "
                                   "(image_id={}, norm={}, q=[{},{},{},{}])",
                                   img.image_id, qnorm, img.qvec[0], img.qvec[1],
                                   img.qvec[2], img.qvec[3]));
        LFS_ASSERT_MSG(std::ranges::all_of(img.tvec, [](const float value) { return std::isfinite(value); }),
                       std::format("COLMAP image translation must be finite "
                                   "(image_id={}, name='{}', t=[{},{},{}])",
                                   img.image_id, img.name, img.tvec[0], img.tvec[1], img.tvec[2]));
        return true;
    }

    bool looks_like_image_metadata_line(const std::string& line) {
        if (line.size() > IMAGE_METADATA_PROBE_LIMIT) {
            return false;
        }

        ImageData img;
        return parse_image_metadata_line(line, img);
    }

    bool read_next_short_metadata_line_or_skip(std::ifstream& file,
                                               std::string& pending_line,
                                               size_t& file_lines) {
        pending_line.clear();

        std::string probe;
        probe.reserve(256);

        char ch = '\0';
        bool read_any = false;
        while (file.get(ch)) {
            read_any = true;
            if (ch == '\n') {
                break;
            }
            if (probe.size() >= IMAGE_METADATA_PROBE_LIMIT) {
                file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                break;
            }
            probe.push_back(ch);
        }

        if (!read_any) {
            return false;
        }

        ++file_lines;
        if (!probe.empty() && probe.back() == '\r') {
            probe.pop_back();
        }

        if (probe.empty() || probe.starts_with("#") || !looks_like_image_metadata_line(probe)) {
            return false;
        }

        pending_line = std::move(probe);
        return true;
    }

    uint64_t parse_point3D_id_token(const std::string& token) {
        if (token == "-1") {
            return INVALID_POINT3D_ID;
        }
        return std::stoull(token);
    }

    std::vector<ImagePoint2D> parse_points2D_text_line(const std::string& line) {
        std::istringstream iss(line);
        std::vector<ImagePoint2D> points;
        while (true) {
            double x = 0.0;
            if (!(iss >> x)) {
                break;
            }
            double y = 0.0;
            std::string point_id;
            LFS_ASSERT_MSG(static_cast<bool>(iss >> y >> point_id),
                           std::format("COLMAP points2D line must contain x/y/id triples "
                                       "(parsed_x={}, line='{}')",
                                       x, line));
            LFS_ASSERT_MSG(std::isfinite(x) && std::isfinite(y),
                           std::format("COLMAP points2D coordinates must be finite "
                                       "(point_index={}, x={}, y={}, point3D_id_token='{}')",
                                       points.size(), x, y, point_id));
            points.push_back(ImagePoint2D{
                .x = x,
                .y = y,
                .point3D_id = parse_point3D_id_token(point_id),
            });
        }
        return points;
    }

    // -----------------------------------------------------------------------------
    //  images.txt
    // -----------------------------------------------------------------------------
    std::vector<ImageData> read_images_text(const std::filesystem::path& file_path,
                                            const LoadOptions& options = {},
                                            const bool parse_points2d = true) {
        LOG_TIMER_TRACE("Read images.txt");
        auto lines = read_text_file(file_path, options);
        const auto parse_start = std::chrono::high_resolution_clock::now();

        std::vector<ImageData> images;
        images.reserve(lines.size());
        std::unordered_set<uint32_t> image_ids;
        std::unordered_set<std::string> image_names;
        size_t total_points2d = 0;

        for (size_t line_idx = 0; line_idx < lines.size(); ++line_idx) {
            if (should_poll_cancel(line_idx)) {
                throw_if_load_cancel_requested(options, "COLMAP image metadata parse cancelled");
            }
            const auto& line = lines[line_idx];

            ImageData img;
            if (!parse_image_metadata_line(line, img)) {
                continue;
            }

            if (line_idx + 1 < lines.size()) {
                if (!parse_points2d) {
                    if (!looks_like_image_metadata_line(lines[line_idx + 1])) {
                        ++line_idx;
                    }
                } else {
                    ImageData maybe_next_image;
                    if (!parse_image_metadata_line(lines[line_idx + 1], maybe_next_image)) {
                        img.points2D = parse_points2D_text_line(lines[line_idx + 1]);
                        total_points2d += img.points2D.size();
                        ++line_idx;
                    }
                }
            }

            LFS_ASSERT_MSG(image_ids.insert(img.image_id).second,
                           std::format("images.txt image ids must be unique "
                                       "(image_id={}, name='{}', source_record={})",
                                       img.image_id, img.name, line_idx + 1));
            LFS_ASSERT_MSG(image_names.insert(img.name).second,
                           std::format("images.txt image names must be unique "
                                       "(name='{}', image_id={}, source_record={})",
                                       img.name, img.image_id, line_idx + 1));
            images.push_back(std::move(img));
        }

        if (images.empty()) {
            LOG_ERROR("No valid images found in {}", lfs::core::path_to_utf8(file_path));
            throw std::runtime_error("No valid images in images.txt");
        }

        LOG_INFO("[COLMAP_LOAD] parse images.txt images={} points2D={} parse_points2D={} source_lines={} elapsed_ms={:.2f}",
                 images.size(),
                 total_points2d,
                 parse_points2d,
                 lines.size(),
                 elapsed_ms(parse_start));
        LOG_DEBUG("Read {} images from text file", images.size());
        return images;
    }

    std::vector<ImageData> read_images_text_camera_metadata_only(const std::filesystem::path& file_path,
                                                                 const LoadOptions& options = {}) {
        LOG_TIMER_TRACE("Read images.txt camera metadata");
        const auto start = std::chrono::high_resolution_clock::now();
        std::error_code file_size_ec;
        const auto byte_size = fs::file_size(file_path, file_size_ec);

        std::ifstream file;
        if (!lfs::core::open_file_for_read(file_path, file)) {
            LOG_ERROR("Failed to open text file: {}", lfs::core::path_to_utf8(file_path));
            throw std::runtime_error("Failed to open " + lfs::core::path_to_utf8(file_path));
        }

        std::vector<ImageData> images;
        std::unordered_set<uint32_t> image_ids;
        std::unordered_set<std::string> image_names;
        std::string line;
        std::string pending_line;
        bool has_pending_line = false;
        size_t file_lines = 0;

        auto read_next_line = [&]() {
            if (has_pending_line) {
                line = std::move(pending_line);
                pending_line.clear();
                has_pending_line = false;
                return true;
            }

            if (!std::getline(file, line)) {
                return false;
            }

            ++file_lines;
            return true;
        };

        while (read_next_line()) {
            if (should_poll_cancel(file_lines)) {
                throw_if_load_cancel_requested(options, "COLMAP image metadata parse cancelled");
            }

            if (line.starts_with("#")) {
                continue;
            }
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (line.empty()) {
                continue;
            }

            ImageData img;
            if (!parse_image_metadata_line(line, img)) {
                continue;
            }

            LFS_ASSERT_MSG(image_ids.insert(img.image_id).second,
                           std::format("images.txt image ids must be unique "
                                       "(image_id={}, name='{}', source_line={})",
                                       img.image_id, img.name, file_lines));
            LFS_ASSERT_MSG(image_names.insert(img.name).second,
                           std::format("images.txt image names must be unique "
                                       "(name='{}', image_id={}, source_line={})",
                                       img.name, img.image_id, file_lines));
            images.push_back(std::move(img));
            has_pending_line = read_next_short_metadata_line_or_skip(file, pending_line, file_lines);
        }

        if (images.empty()) {
            LOG_ERROR("No valid images found in {}", lfs::core::path_to_utf8(file_path));
            throw std::runtime_error("No valid images in images.txt");
        }

        LOG_INFO("[COLMAP_LOAD] parse images.txt camera_metadata_fast images={} file_lines={} bytes={} elapsed_ms={:.2f}",
                 images.size(),
                 file_lines,
                 file_size_ec ? std::string("unknown") : std::format("{}", byte_size),
                 elapsed_ms(start));
        LOG_DEBUG("Read {} images from text file", images.size());
        return images;
    }

    // -----------------------------------------------------------------------------
    //  cameras.txt
    // -----------------------------------------------------------------------------
    std::unordered_map<uint32_t, CameraDataIntermediate>
    read_cameras_text(const std::filesystem::path& file_path,
                      float scale_factor = 1.0f,
                      const LoadOptions& options = {}) {
        LOG_TIMER_TRACE("Read cameras.txt");
        LFS_ASSERT_MSG(std::isfinite(scale_factor) && scale_factor > 0.0f,
                       std::format("COLMAP camera scale factor must be finite and positive "
                                   "(scale_factor={}, path='{}')",
                                   scale_factor, lfs::core::path_to_utf8(file_path)));
        auto lines = read_text_file(file_path, options);

        LOG_DEBUG("Reading {} cameras from text file{}", lines.size(),
                  scale_factor != 1.0f ? std::format(" with scale factor {}", scale_factor) : "");

        std::unordered_map<uint32_t, CameraDataIntermediate> cams;

        for (size_t line_idx = 0; line_idx < lines.size(); ++line_idx) {
            if (should_poll_cancel(line_idx)) {
                throw_if_load_cancel_requested(options, "COLMAP camera metadata parse cancelled");
            }
            const auto& line = lines[line_idx];
            CameraDataIntermediate cam;
            std::string model_name;
            uint64_t camera_id = 0;
            uint64_t width = 0;
            uint64_t height = 0;
            std::istringstream camera_line(line);
            LFS_ASSERT_MSG(static_cast<bool>(camera_line >> camera_id >> model_name >> width >> height),
                           std::format("cameras.txt record must contain id/model/width/height "
                                       "(source_line={}, text='{}')",
                                       line_idx + 1, line));
            LFS_ASSERT_MSG(camera_id > 0 && camera_id <= std::numeric_limits<uint32_t>::max(),
                           std::format("cameras.txt camera id must be non-zero and fit in uint32 "
                                       "(camera_id={}, source_line={}, max={})",
                                       camera_id, line_idx + 1, std::numeric_limits<uint32_t>::max()));
            LFS_ASSERT_MSG(width > 0 && height > 0 &&
                               width <= static_cast<uint64_t>(std::numeric_limits<int>::max()) &&
                               height <= static_cast<uint64_t>(std::numeric_limits<int>::max()),
                           std::format("cameras.txt dimensions must be positive and fit in int "
                                       "(width={}, height={}, source_line={}, max={})",
                                       width, height, line_idx + 1, std::numeric_limits<int>::max()));
            LFS_ASSERT_MSG(camera_model_names.contains(model_name),
                           std::format("Unknown COLMAP camera model '{}'", model_name));

            cam.camera_id = static_cast<uint32_t>(camera_id);
            cam.model_id = static_cast<int>(camera_model_names.at(model_name));
            cam.width = static_cast<int>(width);
            cam.height = static_cast<int>(height);

            double parameter = 0.0;
            while (camera_line >> parameter) {
                LFS_ASSERT_MSG(std::isfinite(parameter) &&
                                   std::abs(parameter) <= std::numeric_limits<float>::max(),
                               std::format("cameras.txt camera parameters must be finite Float32 values "
                                           "(camera_id={}, parameter_index={}, value={})",
                                           camera_id, cam.params.size(), parameter));
                cam.params.push_back(static_cast<float>(parameter));
            }

            if (scale_factor != 1.0f) {
                cam.width = static_cast<int>(cam.width / scale_factor);
                cam.height = static_cast<int>(cam.height / scale_factor);
            }
            LFS_ASSERT_MSG(cam.width > 0 && cam.height > 0,
                           std::format("scaled COLMAP camera dimensions must remain positive "
                                       "(camera_id={}, width={}, height={}, scale_factor={})",
                                       cam.camera_id, cam.width, cam.height, scale_factor));

            auto it = camera_model_ids.find(cam.model_id);
            LFS_ASSERT_MSG(it != camera_model_ids.end() && it->second.second >= 0,
                           std::format("cameras.txt camera model must have a parameter contract "
                                       "(camera_id={}, model_name='{}', model_id={}, source_line={})",
                                       cam.camera_id, model_name, cam.model_id, line_idx + 1));
            LFS_ASSERT_MSG(cam.params.size() == static_cast<size_t>(it->second.second),
                           std::format("cameras.txt model '{}' expects {} parameters, got {}",
                                       model_name, it->second.second, cam.params.size()));
            if (scale_factor != 1.0f) {
                scale_camera_intrinsics(it->second.first, cam.params, scale_factor);
            }

            const auto [_, inserted] = cams.emplace(cam.camera_id, std::move(cam));
            LFS_ASSERT_MSG(inserted,
                           std::format("cameras.txt camera ids must be unique "
                                       "(camera_id={}, source_line={})",
                                       camera_id, line_idx + 1));
        }

        LFS_ASSERT_MSG(!cams.empty(),
                       std::format("cameras.txt must contain at least one camera record "
                                   "(parsed_camera_count={}, source_line_count={}, path='{}')",
                                   cams.size(), lines.size(), lfs::core::path_to_utf8(file_path)));
        return cams;
    }

    // -----------------------------------------------------------------------------
    //  points3D.txt
    // -----------------------------------------------------------------------------
    std::vector<Point3DData> read_point3D_text_records(const std::filesystem::path& file_path,
                                                       const LoadOptions& options = {},
                                                       const TrackParseMode track_mode = TrackParseMode::Full) {
        LOG_TIMER_TRACE("Read points3D.txt");
        auto buffer = read_binary(file_path);
        const auto parse_start = std::chrono::high_resolution_clock::now();

        size_t total_track_elements = 0;
        size_t file_lines = 0;
        std::vector<Point3DData> points;

        if (buffer->size() < POINTS3D_PARALLEL_MIN_BYTES) {
            points.reserve(std::max<size_t>(buffer->size() / 96, 1));
            file_lines = for_each_data_line(
                std::span<const char>(*buffer),
                options,
                "COLMAP point cloud parse cancelled",
                [&](const std::string_view line, size_t) {
                    Point3DData point;
                    parse_point3D_record_line(line, track_mode, point, total_track_elements);
                    points.push_back(std::move(point));
                });
        } else {
            struct RecordChunkResult {
                std::vector<Point3DData> points;
                size_t file_lines = 0;
                size_t track_elements = 0;
            };

            const auto chunks = split_line_aligned_chunks(std::span<const char>(*buffer));
            std::vector<RecordChunkResult> results(chunks.size());
            // oneTBB propagates parse/cancellation exceptions to the caller and cancels sibling work.
            tbb::parallel_for(size_t{0}, chunks.size(), [&](const size_t chunk_index) {
                const auto& chunk = chunks[chunk_index];
                auto& result = results[chunk_index];
                result.points.reserve(std::max<size_t>(static_cast<size_t>(chunk.end - chunk.begin) / 96, 1));
                result.file_lines = for_each_data_line(
                    std::span<const char>(chunk.begin, static_cast<size_t>(chunk.end - chunk.begin)),
                    options,
                    "COLMAP point cloud parse cancelled",
                    [&](const std::string_view line, size_t) {
                        Point3DData point;
                        parse_point3D_record_line(line, track_mode, point, result.track_elements);
                        result.points.push_back(std::move(point));
                    });
            });

            size_t total_points = 0;
            for (const auto& result : results) {
                total_points += result.points.size();
                total_track_elements += result.track_elements;
                file_lines += result.file_lines;
            }

            points.reserve(total_points);
            for (auto& result : results) {
                std::move(result.points.begin(), result.points.end(), std::back_inserter(points));
            }
        }
        buffer.reset();

        if (points.empty()) {
            LOG_ERROR("No valid points found in {}", lfs::core::path_to_utf8(file_path));
            throw std::runtime_error("No valid points in points3D.txt");
        }
        std::unordered_set<uint64_t> point_ids;
        point_ids.reserve(points.size());
        for (size_t point_index = 0; point_index < points.size(); ++point_index) {
            const auto& point = points[point_index];
            LFS_ASSERT_MSG(point_ids.insert(point.point3D_id).second,
                           std::format("points3D.txt point ids must be unique "
                                       "(point_id={}, record_index={}, point_count={}, unique_count={})",
                                       point.point3D_id, point_index, points.size(), point_ids.size()));
        }

        LOG_DEBUG("Reading {} 3D points from text file", points.size());
        const char* mode_name = track_mode == TrackParseMode::Full ? "full" : (track_mode == TrackParseMode::CountOnly ? "count_only" : "none");
        LOG_INFO("[COLMAP_LOAD] parse points3D.txt points={} track_elements={} parse_tracks={} mode={} file_lines={} elapsed_ms={:.2f}",
                 points.size(),
                 total_track_elements,
                 track_mode == TrackParseMode::Full,
                 mode_name,
                 file_lines,
                 elapsed_ms(parse_start));
        return points;
    }

    static Point3DTextPointCloudData parse_points3D_text_point_cloud_fast(
        const std::filesystem::path& file_path,
        const LoadOptions& options) {
        const auto start = std::chrono::high_resolution_clock::now();
        std::error_code file_size_ec;
        const auto byte_size = fs::file_size(file_path, file_size_ec);

        Point3DTextPointCloudData data;
        data.byte_size = file_size_ec ? 0 : byte_size;

        auto buffer = read_binary(file_path);
        if (buffer->size() < POINTS3D_PARALLEL_MIN_BYTES) {
            if (!file_size_ec) {
                const auto estimated_points = static_cast<size_t>(std::max<uintmax_t>(byte_size / 96, 1));
                data.positions.reserve(estimated_points * 3);
                data.colors.reserve(estimated_points * 3);
            }

            data.file_lines = for_each_data_line(
                std::span<const char>(*buffer),
                options,
                "COLMAP point cloud parse cancelled",
                [&](const std::string_view line, size_t) {
                    parse_point3D_point_cloud_line(
                        line, data.positions, data.colors, data.point_ids, data.point_count);
                });
        } else {
            struct PointCloudChunkResult {
                std::vector<float> positions;
                std::vector<uint8_t> colors;
                std::vector<uint64_t> point_ids;
                size_t point_count = 0;
                size_t file_lines = 0;
            };

            const auto chunks = split_line_aligned_chunks(std::span<const char>(*buffer));
            std::vector<PointCloudChunkResult> results(chunks.size());
            // oneTBB propagates parse/cancellation exceptions to the caller and cancels sibling work.
            tbb::parallel_for(size_t{0}, chunks.size(), [&](const size_t chunk_index) {
                const auto& chunk = chunks[chunk_index];
                auto& result = results[chunk_index];
                const auto estimated_points = std::max<size_t>(static_cast<size_t>(chunk.end - chunk.begin) / 96, 1);
                result.positions.reserve(estimated_points * 3);
                result.colors.reserve(estimated_points * 3);
                result.point_ids.reserve(estimated_points);
                result.file_lines = for_each_data_line(
                    std::span<const char>(chunk.begin, static_cast<size_t>(chunk.end - chunk.begin)),
                    options,
                    "COLMAP point cloud parse cancelled",
                    [&](const std::string_view line, size_t) {
                        parse_point3D_point_cloud_line(
                            line, result.positions, result.colors, result.point_ids, result.point_count);
                    });
            });

            size_t total_position_values = 0;
            size_t total_color_values = 0;
            size_t total_point_ids = 0;
            for (const auto& result : results) {
                data.point_count += result.point_count;
                data.file_lines += result.file_lines;
                total_position_values += result.positions.size();
                total_color_values += result.colors.size();
                total_point_ids += result.point_ids.size();
            }

            data.positions.clear();
            data.colors.clear();
            data.positions.reserve(total_position_values);
            data.colors.reserve(total_color_values);
            data.point_ids.reserve(total_point_ids);
            for (auto& result : results) {
                std::move(result.positions.begin(), result.positions.end(), std::back_inserter(data.positions));
                std::move(result.colors.begin(), result.colors.end(), std::back_inserter(data.colors));
                std::move(result.point_ids.begin(), result.point_ids.end(), std::back_inserter(data.point_ids));
            }
        }
        buffer.reset();

        if (data.point_count == 0) {
            LOG_ERROR("No valid points found in {}", lfs::core::path_to_utf8(file_path));
            throw std::runtime_error("No valid points in points3D.txt");
        }
        LFS_ASSERT_MSG(data.positions.size() == data.point_count * 3 &&
                           data.colors.size() == data.point_count * 3 &&
                           data.point_ids.size() == data.point_count,
                       std::format("points3D.txt parsed field counts must match the point count "
                                   "(point_count={}, position_values={}, expected_position_values={}, "
                                   "color_values={}, expected_color_values={}, point_id_count={})",
                                   data.point_count, data.positions.size(), data.point_count * 3,
                                   data.colors.size(), data.point_count * 3, data.point_ids.size()));
        std::unordered_set<uint64_t> unique_point_ids;
        unique_point_ids.reserve(data.point_ids.size());
        for (size_t point_index = 0; point_index < data.point_ids.size(); ++point_index) {
            const uint64_t point_id = data.point_ids[point_index];
            LFS_ASSERT_MSG(unique_point_ids.insert(point_id).second,
                           std::format("points3D.txt point ids must be unique "
                                       "(point_id={}, record_index={}, point_count={}, unique_count={})",
                                       point_id, point_index, data.point_ids.size(),
                                       unique_point_ids.size()));
        }

        LOG_INFO("[COLMAP_LOAD] parse points3D.txt point_cloud_fast points={} file_lines={} bytes={} elapsed_ms={:.2f}",
                 data.point_count,
                 data.file_lines,
                 file_size_ec ? std::string("unknown") : std::format("{}", byte_size),
                 elapsed_ms(start));

        return data;
    }

    PointCloud read_point3D_text(const std::filesystem::path& file_path,
                                 const LoadOptions& options = {}) {
        LOG_TIMER_TRACE("Read points3D.txt point cloud");
        auto data = parse_points3D_text_point_cloud_fast(file_path, options);

        Tensor means = Tensor::from_vector(data.positions, {data.point_count, 3}, Device::CUDA);
        Tensor colors_tensor = Tensor::from_blob(data.colors.data(), {data.point_count, 3}, Device::CPU, DataType::UInt8)
                                   .to(Device::CUDA)
                                   .contiguous();

        return PointCloud(std::move(means), std::move(colors_tensor));
    }

    // -----------------------------------------------------------------------------
    //  Assemble cameras with dimension verification
    // -----------------------------------------------------------------------------
    Result<std::tuple<std::vector<std::shared_ptr<Camera>>, Tensor>>
    assemble_colmap_cameras(const std::filesystem::path& base_path,
                            const std::unordered_map<uint32_t, CameraDataIntermediate>& cam_map,
                            const std::vector<ImageData>& images,
                            const std::string& images_folder,
                            const LoadOptions& options = {}) {

        LOG_TIMER_TRACE("Assemble COLMAP cameras");

        std::filesystem::path images_path = base_path / lfs::core::utf8_to_path(images_folder);

        if (!std::filesystem::exists(images_path)) {
            return make_error(ErrorCode::PATH_NOT_FOUND,
                              "Images folder does not exist", images_path);
        }

        std::vector<std::shared_ptr<Camera>> cameras;
        cameras.reserve(images.size());

        RecursiveFileCache image_cache(images_path, options.cancel_requested);
        MaskDirCache mask_cache(base_path, options.cancel_requested);
        DepthDirCache depth_cache(base_path, options.cancel_requested);
        NormalDirCache normal_cache(base_path, options.cancel_requested);
        bool used_recursive_image_lookup = false;
        size_t depth_matched_count = 0;
        size_t normal_matched_count = 0;

        // Accumulate camera positions for scene center
        std::vector<float> camera_positions;
        camera_positions.reserve(images.size() * 3);

        for (size_t i = 0; i < images.size(); ++i) {
            if (should_poll_cancel(i)) {
                throw_if_load_cancel_requested(options, "COLMAP camera assembly cancelled");
            }
            const ImageData& img = images[i];
            const std::filesystem::path image_rel_path = lfs::core::utf8_to_path(img.name);
            std::filesystem::path image_path = images_path / image_rel_path;

            if (!safe_exists(image_path)) {
                if (auto image_lookup = image_cache.lookup(image_rel_path);
                    image_lookup.found()) {
                    if (!used_recursive_image_lookup) {
                        LOG_WARN("COLMAP images are not in the expected flat layout under '{}'; "
                                 "falling back to recursive image lookup",
                                 lfs::core::path_to_utf8(images_path));
                        used_recursive_image_lookup = true;
                    }
                    image_path = std::move(image_lookup.path);
                } else if (image_lookup.ambiguous()) {
                    return make_ambiguous_image_reference_error(images_path, img.name);
                } else {
                    return make_error(ErrorCode::PATH_NOT_FOUND,
                                      std::format("Image '{}' was not found under '{}'",
                                                  img.name,
                                                  lfs::core::path_to_utf8(images_path)),
                                      image_path);
                }
            }

            auto it = cam_map.find(img.camera_id);
            if (it == cam_map.end()) {
                return make_error(ErrorCode::CORRUPTED_DATA,
                                  std::format("Camera ID {} not found for image '{}'", img.camera_id, img.name),
                                  image_path);
            }

            const auto& cam_data = it->second;

            // Convert quaternion to rotation matrix
            Tensor R = qvec2rotmat(img.qvec);

            // Create translation tensor
            Tensor T = Tensor::from_vector(img.tvec, {3}, Device::CPU);

            // Calculate camera position: -R^T * T
            auto R_cpu = R.cpu();
            auto T_cpu = T.cpu();

            float RT[9];
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    RT[r * 3 + c] = R_cpu.ptr<float>()[c * 3 + r]; // Transpose
                }
            }

            float cam_pos[3] = {0, 0, 0};
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    cam_pos[r] -= RT[r * 3 + c] * T_cpu.ptr<float>()[c];
                }
            }

            camera_positions.push_back(cam_pos[0]);
            camera_positions.push_back(cam_pos[1]);
            camera_positions.push_back(cam_pos[2]);

            // Extract camera parameters based on model
            auto model_it = camera_model_ids.find(cam_data.model_id);
            if (model_it == camera_model_ids.end()) {
                return make_error(ErrorCode::UNSUPPORTED_FORMAT,
                                  std::format("Invalid camera model ID {} for image '{}'", cam_data.model_id, img.name),
                                  image_path);
            }

            CAMERA_MODEL model = model_it->second.first;
            const auto& params = cam_data.params;

            float focal_x = 0, focal_y = 0, center_x = 0, center_y = 0;
            Tensor radial_dist, tangential_dist;
            lfs::core::CameraModelType camera_model_type = lfs::core::CameraModelType::PINHOLE;

            switch (model) {
            case CAMERA_MODEL::SIMPLE_PINHOLE:
                focal_x = focal_y = params[0];
                center_x = params[1];
                center_y = params[2];
                radial_dist = Tensor::empty({0}, Device::CPU);
                tangential_dist = Tensor::empty({0}, Device::CPU);
                break;

            case CAMERA_MODEL::PINHOLE:
                focal_x = params[0];
                focal_y = params[1];
                center_x = params[2];
                center_y = params[3];
                radial_dist = Tensor::empty({0}, Device::CPU);
                tangential_dist = Tensor::empty({0}, Device::CPU);
                break;

            case CAMERA_MODEL::SIMPLE_RADIAL:
                focal_x = focal_y = params[0];
                center_x = params[1];
                center_y = params[2];
                if (params[3] != 0.0f) {
                    radial_dist = Tensor::from_vector({params[3]}, {1}, Device::CPU);
                } else {
                    radial_dist = Tensor::empty({0}, Device::CPU);
                }
                tangential_dist = Tensor::empty({0}, Device::CPU);
                camera_model_type = lfs::core::CameraModelType::PINHOLE;
                break;

            case CAMERA_MODEL::RADIAL:
                focal_x = focal_y = params[0];
                center_x = params[1];
                center_y = params[2];
                radial_dist = make_distortion_tensor({params[3], params[4]});
                tangential_dist = Tensor::empty({0}, Device::CPU);
                break;

            case CAMERA_MODEL::OPENCV:
                focal_x = params[0];
                focal_y = params[1];
                center_x = params[2];
                center_y = params[3];
                radial_dist = make_distortion_tensor({params[4], params[5]});
                tangential_dist = make_distortion_tensor({params[6], params[7]});
                break;

            case CAMERA_MODEL::FULL_OPENCV:
                focal_x = params[0];
                focal_y = params[1];
                center_x = params[2];
                center_y = params[3];
                radial_dist = make_distortion_tensor({params[4], params[5], params[8], params[9], params[10], params[11]});
                tangential_dist = make_distortion_tensor({params[6], params[7]});
                break;

            case CAMERA_MODEL::OPENCV_FISHEYE:
                focal_x = params[0];
                focal_y = params[1];
                center_x = params[2];
                center_y = params[3];
                radial_dist = Tensor::from_vector({params[4], params[5], params[6], params[7]}, {4}, Device::CPU);
                tangential_dist = Tensor::empty({0}, Device::CPU);
                camera_model_type = lfs::core::CameraModelType::FISHEYE;
                break;

            case CAMERA_MODEL::RADIAL_FISHEYE:
                focal_x = focal_y = params[0];
                center_x = params[1];
                center_y = params[2];
                radial_dist = Tensor::from_vector({params[3], params[4]}, {2}, Device::CPU);
                tangential_dist = Tensor::empty({0}, Device::CPU);
                camera_model_type = lfs::core::CameraModelType::FISHEYE;
                break;

            case CAMERA_MODEL::SIMPLE_RADIAL_FISHEYE:
                focal_x = focal_y = params[0];
                center_x = params[1];
                center_y = params[2];
                radial_dist = Tensor::from_vector({params[3]}, {1}, Device::CPU);
                tangential_dist = Tensor::empty({0}, Device::CPU);
                camera_model_type = lfs::core::CameraModelType::FISHEYE;
                break;

            case CAMERA_MODEL::THIN_PRISM_FISHEYE:
                // fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1
                focal_x = params[0];
                focal_y = params[1];
                center_x = params[2];
                center_y = params[3];
                radial_dist = Tensor::from_vector({params[4], params[5], params[8], params[9]}, {4}, Device::CPU);       // k1,k2,k3,k4
                tangential_dist = Tensor::from_vector({params[6], params[7], params[10], params[11]}, {4}, Device::CPU); // p1,p2,sx1,sy1
                camera_model_type = lfs::core::CameraModelType::THIN_PRISM_FISHEYE;
                break;

            case CAMERA_MODEL::FOV:
                return make_error(ErrorCode::UNSUPPORTED_FORMAT,
                                  std::format("FOV camera model not supported for image '{}'", img.name),
                                  image_path);

            case CAMERA_MODEL::EQUIRECTANGULAR:
                // Equirectangular 360 panorama. params = [width, height]; no real
                // intrinsics. The EQUIRECTANGULAR rasterizer derives projection
                // from the image dimensions, so we only set placeholder focal/center.
                focal_x = focal_y = EQUIRECTANGULAR_DUMMY_FOCAL;
                center_x = 0.5f * static_cast<float>(cam_data.width);
                center_y = 0.5f * static_cast<float>(cam_data.height);
                radial_dist = Tensor::empty({0}, Device::CPU);
                tangential_dist = Tensor::empty({0}, Device::CPU);
                camera_model_type = lfs::core::CameraModelType::EQUIRECTANGULAR;
                break;

            default:
                return make_error(ErrorCode::UNSUPPORTED_FORMAT,
                                  std::format("Unsupported camera model for image '{}'", img.name),
                                  image_path);
            }

            std::filesystem::path mask_path;
            if (auto mask_lookup = mask_cache.lookup(img.name); mask_lookup.found()) {
                mask_path = std::move(mask_lookup.path);
            } else if (mask_lookup.ambiguous()) {
                return make_ambiguous_mask_reference_error(base_path, img.name);
            }

            std::filesystem::path depth_path;
            if (auto depth_lookup = depth_cache.lookup(img.name); depth_lookup.found()) {
                depth_path = std::move(depth_lookup.path);
                ++depth_matched_count;
            } else if (depth_lookup.ambiguous()) {
                return make_error(
                    ErrorCode::INVALID_DATASET,
                    std::format("Depth map for image '{}' is ambiguous across the dataset depth folders. "
                                "Keep depth maps in the same relative subdirectories as the images or rename them uniquely.",
                                img.name),
                    base_path);
            }

            std::filesystem::path normal_path;
            if (auto normal_lookup = normal_cache.lookup(img.name); normal_lookup.found()) {
                normal_path = std::move(normal_lookup.path);
                ++normal_matched_count;
            } else if (normal_lookup.ambiguous()) {
                return make_error(
                    ErrorCode::INVALID_DATASET,
                    std::format("Normal map for image '{}' is ambiguous across the dataset normal folders. "
                                "Keep normal maps in the same relative subdirectories as the images or rename them uniquely.",
                                img.name),
                    base_path);
            }

            // Validate mask/depth dimensions match image dimensions
            std::optional<std::tuple<int, int, int>> image_info;
            auto get_image_info_cached = [&]() {
                if (!image_info.has_value()) {
                    image_info = lfs::core::get_image_info(image_path);
                }
                return *image_info;
            };
            if (!mask_path.empty()) {
                auto [img_w, img_h, img_c] = get_image_info_cached();
                auto [mask_w, mask_h, mask_c] = lfs::core::get_image_info(mask_path);
                if (img_w != mask_w || img_h != mask_h) {
                    return make_error(ErrorCode::MASK_SIZE_MISMATCH,
                                      std::format("Mask '{}' is {}x{} but image '{}' is {}x{}",
                                                  lfs::core::path_to_utf8(mask_path.filename()), mask_w, mask_h,
                                                  img.name, img_w, img_h),
                                      mask_path);
                }
            }
            if (!depth_path.empty()) {
                auto [img_w, img_h, img_c] = get_image_info_cached();
                auto [depth_w, depth_h, depth_c] = lfs::core::get_image_info(depth_path);
                if (img_w != depth_w || img_h != depth_h) {
                    return make_error(ErrorCode::DEPTH_SIZE_MISMATCH,
                                      std::format("Depth map '{}' is {}x{} but image '{}' is {}x{}",
                                                  lfs::core::path_to_utf8(depth_path.filename()), depth_w, depth_h,
                                                  img.name, img_w, img_h),
                                      depth_path);
                }
            }
            if (!normal_path.empty()) {
                auto [img_w, img_h, img_c] = get_image_info_cached();
                auto [normal_w, normal_h, normal_c] = lfs::core::get_image_info(normal_path);
                if (img_w != normal_w || img_h != normal_h) {
                    return make_error(ErrorCode::NORMAL_SIZE_MISMATCH,
                                      std::format("Normal map '{}' is {}x{} but image '{}' is {}x{}",
                                                  lfs::core::path_to_utf8(normal_path.filename()), normal_w, normal_h,
                                                  img.name, img_w, img_h),
                                      normal_path);
                }
            }

            // Create Camera
            auto camera = std::make_shared<Camera>(
                R,
                T,
                focal_x, focal_y,
                center_x, center_y,
                radial_dist,
                tangential_dist,
                camera_model_type,
                img.name,
                image_path,
                mask_path,
                cam_data.width,
                cam_data.height,
                static_cast<int>(i),
                static_cast<int>(img.camera_id),
                depth_path,
                normal_path);

            camera->precompute_undistortion();

            cameras.push_back(std::move(camera));
        }

        // Compute scene center as mean of camera positions
        Tensor scene_center_tensor = Tensor::from_vector(camera_positions, {images.size(), 3}, Device::CPU);
        Tensor scene_center = scene_center_tensor.mean({0}, false);

        LOG_INFO("Training with {} images", cameras.size());
        if (depth_cache.has_depth_dirs()) {
            if (depth_matched_count == 0) {
                LOG_WARN("Depth folder found but no depth map matched any of the {} images. "
                         "Depth files must share the image filename or its trailing frame number.",
                         cameras.size());
            } else {
                LOG_INFO("Depth maps matched for {}/{} images", depth_matched_count, cameras.size());
            }
        }
        if (normal_cache.has_normal_dirs()) {
            if (normal_matched_count == 0) {
                LOG_WARN("Normal folder found but no normal map matched any of the {} images. "
                         "Normal files must share the image filename or its trailing frame number.",
                         cameras.size());
            } else {
                LOG_INFO("Normal maps matched for {}/{} images", normal_matched_count, cameras.size());
            }
        }

        return std::make_tuple(std::move(cameras), scene_center);
    }

    struct ColmapSparseModelData {
        std::unordered_map<uint32_t, CameraDataIntermediate> cameras;
        std::vector<ImageData> images;
        std::vector<Point3DData> points3D;
        bool source_binary = true;
    };

    std::vector<uint32_t> sorted_camera_ids(const std::unordered_map<uint32_t, CameraDataIntermediate>& cameras) {
        std::vector<uint32_t> ids;
        ids.reserve(cameras.size());
        for (const auto& [id, _] : cameras) {
            ids.push_back(id);
        }
        std::sort(ids.begin(), ids.end());
        return ids;
    }

    ColmapSparseModelData read_colmap_sparse_model_for_write(const fs::path& source_base) {
        const auto search_paths = get_colmap_search_paths(source_base);
        const fs::path cameras_bin = find_file_in_paths(search_paths, "cameras.bin");
        const fs::path images_bin = find_file_in_paths(search_paths, "images.bin");
        const fs::path points_bin = find_file_in_paths(search_paths, "points3D.bin");
        const fs::path cameras_txt = find_file_in_paths(search_paths, "cameras.txt");
        const fs::path images_txt = find_file_in_paths(search_paths, "images.txt");
        const fs::path points_txt = find_file_in_paths(search_paths, "points3D.txt");

        const bool has_binary_pair = !cameras_bin.empty() && !images_bin.empty();
        const bool has_text_pair = !cameras_txt.empty() && !images_txt.empty();
        if (!has_binary_pair && !has_text_pair) {
            throw std::runtime_error("Missing COLMAP cameras/images pair in source");
        }

        ColmapSparseModelData model;
        model.source_binary = has_binary_pair;
        if (has_binary_pair) {
            model.cameras = read_cameras_binary(cameras_bin, 1.0f);
            model.images = read_images_binary(images_bin);
            if (!points_bin.empty()) {
                model.points3D = read_point3D_binary_records(points_bin);
            } else if (!points_txt.empty()) {
                model.points3D = read_point3D_text_records(points_txt);
            }
        } else {
            model.cameras = read_cameras_text(cameras_txt, 1.0f);
            model.images = read_images_text(images_txt);
            if (!points_txt.empty()) {
                model.points3D = read_point3D_text_records(points_txt);
            } else if (!points_bin.empty()) {
                model.points3D = read_point3D_binary_records(points_bin);
            }
        }

        return model;
    }

    std::vector<float> rotmat_to_qvec(const glm::mat3& R) {
        const double m00 = R[0][0];
        const double m01 = R[1][0];
        const double m02 = R[2][0];
        const double m10 = R[0][1];
        const double m11 = R[1][1];
        const double m12 = R[2][1];
        const double m20 = R[0][2];
        const double m21 = R[1][2];
        const double m22 = R[2][2];

        double qw = 1.0;
        double qx = 0.0;
        double qy = 0.0;
        double qz = 0.0;
        const double trace = m00 + m11 + m22;
        if (trace > 0.0) {
            const double s = std::sqrt(trace + 1.0) * 2.0;
            qw = 0.25 * s;
            qx = (m21 - m12) / s;
            qy = (m02 - m20) / s;
            qz = (m10 - m01) / s;
        } else if (m00 > m11 && m00 > m22) {
            const double s = std::sqrt(1.0 + m00 - m11 - m22) * 2.0;
            qw = (m21 - m12) / s;
            qx = 0.25 * s;
            qy = (m01 + m10) / s;
            qz = (m02 + m20) / s;
        } else if (m11 > m22) {
            const double s = std::sqrt(1.0 + m11 - m00 - m22) * 2.0;
            qw = (m02 - m20) / s;
            qx = (m01 + m10) / s;
            qy = 0.25 * s;
            qz = (m12 + m21) / s;
        } else {
            const double s = std::sqrt(1.0 + m22 - m00 - m11) * 2.0;
            qw = (m10 - m01) / s;
            qx = (m02 + m20) / s;
            qy = (m12 + m21) / s;
            qz = 0.25 * s;
        }

        const double norm = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
        if (norm <= 1e-12) {
            throw std::runtime_error("Cannot convert degenerate rotation to quaternion");
        }
        qw /= norm;
        qx /= norm;
        qy /= norm;
        qz /= norm;
        if (qw < 0.0) {
            qw = -qw;
            qx = -qx;
            qy = -qy;
            qz = -qz;
        }

        return {
            static_cast<float>(qw),
            static_cast<float>(qx),
            static_cast<float>(qy),
            static_cast<float>(qz),
        };
    }

    std::pair<std::vector<float>, std::vector<float>>
    transformed_camera_pose(const Camera& camera, const glm::mat4& data_world_transform) {
        auto R_cpu = camera.R().cpu().contiguous();
        auto T_cpu = camera.T().cpu().contiguous();
        auto R_acc = R_cpu.accessor<float, 2>();
        auto T_acc = T_cpu.accessor<float, 1>();

        glm::mat3 camera_to_world_rotation(1.0f);
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                camera_to_world_rotation[col][row] = R_acc(col, row);
            }
        }

        const glm::vec3 t(T_acc(0), T_acc(1), T_acc(2));
        const glm::vec3 camera_center = -(camera_to_world_rotation * t);

        glm::mat4 camera_to_world(1.0f);
        for (int col = 0; col < 3; ++col) {
            for (int row = 0; row < 3; ++row) {
                camera_to_world[col][row] = camera_to_world_rotation[col][row];
            }
        }
        camera_to_world[3] = glm::vec4(camera_center, 1.0f);

        const glm::mat4 transformed_c2w = data_world_transform * camera_to_world;
        glm::vec3 basis_x(transformed_c2w[0]);
        glm::vec3 basis_y(transformed_c2w[1]);
        glm::vec3 basis_z(transformed_c2w[2]);

        const float len_x = glm::length(basis_x);
        const float len_y = glm::length(basis_y);
        const float len_z = glm::length(basis_z);
        if (len_x <= 1e-8f || len_y <= 1e-8f || len_z <= 1e-8f) {
            throw std::runtime_error("Cannot export camera with degenerate transform");
        }
        basis_x /= len_x;
        basis_y /= len_y;
        basis_z /= len_z;

        if (std::abs(glm::dot(basis_x, basis_y)) > 1e-3f ||
            std::abs(glm::dot(basis_x, basis_z)) > 1e-3f ||
            std::abs(glm::dot(basis_y, basis_z)) > 1e-3f) {
            throw std::runtime_error("COLMAP camera export requires rigid or uniform-scale camera transforms");
        }

        glm::mat3 transformed_camera_to_world(1.0f);
        transformed_camera_to_world[0] = basis_x;
        transformed_camera_to_world[1] = basis_y;
        transformed_camera_to_world[2] = basis_z;

        const glm::mat3 world_to_camera_rotation = glm::transpose(transformed_camera_to_world);
        const glm::vec3 transformed_center(transformed_c2w[3]);
        const glm::vec3 transformed_t = -(world_to_camera_rotation * transformed_center);

        return {
            rotmat_to_qvec(world_to_camera_rotation),
            {transformed_t.x, transformed_t.y, transformed_t.z},
        };
    }

    void update_image_poses_from_cameras(
        std::vector<ImageData>& images,
        const std::vector<ColmapCameraWriteData>& cameras) {
        LFS_ASSERT_MSG(!images.empty(),
                       std::format("COLMAP pose export requires at least one source image "
                                   "(image_count={}, scene_camera_count={})",
                                   images.size(), cameras.size()));
        LFS_ASSERT_MSG(images.size() == cameras.size(),
                       std::format("COLMAP pose export requires exactly one scene camera per source image ({} images, {} cameras)",
                                   images.size(), cameras.size()));
        std::unordered_map<std::string, const ColmapCameraWriteData*> camera_by_name;
        camera_by_name.reserve(cameras.size());
        for (size_t camera_index = 0; camera_index < cameras.size(); ++camera_index) {
            const auto& item = cameras[camera_index];
            LFS_ASSERT_MSG(item.camera != nullptr,
                           std::format("COLMAP pose export requires every scene camera pointer "
                                       "to be non-null (camera_index={}, camera_pointer={}, "
                                       "scene_camera_count={}, image_count={})",
                                       camera_index, static_cast<const void*>(item.camera.get()),
                                       cameras.size(), images.size()));
            const std::string& image_name = item.camera->image_name();
            LFS_ASSERT_MSG(!image_name.empty(),
                           std::format("COLMAP pose export requires every scene camera to have an "
                                       "image name (camera_index={}, image_name='{}', "
                                       "scene_camera_count={})",
                                       camera_index, image_name, cameras.size()));
            const auto [_, inserted] = camera_by_name.emplace(image_name, &item);
            LFS_ASSERT_MSG(inserted,
                           std::format("COLMAP pose export received duplicate scene camera name '{}'", image_name));
            for (int col = 0; col < 4; ++col) {
                for (int row = 0; row < 4; ++row) {
                    LFS_ASSERT_MSG(std::isfinite(item.data_world_transform[col][row]),
                                   std::format("COLMAP pose export transform elements must be finite "
                                               "(camera_index={}, image_name='{}', column={}, row={}, value={})",
                                               camera_index, image_name, col, row,
                                               item.data_world_transform[col][row]));
                }
            }
        }

        size_t matched = 0;
        for (auto& image : images) {
            const auto it = camera_by_name.find(image.name);
            if (it == camera_by_name.end()) {
                throw std::runtime_error(std::format("No current scene camera found for COLMAP image '{}'", image.name));
            }
            auto [qvec, tvec] = transformed_camera_pose(*it->second->camera, it->second->data_world_transform);
            LFS_ASSERT_MSG(std::ranges::all_of(qvec, [](const float value) { return std::isfinite(value); }) &&
                               std::ranges::all_of(tvec, [](const float value) { return std::isfinite(value); }),
                           std::format("COLMAP pose for '{}' is not finite", image.name));
            image.qvec = std::move(qvec);
            image.tvec = std::move(tvec);
            ++matched;
        }

        LFS_ASSERT_MSG(matched == images.size() && matched == camera_by_name.size(),
                       std::format("COLMAP pose export must establish a one-to-one image association "
                                   "(matched_count={}, image_count={}, unique_camera_name_count={}, "
                                   "scene_camera_count={})",
                                   matched, images.size(), camera_by_name.size(), cameras.size()));
    }

    uint8_t float_color_to_u8(const float value) {
        LFS_ASSERT_MSG(std::isfinite(value) && value >= 0.0f && value <= 1.0f,
                       std::format("COLMAP float point colors must be finite and in [0,1] "
                                   "(color_value={}, valid_range=[0,1])",
                                   value));
        const float clamped = std::clamp(value, 0.0f, 1.0f);
        return static_cast<uint8_t>(std::round(clamped * 255.0f));
    }

    std::vector<Point3DData> build_points3D_for_write(
        const std::vector<Point3DData>& source_points,
        const PointCloud* point_cloud,
        const glm::mat4& point_cloud_transform) {
        for (int col = 0; col < 4; ++col) {
            for (int row = 0; row < 4; ++row) {
                LFS_ASSERT_MSG(std::isfinite(point_cloud_transform[col][row]),
                               std::format("COLMAP point transform elements must be finite "
                                           "(column={}, row={}, value={}, source_point_count={}, "
                                           "live_point_count={})",
                                           col, row, point_cloud_transform[col][row],
                                           source_points.size(),
                                           point_cloud != nullptr ? point_cloud->size() : 0));
            }
        }
        if (!point_cloud || point_cloud->size() <= 0 || !point_cloud->means.is_valid()) {
            // No live scene point cloud (training replaced it with a splat
            // model, etc.) — fall back to the COLMAP source points, but still
            // apply the caller-supplied world transform so the exported points
            // stay in the same frame as the exported cameras. Identity is a
            // no-op so the cost is negligible when nothing was reoriented.
            if (point_cloud_transform == glm::mat4(1.0f)) {
                return source_points;
            }
            std::vector<Point3DData> points = source_points;
            for (size_t point_index = 0; point_index < points.size(); ++point_index) {
                auto& p = points[point_index];
                const glm::vec3 local(static_cast<float>(p.xyz[0]),
                                      static_cast<float>(p.xyz[1]),
                                      static_cast<float>(p.xyz[2]));
                const glm::vec3 world = glm::vec3(point_cloud_transform * glm::vec4(local, 1.0f));
                LFS_ASSERT_MSG(std::isfinite(world.x) && std::isfinite(world.y) && std::isfinite(world.z),
                               std::format("COLMAP point transform must produce finite coordinates "
                                           "(point_index={}, point_id={}, local=[{},{},{}], "
                                           "world=[{},{},{}])",
                                           point_index, p.point3D_id, local.x, local.y, local.z,
                                           world.x, world.y, world.z));
                p.xyz[0] = world.x;
                p.xyz[1] = world.y;
                p.xyz[2] = world.z;
            }
            return points;
        }

        const size_t N = static_cast<size_t>(point_cloud->size());
        LFS_ASSERT_MSG(point_cloud->means.dtype() == DataType::Float32 &&
                           point_cloud->means.ndim() == 2 &&
                           point_cloud->means.size(0) == N &&
                           point_cloud->means.size(1) == 3,
                       std::format("COLMAP point export requires PointCloud.means to be Float32 [N,3] "
                                   "(means_dtype={}({}), means_shape={}, expected_rows={}, "
                                   "expected_columns=3, point_cloud_size={})",
                                   dtype_name(point_cloud->means.dtype()),
                                   static_cast<int>(point_cloud->means.dtype()),
                                   point_cloud->means.shape().str(), N, point_cloud->size()));
        std::vector<Point3DData> points;
        if (source_points.size() == N) {
            points = source_points;
        } else {
            points.resize(N);
            for (size_t i = 0; i < N; ++i) {
                points[i].point3D_id = static_cast<uint64_t>(i + 1);
                points[i].error = 0.0;
            }
            LOG_WARN("COLMAP export point count changed from {} to {}; writing new point ids without tracks",
                     source_points.size(), N);
        }

        auto means_cpu = point_cloud->means.cpu().contiguous();
        auto means_acc = means_cpu.accessor<float, 2>();

        const bool has_colors = point_cloud->colors.is_valid();
        Tensor colors_cpu;
        Tensor colors_float_cpu;
        if (has_colors) {
            LFS_ASSERT_MSG(point_cloud->colors.ndim() == 2 &&
                               point_cloud->colors.size(0) == N &&
                               point_cloud->colors.size(1) == 3 &&
                               (point_cloud->colors.dtype() == DataType::UInt8 ||
                                point_cloud->colors.dtype() == DataType::Float32),
                           std::format("COLMAP point export requires colors to be UInt8 or Float32 [N,3] "
                                       "(colors_dtype={}({}), colors_shape={}, expected_rows={}, "
                                       "expected_columns=3, point_cloud_size={})",
                                       dtype_name(point_cloud->colors.dtype()),
                                       static_cast<int>(point_cloud->colors.dtype()),
                                       point_cloud->colors.shape().str(), N, point_cloud->size()));
            colors_cpu = point_cloud->colors.cpu().contiguous();
            if (colors_cpu.dtype() != DataType::UInt8) {
                colors_float_cpu = colors_cpu.to(DataType::Float32).contiguous();
            }
        }

        for (size_t i = 0; i < N; ++i) {
            const glm::vec3 local_point(means_acc(i, 0), means_acc(i, 1), means_acc(i, 2));
            LFS_ASSERT_MSG(std::isfinite(local_point.x) && std::isfinite(local_point.y) &&
                               std::isfinite(local_point.z),
                           std::format("COLMAP point {} has non-finite coordinates", i));
            const glm::vec3 world_point = glm::vec3(point_cloud_transform * glm::vec4(local_point, 1.0f));
            LFS_ASSERT_MSG(std::isfinite(world_point.x) && std::isfinite(world_point.y) &&
                               std::isfinite(world_point.z),
                           std::format("COLMAP transformed point {} is non-finite", i));
            points[i].xyz[0] = world_point.x;
            points[i].xyz[1] = world_point.y;
            points[i].xyz[2] = world_point.z;

            if (!has_colors) {
                points[i].color[0] = 255;
                points[i].color[1] = 255;
                points[i].color[2] = 255;
            } else if (colors_cpu.dtype() == DataType::UInt8) {
                auto color_acc = colors_cpu.accessor<uint8_t, 2>();
                points[i].color[0] = color_acc(i, 0);
                points[i].color[1] = color_acc(i, 1);
                points[i].color[2] = color_acc(i, 2);
            } else {
                auto color_acc = colors_float_cpu.accessor<float, 2>();
                points[i].color[0] = float_color_to_u8(color_acc(i, 0));
                points[i].color[1] = float_color_to_u8(color_acc(i, 1));
                points[i].color[2] = float_color_to_u8(color_acc(i, 2));
            }
        }

        return points;
    }

    bool point_cloud_requires_untracked_export(const std::vector<Point3DData>& source_points,
                                               const PointCloud* point_cloud) {
        return point_cloud && point_cloud->size() > 0 && point_cloud->means.is_valid() &&
               source_points.size() != static_cast<size_t>(point_cloud->size());
    }

    void clear_image_point3D_references(std::vector<ImageData>& images) {
        for (auto& image : images) {
            for (auto& point : image.points2D) {
                point.point3D_id = INVALID_POINT3D_ID;
            }
        }
    }

    double mean_observations_per_image(const std::vector<ImageData>& images) {
        if (images.empty()) {
            return 0.0;
        }
        size_t observations = 0;
        for (const auto& image : images) {
            observations += image.points2D.size();
        }
        return static_cast<double>(observations) / static_cast<double>(images.size());
    }

    double mean_track_length(const std::vector<Point3DData>& points) {
        if (points.empty()) {
            return 0.0;
        }
        size_t track_elements = 0;
        for (const auto& point : points) {
            track_elements += point.track.size();
        }
        return static_cast<double>(track_elements) / static_cast<double>(points.size());
    }

    std::string point3D_id_to_text(const uint64_t point3D_id) {
        if (point3D_id == INVALID_POINT3D_ID) {
            return "-1";
        }
        return std::to_string(point3D_id);
    }

    void validate_colmap_model_for_write(const ColmapSparseModelData& model,
                                         const bool binary) {
        LFS_ASSERT_MSG(!model.cameras.empty(),
                       std::format("COLMAP writer requires at least one camera "
                                   "(camera_count={}, image_count={}, point_count={}, binary={})",
                                   model.cameras.size(), model.images.size(),
                                   model.points3D.size(), binary));
        LFS_ASSERT_MSG(!model.images.empty(),
                       std::format("COLMAP writer requires at least one image "
                                   "(image_count={}, camera_count={}, point_count={}, binary={})",
                                   model.images.size(), model.cameras.size(),
                                   model.points3D.size(), binary));

        for (const auto& [camera_id, camera] : model.cameras) {
            LFS_ASSERT_MSG(camera_id != 0 && camera.camera_id == camera_id,
                           std::format("COLMAP camera map key must match its non-zero record id "
                                       "(map_key={}, record_camera_id={}, camera_count={})",
                                       camera_id, camera.camera_id, model.cameras.size()));
            LFS_ASSERT_MSG(camera.width > 0 && camera.height > 0,
                           std::format("COLMAP camera dimensions must be positive "
                                       "(camera_id={}, width={}, height={}, model_id={})",
                                       camera_id, camera.width, camera.height, camera.model_id));
            const auto camera_model = camera_model_ids.find(camera.model_id);
            LFS_ASSERT_MSG(camera_model != camera_model_ids.end() &&
                               camera_model->second.second >= 0,
                           std::format("COLMAP camera {} has unsupported model id {}",
                                       camera_id, camera.model_id));
            LFS_ASSERT_MSG(camera.params.size() ==
                               static_cast<size_t>(camera_model->second.second),
                           std::format("COLMAP camera {} has the wrong parameter count", camera_id));
            LFS_ASSERT_MSG(std::ranges::all_of(camera.params, [](const float value) {
                               return std::isfinite(value);
                           }),
                           std::format("COLMAP camera {} has non-finite parameters", camera_id));
        }

        std::unordered_map<uint32_t, const ImageData*> image_by_id;
        std::unordered_set<std::string> image_names;
        image_by_id.reserve(model.images.size());
        image_names.reserve(model.images.size());
        for (const auto& image : model.images) {
            LFS_ASSERT_MSG(image.image_id != 0 &&
                               image_by_id.emplace(image.image_id, &image).second,
                           std::format("COLMAP image ids must be non-zero and unique at write time "
                                       "(image_id={}, name='{}')",
                                       image.image_id, image.name));
            LFS_ASSERT_MSG(!image.name.empty() && image_names.insert(image.name).second,
                           std::format("COLMAP image names must be non-empty and unique at write time "
                                       "(image_id={}, name='{}')",
                                       image.image_id, image.name));
            LFS_ASSERT_MSG(image.name.find('\0') == std::string::npos,
                           std::format("COLMAP image names must not contain embedded null bytes "
                                       "(image_id={}, name_length={}, null_offset={}, binary={})",
                                       image.image_id, image.name.size(), image.name.find('\0'), binary));
            LFS_ASSERT_MSG(binary ||
                               (image.name.find('\n') == std::string::npos &&
                                image.name.find('\r') == std::string::npos),
                           std::format("COLMAP text image names must not contain line breaks "
                                       "(image_id={}, name='{}', newline_offset={}, "
                                       "carriage_return_offset={}, binary={})",
                                       image.image_id, image.name, image.name.find('\n'),
                                       image.name.find('\r'), binary));
            LFS_ASSERT_MSG(model.cameras.contains(image.camera_id),
                           std::format("COLMAP image {} references missing camera {}",
                                       image.image_id, image.camera_id));
            LFS_ASSERT_MSG(image.qvec.size() == 4 && image.tvec.size() == 3,
                           std::format("COLMAP image pose must contain a four-component quaternion "
                                       "and three-component translation "
                                       "(image_id={}, quaternion_count={}, translation_count={})",
                                       image.image_id, image.qvec.size(), image.tvec.size()));
            const double qnorm = std::sqrt(std::inner_product(
                image.qvec.begin(), image.qvec.end(), image.qvec.begin(), 0.0));
            LFS_ASSERT_MSG(std::isfinite(qnorm) && std::abs(qnorm - 1.0) <= 1e-4,
                           std::format("COLMAP image quaternion must be finite and normalized "
                                       "(image_id={}, norm={}, q=[{},{},{},{}])",
                                       image.image_id, qnorm, image.qvec[0], image.qvec[1],
                                       image.qvec[2], image.qvec[3]));
            LFS_ASSERT_MSG(std::ranges::all_of(image.tvec, [](const float value) {
                               return std::isfinite(value);
                           }),
                           std::format("COLMAP image {} translation must be finite", image.image_id));
            for (const auto& point : image.points2D) {
                LFS_ASSERT_MSG(std::isfinite(point.x) && std::isfinite(point.y),
                               std::format("COLMAP image {} has non-finite point2D coordinates",
                                           image.image_id));
            }
        }

        std::unordered_map<uint64_t, const Point3DData*> point_by_id;
        point_by_id.reserve(model.points3D.size());
        for (const auto& point : model.points3D) {
            LFS_ASSERT_MSG(point.point3D_id != 0 &&
                               point_by_id.emplace(point.point3D_id, &point).second,
                           std::format("COLMAP point3D ids must be non-zero and unique at write time "
                                       "(point_id={}, point_count={}, unique_count={})",
                                       point.point3D_id, model.points3D.size(), point_by_id.size()));
            LFS_ASSERT_MSG(std::ranges::all_of(point.xyz, [](const double value) {
                               return std::isfinite(value);
                           }),
                           std::format("COLMAP point {} coordinates must be finite", point.point3D_id));
            LFS_ASSERT_MSG(std::isfinite(point.error) && point.error >= 0.0,
                           std::format("COLMAP point {} error must be finite and non-negative",
                                       point.point3D_id));
            LFS_ASSERT_MSG(point.track_count == point.track.size(),
                           std::format("COLMAP point track count must match its stored track "
                                       "(point3D_id={}, declared_count={}, stored_count={})",
                                       point.point3D_id, point.track_count, point.track.size()));
            std::unordered_set<uint32_t> track_images;
            for (const auto& track : point.track) {
                const auto image = image_by_id.find(track.image_id);
                LFS_ASSERT_MSG(image != image_by_id.end(),
                               std::format("COLMAP point {} track references missing image {}",
                                           point.point3D_id, track.image_id));
                LFS_ASSERT_MSG(track_images.insert(track.image_id).second,
                               std::format("COLMAP point {} track repeats image {}",
                                           point.point3D_id, track.image_id));
                LFS_ASSERT_MSG(track.point2D_idx < image->second->points2D.size(),
                               std::format("COLMAP point track index must be in bounds for its image "
                                           "(point3D_id={}, image_id={}, point2D_index={}, point2D_count={})",
                                           point.point3D_id, track.image_id, track.point2D_idx,
                                           image->second->points2D.size()));
                LFS_ASSERT_MSG(image->second->points2D[track.point2D_idx].point3D_id ==
                                   point.point3D_id,
                               std::format("COLMAP point {} track is not associated back from image {}",
                                           point.point3D_id, track.image_id));
            }
        }

        for (const auto& image : model.images) {
            for (size_t point2D_index = 0; point2D_index < image.points2D.size(); ++point2D_index) {
                const uint64_t point_id = image.points2D[point2D_index].point3D_id;
                if (point_id == INVALID_POINT3D_ID)
                    continue;
                const auto point = point_by_id.find(point_id);
                LFS_ASSERT_MSG(point != point_by_id.end(),
                               std::format("COLMAP image {} references missing point {}",
                                           image.image_id, point_id));
                LFS_ASSERT_MSG(std::ranges::any_of(point->second->track, [&](const auto& track) {
                                   return track.image_id == image.image_id &&
                                          track.point2D_idx == point2D_index;
                               }),
                               std::format("COLMAP image {} point {} has no reciprocal track", image.image_id, point_id));
            }
        }
    }

    void validate_colmap_round_trip(const fs::path& output_sparse_path,
                                    const bool binary,
                                    const ColmapSparseModelData& expected) {
        const auto cameras = binary
                                 ? read_cameras_binary(output_sparse_path / "cameras.bin", 1.0f)
                                 : read_cameras_text(output_sparse_path / "cameras.txt", 1.0f);
        const auto images = binary
                                ? read_images_binary(output_sparse_path / "images.bin")
                                : read_images_text(output_sparse_path / "images.txt");
        LFS_ASSERT_MSG(cameras.size() == expected.cameras.size(),
                       std::format("COLMAP writer round-trip must preserve the camera count "
                                   "(observed_count={}, expected_count={}, path='{}', binary={})",
                                   cameras.size(), expected.cameras.size(),
                                   lfs::core::path_to_utf8(output_sparse_path), binary));
        LFS_ASSERT_MSG(images.size() == expected.images.size(),
                       std::format("COLMAP writer round-trip must preserve the image count "
                                   "(observed_count={}, expected_count={}, path='{}', binary={})",
                                   images.size(), expected.images.size(),
                                   lfs::core::path_to_utf8(output_sparse_path), binary));

        for (const auto& [camera_id, source] : expected.cameras) {
            const auto written = cameras.find(camera_id);
            LFS_ASSERT_MSG(written != cameras.end(),
                           std::format("COLMAP writer round-trip must preserve camera ids "
                                       "(missing_camera_id={}, observed_camera_count={}, "
                                       "expected_camera_count={}, path='{}')",
                                       camera_id, cameras.size(), expected.cameras.size(),
                                       lfs::core::path_to_utf8(output_sparse_path)));
            LFS_ASSERT_MSG(written->second.model_id == source.model_id &&
                               written->second.width == source.width &&
                               written->second.height == source.height &&
                               written->second.params == source.params,
                           std::format("COLMAP writer round-trip changed camera {}", camera_id));
        }

        for (size_t i = 0; i < expected.images.size(); ++i) {
            const auto& source = expected.images[i];
            const auto& written = images[i];
            LFS_ASSERT_MSG(written.image_id == source.image_id &&
                               written.camera_id == source.camera_id &&
                               written.name == source.name &&
                               written.points2D.size() == source.points2D.size(),
                           std::format("COLMAP writer round-trip changed image record {}", i));
            for (size_t value = 0; value < source.qvec.size(); ++value) {
                LFS_ASSERT_MSG(std::abs(written.qvec[value] - source.qvec[value]) <= 1e-6f,
                               std::format("COLMAP writer round-trip changed image {} quaternion",
                                           source.image_id));
            }
            for (size_t value = 0; value < source.tvec.size(); ++value) {
                LFS_ASSERT_MSG(std::abs(written.tvec[value] - source.tvec[value]) <= 1e-6f,
                               std::format("COLMAP writer round-trip changed image {} translation",
                                           source.image_id));
            }
            for (size_t point_index = 0; point_index < source.points2D.size(); ++point_index) {
                const auto& source_point = source.points2D[point_index];
                const auto& written_point = written.points2D[point_index];
                LFS_ASSERT_MSG(source_point.x == written_point.x &&
                                   source_point.y == written_point.y &&
                                   source_point.point3D_id == written_point.point3D_id,
                               std::format("COLMAP writer round-trip changed image {} point2D {}",
                                           source.image_id, point_index));
            }
        }

        if (!expected.points3D.empty()) {
            const auto points = binary
                                    ? read_point3D_binary_records(output_sparse_path / "points3D.bin")
                                    : read_point3D_text_records(output_sparse_path / "points3D.txt");
            LFS_ASSERT_MSG(points.size() == expected.points3D.size(),
                           std::format("COLMAP writer round-trip must preserve the point3D count "
                                       "(observed_count={}, expected_count={}, path='{}', binary={})",
                                       points.size(), expected.points3D.size(),
                                       lfs::core::path_to_utf8(output_sparse_path), binary));
            for (size_t i = 0; i < expected.points3D.size(); ++i) {
                const auto& source = expected.points3D[i];
                const auto& written = points[i];
                LFS_ASSERT_MSG(written.point3D_id == source.point3D_id &&
                                   std::equal(std::begin(written.xyz), std::end(written.xyz),
                                              std::begin(source.xyz)) &&
                                   std::equal(std::begin(written.color), std::end(written.color),
                                              std::begin(source.color)) &&
                                   written.error == source.error &&
                                   written.track.size() == source.track.size(),
                               std::format("COLMAP writer round-trip changed point {}", source.point3D_id));
                for (size_t track_index = 0; track_index < source.track.size(); ++track_index) {
                    LFS_ASSERT_MSG(written.track[track_index].image_id ==
                                           source.track[track_index].image_id &&
                                       written.track[track_index].point2D_idx ==
                                           source.track[track_index].point2D_idx,
                                   std::format("COLMAP writer round-trip changed point {} track {}",
                                               source.point3D_id, track_index));
                }
            }
        } else if (binary) {
            const auto written_points = read_point3D_binary_records(
                output_sparse_path / "points3D.bin");
            LFS_ASSERT_MSG(written_points.empty(),
                           std::format("COLMAP writer round-trip must preserve an empty point3D set "
                                       "(observed_count={}, expected_count=0, path='{}', binary=true)",
                                       written_points.size(),
                                       lfs::core::path_to_utf8(output_sparse_path / "points3D.bin")));
        } else {
            const auto points_path = output_sparse_path / "points3D.txt";
            std::ifstream points_stream(points_path);
            LFS_ASSERT_MSG(points_stream.is_open(),
                           std::format("COLMAP writer verification must reopen empty points3D.txt "
                                       "(is_open={}, path='{}')",
                                       points_stream.is_open(),
                                       lfs::core::path_to_utf8(points_path)));
            std::string line;
            size_t line_number = 0;
            while (std::getline(points_stream, line)) {
                ++line_number;
                if (!line.empty() && line.back() == '\r')
                    line.pop_back();
                LFS_ASSERT_MSG(line.empty() || line.starts_with('#'),
                               std::format("COLMAP writer must not emit point records for an empty "
                                           "point3D set (line_number={}, line='{}', path='{}')",
                                           line_number, line,
                                           lfs::core::path_to_utf8(points_path)));
            }
        }
    }
    void finalize_colmap_output(std::ofstream& stream, const fs::path& path) {
        stream.flush();
        LFS_ASSERT_MSG(stream.good(),
                       std::format("Failed to flush COLMAP output '{}'", lfs::core::path_to_utf8(path)));
        stream.close();
        LFS_ASSERT_MSG(!stream.fail(),
                       std::format("Failed to close COLMAP output '{}'", lfs::core::path_to_utf8(path)));
    }

    void write_cameras_binary_file(const fs::path& path,
                                   const std::unordered_map<uint32_t, CameraDataIntermediate>& cameras) {
        std::ofstream stream(path, std::ios::trunc | std::ios::binary);
        if (!stream.is_open()) {
            throw std::runtime_error("Failed to open " + lfs::core::path_to_utf8(path));
        }

        write_pod<uint64_t>(stream, static_cast<uint64_t>(cameras.size()));
        for (const uint32_t camera_id : sorted_camera_ids(cameras)) {
            const auto& camera = cameras.at(camera_id);
            write_pod<uint32_t>(stream, camera_id);
            write_pod<int32_t>(stream, camera.model_id);
            write_pod<uint64_t>(stream, static_cast<uint64_t>(camera.width));
            write_pod<uint64_t>(stream, static_cast<uint64_t>(camera.height));
            for (const float param : camera.params) {
                write_pod<double>(stream, static_cast<double>(param));
            }
        }
        finalize_colmap_output(stream, path);
    }

    void write_images_binary_file(const fs::path& path, const std::vector<ImageData>& images) {
        std::ofstream stream(path, std::ios::trunc | std::ios::binary);
        if (!stream.is_open()) {
            throw std::runtime_error("Failed to open " + lfs::core::path_to_utf8(path));
        }

        write_pod<uint64_t>(stream, static_cast<uint64_t>(images.size()));
        for (const auto& image : images) {
            write_pod<uint32_t>(stream, image.image_id);
            for (const float value : image.qvec) {
                write_pod<double>(stream, static_cast<double>(value));
            }
            for (const float value : image.tvec) {
                write_pod<double>(stream, static_cast<double>(value));
            }
            write_pod<uint32_t>(stream, image.camera_id);
            stream.write(image.name.c_str(), static_cast<std::streamsize>(image.name.size() + 1));
            write_pod<uint64_t>(stream, static_cast<uint64_t>(image.points2D.size()));
            for (const auto& point : image.points2D) {
                write_pod<double>(stream, point.x);
                write_pod<double>(stream, point.y);
                write_pod<uint64_t>(stream, point.point3D_id);
            }
        }
        finalize_colmap_output(stream, path);
    }

    void write_points3D_binary_file(const fs::path& path, const std::vector<Point3DData>& points) {
        std::ofstream stream(path, std::ios::trunc | std::ios::binary);
        if (!stream.is_open()) {
            throw std::runtime_error("Failed to open " + lfs::core::path_to_utf8(path));
        }

        write_pod<uint64_t>(stream, static_cast<uint64_t>(points.size()));
        for (const auto& point : points) {
            write_pod<uint64_t>(stream, point.point3D_id);
            write_pod<double>(stream, point.xyz[0]);
            write_pod<double>(stream, point.xyz[1]);
            write_pod<double>(stream, point.xyz[2]);
            write_pod<uint8_t>(stream, point.color[0]);
            write_pod<uint8_t>(stream, point.color[1]);
            write_pod<uint8_t>(stream, point.color[2]);
            write_pod<double>(stream, point.error);
            write_pod<uint64_t>(stream, static_cast<uint64_t>(point.track.size()));
            for (const auto& track : point.track) {
                write_pod<uint32_t>(stream, track.image_id);
                write_pod<uint32_t>(stream, track.point2D_idx);
            }
        }
        finalize_colmap_output(stream, path);
    }

    void write_cameras_text_file(const fs::path& path,
                                 const std::unordered_map<uint32_t, CameraDataIntermediate>& cameras) {
        std::ofstream stream(path, std::ios::trunc);
        if (!stream.is_open()) {
            throw std::runtime_error("Failed to open " + lfs::core::path_to_utf8(path));
        }

        stream << std::setprecision(17);
        stream << "# Camera list with one line of data per camera:\n";
        stream << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n";
        stream << "# Number of cameras: " << cameras.size() << '\n';
        for (const uint32_t camera_id : sorted_camera_ids(cameras)) {
            const auto& camera = cameras.at(camera_id);
            stream << camera_id << ' '
                   << camera_model_name(camera.model_id) << ' '
                   << camera.width << ' '
                   << camera.height;
            for (const float param : camera.params) {
                stream << ' ' << static_cast<double>(param);
            }
            stream << '\n';
        }
        finalize_colmap_output(stream, path);
    }

    void write_images_text_file(const fs::path& path, const std::vector<ImageData>& images) {
        std::ofstream stream(path, std::ios::trunc);
        if (!stream.is_open()) {
            throw std::runtime_error("Failed to open " + lfs::core::path_to_utf8(path));
        }

        stream << std::setprecision(17);
        stream << "# Image list with two lines of data per image:\n";
        stream << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n";
        stream << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n";
        stream << "# Number of images: " << images.size()
               << ", mean observations per image: " << mean_observations_per_image(images) << '\n';

        for (const auto& image : images) {
            stream << image.image_id;
            for (const float value : image.qvec) {
                stream << ' ' << static_cast<double>(value);
            }
            for (const float value : image.tvec) {
                stream << ' ' << static_cast<double>(value);
            }
            stream << ' ' << image.camera_id << ' ' << image.name << '\n';

            for (size_t i = 0; i < image.points2D.size(); ++i) {
                if (i > 0) {
                    stream << ' ';
                }
                const auto& point = image.points2D[i];
                stream << point.x << ' ' << point.y << ' ' << point3D_id_to_text(point.point3D_id);
            }
            stream << '\n';
        }
        finalize_colmap_output(stream, path);
    }

    void write_points3D_text_file(const fs::path& path, const std::vector<Point3DData>& points) {
        std::ofstream stream(path, std::ios::trunc);
        if (!stream.is_open()) {
            throw std::runtime_error("Failed to open " + lfs::core::path_to_utf8(path));
        }

        stream << std::setprecision(17);
        stream << "# 3D point list with one line of data per point:\n";
        stream << "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n";
        stream << "# Number of points: " << points.size()
               << ", mean track length: " << mean_track_length(points) << '\n';

        for (const auto& point : points) {
            stream << point.point3D_id << ' '
                   << point.xyz[0] << ' '
                   << point.xyz[1] << ' '
                   << point.xyz[2] << ' '
                   << static_cast<int>(point.color[0]) << ' '
                   << static_cast<int>(point.color[1]) << ' '
                   << static_cast<int>(point.color[2]) << ' '
                   << point.error;
            for (const auto& track : point.track) {
                stream << ' ' << track.image_id << ' ' << track.point2D_idx;
            }
            stream << '\n';
        }
        finalize_colmap_output(stream, path);
    }

    void remove_obsolete_sparse_files(const fs::path& output_sparse_path, const bool wrote_binary) {
        auto remove_if_present = [&](const char* file_name) {
            std::error_code ec;
            fs::remove(output_sparse_path / file_name, ec);
            if (ec) {
                throw std::runtime_error(std::format(
                    "Failed to remove stale COLMAP sparse file '{}': {}",
                    file_name,
                    ec.message()));
            }
        };

        if (wrote_binary) {
            remove_if_present("cameras.txt");
            remove_if_present("images.txt");
            remove_if_present("points3D.txt");
        } else {
            remove_if_present("cameras.bin");
            remove_if_present("images.bin");
            remove_if_present("points3D.bin");
        }
    }

    // -----------------------------------------------------------------------------
    //  Public API
    // -----------------------------------------------------------------------------

    static fs::path absolute_sparse_directory(const fs::path& path) {
        std::error_code ec;
        auto canonical = fs::weakly_canonical(path, ec);
        if (!ec && !canonical.empty()) {
            return canonical;
        }

        ec.clear();
        auto absolute = fs::absolute(path, ec);
        if (!ec && !absolute.empty()) {
            return absolute.lexically_normal();
        }

        return path;
    }

    Result<std::filesystem::path> find_colmap_sparse_model_path(
        const std::filesystem::path& source_base) {
        if (source_base.empty()) {
            return make_error(ErrorCode::PATH_NOT_FOUND,
                              "COLMAP source path is empty",
                              source_base);
        }

        try {
            const auto search_paths = get_colmap_search_paths(source_base);
            const fs::path cameras_bin = find_file_in_paths(search_paths, "cameras.bin");
            const fs::path images_bin = find_file_in_paths(search_paths, "images.bin");
            if (!cameras_bin.empty() && !images_bin.empty()) {
                return absolute_sparse_directory(cameras_bin.parent_path());
            }

            const fs::path cameras_txt = find_file_in_paths(search_paths, "cameras.txt");
            const fs::path images_txt = find_file_in_paths(search_paths, "images.txt");
            if (!cameras_txt.empty() && !images_txt.empty()) {
                return absolute_sparse_directory(cameras_txt.parent_path());
            }

            return make_error(ErrorCode::MISSING_REQUIRED_FILES,
                              "Missing required COLMAP metadata pair (cameras.bin/images.bin or cameras.txt/images.txt)",
                              source_base);
        } catch (const std::exception& e) {
            return make_error(ErrorCode::READ_FAILURE,
                              std::format("Failed to locate COLMAP sparse model: {}", e.what()),
                              source_base);
        }
    }

    static fs::path get_sparse_file_path(const fs::path& base, const std::string& filename) {
        auto search_paths = get_colmap_search_paths(base);
        auto found = find_file_in_paths(search_paths, filename);

        if (!found.empty()) {
            LOG_TRACE("Found sparse file at: {}", lfs::core::path_to_utf8(found));
            return found;
        }

        std::string error_msg = std::format("Cannot find '{}' in any location", filename);
        LOG_ERROR("{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    PointCloud read_colmap_point_cloud(const std::filesystem::path& filepath,
                                       const LoadOptions& options) {
        LOG_TIMER_TRACE("Read COLMAP point cloud");
        fs::path points3d_file = get_sparse_file_path(filepath, "points3D.bin");
        return read_point3D_binary(points3d_file, options);
    }

    ColmapPointCloudLoadStats read_colmap_point_cloud_with_stats(
        const std::filesystem::path& filepath,
        const LoadOptions& options) {
        LOG_TIMER_TRACE("Read COLMAP point cloud");
        fs::path points3d_file = get_sparse_file_path(filepath, "points3D.bin");
        return point3D_records_to_point_cloud_with_stats(
            read_point3D_binary_records(points3d_file, options),
            options);
    }

    Result<void> validate_colmap_dataset_layout(const std::filesystem::path& base,
                                                const std::string& images_folder,
                                                const LoadOptions& options) {
        try {
            const auto search_paths = get_colmap_search_paths(base);
            const fs::path cameras_bin = find_file_in_paths(search_paths, "cameras.bin");
            const fs::path images_bin = find_file_in_paths(search_paths, "images.bin");
            const fs::path cameras_txt = find_file_in_paths(search_paths, "cameras.txt");
            const fs::path images_txt = find_file_in_paths(search_paths, "images.txt");

            const bool has_binary_pair = !cameras_bin.empty() && !images_bin.empty();
            const bool has_text_pair = !cameras_txt.empty() && !images_txt.empty();

            if (!has_binary_pair && !has_text_pair) {
                return make_error(ErrorCode::MISSING_REQUIRED_FILES,
                                  "Missing required COLMAP metadata pair (cameras.bin/images.bin or cameras.txt/images.txt)",
                                  base);
            }

            std::vector<ImageData> images;
            if (has_binary_pair) {
                images = read_images_binary(images_bin, options);
            } else {
                images = read_images_text_camera_metadata_only(images_txt, options);
            }

            return validate_colmap_dataset_layout_impl(base, images_folder, images, options);
        } catch (const LoadCancelledError& e) {
            return make_error(ErrorCode::CANCELLED, e.what(), base);
        } catch (const std::exception& e) {
            return make_error(ErrorCode::CORRUPTED_DATA,
                              std::format("Failed to validate COLMAP dataset: {}", e.what()),
                              base);
        }
    }

    Result<std::tuple<std::vector<std::shared_ptr<Camera>>, Tensor>>
    read_colmap_cameras_and_images(const std::filesystem::path& base,
                                   const std::string& images_folder,
                                   const LoadOptions& options) {

        LOG_TIMER_TRACE("Read COLMAP cameras and images");

        const float scale_factor = extract_scale_from_folder(images_folder);

        fs::path cams_file = get_sparse_file_path(base, "cameras.bin");
        fs::path images_file = get_sparse_file_path(base, "images.bin");

        auto cam_map = read_cameras_binary(cams_file, scale_factor, options);
        auto images = read_images_binary(images_file, options);

        LOG_INFO("Read {} cameras and {} images from COLMAP", cam_map.size(), images.size());

        if (auto validation = validate_colmap_dataset_layout_impl(base, images_folder, images, options); !validation) {
            return std::unexpected(validation.error());
        }

        return assemble_colmap_cameras(base, cam_map, images, images_folder, options);
    }

    PointCloud read_colmap_point_cloud_text(const std::filesystem::path& filepath,
                                            const LoadOptions& options) {
        LOG_TIMER_TRACE("Read COLMAP point cloud (text)");
        fs::path points3d_file = get_sparse_file_path(filepath, "points3D.txt");
        return read_point3D_text(points3d_file, options);
    }

    ColmapPointCloudLoadStats read_colmap_point_cloud_text_with_stats(
        const std::filesystem::path& filepath,
        const LoadOptions& options) {
        LOG_TIMER_TRACE("Read COLMAP point cloud (text)");
        fs::path points3d_file = get_sparse_file_path(filepath, "points3D.txt");
        return point3D_records_to_point_cloud_with_stats(
            read_point3D_text_records(points3d_file, options, TrackParseMode::CountOnly),
            options);
    }

    Result<std::tuple<std::vector<std::shared_ptr<Camera>>, Tensor>>
    read_colmap_cameras_and_images_text(const std::filesystem::path& base,
                                        const std::string& images_folder,
                                        const LoadOptions& options) {

        LOG_TIMER_TRACE("Read COLMAP cameras and images (text)");

        const float scale_factor = extract_scale_from_folder(images_folder);

        fs::path cams_file = get_sparse_file_path(base, "cameras.txt");
        fs::path images_file = get_sparse_file_path(base, "images.txt");

        auto cam_map = read_cameras_text(cams_file, scale_factor, options);
        auto images = read_images_text_camera_metadata_only(images_file, options);

        LOG_INFO("Read {} cameras and {} images from COLMAP text files", cam_map.size(), images.size());

        if (auto validation = validate_colmap_dataset_layout_impl(base, images_folder, images, options); !validation) {
            return std::unexpected(validation.error());
        }

        return assemble_colmap_cameras(base, cam_map, images, images_folder, options);
    }

    Result<std::tuple<std::vector<std::shared_ptr<Camera>>, Tensor>>
    read_colmap_cameras_only(const std::filesystem::path& sparse_path, float scale_factor) {
        LOG_TIMER_TRACE("Read COLMAP cameras only");

        std::unordered_map<uint32_t, CameraDataIntermediate> cam_map;
        std::vector<ImageData> images;

        const bool has_binary = fs::exists(sparse_path / "cameras.bin") && fs::exists(sparse_path / "images.bin");
        const bool has_text = fs::exists(sparse_path / "cameras.txt") && fs::exists(sparse_path / "images.txt");

        if (!has_binary && !has_text) {
            return make_error(ErrorCode::PATH_NOT_FOUND,
                              "Missing cameras.bin/images.bin or cameras.txt/images.txt",
                              sparse_path);
        }

        if (has_binary) {
            cam_map = read_cameras_binary(sparse_path / "cameras.bin", scale_factor);
            images = read_images_binary(sparse_path / "images.bin");
        } else {
            cam_map = read_cameras_text(sparse_path / "cameras.txt", scale_factor);
            images = read_images_text_camera_metadata_only(sparse_path / "images.txt");
        }

        LOG_INFO("Read {} cameras and {} images from COLMAP", cam_map.size(), images.size());

        std::vector<std::shared_ptr<Camera>> cameras;
        cameras.reserve(images.size());

        std::vector<float> camera_positions;
        camera_positions.reserve(images.size() * 3);

        for (size_t i = 0; i < images.size(); ++i) {
            const ImageData& img = images[i];

            auto it = cam_map.find(img.camera_id);
            if (it == cam_map.end()) {
                return make_error(ErrorCode::CORRUPTED_DATA,
                                  std::format("Camera ID {} not found for image '{}'", img.camera_id, img.name),
                                  sparse_path);
            }

            const auto& cam_data = it->second;

            Tensor R = qvec2rotmat(img.qvec);
            Tensor T = Tensor::from_vector(img.tvec, {3}, Device::CPU);

            auto R_cpu = R.cpu();
            auto T_cpu = T.cpu();

            float RT[9];
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    RT[r * 3 + c] = R_cpu.ptr<float>()[c * 3 + r];
                }
            }

            float cam_pos[3] = {0, 0, 0};
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    cam_pos[r] -= RT[r * 3 + c] * T_cpu.ptr<float>()[c];
                }
            }

            camera_positions.push_back(cam_pos[0]);
            camera_positions.push_back(cam_pos[1]);
            camera_positions.push_back(cam_pos[2]);

            auto model_it = camera_model_ids.find(cam_data.model_id);
            if (model_it == camera_model_ids.end()) {
                return make_error(ErrorCode::UNSUPPORTED_FORMAT,
                                  std::format("Invalid camera model ID {} for image '{}'", cam_data.model_id, img.name),
                                  sparse_path);
            }

            CAMERA_MODEL model = model_it->second.first;
            const auto& params = cam_data.params;

            float focal_x = 0, focal_y = 0, center_x = 0, center_y = 0;
            Tensor radial_dist, tangential_dist;
            lfs::core::CameraModelType camera_model_type = lfs::core::CameraModelType::PINHOLE;

            switch (model) {
            case CAMERA_MODEL::SIMPLE_PINHOLE:
                focal_x = focal_y = params[0];
                center_x = params[1];
                center_y = params[2];
                radial_dist = Tensor::empty({0}, Device::CPU);
                tangential_dist = Tensor::empty({0}, Device::CPU);
                break;

            case CAMERA_MODEL::PINHOLE:
                focal_x = params[0];
                focal_y = params[1];
                center_x = params[2];
                center_y = params[3];
                radial_dist = Tensor::empty({0}, Device::CPU);
                tangential_dist = Tensor::empty({0}, Device::CPU);
                break;

            case CAMERA_MODEL::SIMPLE_RADIAL:
                focal_x = focal_y = params[0];
                center_x = params[1];
                center_y = params[2];
                if (params[3] != 0.0f) {
                    radial_dist = Tensor::from_vector({params[3]}, {1}, Device::CPU);
                } else {
                    radial_dist = Tensor::empty({0}, Device::CPU);
                }
                tangential_dist = Tensor::empty({0}, Device::CPU);
                break;

            case CAMERA_MODEL::RADIAL:
                focal_x = focal_y = params[0];
                center_x = params[1];
                center_y = params[2];
                radial_dist = make_distortion_tensor({params[3], params[4]});
                tangential_dist = Tensor::empty({0}, Device::CPU);
                break;

            case CAMERA_MODEL::OPENCV:
                focal_x = params[0];
                focal_y = params[1];
                center_x = params[2];
                center_y = params[3];
                radial_dist = make_distortion_tensor({params[4], params[5]});
                tangential_dist = make_distortion_tensor({params[6], params[7]});
                break;

            case CAMERA_MODEL::FULL_OPENCV:
                focal_x = params[0];
                focal_y = params[1];
                center_x = params[2];
                center_y = params[3];
                radial_dist = make_distortion_tensor({params[4], params[5], params[8], params[9], params[10], params[11]});
                tangential_dist = make_distortion_tensor({params[6], params[7]});
                break;

            case CAMERA_MODEL::OPENCV_FISHEYE:
                focal_x = params[0];
                focal_y = params[1];
                center_x = params[2];
                center_y = params[3];
                radial_dist = Tensor::from_vector({params[4], params[5], params[6], params[7]}, {4}, Device::CPU);
                tangential_dist = Tensor::empty({0}, Device::CPU);
                camera_model_type = lfs::core::CameraModelType::FISHEYE;
                break;

            case CAMERA_MODEL::RADIAL_FISHEYE:
                focal_x = focal_y = params[0];
                center_x = params[1];
                center_y = params[2];
                radial_dist = Tensor::from_vector({params[3], params[4]}, {2}, Device::CPU);
                tangential_dist = Tensor::empty({0}, Device::CPU);
                camera_model_type = lfs::core::CameraModelType::FISHEYE;
                break;

            case CAMERA_MODEL::SIMPLE_RADIAL_FISHEYE:
                focal_x = focal_y = params[0];
                center_x = params[1];
                center_y = params[2];
                radial_dist = Tensor::from_vector({params[3]}, {1}, Device::CPU);
                tangential_dist = Tensor::empty({0}, Device::CPU);
                camera_model_type = lfs::core::CameraModelType::FISHEYE;
                break;

            case CAMERA_MODEL::THIN_PRISM_FISHEYE:
                focal_x = params[0];
                focal_y = params[1];
                center_x = params[2];
                center_y = params[3];
                radial_dist = Tensor::from_vector({params[4], params[5], params[8], params[9]}, {4}, Device::CPU);
                tangential_dist = Tensor::from_vector({params[6], params[7], params[10], params[11]}, {4}, Device::CPU);
                camera_model_type = lfs::core::CameraModelType::THIN_PRISM_FISHEYE;
                break;

            case CAMERA_MODEL::FOV:
                return make_error(ErrorCode::UNSUPPORTED_FORMAT,
                                  std::format("FOV camera model not supported for image '{}'", img.name),
                                  sparse_path);

            case CAMERA_MODEL::EQUIRECTANGULAR:
                // Equirectangular 360 panorama. params = [width, height]; no real
                // intrinsics. The EQUIRECTANGULAR rasterizer derives projection
                // from the image dimensions, so we only set placeholder focal/center.
                focal_x = focal_y = EQUIRECTANGULAR_DUMMY_FOCAL;
                center_x = 0.5f * static_cast<float>(cam_data.width);
                center_y = 0.5f * static_cast<float>(cam_data.height);
                radial_dist = Tensor::empty({0}, Device::CPU);
                tangential_dist = Tensor::empty({0}, Device::CPU);
                camera_model_type = lfs::core::CameraModelType::EQUIRECTANGULAR;
                break;

            default:
                return make_error(ErrorCode::UNSUPPORTED_FORMAT,
                                  std::format("Unsupported camera model for image '{}'", img.name),
                                  sparse_path);
            }

            auto camera = std::make_shared<Camera>(
                R,
                T,
                focal_x, focal_y,
                center_x, center_y,
                radial_dist,
                tangential_dist,
                camera_model_type,
                img.name,
                fs::path{}, // Empty image path
                fs::path{}, // Empty mask path
                cam_data.width,
                cam_data.height,
                static_cast<int>(i));

            cameras.push_back(std::move(camera));
        }

        Tensor scene_center_tensor = Tensor::from_vector(camera_positions, {images.size(), 3}, Device::CPU);
        Tensor scene_center = scene_center_tensor.mean({0}, false);

        LOG_INFO("Loaded {} cameras (no images required)", cameras.size());

        return std::make_tuple(std::move(cameras), scene_center);
    }

    Result<void> write_colmap_reconstruction(
        const std::filesystem::path& source_base,
        const std::filesystem::path& output_sparse_path,
        const std::vector<ColmapCameraWriteData>& cameras,
        const PointCloud* point_cloud,
        const glm::mat4& point_cloud_transform,
        const ColmapWriteOptions& options) {
        LOG_TIMER_TRACE("Write COLMAP reconstruction");

        if (source_base.empty()) {
            return make_error(ErrorCode::PATH_NOT_FOUND,
                              "COLMAP source path is required for write-back",
                              source_base);
        }
        if (output_sparse_path.empty()) {
            return make_error(ErrorCode::PATH_NOT_FOUND,
                              "COLMAP output sparse path is required",
                              output_sparse_path);
        }
        if (cameras.empty()) {
            return make_error(ErrorCode::EMPTY_DATASET,
                              "Cannot write COLMAP reconstruction without cameras",
                              output_sparse_path);
        }

        try {
            ColmapSparseModelData model = read_colmap_sparse_model_for_write(source_base);
            update_image_poses_from_cameras(model.images, cameras);
            const bool clear_observation_links = point_cloud_requires_untracked_export(model.points3D, point_cloud);
            model.points3D = build_points3D_for_write(model.points3D, point_cloud, point_cloud_transform);
            if (clear_observation_links || model.points3D.empty()) {
                clear_image_point3D_references(model.images);
            }

            std::error_code ec;
            const auto output_parent = output_sparse_path.parent_path().empty()
                                           ? fs::path{"."}
                                           : output_sparse_path.parent_path();
            fs::create_directories(output_parent, ec);
            if (ec) {
                return make_error(ErrorCode::PERMISSION_DENIED,
                                  std::format("Cannot create COLMAP output parent directory: {}", ec.message()),
                                  output_sparse_path);
            }

            const bool write_binary = options.format == ColmapWriteFormat::Binary ||
                                      (options.format == ColmapWriteFormat::Auto && model.source_binary);
            validate_colmap_model_for_write(model, write_binary);

            const auto staging_path = make_atomic_temp_output_path(output_sparse_path);
            ScopedStagingDirectory staging_cleanup(staging_path);
            clone_sparse_directory_for_staging(output_sparse_path, staging_path);

            if (write_binary) {
                write_cameras_binary_file(staging_path / "cameras.bin", model.cameras);
                write_images_binary_file(staging_path / "images.bin", model.images);
                write_points3D_binary_file(staging_path / "points3D.bin", model.points3D);
            } else {
                write_cameras_text_file(staging_path / "cameras.txt", model.cameras);
                write_images_text_file(staging_path / "images.txt", model.images);
                write_points3D_text_file(staging_path / "points3D.txt", model.points3D);
            }
            remove_obsolete_sparse_files(staging_path, write_binary);
            validate_colmap_round_trip(staging_path, write_binary, model);
            sync_colmap_generation(staging_path, write_binary);
            publish_staged_colmap_generation(staging_path, output_sparse_path);

            LOG_INFO("Wrote COLMAP {} reconstruction to '{}' ({} cameras, {} images, {} points)",
                     write_binary ? "binary" : "text",
                     lfs::core::path_to_utf8(output_sparse_path),
                     model.cameras.size(),
                     model.images.size(),
                     model.points3D.size());
            return {};
        } catch (const std::exception& e) {
            return make_error(ErrorCode::WRITE_FAILURE,
                              std::format("Failed to write COLMAP reconstruction: {}", e.what()),
                              output_sparse_path);
        }
    }

} // namespace lfs::io
