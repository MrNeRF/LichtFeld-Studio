/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "colmap.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "io/filesystem_utils.hpp"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

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

        [[nodiscard]] bool should_poll_cancel(const size_t index) {
            return (index % CANCEL_POLL_INTERVAL) == 0;
        }
    } // namespace

    // -----------------------------------------------------------------------------
    //  Quaternion to rotation matrix (torch-free)
    // -----------------------------------------------------------------------------
    inline Tensor qvec2rotmat(const std::vector<float>& q_raw) {
        if (q_raw.size() != 4) {
            LOG_ERROR("Quaternion must have 4 elements");
            throw std::runtime_error("Invalid quaternion size");
        }

        // Normalize quaternion
        float len = std::sqrt(q_raw[0] * q_raw[0] + q_raw[1] * q_raw[1] +
                              q_raw[2] * q_raw[2] + q_raw[3] * q_raw[3]);
        if (len < 1e-8f) {
            LOG_ERROR("Quaternion has zero length");
            throw std::runtime_error("Zero-length quaternion");
        }

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
        std::vector<Point3DTrackElement> track;
    };

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
    static inline uint64_t read_u64(const char*& p) {
        uint64_t v;
        std::memcpy(&v, p, 8);
        p += 8;
        return v;
    }
    static inline uint32_t read_u32(const char*& p) {
        uint32_t v;
        std::memcpy(&v, p, 4);
        p += 4;
        return v;
    }
    static inline int32_t read_i32(const char*& p) {
        int32_t v;
        std::memcpy(&v, p, 4);
        p += 4;
        return v;
    }
    static inline double read_f64(const char*& p) {
        double v;
        std::memcpy(&v, p, 8);
        p += 8;
        return v;
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
        UNDEFINED = 11
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
        {11, {CAMERA_MODEL::UNDEFINED, -1}}};

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
        {"THIN_PRISM_FISHEYE", CAMERA_MODEL::THIN_PRISM_FISHEYE}};

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

        const fs::path images_path = base / lfs::core::utf8_to_path(images_folder);
        if (!safe_is_directory(images_path)) {
            return make_error(ErrorCode::PATH_NOT_FOUND, "Images folder does not exist", images_path);
        }

        const auto basename_layout = scan_image_basename_layout(images_path, options);
        RecursiveFileCache image_cache(images_path, options.cancel_requested);
        MaskDirCache mask_cache(base, options.cancel_requested);

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

        auto sz = static_cast<std::streamsize>(f.tellg());
        auto buf = std::make_unique<std::vector<char>>(static_cast<size_t>(sz));

        f.seekg(0, std::ios::beg);
        f.read(buf->data(), sz);
        if (!f) {
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

        uint64_t n_images = read_u64(cur);
        LOG_DEBUG("Reading {} images from binary file", n_images);
        std::vector<ImageData> images;
        images.reserve(n_images);

        for (uint64_t i = 0; i < n_images; ++i) {
            if (should_poll_cancel(static_cast<size_t>(i))) {
                throw_if_load_cancel_requested(options, "COLMAP image metadata load cancelled");
            }
            ImageData img;
            img.image_id = read_u32(cur);

            // Read quaternion [w, x, y, z]
            for (int k = 0; k < 4; ++k) {
                img.qvec[k] = static_cast<float>(read_f64(cur));
            }

            // Read translation [x, y, z]
            for (int k = 0; k < 3; ++k) {
                img.tvec[k] = static_cast<float>(read_f64(cur));
            }

            img.camera_id = read_u32(cur);

            img.name.assign(cur);
            cur += img.name.size() + 1;

            uint64_t npts = read_u64(cur);
            img.points2D.reserve(npts);
            for (uint64_t j = 0; j < npts; ++j) {
                ImagePoint2D point;
                point.x = read_f64(cur);
                point.y = read_f64(cur);
                point.point3D_id = read_u64(cur);
                img.points2D.push_back(point);
            }

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
        auto buf_owner = read_binary(file_path);
        const char* cur = buf_owner->data();
        const char* end = cur + buf_owner->size();

        uint64_t n_cams = read_u64(cur);
        LOG_DEBUG("Reading {} cameras from binary file{}", n_cams,
                  scale_factor != 1.0f ? std::format(" with scale factor {}", scale_factor) : "");

        std::unordered_map<uint32_t, CameraDataIntermediate> cams;
        cams.reserve(n_cams);

        for (uint64_t i = 0; i < n_cams; ++i) {
            if (should_poll_cancel(static_cast<size_t>(i))) {
                throw_if_load_cancel_requested(options, "COLMAP camera metadata load cancelled");
            }
            CameraDataIntermediate cam;
            cam.camera_id = read_u32(cur);
            cam.model_id = read_i32(cur);
            cam.width = static_cast<int>(read_u64(cur));
            cam.height = static_cast<int>(read_u64(cur));

            if (scale_factor != 1.0f) {
                cam.width = static_cast<int>(cam.width / scale_factor);
                cam.height = static_cast<int>(cam.height / scale_factor);
            }

            auto it = camera_model_ids.find(cam.model_id);
            if (it == camera_model_ids.end() || it->second.second < 0) {
                LOG_ERROR("Unsupported camera-model id: {}", cam.model_id);
                throw std::runtime_error("Unsupported camera-model id");
            }

            int32_t param_cnt = it->second.second;
            cam.params.resize(param_cnt);

            for (int j = 0; j < param_cnt; j++) {
                cam.params[j] = static_cast<float>(read_f64(cur));
            }

            if (scale_factor != 1.0f) {
                scale_camera_intrinsics(it->second.first, cam.params, scale_factor);
            }

            cams.emplace(cam.camera_id, std::move(cam));
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

        uint64_t N = read_u64(cur);
        LOG_DEBUG("Reading {} 3D points from binary file", N);

        std::vector<Point3DData> points;
        points.reserve(N);

        for (uint64_t i = 0; i < N; ++i) {
            if (should_poll_cancel(static_cast<size_t>(i))) {
                throw_if_load_cancel_requested(options, "COLMAP point cloud load cancelled");
            }
            Point3DData point;
            point.point3D_id = read_u64(cur);

            point.xyz[0] = read_f64(cur);
            point.xyz[1] = read_f64(cur);
            point.xyz[2] = read_f64(cur);

            point.color[0] = static_cast<uint8_t>(*cur++);
            point.color[1] = static_cast<uint8_t>(*cur++);
            point.color[2] = static_cast<uint8_t>(*cur++);

            point.error = read_f64(cur);
            const uint64_t track_len = read_u64(cur);
            point.track.reserve(track_len);
            for (uint64_t j = 0; j < track_len; ++j) {
                Point3DTrackElement track;
                track.image_id = read_u32(cur);
                track.point2D_idx = read_u32(cur);
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
        return std::isalpha(static_cast<unsigned char>(img.name[dot_pos + 1]));
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
        double x = 0.0;
        double y = 0.0;
        std::string point_id;
        while (iss >> x >> y >> point_id) {
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
                                            const LoadOptions& options = {}) {
        LOG_TIMER_TRACE("Read images.txt");
        auto lines = read_text_file(file_path, options);

        std::vector<ImageData> images;
        images.reserve(lines.size());

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
                ImageData maybe_next_image;
                if (!parse_image_metadata_line(lines[line_idx + 1], maybe_next_image)) {
                    img.points2D = parse_points2D_text_line(lines[line_idx + 1]);
                    ++line_idx;
                }
            }

            images.push_back(std::move(img));
        }

        if (images.empty()) {
            LOG_ERROR("No valid images found in {}", lfs::core::path_to_utf8(file_path));
            throw std::runtime_error("No valid images in images.txt");
        }

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
        auto lines = read_text_file(file_path, options);

        LOG_DEBUG("Reading {} cameras from text file{}", lines.size(),
                  scale_factor != 1.0f ? std::format(" with scale factor {}", scale_factor) : "");

        std::unordered_map<uint32_t, CameraDataIntermediate> cams;

        for (size_t line_idx = 0; line_idx < lines.size(); ++line_idx) {
            if (should_poll_cancel(line_idx)) {
                throw_if_load_cancel_requested(options, "COLMAP camera metadata parse cancelled");
            }
            const auto& line = lines[line_idx];
            const auto tokens = split_string(line, ' ');
            if (tokens.size() < 4) {
                LOG_ERROR("Invalid format in cameras.txt: {}", line);
                throw std::runtime_error("Invalid format in cameras.txt");
            }

            CameraDataIntermediate cam;
            cam.camera_id = std::stoul(tokens[0]);

            if (!camera_model_names.contains(tokens[1])) {
                LOG_ERROR("Unknown camera model: {}", tokens[1]);
                throw std::runtime_error("Unknown camera model");
            }

            cam.model_id = static_cast<int>(camera_model_names.at(tokens[1]));
            cam.width = std::stoi(tokens[2]);
            cam.height = std::stoi(tokens[3]);

            if (scale_factor != 1.0f) {
                cam.width = static_cast<int>(cam.width / scale_factor);
                cam.height = static_cast<int>(cam.height / scale_factor);
            }

            for (size_t j = 4; j < tokens.size(); ++j) {
                cam.params.push_back(std::stof(tokens[j]));
            }

            auto it = camera_model_ids.find(cam.model_id);
            if (it != camera_model_ids.end() && scale_factor != 1.0f) {
                scale_camera_intrinsics(it->second.first, cam.params, scale_factor);
            }

            cams.emplace(cam.camera_id, std::move(cam));
        }

        return cams;
    }

    // -----------------------------------------------------------------------------
    //  points3D.txt
    // -----------------------------------------------------------------------------
    std::vector<Point3DData> read_point3D_text_records(const std::filesystem::path& file_path,
                                                       const LoadOptions& options = {}) {
        LOG_TIMER_TRACE("Read points3D.txt");
        auto lines = read_text_file(file_path, options);
        uint64_t N = lines.size();
        LOG_DEBUG("Reading {} 3D points from text file", N);

        std::vector<Point3DData> points;
        points.reserve(N);

        for (uint64_t i = 0; i < N; ++i) {
            if (should_poll_cancel(static_cast<size_t>(i))) {
                throw_if_load_cancel_requested(options, "COLMAP point cloud parse cancelled");
            }
            const auto& line = lines[i];
            const auto tokens = split_string(line, ' ');

            if (tokens.size() < 8) {
                LOG_ERROR("Invalid format in points3D.txt: {}", line);
                throw std::runtime_error("Invalid format in points3D.txt");
            }

            Point3DData point;
            point.point3D_id = std::stoull(tokens[0]);
            point.xyz[0] = std::stod(tokens[1]);
            point.xyz[1] = std::stod(tokens[2]);
            point.xyz[2] = std::stod(tokens[3]);

            point.color[0] = static_cast<uint8_t>(std::stoi(tokens[4]));
            point.color[1] = static_cast<uint8_t>(std::stoi(tokens[5]));
            point.color[2] = static_cast<uint8_t>(std::stoi(tokens[6]));
            point.error = std::stod(tokens[7]);

            for (size_t j = 8; j + 1 < tokens.size(); j += 2) {
                point.track.push_back(Point3DTrackElement{
                    .image_id = static_cast<uint32_t>(std::stoul(tokens[j])),
                    .point2D_idx = static_cast<uint32_t>(std::stoul(tokens[j + 1])),
                });
            }

            points.push_back(std::move(point));
        }

        return points;
    }

    PointCloud read_point3D_text(const std::filesystem::path& file_path,
                                 const LoadOptions& options = {}) {
        return point3D_records_to_point_cloud(read_point3D_text_records(file_path, options));
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
        bool used_recursive_image_lookup = false;

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

            // Validate mask dimensions match image dimensions
            if (!mask_path.empty()) {
                auto [img_w, img_h, img_c] = lfs::core::get_image_info(image_path);
                auto [mask_w, mask_h, mask_c] = lfs::core::get_image_info(mask_path);
                if (img_w != mask_w || img_h != mask_h) {
                    return make_error(ErrorCode::MASK_SIZE_MISMATCH,
                                      std::format("Mask '{}' is {}x{} but image '{}' is {}x{}",
                                                  lfs::core::path_to_utf8(mask_path.filename()), mask_w, mask_h,
                                                  img.name, img_w, img_h),
                                      mask_path);
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
                static_cast<int>(img.camera_id));

            camera->precompute_undistortion();

            cameras.push_back(std::move(camera));
        }

        // Compute scene center as mean of camera positions
        Tensor scene_center_tensor = Tensor::from_vector(camera_positions, {images.size(), 3}, Device::CPU);
        Tensor scene_center = scene_center_tensor.mean({0}, false);

        LOG_INFO("Training with {} images", cameras.size());

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
        std::unordered_map<std::string, const ColmapCameraWriteData*> camera_by_name;
        camera_by_name.reserve(cameras.size());
        for (const auto& item : cameras) {
            if (!item.camera) {
                continue;
            }
            camera_by_name.emplace(item.camera->image_name(), &item);
        }

        size_t matched = 0;
        for (auto& image : images) {
            const auto it = camera_by_name.find(image.name);
            if (it == camera_by_name.end()) {
                throw std::runtime_error(std::format("No current scene camera found for COLMAP image '{}'", image.name));
            }
            auto [qvec, tvec] = transformed_camera_pose(*it->second->camera, it->second->data_world_transform);
            image.qvec = std::move(qvec);
            image.tvec = std::move(tvec);
            ++matched;
        }

        if (matched == 0) {
            throw std::runtime_error("No COLMAP image poses were matched for export");
        }
        if (matched != camera_by_name.size()) {
            LOG_WARN("COLMAP export matched {} source images but received {} scene cameras",
                     matched, camera_by_name.size());
        }
    }

    uint8_t float_color_to_u8(const float value) {
        const float clamped = std::clamp(value, 0.0f, 1.0f);
        return static_cast<uint8_t>(std::round(clamped * 255.0f));
    }

    std::vector<Point3DData> build_points3D_for_write(
        const std::vector<Point3DData>& source_points,
        const PointCloud* point_cloud,
        const glm::mat4& point_cloud_transform) {
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
            for (auto& p : points) {
                const glm::vec3 local(static_cast<float>(p.xyz[0]),
                                      static_cast<float>(p.xyz[1]),
                                      static_cast<float>(p.xyz[2]));
                const glm::vec3 world = glm::vec3(point_cloud_transform * glm::vec4(local, 1.0f));
                p.xyz[0] = world.x;
                p.xyz[1] = world.y;
                p.xyz[2] = world.z;
            }
            return points;
        }

        const size_t N = static_cast<size_t>(point_cloud->size());
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

        auto means_cpu = point_cloud->means.to(DataType::Float32).cpu().contiguous();
        auto means_acc = means_cpu.accessor<float, 2>();

        const bool has_colors = point_cloud->colors.is_valid() &&
                                static_cast<size_t>(point_cloud->colors.numel()) >= N * 3;
        Tensor colors_cpu;
        Tensor colors_float_cpu;
        if (has_colors) {
            colors_cpu = point_cloud->colors.cpu().contiguous();
            if (colors_cpu.dtype() != DataType::UInt8) {
                colors_float_cpu = colors_cpu.to(DataType::Float32).contiguous();
            }
        }

        for (size_t i = 0; i < N; ++i) {
            const glm::vec3 local_point(means_acc(i, 0), means_acc(i, 1), means_acc(i, 2));
            const glm::vec3 world_point = glm::vec3(point_cloud_transform * glm::vec4(local_point, 1.0f));
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
                images = read_images_text(images_txt, options);
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

    Result<std::tuple<std::vector<std::shared_ptr<Camera>>, Tensor>>
    read_colmap_cameras_and_images_text(const std::filesystem::path& base,
                                        const std::string& images_folder,
                                        const LoadOptions& options) {

        LOG_TIMER_TRACE("Read COLMAP cameras and images (text)");

        const float scale_factor = extract_scale_from_folder(images_folder);

        fs::path cams_file = get_sparse_file_path(base, "cameras.txt");
        fs::path images_file = get_sparse_file_path(base, "images.txt");

        auto cam_map = read_cameras_text(cams_file, scale_factor, options);
        auto images = read_images_text(images_file, options);

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
            images = read_images_text(sparse_path / "images.txt");
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
            if (clear_observation_links) {
                clear_image_point3D_references(model.images);
            }

            std::error_code ec;
            fs::create_directories(output_sparse_path, ec);
            if (ec) {
                return make_error(ErrorCode::PERMISSION_DENIED,
                                  std::format("Cannot create COLMAP output directory: {}", ec.message()),
                                  output_sparse_path);
            }

            const bool write_binary = options.format == ColmapWriteFormat::Binary ||
                                      (options.format == ColmapWriteFormat::Auto && model.source_binary);

            if (write_binary) {
                write_cameras_binary_file(output_sparse_path / "cameras.bin", model.cameras);
                write_images_binary_file(output_sparse_path / "images.bin", model.images);
                write_points3D_binary_file(output_sparse_path / "points3D.bin", model.points3D);
            } else {
                write_cameras_text_file(output_sparse_path / "cameras.txt", model.cameras);
                write_images_text_file(output_sparse_path / "images.txt", model.images);
                write_points3D_text_file(output_sparse_path / "points3D.txt", model.points3D);
            }
            remove_obsolete_sparse_files(output_sparse_path, write_binary);

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
