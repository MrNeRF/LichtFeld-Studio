/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "transforms.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/tensor.hpp"
#include "formats/colmap.hpp"
#include "tinyply.hpp"
#include <array>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <limits>
#include <nlohmann/json.hpp>
#include <numbers>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lfs::io {

    // Import types from lfs::core for convenience
    using lfs::core::Camera;
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::PointCloud;
    using lfs::core::Tensor;

    constexpr int DEFAULT_NUM_INIT_GAUSSIAN = 10000;
    constexpr uint64_t DEFAULT_RANDOM_SEED = 8128;
    constexpr float EQUIRECTANGULAR_DUMMY_FOCAL = 20.0f;

    namespace {
        constexpr uintmax_t MAX_TRANSFORMS_JSON_BYTES = 64ULL * 1024ULL * 1024ULL;
        constexpr size_t MAX_TRANSFORMS_FRAMES = 1'000'000;
        constexpr size_t MAX_TRANSFORMS_PATH_BYTES = 4096;
        constexpr float MAX_TRANSFORM_COMPONENT = 1.0e12f;
        constexpr float MAX_DISTORTION_MAGNITUDE = 1.0e8f;
        constexpr float MAX_INTRINSIC_DIMENSION_MULTIPLIER = 1.0e6f;
        constexpr double MIN_AFFINE_DETERMINANT = 1.0e-8;
        constexpr float AFFINE_ROW_TOLERANCE = 1.0e-4f;

        struct ValidatedTransformFrame {
            std::string file_path;
            std::array<float, 16> matrix{};
        };

        [[nodiscard]] float finite_json_float_value(
            const nlohmann::json& value,
            const std::string_view key) {
            if (!value.is_number())
                throw std::runtime_error("Transforms field '" + std::string(key) + "' must be numeric");
            const double parsed = value.get<double>();
            if (!std::isfinite(parsed) || std::abs(parsed) > std::numeric_limits<float>::max())
                throw std::runtime_error("Transforms field '" + std::string(key) + "' must be finite float32");
            return static_cast<float>(parsed);
        }

        [[nodiscard]] float finite_json_float(
            const nlohmann::json& object,
            const std::string_view key) {
            return finite_json_float_value(object.at(std::string(key)), key);
        }

        [[nodiscard]] int positive_image_dimension(
            const nlohmann::json& object,
            const std::string_view key) {
            const auto& value = object.at(std::string(key));
            if (!value.is_number())
                throw std::runtime_error("Transforms field '" + std::string(key) + "' must be numeric");
            const double parsed = value.get<double>();
            if (!std::isfinite(parsed) || parsed <= 0.0 || std::floor(parsed) != parsed ||
                parsed > static_cast<double>(std::numeric_limits<int>::max())) {
                throw std::runtime_error("Transforms field '" + std::string(key) + "' must be a positive int");
            }
            return static_cast<int>(parsed);
        }

        [[nodiscard]] std::array<float, 16> validated_transform_matrix(
            const nlohmann::json& matrix,
            const size_t frame_index) {
            if (!matrix.is_array() || matrix.size() != 4)
                throw std::runtime_error(std::format("Frame {} transform_matrix must be a 4x4 array", frame_index));

            std::array<float, 16> result{};
            for (size_t row = 0; row < 4; ++row) {
                if (!matrix[row].is_array() || matrix[row].size() != 4) {
                    throw std::runtime_error(
                        std::format("Frame {} transform_matrix row {} must contain four values", frame_index, row));
                }
                for (size_t column = 0; column < 4; ++column) {
                    const auto& value = matrix[row][column];
                    if (!value.is_number())
                        throw std::runtime_error(std::format(
                            "Frame {} transform_matrix[{}][{}] must be numeric", frame_index, row, column));
                    const double parsed = value.get<double>();
                    if (!std::isfinite(parsed) || std::abs(parsed) > MAX_TRANSFORM_COMPONENT) {
                        throw std::runtime_error(std::format(
                            "Frame {} transform_matrix[{}][{}] is not a bounded finite value",
                            frame_index, row, column));
                    }
                    result[row * 4 + column] = static_cast<float>(parsed);
                }
            }

            for (size_t column = 0; column < 4; ++column) {
                const float expected = column == 3 ? 1.0f : 0.0f;
                if (std::abs(result[12 + column] - expected) > AFFINE_ROW_TOLERANCE) {
                    throw std::runtime_error(
                        std::format("Frame {} transform_matrix must have affine last row [0,0,0,1]", frame_index));
                }
            }

            const double a = result[0], b = result[1], c = result[2];
            const double d = result[4], e = result[5], f = result[6];
            const double g = result[8], h = result[9], i = result[10];
            const double determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
            // Some exporters encode scale in camera matrices, so invertibility is the
            // compatibility-safe boundary. Do not silently orthogonalize their basis.
            if (!std::isfinite(determinant) || std::abs(determinant) < MIN_AFFINE_DETERMINANT) {
                throw std::runtime_error(std::format("Frame {} transform_matrix is singular", frame_index));
            }
            return result;
        }

        [[nodiscard]] std::vector<ValidatedTransformFrame> validated_transform_frames(
            const nlohmann::json& transforms,
            const LoadOptions& options) {
            if (!transforms.contains("frames") || !transforms["frames"].is_array() ||
                transforms["frames"].empty()) {
                throw std::runtime_error("Transforms JSON must contain a non-empty frames array");
            }
            const auto& frames = transforms["frames"];
            if (frames.size() > MAX_TRANSFORMS_FRAMES)
                throw std::runtime_error("Transforms JSON exceeds the frame-count budget");

            std::vector<ValidatedTransformFrame> result;
            result.reserve(frames.size());
            for (size_t frame_index = 0; frame_index < frames.size(); ++frame_index) {
                if ((frame_index % 256) == 0)
                    throw_if_load_cancel_requested(options, "Transforms schema validation cancelled");
                const auto& frame = frames[frame_index];
                if (!frame.is_object() || !frame.contains("file_path") || !frame["file_path"].is_string())
                    throw std::runtime_error(std::format("Frame {} must contain a string file_path", frame_index));
                std::string file_path = frame["file_path"].get<std::string>();
                if (file_path.empty() || file_path.size() > MAX_TRANSFORMS_PATH_BYTES ||
                    file_path.find('\0') != std::string::npos) {
                    throw std::runtime_error(std::format("Frame {} file_path is empty or too long", frame_index));
                }
                if (!frame.contains("transform_matrix"))
                    throw std::runtime_error(std::format("Frame {} is missing transform_matrix", frame_index));
                result.push_back({
                    .file_path = std::move(file_path),
                    .matrix = validated_transform_matrix(frame["transform_matrix"], frame_index),
                });
            }
            return result;
        }

        [[nodiscard]] bool matrix_is_finite(const glm::mat4& matrix) {
            for (int column = 0; column < 4; ++column) {
                for (int row = 0; row < 4; ++row) {
                    if (!std::isfinite(matrix[column][row]))
                        return false;
                }
            }
            return true;
        }
    } // namespace

    // Helper: Tensor to glm::mat4
    glm::mat4 tensor_to_mat4(const Tensor& t) {
        glm::mat4 mat;
        const float* data = t.ptr<float>();
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                mat[j][i] = data[i * 4 + j]; // GLM is column-major
            }
        }
        return mat;
    }

    // Helper function to convert glm::mat4 to Tensor
    Tensor mat4_to_tensor(const glm::mat4& mat) {
        Tensor t = Tensor::empty({4, 4}, Device::CPU, DataType::Float32);
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                t[i][j] = mat[j][i]; // GLM is column-major
            }
        }
        return t;
    }

    float fov_deg_to_focal_length(int resolution, float fov_deg) {
        return 0.5f * (float)resolution / tanf(0.5f * fov_deg * static_cast<float>(std::numbers::pi) / 180.0f);
    }

    float fov_rad_to_focal_length(int resolution, float fov_rad) {
        return 0.5f * (float)resolution / tanf(0.5f * fov_rad);
    }

    std::filesystem::path GetTransformImagePath(
        const std::filesystem::path& dir_path,
        const std::string& frame_file_path) {
        // Use utf8_to_path for proper Unicode handling since JSON is UTF-8 encoded
        const auto file_path = lfs::core::utf8_to_path(frame_file_path);
        auto image_path = dir_path / file_path;
        auto images_image_path = dir_path / "images" / file_path;
        // Use path concatenation for proper Unicode handling
        auto image_path_png = image_path;
        image_path_png += ".png";
        if (std::filesystem::exists(image_path_png)) {
            // blender data set has not extension, must assumes png
            image_path = image_path_png;
            LOG_TRACE("Using PNG extension for image: {}", lfs::core::path_to_utf8(image_path));
        }
        if (std::filesystem::exists(images_image_path) && std::filesystem::is_regular_file(images_image_path)) {
            image_path = images_image_path;
        }
        return image_path;
    }

    namespace {
        // Resolved camera intrinsics for a single frame. Datasets exported by
        // nerfstudio/Metashape store these per-frame instead of once at the root
        // (each image may have a different sensor, resolution and distortion).
        struct FrameIntrinsics {
            int width = 0;
            int height = 0;
            float fl_x = -1.0f;
            float fl_y = -1.0f;
            float cx = -1.0f;
            float cy = -1.0f;
            float k1 = 0.0f;
            float k2 = 0.0f;
            float k3 = 0.0f;
            float p1 = 0.0f;
            float p2 = 0.0f;
            bool is_distorted = false;
            lfs::core::CameraModelType camera_model = lfs::core::CameraModelType::PINHOLE;
        };

        // Mutable load-scoped warning state so per-frame resolution can warn once.
        struct IntrinsicsWarningState {
            bool equirectangular_warned = false;
            std::set<std::string> unknown_camera_models;
            std::set<std::string> unsupported_distortion_keys;
        };

        // Look up an intrinsics field, preferring the per-frame value and falling
        // back to the global (root) object. Returns nullptr when neither defines it.
        [[nodiscard]] const nlohmann::json* find_intrinsic(
            const nlohmann::json& frame,
            const nlohmann::json& global,
            const std::string_view key) {
            const std::string name(key);
            if (const auto it = frame.find(name); it != frame.end())
                return &*it;
            if (const auto it = global.find(name); it != global.end())
                return &*it;
            return nullptr;
        }

        // Resolve the pixel dimensions for a frame: per-frame w/h override the
        // global w/h, which in turn fall back to reading the image header.
        [[nodiscard]] std::pair<int, int> resolve_frame_dimensions(
            const nlohmann::json& frame,
            const int global_w,
            const int global_h,
            const std::filesystem::path& dir_path,
            const std::string& file_path) {
            int w = -1, h = -1;
            const bool has_width = frame.contains("w");
            const bool has_height = frame.contains("h");
            if (has_width != has_height)
                throw std::runtime_error("Transforms frame must specify both w and h or neither");
            if (has_width) {
                w = positive_image_dimension(frame, "w");
                h = positive_image_dimension(frame, "h");
            } else if (global_w > 0 && global_h > 0) {
                w = global_w;
                h = global_h;
            } else {
                const auto image_path = GetTransformImagePath(dir_path, file_path);
                const auto info = lfs::core::get_image_info(image_path);
                w = std::get<0>(info);
                h = std::get<1>(info);
            }
            if (w <= 0 || h <= 0 ||
                static_cast<uint64_t>(w) * static_cast<uint64_t>(h) > std::numeric_limits<int>::max())
                throw std::runtime_error("Transforms image dimensions exceed the signed pixel-index budget");
            return {w, h};
        }

        // Resolve intrinsics for a frame, preferring per-frame fields and falling
        // back to the global object. Mirrors the historical global-only parsing so
        // datasets that only carry root-level intrinsics behave identically.
        [[nodiscard]] FrameIntrinsics resolve_frame_intrinsics(
            const nlohmann::json& frame,
            const nlohmann::json& global,
            const int w,
            const int h,
            IntrinsicsWarningState& warnings) {
            FrameIntrinsics intrinsics;
            intrinsics.width = w;
            intrinsics.height = h;

            const auto find = [&](const std::string_view key) {
                return find_intrinsic(frame, global, key);
            };

            // Camera model (nerfstudio/COLMAP naming). Pinhole variants (including
            // OPENCV, which is pinhole + Brown-Conrady distortion) map to PINHOLE.
            if (const auto* model_value = find("camera_model")) {
                if (!model_value->is_string())
                    throw std::runtime_error("Transforms camera_model must be a string");
                const std::string model_str = model_value->get<std::string>();
                if (model_str == "EQUIRECTANGULAR") {
                    intrinsics.camera_model = lfs::core::CameraModelType::EQUIRECTANGULAR;
                } else if (model_str == "FISHEYE" || model_str == "OPENCV_FISHEYE") {
                    intrinsics.camera_model = lfs::core::CameraModelType::FISHEYE;
                } else if (model_str != "PINHOLE" && model_str != "SIMPLE_PINHOLE" &&
                           model_str != "OPENCV" && model_str != "FULL_OPENCV") {
                    if (warnings.unknown_camera_models.insert(model_str).second)
                        LOG_WARN("Unknown camera_model '{}', defaulting to PINHOLE", model_str);
                }
            }

            const auto* fl_x_value = find("fl_x");
            const auto* angle_x_value = find("camera_angle_x");
            const auto* fl_y_value = find("fl_y");
            const auto* angle_y_value = find("camera_angle_y");

            float fl_x = -1.0f, fl_y = -1.0f;
            if (fl_x_value) {
                fl_x = finite_json_float_value(*fl_x_value, "fl_x");
            } else if (angle_x_value) {
                const float angle = finite_json_float_value(*angle_x_value, "camera_angle_x");
                if (angle <= 0.0f || angle >= std::numbers::pi_v<float>)
                    throw std::runtime_error("Transforms camera_angle_x must be in (0, pi)");
                fl_x = fov_rad_to_focal_length(w, angle);
            }

            if (fl_y_value) {
                fl_y = finite_json_float_value(*fl_y_value, "fl_y");
            } else if (angle_y_value) {
                const float angle = finite_json_float_value(*angle_y_value, "camera_angle_y");
                if (angle <= 0.0f || angle >= std::numbers::pi_v<float>)
                    throw std::runtime_error("Transforms camera_angle_y must be in (0, pi)");
                fl_y = fov_rad_to_focal_length(h, angle);
            } else {
                const bool no_intrinsics = !fl_x_value && !angle_x_value && !fl_y_value && !angle_y_value;
                if (no_intrinsics) {
                    // Auto-detect equirectangular if not explicitly set
                    if (intrinsics.camera_model != lfs::core::CameraModelType::EQUIRECTANGULAR) {
                        if (!warnings.equirectangular_warned) {
                            LOG_WARN("No camera intrinsics found, assuming equirectangular");
                            warnings.equirectangular_warned = true;
                        }
                        intrinsics.camera_model = lfs::core::CameraModelType::EQUIRECTANGULAR;
                    }
                    fl_x = fl_y = EQUIRECTANGULAR_DUMMY_FOCAL;
                } else {
                    if (w != h)
                        throw std::runtime_error("Transforms JSON is missing vertical intrinsics for a non-square image");
                    fl_y = fl_x;
                }
            }

            if (fl_x < 0.0f && fl_y > 0.0f) {
                if (w != h)
                    throw std::runtime_error("Transforms JSON is missing horizontal intrinsics for a non-square image");
                fl_x = fl_y;
            }
            const float max_focal = static_cast<float>(std::max(w, h)) * MAX_INTRINSIC_DIMENSION_MULTIPLIER;
            if (!std::isfinite(fl_x) || !std::isfinite(fl_y) || fl_x <= 0.0f || fl_y <= 0.0f ||
                fl_x > max_focal || fl_y > max_focal) {
                throw std::runtime_error("Transforms focal lengths must be positive bounded finite values");
            }
            intrinsics.fl_x = fl_x;
            intrinsics.fl_y = fl_y;

            if (const auto* cx_value = find("cx")) {
                intrinsics.cx = finite_json_float_value(*cx_value, "cx");
            } else {
                intrinsics.cx = 0.5f * static_cast<float>(w);
            }
            if (const auto* cy_value = find("cy")) {
                intrinsics.cy = finite_json_float_value(*cy_value, "cy");
            } else {
                intrinsics.cy = 0.5f * static_cast<float>(h);
            }
            const float max_center = static_cast<float>(std::max(w, h)) * MAX_INTRINSIC_DIMENSION_MULTIPLIER;
            if (!std::isfinite(intrinsics.cx) || !std::isfinite(intrinsics.cy) ||
                std::abs(intrinsics.cx) > max_center || std::abs(intrinsics.cy) > max_center)
                throw std::runtime_error("Transforms principal point must be a bounded finite value");

            if (const auto* value = find("k1"))
                intrinsics.k1 = finite_json_float_value(*value, "k1");
            if (const auto* value = find("k2"))
                intrinsics.k2 = finite_json_float_value(*value, "k2");
            if (const auto* value = find("k3"))
                intrinsics.k3 = finite_json_float_value(*value, "k3");
            if (const auto* value = find("p1"))
                intrinsics.p1 = finite_json_float_value(*value, "p1");
            if (const auto* value = find("p2"))
                intrinsics.p2 = finite_json_float_value(*value, "p2");
            for (const float coefficient : {intrinsics.k1, intrinsics.k2, intrinsics.k3, intrinsics.p1, intrinsics.p2}) {
                if (std::abs(coefficient) > MAX_DISTORTION_MAGNITUDE)
                    throw std::runtime_error("Transforms distortion coefficient exceeds the supported range");
            }
            intrinsics.is_distorted = (intrinsics.k1 != 0.0f) || (intrinsics.k2 != 0.0f) ||
                                      (intrinsics.k3 != 0.0f) || (intrinsics.p1 != 0.0f) || (intrinsics.p2 != 0.0f);

            // Metashape/nerfstudio may emit higher-order distortion terms we do not model.
            static constexpr std::array<const char*, 5> unsupported_distortion_keys = {
                "k4", "k5", "k6", "b1", "b2"};
            for (const char* key : unsupported_distortion_keys) {
                const auto* value = find(key);
                if (!value || !value->is_number())
                    continue;
                const double parsed = value->get<double>();
                if (std::isfinite(parsed) && parsed != 0.0)
                    warnings.unsupported_distortion_keys.insert(key);
            }

            return intrinsics;
        }
    } // namespace

    std::tuple<std::vector<CameraData>, lfs::core::Tensor, std::optional<std::tuple<std::vector<std::string>, std::vector<std::string>>>> read_transforms_cameras_and_images(
        const std::filesystem::path& transPath,
        const LoadOptions& options) {

        LOG_TIMER_TRACE("Read transforms file");

        std::filesystem::path transformsFile = transPath;
        if (std::filesystem::is_directory(transPath)) {
            if (std::filesystem::is_regular_file(transPath / "transforms_train.json")) {
                transformsFile = transPath / "transforms_train.json";
            } else if (std::filesystem::is_regular_file(transPath / "transforms.json")) {
                transformsFile = transPath / "transforms.json";
            } else {
                LOG_ERROR("Could not find transforms file in: {}", lfs::core::path_to_utf8(transPath));
                throw std::runtime_error("could not find transforms_train.json nor transforms.json in " + lfs::core::path_to_utf8(transPath));
            }
        }

        if (!std::filesystem::is_regular_file(transformsFile)) {
            LOG_ERROR("Not a valid file: {}", lfs::core::path_to_utf8(transformsFile));
            throw std::runtime_error(lfs::core::path_to_utf8(transformsFile) + " is not a valid file");
        }

        throw_if_load_cancel_requested(options, "Transforms dataset load cancelled");

        LOG_DEBUG("Reading transforms from: {}", lfs::core::path_to_utf8(transformsFile));
        std::ifstream trans_file;
        if (!lfs::core::open_file_for_read(transformsFile, std::ios::in | std::ios::binary, trans_file)) {
            throw std::runtime_error("Failed to open: " + lfs::core::path_to_utf8(transformsFile));
        }

        std::filesystem::path dir_path = transformsFile.parent_path();

        std::error_code file_size_error;
        const uintmax_t transforms_file_size = std::filesystem::file_size(transformsFile, file_size_error);
        if (file_size_error || transforms_file_size == 0 || transforms_file_size > MAX_TRANSFORMS_JSON_BYTES)
            throw std::runtime_error("Transforms JSON is empty or exceeds the 64 MiB input budget");

        std::vector<char> json_bytes(static_cast<size_t>(transforms_file_size) + 1);
        trans_file.read(json_bytes.data(), static_cast<std::streamsize>(json_bytes.size()));
        const size_t bytes_read = static_cast<size_t>(trans_file.gcount());
        if (bytes_read != transforms_file_size)
            throw std::runtime_error("Transforms JSON changed size while it was being read");

        // Parse only the validated byte range; the extra byte detects a file that grew after stat().
        nlohmann::json transforms = nlohmann::json::parse(
            json_bytes.begin(), json_bytes.begin() + static_cast<ptrdiff_t>(bytes_read), nullptr, true, true);
        if (!transforms.is_object())
            throw std::runtime_error("Transforms JSON root must be an object");
        const auto validated_frames = validated_transform_frames(transforms, options);
        throw_if_load_cancel_requested(options, "Transforms dataset parse cancelled");

        // Global w/h are defaults; individual frames may override. nerfstudio/Metashape
        // often store intrinsics only per-frame with no root-level dimensions.
        int global_w = -1, global_h = -1;
        const bool has_width = transforms.contains("w");
        const bool has_height = transforms.contains("h");
        if (has_width != has_height)
            throw std::runtime_error("Transforms JSON must specify both w and h or neither");
        if (has_width) {
            global_w = positive_image_dimension(transforms, "w");
            global_h = positive_image_dimension(transforms, "h");
            if (global_w <= 0 || global_h <= 0 ||
                static_cast<uint64_t>(global_w) * static_cast<uint64_t>(global_h) > std::numeric_limits<int>::max())
                throw std::runtime_error("Transforms image dimensions exceed the signed pixel-index budget");
        }

        // Validate remaining scalar metadata before allocating any camera tensors.
        if (transforms.contains("aabb_scale")) {
            const float aabb_scale = finite_json_float(transforms, "aabb_scale");
            if (aabb_scale <= 0.0f)
                throw std::runtime_error("Transforms aabb_scale must be positive");
            LOG_DEBUG("Found aabb_scale: {}", aabb_scale);
        }

        IntrinsicsWarningState intrinsics_warnings;
        std::vector<CameraData> camerasdata;
        {
            uint64_t counter = 0;
            LOG_DEBUG("Processing {} frames", validated_frames.size());
            camerasdata.reserve(validated_frames.size());

            for (size_t frameInd = 0; frameInd < validated_frames.size(); ++frameInd) {
                if ((frameInd % 64) == 0) {
                    throw_if_load_cancel_requested(options, "Transforms camera assembly cancelled");
                }
                CameraData camdata;
                const auto& frame = validated_frames[frameInd];

                // Resolve intrinsics for this frame. Per-frame fields take priority
                // over the global defaults so datasets with mixed sensors/resolutions
                // load correctly; global-only datasets resolve identically as before.
                const auto& frame_json = transforms["frames"][frameInd];
                const auto [frame_w, frame_h] =
                    resolve_frame_dimensions(frame_json, global_w, global_h, dir_path, frame.file_path);
                const FrameIntrinsics intrinsics =
                    resolve_frame_intrinsics(frame_json, transforms, frame_w, frame_h, intrinsics_warnings);

                // Create camera-to-world transform matrix
                lfs::core::Tensor c2w = lfs::core::Tensor::empty({4, 4}, Device::CPU, DataType::Float32);

                // Fill the c2w matrix from the JSON data
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        c2w[i][j] = frame.matrix[static_cast<size_t>(i) * 4 + static_cast<size_t>(j)];
                    }
                }

                // Change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                // c2w[:3, 1:3] *= -1
                float* c2w_data = c2w.ptr<float>();
                for (int i = 0; i < 3; ++i) {
                    c2w_data[i * 4 + 1] *= -1.0f;
                    c2w_data[i * 4 + 2] *= -1.0f;
                }

                // Get the world-to-camera transform by computing inverse of c2w
                glm::mat4 c2w_glm = tensor_to_mat4(c2w);
                glm::mat4 w2c_glm = glm::inverse(c2w_glm);
                if (!matrix_is_finite(w2c_glm))
                    throw std::runtime_error(std::format("Frame {} inverse transform is non-finite", frameInd));
                Tensor w2c = mat4_to_tensor(w2c_glm);

                // transforms datasets convert their point cloud into the repo's
                // "data/COLMAP" world basis via convert_transforms_point_cloud_to_colmap_world
                // (diag(1,-1,-1) on Y/Z). Right-multiply the same worldAxesFlip into w2c so
                // camera extrinsics share exactly that world basis and stay aligned with
                // PLY-initialized point clouds. An identity JSON transform_matrix then yields
                // identity extrinsics (R=I, T=0).
                //
                // This is a world-basis change, so it must be right-multiplied into the world->camera matrix.
                Tensor worldAxesFlip = lfs::core::Tensor::eye(4, Device::CPU);
                worldAxesFlip[1][1] = -1.0f;
                worldAxesFlip[2][2] = -1.0f;
                w2c = w2c.mm(worldAxesFlip);

                // Extract rotation matrix R (transposed due to 'glm' in CUDA code)
                // R = np.transpose(w2c[:3,:3])
                lfs::core::Tensor R = w2c.slice(0, 0, 3).slice(1, 0, 3);

                // Extract translation vector T
                // T = w2c[:3, 3]
                lfs::core::Tensor T = w2c.slice(0, 0, 3).slice(1, 3, 4).squeeze(1);

                camdata._image_path = GetTransformImagePath(dir_path, frame.file_path);

                camdata._image_name = lfs::core::path_to_utf8(camdata._image_path.filename());

                camdata._width = intrinsics.width;
                camdata._height = intrinsics.height;

                camdata._T = T.contiguous();
                camdata._R = R.contiguous();

                if (intrinsics.is_distorted) {
                    camdata._radial_distortion =
                        Tensor::from_vector({intrinsics.k1, intrinsics.k2, intrinsics.k3}, {3}, Device::CPU);
                    camdata._tangential_distortion =
                        Tensor::from_vector({intrinsics.p1, intrinsics.p2}, {2}, Device::CPU);
                } else {
                    camdata._radial_distortion = Tensor::empty({0}, Device::CPU);
                    camdata._tangential_distortion = Tensor::empty({0}, Device::CPU);
                }

                camdata._focal_x = intrinsics.fl_x;
                camdata._focal_y = intrinsics.fl_y;

                camdata._center_x = intrinsics.cx;
                camdata._center_y = intrinsics.cy;

                camdata._camera_model_type = intrinsics.camera_model;
                camdata._camera_ID = static_cast<uint32_t>(counter++);

                camerasdata.push_back(camdata);
                LOG_TRACE("Processed frame {}: {}", frameInd, camdata._image_name);
            }
        }

        if (!intrinsics_warnings.unsupported_distortion_keys.empty()) {
            std::string keys;
            for (const auto& key : intrinsics_warnings.unsupported_distortion_keys) {
                if (!keys.empty())
                    keys += ", ";
                keys += key;
            }
            LOG_WARN("Unsupported distortion coefficients ignored: {}", keys);
        }

        auto center = lfs::core::Tensor::zeros({3}, Device::CPU, DataType::Float32);

        LOG_INFO("Loaded {} cameras from transforms file", camerasdata.size());

        if (std::filesystem::is_regular_file(dir_path / "train.txt") &&
            std::filesystem::is_regular_file(dir_path / "test.txt")) {
            throw_if_load_cancel_requested(options, "Transforms split metadata load cancelled");
            LOG_DEBUG("Found train.txt and test.txt files, loading image splits");

            std::ifstream train_file;
            std::ifstream val_file;
            if (!lfs::core::open_file_for_read(dir_path / "train.txt", train_file)) {
                LOG_WARN("Failed to open train.txt");
            }
            if (!lfs::core::open_file_for_read(dir_path / "test.txt", val_file)) {
                LOG_WARN("Failed to open test.txt");
            }

            std::vector<std::string> train_images;
            std::vector<std::string> val_images;

            std::string line;
            while (std::getline(train_file, line)) {
                if (!line.empty()) {
                    train_images.push_back(line);
                }
            }
            while (std::getline(val_file, line)) {
                if (!line.empty()) {
                    val_images.push_back(line);
                }
            }

            LOG_INFO("Loaded {} training images and {} validation images", train_images.size(), val_images.size());

            return {camerasdata, center, std::make_tuple(train_images, val_images)};
        }

        return {camerasdata, center, std::nullopt};
    }

    PointCloud generate_random_point_cloud() {
        LOG_DEBUG("Generating random point cloud with {} points", DEFAULT_NUM_INIT_GAUSSIAN);

        int numInitGaussian = DEFAULT_NUM_INIT_GAUSSIAN;

        uint64_t seed = DEFAULT_RANDOM_SEED;
        // Set random seed for reproducibility
        Tensor::manual_seed(seed);

        lfs::core::Tensor positions = Tensor::rand({static_cast<size_t>(numInitGaussian), 3}, Device::CPU); // in [0, 1]
        positions = positions * 2.0 - 1.0;                                                                  // now in [-1, 1]
        // Random RGB colors
        lfs::core::Tensor colors = Tensor::randint({static_cast<size_t>(numInitGaussian), 3}, 0, 256, Device::CPU, DataType::UInt8);

        return PointCloud(positions, colors);
    }

    PointCloud load_simple_ply_point_cloud(const std::filesystem::path& filepath,
                                           const LoadOptions& options) {
        LOG_DEBUG("Loading simple PLY point cloud from: {}", lfs::core::path_to_utf8(filepath));

        if (!std::filesystem::exists(filepath)) {
            throw std::runtime_error(std::format("PLY file not found: {}", lfs::core::path_to_utf8(filepath)));
        }

        try {
            // Open the PLY file
            std::ifstream ss;
            if (!lfs::core::open_file_for_read(filepath, std::ios::binary, ss)) {
                throw std::runtime_error(std::format("Failed to open PLY file: {}", lfs::core::path_to_utf8(filepath)));
            }

            // Parse PLY header
            tinyply::PlyFile file;
            throw_if_load_cancel_requested(options, "Transforms PLY header parse cancelled");
            file.parse_header(ss);

            // Request vertex positions (x, y, z)
            std::shared_ptr<tinyply::PlyData> vertices;
            try {
                vertices = file.request_properties_from_element("vertex", {"x", "y", "z"});
            } catch (const std::exception& e) {
                throw std::runtime_error(std::format("PLY file missing vertex positions: {}", e.what()));
            }

            // Try to get colors (red, green, blue) - optional
            std::shared_ptr<tinyply::PlyData> colors;
            bool has_colors = false;
            try {
                colors = file.request_properties_from_element("vertex", {"red", "green", "blue"});
                has_colors = true;
            } catch (const std::exception&) {
                // Colors are optional, we'll use default white color if not present
                LOG_DEBUG("PLY file has no color data, using default white color");
            }

            // Read the actual data
            throw_if_load_cancel_requested(options, "Transforms PLY read cancelled");
            file.read(ss);
            throw_if_load_cancel_requested(options, "Transforms PLY read cancelled");

            // Get vertex count
            const size_t vertex_count = vertices->count;
            LOG_DEBUG("Loaded {} vertices from PLY file", vertex_count);

            // Create position tensor from vertex data
            lfs::core::Tensor positions = lfs::core::Tensor::zeros({static_cast<size_t>(vertex_count), 3}, Device::CPU, DataType::Float32);
            float* pos_ptr = positions.ptr<float>();

            // Copy and convert vertex data according to its type
            switch (vertices->t) {
            case tinyply::Type::FLOAT32: {
                const float* vertex_data = reinterpret_cast<const float*>(vertices->buffer.get());
                std::memcpy(pos_ptr, vertex_data, vertex_count * 3 * sizeof(float));
                break;
            }
            case tinyply::Type::FLOAT64: {
                const double* vertex_data = reinterpret_cast<const double*>(vertices->buffer.get());
                for (size_t i = 0; i < vertex_count * 3; ++i) {
                    if ((i % 4096) == 0) {
                        throw_if_load_cancel_requested(options, "Transforms PLY vertex conversion cancelled");
                    }
                    pos_ptr[i] = static_cast<float>(vertex_data[i]);
                }
                break;
            }
            case tinyply::Type::INT32: {
                const int32_t* vertex_data = reinterpret_cast<const int32_t*>(vertices->buffer.get());
                for (size_t i = 0; i < vertex_count * 3; ++i) {
                    if ((i % 4096) == 0) {
                        throw_if_load_cancel_requested(options, "Transforms PLY vertex conversion cancelled");
                    }
                    pos_ptr[i] = static_cast<float>(vertex_data[i]);
                }
                break;
            }
            case tinyply::Type::UINT8: {
                const uint8_t* vertex_data = reinterpret_cast<const uint8_t*>(vertices->buffer.get());
                for (size_t i = 0; i < vertex_count * 3; ++i) {
                    if ((i % 4096) == 0) {
                        throw_if_load_cancel_requested(options, "Transforms PLY vertex conversion cancelled");
                    }
                    pos_ptr[i] = static_cast<float>(vertex_data[i]);
                }
                break;
            }
            // Add more cases as needed for other types
            default:
                throw std::runtime_error("Unsupported vertex type in PLY file");
            }

            // Create color tensor
            lfs::core::Tensor color_tensor;
            if (has_colors && colors && colors->count == vertex_count) {
                // Check if colors are float or uint8
                if (colors->t == tinyply::Type::FLOAT32) {
                    // Float colors [0, 1] - convert to uint8
                    lfs::core::Tensor float_colors = lfs::core::Tensor::zeros({static_cast<size_t>(vertex_count), 3}, Device::CPU, DataType::Float32);
                    float* color_ptr = float_colors.ptr<float>();
                    const float* color_data = reinterpret_cast<const float*>(colors->buffer.get());
                    std::memcpy(color_ptr, color_data, vertex_count * 3 * sizeof(float));

                    // Convert to uint8 [0, 255]
                    color_tensor = (float_colors * 255.0f).clamp(0, 255).to(lfs::core::DataType::UInt8);
                } else if (colors->t == tinyply::Type::UINT8 || colors->t == tinyply::Type::INT8) {
                    // Already uint8
                    color_tensor = lfs::core::Tensor::zeros({static_cast<size_t>(vertex_count), 3}, Device::CPU, DataType::UInt8);
                    uint8_t* color_ptr = color_tensor.ptr<uint8_t>();
                    const uint8_t* color_data = reinterpret_cast<const uint8_t*>(colors->buffer.get());
                    std::memcpy(color_ptr, color_data, vertex_count * 3 * sizeof(uint8_t));
                } else {
                    // Unsupported color type, use white
                    LOG_WARN("Unsupported color type in PLY file, using default white color");
                    color_tensor = lfs::core::Tensor::full({static_cast<size_t>(vertex_count), 3}, 255, Device::CPU, DataType::UInt8);
                }
            } else {
                // No colors or count mismatch, use white
                color_tensor = lfs::core::Tensor::full({static_cast<size_t>(vertex_count), 3}, 255, Device::CPU, DataType::UInt8);
            }

            LOG_INFO("Successfully loaded PLY point cloud with {} points", vertex_count);
            throw_if_load_cancel_requested(options, "Transforms PLY upload cancelled");

            // Move to CUDA for GPU rendering
            return PointCloud(positions.cuda(), color_tensor.cuda());

        } catch (const std::exception& e) {
            throw std::runtime_error(std::format("Failed to load PLY file {}: {}", lfs::core::path_to_utf8(filepath), e.what()));
        }
    }

    PointCloud convert_transforms_point_cloud_to_colmap_world(PointCloud point_cloud) {
        if (!point_cloud.means.is_valid() || point_cloud.size() == 0) {
            return point_cloud;
        }

        const auto device = point_cloud.means.device();
        auto means_cpu = point_cloud.means.cpu().contiguous();
        auto acc = means_cpu.accessor<float, 2>();
        const size_t point_count = means_cpu.size(0);
        for (size_t i = 0; i < point_count; ++i) {
            acc(i, 1) = -acc(i, 1);
            acc(i, 2) = -acc(i, 2);
        }

        point_cloud.means = means_cpu.to(device).contiguous();
        return point_cloud;
    }

} // namespace lfs::io
