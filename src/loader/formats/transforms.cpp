/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "transforms.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "external/tinyply.hpp"
#include "formats/colmap.hpp"
#include <filesystem>
#include <format>
#include <fstream>
#include <nlohmann/json.hpp>
#include <numbers>
#include <torch/torch.h>

namespace gs::loader {

    namespace F = torch::nn::functional;

    // Constants for random point cloud generation
    constexpr int DEFAULT_NUM_INIT_GAUSSIAN = 10000;
    constexpr uint64_t DEFAULT_RANDOM_SEED = 8128;

    // Use std::numbers::pi instead of a custom PI constant.

    float fov_deg_to_focal_length(int resolution, float fov_deg) {
        return 0.5f * (float)resolution / tanf(0.5f * fov_deg * std::numbers::pi / 180.0f);
    }

    float fov_rad_to_focal_length(int resolution, float fov_rad) {
        return 0.5f * (float)resolution / tanf(0.5f * fov_rad);
    }

    // Function to create a 3x3 rotation matrix around Y-axis embeded in 4x4 matrix
    torch::Tensor createYRotationMatrix(float angle_radians) {
        torch::Tensor rotMat = torch::eye(4);
        float cos_angle = std::cos(angle_radians);
        float sin_angle = std::sin(angle_radians);

        // Rotation matrix around Y-axis by angle θ:
        // [cos(θ)   0   sin(θ) 0]
        // [  0      1     0    0]
        // [-sin(θ)  0   cos(θ) 0]
        // [0        0   0      1]

        rotMat[0][0] = cos_angle;  // cos(θ)
        rotMat[0][1] = 0.0f;       // 0
        rotMat[0][2] = sin_angle;  // sin(θ)
        rotMat[1][0] = 0.0f;       // 0
        rotMat[1][1] = 1.0f;       // 1
        rotMat[1][2] = 0.0f;       // 0
        rotMat[2][0] = -sin_angle; // -sin(θ)
        rotMat[2][1] = 0.0f;       // 0
        rotMat[2][2] = cos_angle;  // cos(θ)

        return rotMat;
    }

    std::filesystem::path GetTransformImagePath(const std::filesystem::path& dir_path, const nlohmann::json& frame) {
        auto image_path = dir_path / frame["file_path"];
        auto images_image_path = dir_path / "images" / frame["file_path"];
        auto image_path_png = std::filesystem::path(image_path.string() + ".png");
        if (std::filesystem::exists(image_path_png)) {
            // blender data set has not extension, must assumes png
            image_path = image_path_png;
            LOG_TRACE("Using PNG extension for image: {}", image_path.string());
        }
        if (std::filesystem::exists(images_image_path) && std::filesystem::is_regular_file(images_image_path)) {
            image_path = images_image_path;
        }
        return image_path;
    }

    std::tuple<std::vector<CameraData>, torch::Tensor, std::optional<std::tuple<std::vector<std::string>, std::vector<std::string>>>> read_transforms_cameras_and_images(
        const std::filesystem::path& transPath) {

        LOG_TIMER_TRACE("Read transforms file");

        std::filesystem::path transformsFile = transPath;
        if (std::filesystem::is_directory(transPath)) {
            if (std::filesystem::is_regular_file(transPath / "transforms_train.json")) {
                transformsFile = transPath / "transforms_train.json";
            } else if (std::filesystem::is_regular_file(transPath / "transforms.json")) {
                transformsFile = transPath / "transforms.json";
            } else {
                LOG_ERROR("Could not find transforms file in: {}", transPath.string());
                throw std::runtime_error("could not find transforms_train.json nor transforms.json in " + transPath.string());
            }
        }

        if (!std::filesystem::is_regular_file(transformsFile)) {
            LOG_ERROR("Not a valid file: {}", transformsFile.string());
            throw std::runtime_error(transformsFile.string() + " is not a valid file");
        }

        LOG_DEBUG("Reading transforms from: {}", transformsFile.string());
        std::ifstream trans_file{transformsFile.string()};

        std::filesystem::path dir_path = transformsFile.parent_path();

        // should throw if parse fails
        nlohmann::json transforms = nlohmann::json::parse(trans_file, nullptr, true, true);
        int w = -1, h = -1;
        if (!transforms.contains("w") or !transforms.contains("h")) {

            try {
                LOG_DEBUG("Width/height not in transforms.json, reading from first image");
                auto first_frame_img_path = GetTransformImagePath(dir_path, transforms["frames"][0]);
                auto result = get_image_info(first_frame_img_path);

                w = std::get<0>(result);
                h = std::get<1>(result);

                LOG_DEBUG("Got image dimensions: {}x{}", w, h);
            } catch (const std::exception& e) {
                std::string error_msg = "Error while trying to read image dimensions: " + std::string(e.what());
                LOG_ERROR("{}", error_msg);
                throw std::runtime_error(error_msg);
            } catch (...) {
                std::string error_msg = "Unknown error while trying to read image dimensions";
                LOG_ERROR("{}", error_msg);
                throw std::runtime_error(error_msg);
            }
        } else {
            w = int(transforms["w"]);
            h = int(transforms["h"]);
        }

        float fl_x = -1, fl_y = -1;
        auto camera_model = gsplat::CameraModelType::PINHOLE;
        if (transforms.contains("fl_x")) {
            fl_x = float(transforms["fl_x"]);
        } else if (transforms.contains("camera_angle_x")) {
            fl_x = fov_rad_to_focal_length(w, float(transforms["camera_angle_x"]));
        }

        if (transforms.contains("fl_y")) {
            fl_y = float(transforms["fl_y"]);
        } else if (transforms.contains("camera_angle_y")) {
            fl_y = fov_rad_to_focal_length(h, float(transforms["camera_angle_y"]));
        } else {
            // OmniBlender no intrinsics
            if (!transforms.contains("fl_x") && !transforms.contains("camera_angle_x") &&
                !transforms.contains("fl_y") && !transforms.contains("camera_angle_y")) {
                LOG_WARN("No camera intrinsics found, assuming equirectangular");
                fl_x = 20.0;
                fl_y = 20.0;
                camera_model = gsplat::CameraModelType::EQUIRECTANGULAR;
            } else {
                // we should be  here in this scope only for blender - if w!=h then we must throw exception
                if (w != h) {
                    LOG_ERROR("No camera_angle_y but w!=h: {}!={}", w, h);
                    throw std::runtime_error("no camera_angle_y expected w!=h");
                }
                fl_y = fl_x;
            }
        }

        float cx = -1, cy = -1;
        if (transforms.contains("cx")) {
            cx = float(transforms["cx"]);
        } else {
            cx = 0.5 * w;
        }

        if (transforms.contains("cy")) {
            cy = float(transforms["cy"]);
        } else {
            cy = 0.5 * h;
        }

        float k1 = 0;
        float k2 = 0;
        float p1 = 0;
        float p2 = 0;
        if (transforms.contains("k1")) {
            k1 = float(transforms["k1"]);
        }
        if (transforms.contains("k2")) {
            k2 = float(transforms["k2"]);
        }
        if (transforms.contains("p1")) {
            p1 = float(transforms["p1"]);
        }
        if (transforms.contains("p2")) {
            p2 = float(transforms["p2"]);
        }
        if (k1 > 0 || k2 > 0 || p1 > 0 || p2 > 0) {
            LOG_ERROR("Distortion parameters not supported: k1={}, k2={}, p1={}, p2={}", k1, k2, p1, p2);
            throw std::runtime_error(std::format("GS don't support distortion for now: k1={}, k2={}, p1={}, p2={}", k1, k2, p1, p2));
        }

        std::vector<CameraData> camerasdata;
        if (transforms.contains("frames") && transforms["frames"].is_array()) {
            uint64_t counter = 0;
            LOG_DEBUG("Processing {} frames", transforms["frames"].size());

            for (size_t frameInd = 0; frameInd < transforms["frames"].size(); ++frameInd) {
                CameraData camdata;
                auto& frame = transforms["frames"][frameInd];
                if (!frame.contains("transform_matrix")) {
                    LOG_ERROR("Frame {} missing transform_matrix", frameInd);
                    throw std::runtime_error("expected all frames to contain transform_matrix");
                }
                if (!(frame["transform_matrix"].is_array() and frame["transform_matrix"].size() == 4)) {
                    LOG_ERROR("Frame {} has invalid transform_matrix dimensions", frameInd);
                    throw std::runtime_error("transform_matrix has the wrong dimensions");
                }

                // Create camera-to-world transform matrix
                torch::Tensor c2w = torch::empty({4, 4}, torch::kFloat32);

                // Fill the c2w matrix from the JSON data
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        c2w[i][j] = float(frame["transform_matrix"][i][j]);
                    }
                }

                // Change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                // c2w[:3, 1:3] *= -1
                c2w.slice(0, 0, 3).slice(1, 1, 3) *= -1;

                // Get the world-to-camera transform by computing inverse of c2w
                torch::Tensor w2c = torch::inverse(c2w);

                // fix so that the z direction will be the same (currently it is faceing downward)
                torch::Tensor fixMat = createYRotationMatrix(M_PI);
                w2c = torch::mm(w2c, fixMat);

                // Extract rotation matrix R (transposed due to 'glm' in CUDA code)
                // R = np.transpose(w2c[:3,:3])
                torch::Tensor R = w2c.slice(0, 0, 3).slice(1, 0, 3);

                // Extract translation vector T
                // T = w2c[:3, 3]
                torch::Tensor T = w2c.slice(0, 0, 3).slice(1, 3, 4).squeeze(1);

                camdata._image_path = GetTransformImagePath(dir_path, frame);

                camdata._image_name = std::filesystem::path(camdata._image_path).filename().string();

                camdata._width = w;
                camdata._height = h;

                camdata._T = T;
                camdata._R = R;

                camdata._focal_x = fl_x;
                camdata._focal_y = fl_y;

                camdata._center_x = cx;
                camdata._center_y = cy;

                camdata._camera_model_type = camera_model;
                camdata._camera_ID = counter++;

                camerasdata.push_back(camdata);
                LOG_TRACE("Processed frame {}: {}", frameInd, camdata._image_name);
            }
        }

        auto center = torch::zeros({3}, torch::kFloat32);

        // Check for aabb_scale (used in some NeRF datasets for scene scaling)
        float aabb_scale = 1.0f;
        if (transforms.contains("aabb_scale")) {
            aabb_scale = float(transforms["aabb_scale"]);
            LOG_DEBUG("Found aabb_scale: {}", aabb_scale);
        }

        LOG_INFO("Loaded {} cameras from transforms file", camerasdata.size());

        if (std::filesystem::is_regular_file(dir_path / "train.txt") &&
            std::filesystem::is_regular_file(dir_path / "test.txt")) {
            LOG_DEBUG("Found train.txt and test.txt files, loading image splits");

            std::ifstream train_file(dir_path / "train.txt");
            std::ifstream val_file(dir_path / "test.txt");

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
        torch::manual_seed(seed);

        torch::Tensor positions = torch::rand({numInitGaussian, 3}); // in [0, 1]
        positions = positions * 2.0 - 1.0;                           // now in [-1, 1]
        // Random RGB colors
        torch::Tensor colors = torch::randint(0, 256, {numInitGaussian, 3}, torch::kUInt8);

        return PointCloud(positions, colors);
    }

    PointCloud load_simple_ply_point_cloud(const std::filesystem::path& filepath) {
        LOG_DEBUG("Loading simple PLY point cloud from: {}", filepath.string());

        if (!std::filesystem::exists(filepath)) {
            throw std::runtime_error(std::format("PLY file not found: {}", filepath.string()));
        }

        try {
            // Open the PLY file
            std::ifstream ss(filepath, std::ios::binary);
            if (!ss) {
                throw std::runtime_error(std::format("Failed to open PLY file: {}", filepath.string()));
            }

            // Parse PLY header
            tinyply::PlyFile file;
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
            file.read(ss);

            // Get vertex count
            const size_t vertex_count = vertices->count;
            LOG_DEBUG("Loaded {} vertices from PLY file", vertex_count);

            // Create position tensor from vertex data
            torch::Tensor positions = torch::zeros({static_cast<long>(vertex_count), 3}, torch::kFloat32);
            float* pos_ptr = positions.data_ptr<float>();

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
                    pos_ptr[i] = static_cast<float>(vertex_data[i]);
                }
                break;
            }
            case tinyply::Type::INT32: {
                const int32_t* vertex_data = reinterpret_cast<const int32_t*>(vertices->buffer.get());
                for (size_t i = 0; i < vertex_count * 3; ++i) {
                    pos_ptr[i] = static_cast<float>(vertex_data[i]);
                }
                break;
            }
            case tinyply::Type::UINT8: {
                const uint8_t* vertex_data = reinterpret_cast<const uint8_t*>(vertices->buffer.get());
                for (size_t i = 0; i < vertex_count * 3; ++i) {
                    pos_ptr[i] = static_cast<float>(vertex_data[i]);
                }
                break;
            }
            // Add more cases as needed for other types
            default:
                throw std::runtime_error("Unsupported vertex type in PLY file");
            }

            // Create color tensor
            torch::Tensor color_tensor;
            if (has_colors && colors && colors->count == vertex_count) {
                // Check if colors are float or uint8
                if (colors->t == tinyply::Type::FLOAT32) {
                    // Float colors [0, 1] - convert to uint8
                    torch::Tensor float_colors = torch::zeros({static_cast<long>(vertex_count), 3}, torch::kFloat32);
                    float* color_ptr = float_colors.data_ptr<float>();
                    const float* color_data = reinterpret_cast<const float*>(colors->buffer.get());
                    std::memcpy(color_ptr, color_data, vertex_count * 3 * sizeof(float));

                    // Convert to uint8 [0, 255]
                    color_tensor = (float_colors * 255.0f).clamp(0, 255).to(torch::kUInt8);
                } else if (colors->t == tinyply::Type::UINT8 || colors->t == tinyply::Type::INT8) {
                    // Already uint8
                    color_tensor = torch::zeros({static_cast<long>(vertex_count), 3}, torch::kUInt8);
                    uint8_t* color_ptr = color_tensor.data_ptr<uint8_t>();
                    const uint8_t* color_data = reinterpret_cast<const uint8_t*>(colors->buffer.get());
                    std::memcpy(color_ptr, color_data, vertex_count * 3 * sizeof(uint8_t));
                } else {
                    // Unsupported color type, use white
                    LOG_WARN("Unsupported color type in PLY file, using default white color");
                    color_tensor = torch::full({static_cast<long>(vertex_count), 3}, 255, torch::kUInt8);
                }
            } else {
                // No colors or count mismatch, use white
                color_tensor = torch::full({static_cast<long>(vertex_count), 3}, 255, torch::kUInt8);
            }

            LOG_INFO("Successfully loaded PLY point cloud with {} points", vertex_count);
            return PointCloud(positions, color_tensor);

        } catch (const std::exception& e) {
            throw std::runtime_error(std::format("Failed to load PLY file {}: {}", filepath.string(), e.what()));
        }
    }

} // namespace gs::loader