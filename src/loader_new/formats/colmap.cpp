/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "colmap.hpp"
#include "core_new/image_io.hpp"
#include "core_new/logger.hpp"
#include "loader_new/filesystem_utils.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace lfs::loader {

    // Import types from lfs::core for convenience
    using lfs::core::Camera;
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::PointCloud;
    using lfs::core::Tensor;

    namespace fs = std::filesystem;

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
    struct ImageData {
        uint32_t image_id = 0;
        uint32_t camera_id = 0;
        std::string name;
        std::vector<float> qvec = {1.0f, 0.0f, 0.0f, 0.0f}; // [w, x, y, z]
        std::vector<float> tvec = {0.0f, 0.0f, 0.0f};
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

    // -----------------------------------------------------------------------------
    //  Binary-file loader
    // -----------------------------------------------------------------------------
    static std::unique_ptr<std::vector<char>>
    read_binary(const std::filesystem::path& p) {
        LOG_TRACE("Reading binary file: {}", p.string());
        std::ifstream f(p, std::ios::binary | std::ios::ate);
        if (!f) {
            LOG_ERROR("Failed to open binary file: {}", p.string());
            throw std::runtime_error("Failed to open " + p.string());
        }

        auto sz = static_cast<std::streamsize>(f.tellg());
        auto buf = std::make_unique<std::vector<char>>(static_cast<size_t>(sz));

        f.seekg(0, std::ios::beg);
        f.read(buf->data(), sz);
        if (!f) {
            LOG_ERROR("Short read on binary file: {}", p.string());
            throw std::runtime_error("Short read on " + p.string());
        }
        LOG_TRACE("Read {} bytes from {}", sz, p.string());
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
    std::vector<ImageData> read_images_binary(const std::filesystem::path& file_path) {
        LOG_TIMER_TRACE("Read images.bin");
        auto buf_owner = read_binary(file_path);
        const char* cur = buf_owner->data();
        const char* end = cur + buf_owner->size();

        uint64_t n_images = read_u64(cur);
        LOG_DEBUG("Reading {} images from binary file", n_images);
        std::vector<ImageData> images;
        images.reserve(n_images);

        for (uint64_t i = 0; i < n_images; ++i) {
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
            cur += npts * (sizeof(double) * 2 + sizeof(uint64_t));

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
    read_cameras_binary(const std::filesystem::path& file_path, float scale_factor = 1.0f) {
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
    PointCloud read_point3D_binary(const std::filesystem::path& file_path) {
        LOG_TIMER_TRACE("Read points3D.bin");
        auto buf_owner = read_binary(file_path);
        const char* cur = buf_owner->data();
        const char* end = cur + buf_owner->size();

        uint64_t N = read_u64(cur);
        LOG_DEBUG("Reading {} 3D points from binary file", N);

        std::vector<float> positions(N * 3);
        std::vector<uint8_t> colors(N * 3);

        for (uint64_t i = 0; i < N; ++i) {
            cur += 8; // skip point ID

            positions[i * 3 + 0] = static_cast<float>(read_f64(cur));
            positions[i * 3 + 1] = static_cast<float>(read_f64(cur));
            positions[i * 3 + 2] = static_cast<float>(read_f64(cur));

            // Store colors as uint8 [0,255] to match old loader
            colors[i * 3 + 0] = static_cast<uint8_t>(*cur++);
            colors[i * 3 + 1] = static_cast<uint8_t>(*cur++);
            colors[i * 3 + 2] = static_cast<uint8_t>(*cur++);

            cur += 8;                                    // skip reprojection error
            cur += read_u64(cur) * sizeof(uint32_t) * 2; // skip track
        }

        if (cur != end) {
            LOG_ERROR("points3D.bin has trailing bytes");
            throw std::runtime_error("points3D.bin: trailing bytes");
        }

        Tensor means = Tensor::from_vector(positions, {N, 3}, Device::CUDA);
        Tensor colors_tensor = Tensor::from_blob(colors.data(), {N, 3}, Device::CPU, DataType::UInt8)
                                   .to(Device::CUDA)
                                   .contiguous();

        return PointCloud(std::move(means), std::move(colors_tensor));
    }

    // -----------------------------------------------------------------------------
    //  Text-file helpers
    // -----------------------------------------------------------------------------
    std::vector<std::string> read_text_file(const std::filesystem::path& file_path) {
        LOG_TRACE("Reading text file: {}", file_path.string());
        std::ifstream file(file_path);
        if (!file.is_open()) {
            LOG_ERROR("Failed to open text file: {}", file_path.string());
            throw std::runtime_error("Failed to open " + file_path.string());
        }

        std::vector<std::string> lines;
        std::string line;
        while (std::getline(file, line)) {
            if (line.starts_with("#"))
                continue;
            if (!line.empty() && line.back() == '\r')
                line.pop_back();
            lines.push_back(line);
        }

        if (lines.empty()) {
            LOG_ERROR("File is empty: {}", file_path.string());
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

    // -----------------------------------------------------------------------------
    //  images.txt
    // -----------------------------------------------------------------------------
    std::vector<ImageData> read_images_text(const std::filesystem::path& file_path) {
        LOG_TIMER_TRACE("Read images.txt");
        auto lines = read_text_file(file_path);

        if (lines.size() % 2 != 0) {
            LOG_ERROR("images.txt should have an even number of lines");
            throw std::runtime_error("images.txt should have an even number of lines");
        }

        uint64_t n_images = lines.size() / 2;
        LOG_DEBUG("Reading {} images from text file", n_images);
        std::vector<ImageData> images;
        images.reserve(n_images);

        for (uint64_t i = 0; i < n_images; ++i) {
            const auto& line = lines[i * 2];
            const auto tokens = split_string(line, ' ');

            if (tokens.size() != 10) {
                LOG_ERROR("Invalid format in images.txt line {}", i * 2 + 1);
                throw std::runtime_error("Invalid format in images.txt");
            }

            ImageData img;
            img.image_id = std::stoul(tokens[0]);
            img.qvec = {std::stof(tokens[1]), std::stof(tokens[2]),
                        std::stof(tokens[3]), std::stof(tokens[4])};
            img.tvec = {std::stof(tokens[5]), std::stof(tokens[6]), std::stof(tokens[7])};
            img.camera_id = std::stoul(tokens[8]);
            img.name = tokens[9];

            images.push_back(std::move(img));
        }

        return images;
    }

    // -----------------------------------------------------------------------------
    //  cameras.txt
    // -----------------------------------------------------------------------------
    std::unordered_map<uint32_t, CameraDataIntermediate>
    read_cameras_text(const std::filesystem::path& file_path, float scale_factor = 1.0f) {
        LOG_TIMER_TRACE("Read cameras.txt");
        auto lines = read_text_file(file_path);

        LOG_DEBUG("Reading {} cameras from text file{}", lines.size(),
                  scale_factor != 1.0f ? std::format(" with scale factor {}", scale_factor) : "");

        std::unordered_map<uint32_t, CameraDataIntermediate> cams;

        for (const auto& line : lines) {
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
    PointCloud read_point3D_text(const std::filesystem::path& file_path) {
        LOG_TIMER_TRACE("Read points3D.txt");
        auto lines = read_text_file(file_path);
        uint64_t N = lines.size();
        LOG_DEBUG("Reading {} 3D points from text file", N);

        std::vector<float> positions(N * 3);
        std::vector<uint8_t> colors(N * 3);

        for (uint64_t i = 0; i < N; ++i) {
            const auto& line = lines[i];
            const auto tokens = split_string(line, ' ');

            if (tokens.size() < 8) {
                LOG_ERROR("Invalid format in points3D.txt: {}", line);
                throw std::runtime_error("Invalid format in points3D.txt");
            }

            positions[i * 3 + 0] = std::stof(tokens[1]);
            positions[i * 3 + 1] = std::stof(tokens[2]);
            positions[i * 3 + 2] = std::stof(tokens[3]);

            // Store colors as uint8 [0,255] to match old loader
            colors[i * 3 + 0] = static_cast<uint8_t>(std::stoi(tokens[4]));
            colors[i * 3 + 1] = static_cast<uint8_t>(std::stoi(tokens[5]));
            colors[i * 3 + 2] = static_cast<uint8_t>(std::stoi(tokens[6]));
        }

        Tensor means = Tensor::from_vector(positions, {N, 3}, Device::CUDA);
        Tensor colors_tensor = Tensor::from_blob(colors.data(), {N, 3}, Device::CPU, DataType::UInt8)
                                   .to(Device::CUDA)
                                   .contiguous();

        return PointCloud(std::move(means), std::move(colors_tensor));
    }

    // -----------------------------------------------------------------------------
    //  Assemble cameras with dimension verification
    // -----------------------------------------------------------------------------
    std::tuple<std::vector<std::shared_ptr<Camera>>, Tensor>
    assemble_colmap_cameras(const std::filesystem::path& base_path,
                            const std::unordered_map<uint32_t, CameraDataIntermediate>& cam_map,
                            const std::vector<ImageData>& images,
                            const std::string& images_folder) {

        LOG_TIMER_TRACE("Assemble COLMAP cameras");

        std::filesystem::path images_path = base_path / images_folder;

        if (!std::filesystem::exists(images_path)) {
            LOG_ERROR("Images folder does not exist: {}", images_path.string());
            throw std::runtime_error("Images folder does not exist");
        }

        std::vector<std::shared_ptr<Camera>> cameras;
        cameras.reserve(images.size());

        // Accumulate camera positions for scene center
        std::vector<float> camera_positions;
        camera_positions.reserve(images.size() * 3);

        for (size_t i = 0; i < images.size(); ++i) {
            const ImageData& img = images[i];

            auto it = cam_map.find(img.camera_id);
            if (it == cam_map.end()) {
                LOG_ERROR("Camera ID {} not found", img.camera_id);
                throw std::runtime_error("Camera ID not found");
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
                LOG_ERROR("Invalid camera model ID: {}", cam_data.model_id);
                throw std::runtime_error("Invalid camera model ID");
            }

            CAMERA_MODEL model = model_it->second.first;
            const auto& params = cam_data.params;

            float focal_x = 0, focal_y = 0, center_x = 0, center_y = 0;
            Tensor radial_dist, tangential_dist;
            gsplat::CameraModelType camera_model_type = gsplat::CameraModelType::PINHOLE;

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
                radial_dist = Tensor::from_vector({params[3]}, {1}, Device::CPU);
                tangential_dist = Tensor::empty({0}, Device::CPU);
                break;

            case CAMERA_MODEL::RADIAL:
                focal_x = focal_y = params[0];
                center_x = params[1];
                center_y = params[2];
                radial_dist = Tensor::from_vector({params[3], params[4]}, {2}, Device::CPU);
                tangential_dist = Tensor::empty({0}, Device::CPU);
                break;

            case CAMERA_MODEL::OPENCV:
                focal_x = params[0];
                focal_y = params[1];
                center_x = params[2];
                center_y = params[3];
                radial_dist = Tensor::from_vector({params[4], params[5]}, {2}, Device::CPU);
                tangential_dist = Tensor::from_vector({params[6], params[7]}, {2}, Device::CPU);
                break;

            case CAMERA_MODEL::FULL_OPENCV:
                focal_x = params[0];
                focal_y = params[1];
                center_x = params[2];
                center_y = params[3];
                radial_dist = Tensor::from_vector({params[4], params[5], params[8], params[9], params[10], params[11]}, {6}, Device::CPU);
                tangential_dist = Tensor::from_vector({params[6], params[7]}, {2}, Device::CPU);
                break;

            case CAMERA_MODEL::OPENCV_FISHEYE:
                focal_x = params[0];
                focal_y = params[1];
                center_x = params[2];
                center_y = params[3];
                radial_dist = Tensor::from_vector({params[4], params[5], params[6], params[7]}, {4}, Device::CPU);
                tangential_dist = Tensor::empty({0}, Device::CPU);
                camera_model_type = gsplat::CameraModelType::FISHEYE;
                break;

            case CAMERA_MODEL::RADIAL_FISHEYE:
                focal_x = focal_y = params[0];
                center_x = params[1];
                center_y = params[2];
                radial_dist = Tensor::from_vector({params[3], params[4]}, {2}, Device::CPU);
                tangential_dist = Tensor::empty({0}, Device::CPU);
                camera_model_type = gsplat::CameraModelType::FISHEYE;
                break;

            case CAMERA_MODEL::SIMPLE_RADIAL_FISHEYE:
                focal_x = focal_y = params[0];
                center_x = params[1];
                center_y = params[2];
                radial_dist = Tensor::from_vector({params[3]}, {1}, Device::CPU);
                tangential_dist = Tensor::empty({0}, Device::CPU);
                camera_model_type = gsplat::CameraModelType::FISHEYE;
                break;

            case CAMERA_MODEL::THIN_PRISM_FISHEYE:
                LOG_ERROR("THIN_PRISM_FISHEYE not supported");
                throw std::runtime_error("THIN_PRISM_FISHEYE not supported");

            case CAMERA_MODEL::FOV:
                LOG_ERROR("FOV camera model not supported");
                throw std::runtime_error("FOV camera model not supported");

            default:
                LOG_ERROR("Unsupported camera model");
                throw std::runtime_error("Unsupported camera model");
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
                images_path / img.name,
                cam_data.width,
                cam_data.height,
                static_cast<int>(i));

            cameras.push_back(std::move(camera));
        }

        // Compute scene center as mean of camera positions
        Tensor scene_center_tensor = Tensor::from_vector(camera_positions, {images.size(), 3}, Device::CPU);
        Tensor scene_center = scene_center_tensor.mean({0}, false);

        LOG_INFO("Training with {} images", cameras.size());

        return {std::move(cameras), scene_center};
    }

    // -----------------------------------------------------------------------------
    //  Public API
    // -----------------------------------------------------------------------------

    static fs::path get_sparse_file_path(const fs::path& base, const std::string& filename) {
        auto search_paths = get_colmap_search_paths(base);
        auto found = find_file_in_paths(search_paths, filename);

        if (!found.empty()) {
            LOG_TRACE("Found sparse file at: {}", found.string());
            return found;
        }

        std::string error_msg = std::format("Cannot find '{}' in any location", filename);
        LOG_ERROR("{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    PointCloud read_colmap_point_cloud(const std::filesystem::path& filepath) {
        LOG_TIMER_TRACE("Read COLMAP point cloud");
        fs::path points3d_file = get_sparse_file_path(filepath, "points3D.bin");
        return read_point3D_binary(points3d_file);
    }

    std::tuple<std::vector<std::shared_ptr<Camera>>, Tensor>
    read_colmap_cameras_and_images(const std::filesystem::path& base,
                                   const std::string& images_folder) {

        LOG_TIMER_TRACE("Read COLMAP cameras and images");

        const float scale_factor = extract_scale_from_folder(images_folder);

        fs::path cams_file = get_sparse_file_path(base, "cameras.bin");
        fs::path images_file = get_sparse_file_path(base, "images.bin");

        auto cam_map = read_cameras_binary(cams_file, scale_factor);
        auto images = read_images_binary(images_file);

        LOG_INFO("Read {} cameras and {} images from COLMAP", cam_map.size(), images.size());

        return assemble_colmap_cameras(base, cam_map, images, images_folder);
    }

    PointCloud read_colmap_point_cloud_text(const std::filesystem::path& filepath) {
        LOG_TIMER_TRACE("Read COLMAP point cloud (text)");
        fs::path points3d_file = get_sparse_file_path(filepath, "points3D.txt");
        return read_point3D_text(points3d_file);
    }

    std::tuple<std::vector<std::shared_ptr<Camera>>, Tensor>
    read_colmap_cameras_and_images_text(const std::filesystem::path& base,
                                        const std::string& images_folder) {

        LOG_TIMER_TRACE("Read COLMAP cameras and images (text)");

        const float scale_factor = extract_scale_from_folder(images_folder);

        fs::path cams_file = get_sparse_file_path(base, "cameras.txt");
        fs::path images_file = get_sparse_file_path(base, "images.txt");

        auto cam_map = read_cameras_text(cams_file, scale_factor);
        auto images = read_images_text(images_file);

        LOG_INFO("Read {} cameras and {} images from COLMAP text files", cam_map.size(), images.size());

        return assemble_colmap_cameras(base, cam_map, images, images_folder);
    }

} // namespace lfs::loader