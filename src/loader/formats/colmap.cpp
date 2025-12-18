/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "colmap.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/point_cloud.hpp"
#include "core/torch_shapes.hpp"
#include "loader/filesystem_utils.hpp"
#include <algorithm>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

namespace gs::loader {

    namespace fs = std::filesystem;
    namespace F = torch::nn::functional;

    // -----------------------------------------------------------------------------
    //  Quaternion to rotation matrix
    // -----------------------------------------------------------------------------
    inline torch::Tensor qvec2rotmat(const torch::Tensor& qraw) {
        assert_vec(qraw, 4, "qvec");

        auto q = F::normalize(qraw.to(torch::kFloat32),
                              F::NormalizeFuncOptions().dim(0));

        auto w = q[0], x = q[1], y = q[2], z = q[3];

        torch::Tensor R = torch::empty({3, 3}, torch::kFloat32);
        R[0][0] = 1 - 2 * (y * y + z * z);
        R[0][1] = 2 * (x * y - z * w);
        R[0][2] = 2 * (x * z + y * w);

        R[1][0] = 2 * (x * y + z * w);
        R[1][1] = 1 - 2 * (x * x + z * z);
        R[1][2] = 2 * (y * z - x * w);

        R[2][0] = 2 * (x * z - y * w);
        R[2][1] = 2 * (y * z + x * w);
        R[2][2] = 1 - 2 * (x * x + y * y);
        return R;
    }

    class Image {
    public:
        Image() = default;
        explicit Image(uint32_t id)
            : _image_ID(id) {}

        uint32_t _camera_id = 0;
        std::string _name;

        torch::Tensor _qvec = torch::tensor({1.f, 0.f, 0.f, 0.f}, torch::kFloat32);
        torch::Tensor _tvec = torch::zeros({3}, torch::kFloat32);

    private:
        uint32_t _image_ID = 0;
    };

    // -----------------------------------------------------------------------------
    //  Build 4x4 world-to-camera matrix
    // -----------------------------------------------------------------------------
    inline torch::Tensor getWorld2View(const torch::Tensor& R,
                                       const torch::Tensor& T) {
        assert_mat(R, 3, 3, "R");
        assert_vec(T, 3, "T");

        torch::Tensor M = torch::eye(4, torch::kFloat32);
        M.index_put_({torch::indexing::Slice(0, 3),
                      torch::indexing::Slice(0, 3)},
                     R);
        M.index_put_({torch::indexing::Slice(0, 3), 3},
                     (-torch::matmul(R, T)).reshape({3}));
        return M;
    }

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
    //  Helper to scale camera intrinsics based on model
    // -----------------------------------------------------------------------------
    static void scale_camera_intrinsics(CAMERA_MODEL model, std::vector<double>& params, float factor) {
        switch (model) {
        case CAMERA_MODEL::SIMPLE_PINHOLE:
            // params: [f, cx, cy]
            params[0] /= factor; // f
            params[1] /= factor; // cx
            params[2] /= factor; // cy
            break;

        case CAMERA_MODEL::PINHOLE:
            // params: [fx, fy, cx, cy]
            params[0] /= factor; // fx
            params[1] /= factor; // fy
            params[2] /= factor; // cx
            params[3] /= factor; // cy
            break;

        case CAMERA_MODEL::SIMPLE_RADIAL:
            // params: [f, cx, cy, k1]
            params[0] /= factor; // f
            params[1] /= factor; // cx
            params[2] /= factor; // cy
            // k1 unchanged
            break;

        case CAMERA_MODEL::RADIAL:
            // params: [f, cx, cy, k1, k2]
            params[0] /= factor; // f
            params[1] /= factor; // cx
            params[2] /= factor; // cy
            // k1, k2 unchanged
            break;

        case CAMERA_MODEL::OPENCV:
        case CAMERA_MODEL::OPENCV_FISHEYE:
            // params: [fx, fy, cx, cy, k1, k2, p1, p2, ...]
            params[0] /= factor; // fx
            params[1] /= factor; // fy
            params[2] /= factor; // cx
            params[3] /= factor; // cy
            // distortion params unchanged
            break;

        case CAMERA_MODEL::FULL_OPENCV:
            // params: [fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6]
            params[0] /= factor; // fx
            params[1] /= factor; // fy
            params[2] /= factor; // cx
            params[3] /= factor; // cy
            // distortion params unchanged
            break;

        case CAMERA_MODEL::SIMPLE_RADIAL_FISHEYE:
        case CAMERA_MODEL::RADIAL_FISHEYE:
            // params: [f, cx, cy, k1, ...]
            params[0] /= factor; // f
            params[1] /= factor; // cx
            params[2] /= factor; // cy
            // distortion params unchanged
            break;

        case CAMERA_MODEL::FOV:
            // params: [fx, fy, cx, cy, omega]
            params[0] /= factor; // fx
            params[1] /= factor; // fy
            params[2] /= factor; // cx
            params[3] /= factor; // cy
            // omega unchanged
            break;

        case CAMERA_MODEL::THIN_PRISM_FISHEYE:
            // params: [fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1]
            params[0] /= factor; // fx
            params[1] /= factor; // fy
            params[2] /= factor; // cx
            params[3] /= factor; // cy
            // distortion params unchanged
            break;

        default:
            LOG_WARN("Unknown camera model for scaling: {}", static_cast<int>(model));
            // At minimum, try to scale principal point if present
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
        // Check for pattern like "images_4" where 4 is the downscale factor
        size_t underscore_pos = folder_name.rfind('_');
        if (underscore_pos != std::string::npos) {
            std::string suffix = folder_name.substr(underscore_pos + 1);
            try {
                float factor = std::stof(suffix);
                if (factor > 0 && factor <= 16) { // Sanity check
                    LOG_DEBUG("Extracted scale factor {} from folder name", factor);
                    return factor;
                }
            } catch (...) {
                // Not a valid number
            }
        }
        return 1.0f;
    }

    // -----------------------------------------------------------------------------
    //  Helper to apply dimension correction to camera
    // -----------------------------------------------------------------------------
    static void apply_dimension_correction(CameraData& cam, float scale_x, float scale_y,
                                           int actual_w, int actual_h) {
        // Update dimensions
        cam._width = actual_w;
        cam._height = actual_h;

        // Update intrinsics that were already extracted
        cam._focal_x *= scale_x;
        cam._focal_y *= scale_y;
        cam._center_x *= scale_x;
        cam._center_y *= scale_y;

        // No need to update distortion parameters as they are dimensionless
        LOG_TRACE("Applied dimension correction to camera: scale_x={:.3f}, scale_y={:.3f}", scale_x, scale_y);
    }

    // -----------------------------------------------------------------------------
    //  images.bin
    // -----------------------------------------------------------------------------
    std::vector<Image> read_images_binary(const std::filesystem::path& file_path) {
        LOG_TIMER_TRACE("Read images.bin");
        auto buf_owner = read_binary(file_path);
        const char* cur = buf_owner->data();
        const char* end = cur + buf_owner->size();

        uint64_t n_images = read_u64(cur);
        LOG_DEBUG("Reading {} images from binary file", n_images);
        std::vector<Image> images;
        images.reserve(n_images);

        for (uint64_t i = 0; i < n_images; ++i) {
            uint32_t id = read_u32(cur);
            auto& img = images.emplace_back(id);

            torch::Tensor q = torch::empty({4}, torch::kFloat32);
            for (int k = 0; k < 4; ++k)
                q[k] = static_cast<float>(read_f64(cur));

            img._qvec = q;

            torch::Tensor t = torch::empty({3}, torch::kFloat32);
            for (int k = 0; k < 3; ++k)
                t[k] = static_cast<float>(read_f64(cur));
            img._tvec = t;

            img._camera_id = read_u32(cur);

            img._name.assign(cur);
            cur += img._name.size() + 1; // skip '\0'

            uint64_t npts = read_u64(cur); // skip 2-D points
            cur += npts * (sizeof(double) * 2 + sizeof(uint64_t));
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
    std::unordered_map<uint32_t, CameraData>
    read_cameras_binary(const std::filesystem::path& file_path, float scale_factor = 1.0f) {
        LOG_TIMER_TRACE("Read cameras.bin");
        auto buf_owner = read_binary(file_path);
        const char* cur = buf_owner->data();
        const char* end = cur + buf_owner->size();

        uint64_t n_cams = read_u64(cur);
        LOG_DEBUG("Reading {} cameras from binary file{}", n_cams,
                  scale_factor != 1.0f ? std::format(" with scale factor {}", scale_factor) : "");
        std::unordered_map<uint32_t, CameraData> cams;
        cams.reserve(n_cams);

        for (uint64_t i = 0; i < n_cams; ++i) {
            CameraData cam;
            cam._camera_ID = read_u32(cur);

            int32_t model_id = read_i32(cur);
            cam._width = read_u64(cur);
            cam._height = read_u64(cur);

            // Apply scaling to dimensions if needed
            if (scale_factor != 1.0f) {
                cam._width = static_cast<uint64_t>(cam._width / scale_factor);
                cam._height = static_cast<uint64_t>(cam._height / scale_factor);
                LOG_TRACE("Scaled camera {} dimensions to {}x{}",
                          cam._camera_ID, cam._width, cam._height);
            }

            auto it = camera_model_ids.find(model_id);
            if (it == camera_model_ids.end() || it->second.second < 0) {
                LOG_ERROR("Unsupported camera-model id: {}", model_id);
                throw std::runtime_error("Unsupported camera-model id " + std::to_string(model_id));
            }

            cam._camera_model = it->second.first;
            int32_t param_cnt = it->second.second;

            // Read raw parameters
            std::vector<double> raw_params(param_cnt);
            for (int j = 0; j < param_cnt; j++) {
                raw_params[j] = read_f64(cur);
            }

            // Scale intrinsics based on camera model
            if (scale_factor != 1.0f) {
                scale_camera_intrinsics(cam._camera_model, raw_params, scale_factor);
            }

            cam._params = torch::from_blob(raw_params.data(), {param_cnt}, torch::kFloat64)
                              .clone()
                              .to(torch::kFloat32);

            cams.emplace(cam._camera_ID, std::move(cam));
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

        // Pre-allocate tensors directly
        torch::Tensor positions = torch::empty({static_cast<int64_t>(N), 3}, torch::kFloat32);
        torch::Tensor colors = torch::empty({static_cast<int64_t>(N), 3}, torch::kUInt8);

        // Get raw pointers for efficient access
        float* pos_data = positions.data_ptr<float>();
        uint8_t* col_data = colors.data_ptr<uint8_t>();

        for (uint64_t i = 0; i < N; ++i) {
            cur += 8; // skip point ID

            // Read position directly into tensor
            pos_data[i * 3 + 0] = static_cast<float>(read_f64(cur));
            pos_data[i * 3 + 1] = static_cast<float>(read_f64(cur));
            pos_data[i * 3 + 2] = static_cast<float>(read_f64(cur));

            // Read color directly into tensor
            col_data[i * 3 + 0] = *cur++;
            col_data[i * 3 + 1] = *cur++;
            col_data[i * 3 + 2] = *cur++;

            cur += 8;                                    // skip reprojection error
            cur += read_u64(cur) * sizeof(uint32_t) * 2; // skip track
        }

        if (cur != end) {
            LOG_ERROR("points3D.bin has trailing bytes");
            throw std::runtime_error("points3D.bin: trailing bytes");
        }

        return PointCloud(positions, colors);
    }

    // -----------------------------------------------------------------------------
    //  Text-file loader
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
            if (line.starts_with("#")) {
                continue; // Skip comment lines
            }
            if (!line.empty() && line.back() == '\r') {
                line.pop_back(); // Remove trailing carriage return
            }
            lines.push_back(line);
        }
        file.close();
        if (lines.empty()) {
            LOG_ERROR("File is empty or contains no valid lines: {}", file_path.string());
            throw std::runtime_error("File " + file_path.string() + " is empty or contains no valid lines");
        }
        // Ensure the last line is not empty
        if (lines.back().empty()) {
            lines.pop_back(); // Remove last empty line if it exists
        }
        LOG_TRACE("Read {} lines from text file", lines.size());
        return lines;
    }

    std::vector<std::string> split_string(const std::string& s, char delimiter) {
        std::vector<std::string> tokens;
        std::string token;
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
    //  Image list with two lines of data per image:
    //   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    //   POINTS2D[] as (X, Y, POINT3D_ID)
    // -----------------------------------------------------------------------------
    std::vector<Image> read_images_text(const std::filesystem::path& file_path) {
        LOG_TIMER_TRACE("Read images.txt");
        auto lines = read_text_file(file_path);
        std::vector<Image> images;
        if (lines.size() % 2 != 0) {
            LOG_ERROR("images.txt should have an even number of lines");
            throw std::runtime_error("images.txt should have an even number of lines");
        }
        uint64_t n_images = lines.size() / 2;
        LOG_DEBUG("Reading {} images from text file", n_images);

        for (uint64_t i = 0; i < n_images; ++i) {
            const auto& line = lines[i * 2];

            const auto tokens = split_string(line, ' ');
            if (tokens.size() != 10) {
                LOG_ERROR("Invalid format in images.txt line {}", i * 2 + 1);
                throw std::runtime_error("Invalid format in images.txt line " + std::to_string(i * 2 + 1));
            }

            auto& img = images.emplace_back(std::stoul(tokens[0]));
            img._qvec = torch::tensor({std::stof(tokens[1]), std::stof(tokens[2]),
                                       std::stof(tokens[3]), std::stof(tokens[4])},
                                      torch::kFloat32);

            img._tvec = torch::tensor({std::stof(tokens[5]), std::stof(tokens[6]),
                                       std::stof(tokens[7])},
                                      torch::kFloat32);

            img._camera_id = std::stoul(tokens[8]);
            img._name = tokens[9];
        }
        return images;
    }

    // -----------------------------------------------------------------------------
    //  cameras.txt with optional scaling
    // -----------------------------------------------------------------------------
    std::unordered_map<uint32_t, CameraData>
    read_cameras_text(const std::filesystem::path& file_path, float scale_factor = 1.0f) {
        LOG_TIMER_TRACE("Read cameras.txt");
        auto lines = read_text_file(file_path);
        std::unordered_map<uint32_t, CameraData> cams;
        LOG_DEBUG("Reading {} cameras from text file{}", lines.size(),
                  scale_factor != 1.0f ? std::format(" with scale factor {}", scale_factor) : "");

        for (const auto& line : lines) {
            const auto tokens = split_string(line, ' ');
            if (tokens.size() < 4) {
                LOG_ERROR("Invalid format in cameras.txt: {}", line);
                throw std::runtime_error("Invalid format in cameras.txt: " + line);
            }

            CameraData cam;
            cam._camera_ID = std::stoul(tokens[0]);
            if (!camera_model_names.contains(tokens[1])) {
                LOG_ERROR("Unknown camera model in cameras.txt: {}", tokens[1]);
                throw std::runtime_error("Invalid format in cameras.txt: " + line);
            }
            cam._camera_model = camera_model_names.at(tokens[1]);
            cam._width = std::stoi(tokens[2]);
            cam._height = std::stoi(tokens[3]);

            // Apply scaling to dimensions if needed
            if (scale_factor != 1.0f) {
                cam._width = static_cast<uint64_t>(cam._width / scale_factor);
                cam._height = static_cast<uint64_t>(cam._height / scale_factor);
                LOG_TRACE("Scaled camera {} dimensions to {}x{}",
                          cam._camera_ID, cam._width, cam._height);
            }

            // Read parameters
            std::vector<double> raw_params;
            for (uint64_t j = 4; j < tokens.size(); ++j) {
                raw_params.push_back(std::stod(tokens[j]));
            }

            // Scale intrinsics based on camera model
            if (scale_factor != 1.0f) {
                scale_camera_intrinsics(cam._camera_model, raw_params, scale_factor);
            }

            cam._params = torch::empty({static_cast<int64_t>(raw_params.size())}, torch::kFloat32);
            for (size_t j = 0; j < raw_params.size(); ++j) {
                cam._params[j] = static_cast<float>(raw_params[j]);
            }

            cams.emplace(cam._camera_ID, std::move(cam));
        }
        return cams;
    }

    // -----------------------------------------------------------------------------
    //  point3D.txt
    //  3D point list with one line of data per point:
    //    POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    // -----------------------------------------------------------------------------
    PointCloud read_point3D_text(const std::filesystem::path& file_path) {
        LOG_TIMER_TRACE("Read points3D.txt");
        auto lines = read_text_file(file_path);
        uint64_t N = lines.size();
        LOG_DEBUG("Reading {} 3D points from text file", N);

        torch::Tensor positions = torch::empty({static_cast<int64_t>(N), 3}, torch::kFloat32);
        torch::Tensor colors = torch::empty({static_cast<int64_t>(N), 3}, torch::kUInt8);

        float* pos_data = positions.data_ptr<float>();
        uint8_t* col_data = colors.data_ptr<uint8_t>();

        for (uint64_t i = 0; i < N; ++i) {
            const auto& line = lines[i];
            const auto tokens = split_string(line, ' ');

            if (tokens.size() < 8) {
                LOG_ERROR("Invalid format in points3D.txt: {}", line);
                throw std::runtime_error("Invalid format in point3D.txt: " + line);
            }

            pos_data[i * 3 + 0] = std::stof(tokens[1]);
            pos_data[i * 3 + 1] = std::stof(tokens[2]);
            pos_data[i * 3 + 2] = std::stof(tokens[3]);

            col_data[i * 3 + 0] = std::stoi(tokens[4]);
            col_data[i * 3 + 1] = std::stoi(tokens[5]);
            col_data[i * 3 + 2] = std::stoi(tokens[6]);
        }
        return PointCloud(positions, colors);
    }

    // -----------------------------------------------------------------------------
    //  Assemble per-image camera information with dimension verification
    // -----------------------------------------------------------------------------
    std::tuple<std::vector<CameraData>, torch::Tensor>
    read_colmap_cameras(const std::filesystem::path base_path,
                        const std::unordered_map<uint32_t, CameraData>& cams,
                        const std::vector<Image>& images,
                        const std::string& images_folder = "images") {
        LOG_TIMER_TRACE("Assemble COLMAP cameras");
        std::vector<CameraData> out(images.size());

        std::filesystem::path images_path = base_path / images_folder;
        std::filesystem::path masks_path = base_path / "masks";

        // Prepare tensor to store all camera locations [N, 3]
        torch::Tensor camera_locations = torch::zeros({static_cast<int64_t>(images.size()), 3}, torch::kFloat32);

        // Check if the specified images folder exists
        if (!std::filesystem::exists(images_path)) {
            LOG_ERROR("Images folder does not exist: {}", images_path.string());
            throw std::runtime_error("Images folder does not exist: " + images_path.string());
        }

        for (size_t i = 0; i < images.size(); ++i) {
            const Image& img = images[i];
            auto it = cams.find(img._camera_id);
            if (it == cams.end()) {
                LOG_ERROR("Camera ID {} not found", img._camera_id);
                throw std::runtime_error("Camera ID " + std::to_string(img._camera_id) + " not found");
            }

            out[i] = it->second;
            out[i]._image_path = images_path / img._name;
            out[i]._image_name = img._name;

            // Try both mask formats: prefer img_name.png (e.g., cam_1/00.png.png), fallback to img_name (e.g., cam_1/00.png)
            std::filesystem::path mask_path_png = masks_path / img._name;
            std::filesystem::path mask_path_png_rs = masks_path / img._name;

            mask_path_png += ".png";  // Append .png to get cam_1/00.png.png
            mask_path_png_rs += ".mask.png";  // Append .mask.png to get cam_1/00.png.mask.png

            std::filesystem::path mask_path_same = masks_path / img._name;

            if (std::filesystem::exists(mask_path_png)) {
                out[i]._mask_path = mask_path_png;  // Prefer .ext.png format (e.g., cam_1/00.png.png)
            } else if (std::filesystem::exists( mask_path_png_rs)) {
                out[i]._mask_path =  mask_path_png_rs;  // Fallback to .mask.png format from reality scan (e.g., cam_1/00.png.mask.png)
            } else if (std::filesystem::exists(mask_path_same)) {
                out[i]._mask_path = mask_path_same;  // Fallback to .ext format (e.g., cam_1/00.png)
            } else {
                // No mask found, leave empty
                out[i]._mask_path = "";
            }

            out[i]._R = qvec2rotmat(img._qvec);
            out[i]._T = img._tvec.clone();

            // Camera location in world space = -R^T * T
            // This is equivalent to extracting camtoworlds[:, :3, 3] after inverting w2c
            camera_locations[i] = -torch::matmul(out[i]._R.t(), out[i]._T);

            switch (out[i]._camera_model) {
            // f, cx, cy
            case CAMERA_MODEL::SIMPLE_PINHOLE: {
                float fx = out[i]._params[0].item<float>();
                out[i]._focal_x = fx;
                out[i]._focal_y = fx;
                out[i]._center_x = out[i]._params[1].item<float>();
                out[i]._center_y = out[i]._params[2].item<float>();
                out[i]._camera_model_type = gsplat::CameraModelType::PINHOLE;
                break;
            }
            // fx, fy, cx, cy
            case CAMERA_MODEL::PINHOLE: {
                out[i]._focal_x = out[i]._params[0].item<float>();
                out[i]._focal_y = out[i]._params[1].item<float>();
                out[i]._center_x = out[i]._params[2].item<float>();
                out[i]._center_y = out[i]._params[3].item<float>();
                out[i]._camera_model_type = gsplat::CameraModelType::PINHOLE;
                break;
            }
            // f, cx, cy, k1
            case CAMERA_MODEL::SIMPLE_RADIAL: {
                float fx = out[i]._params[0].item<float>();
                out[i]._focal_x = fx;
                out[i]._focal_y = fx;
                out[i]._center_x = out[i]._params[1].item<float>();
                out[i]._center_y = out[i]._params[2].item<float>();
                float k1 = out[i]._params[3].item<float>();
                // k1 should be zero for COLMAP's SIMPLE_RADIAL to match a pinhole model
                if (k1 != 0.0f) {
                    LOG_WARN("Camera {} uses SIMPLE_RADIAL model with non-zero k1 distortion ({})",
                             out[i]._camera_ID, k1);
                    out[i]._radial_distortion = torch::tensor({k1}, torch::kFloat32);
                }
                out[i]._camera_model_type = gsplat::CameraModelType::PINHOLE;
                break;
            }
            // f, cx, cy, k1, k2
            case CAMERA_MODEL::RADIAL: {
                float fx = out[i]._params[0].item<float>();
                out[i]._focal_x = fx;
                out[i]._focal_y = fx;
                out[i]._center_x = out[i]._params[1].item<float>();
                out[i]._center_y = out[i]._params[2].item<float>();
                float k1 = out[i]._params[3].item<float>();
                float k2 = out[i]._params[4].item<float>();
                out[i]._radial_distortion = torch::tensor({k1, k2}, torch::kFloat32);
                out[i]._camera_model_type = gsplat::CameraModelType::PINHOLE;
                break;
            }
            // fx, fy, cx, cy, k1, k2, p1, p2
            case CAMERA_MODEL::OPENCV: {
                out[i]._focal_x = out[i]._params[0].item<float>();
                out[i]._focal_y = out[i]._params[1].item<float>();
                out[i]._center_x = out[i]._params[2].item<float>();
                out[i]._center_y = out[i]._params[3].item<float>();

                float k1 = out[i]._params[4].item<float>();
                float k2 = out[i]._params[5].item<float>();
                out[i]._radial_distortion = torch::tensor({k1, k2}, torch::kFloat32);

                float p1 = out[i]._params[6].item<float>();
                float p2 = out[i]._params[7].item<float>();
                out[i]._tangential_distortion = torch::tensor({p1, p2}, torch::kFloat32);

                out[i]._camera_model_type = gsplat::CameraModelType::PINHOLE;
                break;
            }
            // fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
            case CAMERA_MODEL::FULL_OPENCV: {
                out[i]._focal_x = out[i]._params[0].item<float>();
                out[i]._focal_y = out[i]._params[1].item<float>();
                out[i]._center_x = out[i]._params[2].item<float>();
                out[i]._center_y = out[i]._params[3].item<float>();

                float k1 = out[i]._params[4].item<float>();
                float k2 = out[i]._params[5].item<float>();
                float k3 = out[i]._params[8].item<float>();
                float k4 = out[i]._params[9].item<float>();
                float k5 = out[i]._params[10].item<float>();
                float k6 = out[i]._params[11].item<float>();
                out[i]._radial_distortion = torch::tensor({k1, k2, k3, k4, k5, k6}, torch::kFloat32);

                float p1 = out[i]._params[6].item<float>();
                float p2 = out[i]._params[7].item<float>();
                out[i]._tangential_distortion = torch::tensor({p1, p2}, torch::kFloat32);
                out[i]._camera_model_type = gsplat::CameraModelType::PINHOLE;
                break;
            }
            // fx, fy, cx, cy, k1, k2, k3, k4
            case CAMERA_MODEL::OPENCV_FISHEYE: {
                out[i]._focal_x = out[i]._params[0].item<float>();
                out[i]._focal_y = out[i]._params[1].item<float>();
                out[i]._center_x = out[i]._params[2].item<float>();
                out[i]._center_y = out[i]._params[3].item<float>();

                float k1 = out[i]._params[4].item<float>();
                float k2 = out[i]._params[5].item<float>();
                float k3 = out[i]._params[6].item<float>();
                float k4 = out[i]._params[7].item<float>();
                out[i]._radial_distortion = torch::tensor({k1, k2, k3, k4}, torch::kFloat32);
                out[i]._camera_model_type = gsplat::CameraModelType::FISHEYE;
                break;
            }
            // f, cx, cy, k1, k2
            case CAMERA_MODEL::RADIAL_FISHEYE: {
                float fx = out[i]._params[0].item<float>();
                out[i]._focal_x = fx;
                out[i]._focal_y = fx;
                out[i]._center_x = out[i]._params[1].item<float>();
                out[i]._center_y = out[i]._params[2].item<float>();
                float k1 = out[i]._params[3].item<float>();
                float k2 = out[i]._params[4].item<float>();
                out[i]._radial_distortion = torch::tensor({k1, k2}, torch::kFloat32);
                out[i]._camera_model_type = gsplat::CameraModelType::FISHEYE;
                break;
            }
            // f, cx, cy, k
            case CAMERA_MODEL::SIMPLE_RADIAL_FISHEYE: {
                float fx = out[i]._params[0].item<float>();
                out[i]._focal_x = fx;
                out[i]._focal_y = fx;
                out[i]._center_x = out[i]._params[1].item<float>();
                out[i]._center_y = out[i]._params[2].item<float>();
                float k = out[i]._params[3].item<float>();
                out[i]._radial_distortion = torch::tensor({k}, torch::kFloat32);
                out[i]._camera_model_type = gsplat::CameraModelType::FISHEYE;
                break;
            }
            // fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1
            case CAMERA_MODEL::THIN_PRISM_FISHEYE: {
                throw std::runtime_error("THIN_PRISM_FISHEYE camera model is not supported but could be implemented in 3DGUT pretty easily");
                out[i]._focal_x = out[i]._params[0].item<float>();
                out[i]._focal_y = out[i]._params[1].item<float>();
                out[i]._center_x = out[i]._params[2].item<float>();
                out[i]._center_y = out[i]._params[3].item<float>();

                float k1 = out[i]._params[4].item<float>();
                float k2 = out[i]._params[5].item<float>();
                float k3 = out[i]._params[8].item<float>();
                float k4 = out[i]._params[9].item<float>();
                out[i]._radial_distortion = torch::tensor({k1, k2, k3, k4}, torch::kFloat32);

                float p1 = out[i]._params[6].item<float>();
                float p2 = out[i]._params[7].item<float>();
                out[i]._tangential_distortion = torch::tensor({p1, p2}, torch::kFloat32);
                out[i]._camera_model_type = gsplat::CameraModelType::FISHEYE;
                break;
            }
            // fx, fy, cx, cy, omega
            case CAMERA_MODEL::FOV: {
                throw std::runtime_error("FOV camera model is not supported.");
                out[i]._focal_x = out[i]._params[0].item<float>();
                out[i]._focal_y = out[i]._params[1].item<float>();
                out[i]._center_x = out[i]._params[2].item<float>();
                out[i]._center_y = out[i]._params[3].item<float>();
                // float omega = out[i]._params[4].item<float>();
                // out[i]._camera_model_type = ;
                break;
            }
            default:
                LOG_ERROR("Unsupported camera model");
                throw std::runtime_error("Unsupported camera model");
            }

            out[i]._img_w = out[i]._img_h = out[i]._channels = 0;
            out[i]._img_data = nullptr;
        }

        // Verify actual image dimensions and apply correction if needed
        if (!out.empty() && std::filesystem::exists(out[0]._image_path)) {
            LOG_DEBUG("Verifying actual image dimensions against COLMAP database");

            // Load first image to check actual dimensions
            auto [actual_w, actual_h, channels] = get_image_info(out[0]._image_path);

            int expected_w = out[0]._width;
            int expected_h = out[0]._height;

            float scale_x = static_cast<float>(actual_w) / expected_w;
            float scale_y = static_cast<float>(actual_h) / expected_h;

            if (std::abs(scale_x - 1.0f) > 1e-5 || std::abs(scale_y - 1.0f) > 1e-5) {
                LOG_WARN("Image dimension mismatch detected!");
                LOG_INFO("  Expected (from COLMAP): {}x{}", expected_w, expected_h);
                LOG_INFO("  Actual (from image file): {}x{}", actual_w, actual_h);
                LOG_INFO("  Applying correction scale: {:.3f}x{:.3f}", scale_x, scale_y);

                // Apply correction to all cameras
                for (auto& cam : out) {
                    apply_dimension_correction(cam, scale_x, scale_y, actual_w, actual_h);
                }
            } else {
                LOG_DEBUG("Image dimensions match COLMAP database ({}x{})", actual_w, actual_h);
            }
        }

        LOG_INFO("Training with {} images", out.size());
        return {std::move(out), camera_locations.mean(0)};
    }

    // -----------------------------------------------------------------------------
    //  Public API functions
    // -----------------------------------------------------------------------------

    static fs::path get_sparse_file_path(const fs::path& base, const std::string& filename) {
        auto search_paths = get_colmap_search_paths(base);
        auto found = find_file_in_paths(search_paths, filename);

        if (!found.empty()) {
            LOG_TRACE("Found sparse file at: {}", found.string());
            return found;
        }

        // Build error message showing all attempted locations
        std::string error_msg = std::format("Cannot find '{}' in any of these locations:\n", filename);
        for (const auto& dir : search_paths) {
            error_msg += std::format("  - {}\n", (dir / filename).string());
        }
        error_msg += "Searched case-insensitively for: " + filename;

        LOG_ERROR("{}", error_msg);
        throw std::runtime_error(error_msg);
    }

    PointCloud read_colmap_point_cloud(const std::filesystem::path& filepath) {
        LOG_TIMER_TRACE("Read COLMAP point cloud");
        fs::path points3d_file = get_sparse_file_path(filepath, "points3D.bin");
        return read_point3D_binary(points3d_file);
    }

    std::tuple<std::vector<CameraData>, torch::Tensor> read_colmap_cameras_and_images(
        const std::filesystem::path& base,
        const std::string& images_folder) {

        LOG_TIMER_TRACE("Read COLMAP cameras and images");

        // Extract scale factor from folder name if present
        const float scale_factor = extract_scale_from_folder(images_folder);

        fs::path cams_file = get_sparse_file_path(base, "cameras.bin");
        fs::path images_file = get_sparse_file_path(base, "images.bin");

        auto cams = read_cameras_binary(cams_file, scale_factor);
        auto images = read_images_binary(images_file);

        LOG_INFO("Read {} cameras and {} images from COLMAP", cams.size(), images.size());

        return read_colmap_cameras(base, cams, images, images_folder);
    }

    PointCloud read_colmap_point_cloud_text(const std::filesystem::path& filepath) {
        LOG_TIMER_TRACE("Read COLMAP point cloud (text)");
        fs::path points3d_file = get_sparse_file_path(filepath, "points3D.txt");
        return read_point3D_text(points3d_file);
    }

    std::tuple<std::vector<CameraData>, torch::Tensor> read_colmap_cameras_and_images_text(
        const std::filesystem::path& base,
        const std::string& images_folder) {

        LOG_TIMER_TRACE("Read COLMAP cameras and images (text)");

        // Extract scale factor from folder name if present
        const float scale_factor = extract_scale_from_folder(images_folder);

        fs::path cams_file = get_sparse_file_path(base, "cameras.txt");
        fs::path images_file = get_sparse_file_path(base, "images.txt");

        auto cams = read_cameras_text(cams_file, scale_factor);
        auto images = read_images_text(images_file);

        LOG_INFO("Read {} cameras and {} images from COLMAP text files", cams.size(), images.size());

        return read_colmap_cameras(base, cams, images, images_folder);
    }

} // namespace gs::loader
