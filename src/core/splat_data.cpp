/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_data.hpp"
#include "core/logger.hpp"
#include "core/parameters.hpp"
#include "core/point_cloud.hpp"
#include "core/sogs.hpp"

#include "external/nanoflann.hpp"
#include "external/tinyply.hpp"
#include <algorithm>
#include <cmath>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <future>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <print>
#include <string>
#include <thread>
#include <torch/torch.h>
#include <vector>

namespace {
    std::string tensor_sizes_to_string(const c10::ArrayRef<int64_t>& sizes) {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < sizes.size(); ++i) {
            if (i > 0)
                oss << ", ";
            oss << sizes[i];
        }
        oss << "]";
        return oss.str();
    }

    // Point cloud adaptor for nanoflann
    struct PointCloudAdaptor {
        const float* points;
        size_t num_points;

        PointCloudAdaptor(const float* pts, size_t n) : points(pts),
                                                        num_points(n) {}

        inline size_t kdtree_get_point_count() const { return num_points; }

        inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
            return points[idx * 3 + dim];
        }

        template <class BBOX>
        bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
    };

    // Fixed: KDTree typedef on single line
    using KDTree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloudAdaptor>, PointCloudAdaptor, 3>;

    // Compute mean distance to 3 nearest neighbors for each point
    torch::Tensor compute_mean_neighbor_distances(const torch::Tensor& points) {
        auto cpu_points = points.to(torch::kCPU).contiguous();
        const int num_points = cpu_points.size(0);

        TORCH_CHECK(cpu_points.dim() == 2 && cpu_points.size(1) == 3,
                    "Input points must have shape [N, 3]");
        TORCH_CHECK(cpu_points.dtype() == torch::kFloat32,
                    "Input points must be float32");

        if (num_points <= 1) {
            return torch::full({num_points}, 0.01f, points.options());
        }

        const float* data = cpu_points.data_ptr<float>();

        PointCloudAdaptor cloud(data, num_points);
        KDTree index(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        index.buildIndex();

        auto result = torch::zeros({num_points}, torch::kFloat32);
        float* result_data = result.data_ptr<float>();

#pragma omp parallel for if (num_points > 1000)
        for (int i = 0; i < num_points; i++) {
            const float query_pt[3] = {data[i * 3 + 0], data[i * 3 + 1], data[i * 3 + 2]};

            const size_t num_results = std::min(4, num_points);
            std::vector<size_t> ret_indices(num_results);
            std::vector<float> out_dists_sqr(num_results);

            nanoflann::KNNResultSet<float> resultSet(num_results);
            resultSet.init(&ret_indices[0], &out_dists_sqr[0]);
            index.findNeighbors(resultSet, &query_pt[0], nanoflann::SearchParameters(10));

            float sum_dist = 0.0f;
            int valid_neighbors = 0;

            for (size_t j = 0; j < num_results && valid_neighbors < 3; j++) {
                if (out_dists_sqr[j] > 1e-8f) {
                    sum_dist += std::sqrt(out_dists_sqr[j]);
                    valid_neighbors++;
                }
            }

            result_data[i] = (valid_neighbors > 0) ? (sum_dist / valid_neighbors) : 0.01f;
        }

        return result.to(points.device());
    }

    void write_ply_impl(const gs::PointCloud& pc,
                        const std::filesystem::path& root,
                        int iteration, const std::string& stem) {
        namespace fs = std::filesystem;
        fs::create_directories(root);

        std::vector<torch::Tensor> tensors;
        tensors.push_back(pc.means);

        if (pc.normals.defined())
            tensors.push_back(pc.normals);
        if (pc.sh0.defined())
            tensors.push_back(pc.sh0);
        if (pc.shN.defined())
            tensors.push_back(pc.shN);
        if (pc.opacity.defined())
            tensors.push_back(pc.opacity);
        if (pc.scaling.defined())
            tensors.push_back(pc.scaling);
        if (pc.rotation.defined())
            tensors.push_back(pc.rotation);

        auto write_output_ply =
            [](const fs::path& file_path,
               const std::vector<torch::Tensor>& data,
               const std::vector<std::string>& attr_names) {
                tinyply::PlyFile ply;
                size_t attr_off = 0;

                for (const auto& tensor : data) {
                    const size_t cols = tensor.size(1);
                    std::vector<std::string> attrs(attr_names.begin() + attr_off,
                                                   attr_names.begin() + attr_off + cols);

                    ply.add_properties_to_element(
                        "vertex",
                        attrs,
                        tinyply::Type::FLOAT32,
                        tensor.size(0),
                        reinterpret_cast<uint8_t*>(tensor.data_ptr<float>()),
                        tinyply::Type::INVALID, 0);

                    attr_off += cols;
                }

                std::filebuf fb;
                fb.open(file_path, std::ios::out | std::ios::binary);
                std::ostream out_stream(&fb);
                ply.write(out_stream, /*binary=*/true);
            };

        if (stem.empty()) {
            write_output_ply(root / ("splat_" + std::to_string(iteration) + ".ply"), tensors, pc.attribute_names);
        } else {
            write_output_ply(root / std::string(stem + ".ply"), tensors, pc.attribute_names);
        }
    }

    // returns the output path
    std::filesystem::path write_sog_impl(const gs::SplatData& splat_data,
                                         const std::filesystem::path& root,
                                         int iteration,
                                         int kmeans_iterations) {
        namespace fs = std::filesystem;

        // Create SOG subdirectory
        fs::path sog_dir = root / "sog";
        fs::create_directories(sog_dir);

        // Set up SOG write options - use .sog extension to create bundle
        std::filesystem::path sog_out_path = sog_dir / ("splat_" + std::to_string(iteration) + "_sog.sog");
        gs::core::SogWriteOptions options{
            .iterations = kmeans_iterations,
            .output_path = sog_out_path};

        // Write SOG format
        auto result = gs::core::write_sog(splat_data, options);
        if (!result) {
            LOG_ERROR("Failed to write SOG format: {}", result.error());
        } else {
            LOG_DEBUG("Successfully wrote SOG format for iteration {}", iteration);
        }

        return sog_out_path;
    }
} // namespace

namespace gs {
    // Constructor from tensors
    SplatData::SplatData(int sh_degree,
                         torch::Tensor means,
                         torch::Tensor sh0,
                         torch::Tensor shN,
                         torch::Tensor scaling,
                         torch::Tensor rotation,
                         torch::Tensor opacity,
                         float scene_scale)
        : _max_sh_degree{sh_degree},
          _active_sh_degree{0},
          _scene_scale{scene_scale},
          _means{std::move(means)},
          _sh0{std::move(sh0)},
          _shN{std::move(shN)},
          _scaling{std::move(scaling)},
          _rotation{std::move(rotation)},
          _opacity{std::move(opacity)} {}

    // Move constructor
    SplatData::SplatData(SplatData&& other) noexcept
        : _active_sh_degree(other._active_sh_degree),
          _max_sh_degree(other._max_sh_degree),
          _scene_scale(other._scene_scale),
          _means(std::move(other._means)),
          _sh0(std::move(other._sh0)),
          _shN(std::move(other._shN)),
          _scaling(std::move(other._scaling)),
          _rotation(std::move(other._rotation)),
          _opacity(std::move(other._opacity)),
          _densification_info(std::move(other._densification_info))
    // Note: _save_mutex and _save_futures are default constructed
    {
        // Don't move the mutex or futures - each instance should have its own
    }

    // Move assignment operator
    SplatData& SplatData::operator=(SplatData&& other) noexcept {
        if (this != &other) {
            // Wait for any pending saves to complete
            wait_for_saves();

            // Move scalar members
            _active_sh_degree = other._active_sh_degree;
            _max_sh_degree = other._max_sh_degree;
            _scene_scale = other._scene_scale;

            // Move tensors
            _means = std::move(other._means);
            _sh0 = std::move(other._sh0);
            _shN = std::move(other._shN);
            _scaling = std::move(other._scaling);
            _rotation = std::move(other._rotation);
            _opacity = std::move(other._opacity);
            _densification_info = std::move(other._densification_info);

            // Don't move the mutex or futures
        }
        return *this;
    }

    SplatData::~SplatData() {
        wait_for_saves();
    }

    // Computed getters
    torch::Tensor SplatData::get_means() const {
        return _means;
    }

    torch::Tensor SplatData::get_opacity() const {
        return torch::sigmoid(_opacity).squeeze(-1);
    }

    torch::Tensor SplatData::get_rotation() const {
        return torch::nn::functional::normalize(_rotation,
                                                torch::nn::functional::NormalizeFuncOptions().dim(-1));
    }

    torch::Tensor SplatData::get_scaling() const {
        return torch::exp(_scaling);
    }

    torch::Tensor SplatData::get_shs() const {
        return torch::cat({_sh0, _shN}, 1);
    }

    SplatData& SplatData::transform(const glm::mat4& transform_matrix) {
        LOG_TIMER("SplatData::transform");

        if (_means.size(0) == 0) {
            return *this; // Nothing to transform
        }

        const int num_points = _means.size(0);

        // Keep everything on GPU for efficiency
        auto device = _means.device();

        // 1. Transform positions (means)
        // Convert transform matrix to tensor
        auto transform_tensor = torch::tensor({transform_matrix[0][0], transform_matrix[0][1], transform_matrix[0][2], transform_matrix[0][3],
                                               transform_matrix[1][0], transform_matrix[1][1], transform_matrix[1][2], transform_matrix[1][3],
                                               transform_matrix[2][0], transform_matrix[2][1], transform_matrix[2][2], transform_matrix[2][3],
                                               transform_matrix[3][0], transform_matrix[3][1], transform_matrix[3][2], transform_matrix[3][3]},
                                              torch::TensorOptions().dtype(torch::kFloat32).device(device))
                                    .reshape({4, 4});

        // Add homogeneous coordinate
        auto means_homo = torch::cat({_means, torch::ones({num_points, 1}, _means.options())}, 1);

        // Apply transform: (4x4) @ (Nx4)^T = (4xN), then transpose back
        auto transformed_means = torch::matmul(transform_tensor, means_homo.t()).t();

        // Extract xyz and update in-place
        _means.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, 3)},
                          transformed_means.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)}));

        // 2. Extract rotation from transform matrix (simple method without decompose)
        glm::mat3 rot_mat(transform_matrix);

        // Normalize columns to remove scale
        glm::vec3 scale;
        for (int i = 0; i < 3; ++i) {
            scale[i] = glm::length(rot_mat[i]);
            if (scale[i] > 0.0f) {
                rot_mat[i] /= scale[i];
            }
        }

        // Convert rotation matrix to quaternion
        glm::quat rotation = glm::quat_cast(rot_mat);

        // 3. Transform rotations (quaternions) if there's rotation
        if (std::abs(rotation.w - 1.0f) > 1e-6f) {
            auto rot_tensor = torch::tensor({rotation.w, rotation.x, rotation.y, rotation.z},
                                            torch::TensorOptions().dtype(torch::kFloat32).device(device));

            // Quaternion multiplication: q_new = q_transform * q_original
            auto q = _rotation; // Shape: [N, 4] in [w, x, y, z] format

            // Expand rotation quaternion to match batch size
            auto q_rot = rot_tensor.unsqueeze(0).expand({num_points, 4});

            // Quaternion multiplication formula
            auto w1 = q_rot.index({torch::indexing::Slice(), 0});
            auto x1 = q_rot.index({torch::indexing::Slice(), 1});
            auto y1 = q_rot.index({torch::indexing::Slice(), 2});
            auto z1 = q_rot.index({torch::indexing::Slice(), 3});

            auto w2 = q.index({torch::indexing::Slice(), 0});
            auto x2 = q.index({torch::indexing::Slice(), 1});
            auto y2 = q.index({torch::indexing::Slice(), 2});
            auto z2 = q.index({torch::indexing::Slice(), 3});

            _rotation.index_put_({torch::indexing::Slice(), 0}, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2);
            _rotation.index_put_({torch::indexing::Slice(), 1}, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2);
            _rotation.index_put_({torch::indexing::Slice(), 2}, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2);
            _rotation.index_put_({torch::indexing::Slice(), 3}, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2);
        }

        // 4. Transform scaling (if non-uniform scale is present)
        if (std::abs(scale.x - 1.0f) > 1e-6f ||
            std::abs(scale.y - 1.0f) > 1e-6f ||
            std::abs(scale.z - 1.0f) > 1e-6f) {

            // Average scale factor (for isotropic gaussian scaling)
            float avg_scale = (scale.x + scale.y + scale.z) / 3.0f;

            // Since _scaling is log(scale), we add log of the scale factor
            _scaling = _scaling + std::log(avg_scale);
        }

        // 5. Update scene scale if significant change
        torch::Tensor scene_center = _means.mean(0);
        torch::Tensor dists = torch::norm(_means - scene_center, 2, 1);
        float new_scene_scale = dists.median().item<float>();
        if (std::abs(new_scene_scale - _scene_scale) > _scene_scale * 0.1f) {
            _scene_scale = new_scene_scale;
        }

        LOG_DEBUG("Transformed {} gaussians", num_points);
        return *this;
    }

    // Utility method
    void SplatData::increment_sh_degree() {
        if (_active_sh_degree < _max_sh_degree) {
            _active_sh_degree++;
        }
    }

    void SplatData::set_active_sh_degree(int sh_degree = 0) {
        if (sh_degree <= _max_sh_degree) {
            _active_sh_degree = sh_degree;
        } else {
            _active_sh_degree = _max_sh_degree;
        }
    }

    // Get attribute names for PLY format
    std::vector<std::string> SplatData::get_attribute_names() const {
        std::vector<std::string> a{"x", "y", "z", "nx", "ny", "nz"};

        for (int i = 0; i < _sh0.size(1) * _sh0.size(2); ++i)
            a.emplace_back("f_dc_" + std::to_string(i));
        for (int i = 0; i < _shN.size(1) * _shN.size(2); ++i)
            a.emplace_back("f_rest_" + std::to_string(i));

        a.emplace_back("opacity");

        for (int i = 0; i < _scaling.size(1); ++i)
            a.emplace_back("scale_" + std::to_string(i));
        for (int i = 0; i < _rotation.size(1); ++i)
            a.emplace_back("rot_" + std::to_string(i));

        return a;
    }

    void SplatData::wait_for_saves() const {
        std::lock_guard<std::mutex> lock(_save_mutex);

        // Wait for all pending saves
        for (auto& future : _save_futures) {
            if (future.valid()) {
                try {
                    future.wait();
                } catch (const std::exception& e) {
                    LOG_ERROR("Error waiting for save to complete: {}", e.what());
                }
            }
        }
        _save_futures.clear();
    }

    void SplatData::cleanup_finished_saves() const {
        std::lock_guard<std::mutex> lock(_save_mutex);

        // Remove completed futures
        _save_futures.erase(
            std::remove_if(_save_futures.begin(), _save_futures.end(),
                           [](const std::future<void>& f) {
                               return !f.valid() ||
                                      f.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
                           }),
            _save_futures.end());

        // Log if we have many pending saves (might indicate a problem)
        if (_save_futures.size() > 5) {
            LOG_WARN("Multiple saves pending: {} operations in queue", _save_futures.size());
        }
    }

    // Export to PLY
    void SplatData::save_ply(const std::filesystem::path& root, int iteration, bool join_threads, std::string stem) const {
        auto pc = to_point_cloud();

        if (join_threads) {
            // Synchronous save - wait for completion
            write_ply_impl(pc, root, iteration, stem);
        } else {
            // Asynchronous save
            cleanup_finished_saves();

            std::lock_guard<std::mutex> lock(_save_mutex);
            _save_futures.emplace_back(
                std::async(std::launch::async, [pc = std::move(pc), root, iteration, stem]() {
                    try {
                        write_ply_impl(pc, root, iteration, stem);
                    } catch (const std::exception& e) {
                        LOG_ERROR("Failed to save PLY for iteration {}: {}", iteration, e.what());
                    }
                }));
        }
    }

    // Export to SOG
    std::filesystem::path SplatData::save_sog(const std::filesystem::path& root, int iteration, int kmeans_iterations, bool join_threads) const {
        // SOG must always be synchronous - k-means clustering is too heavy for async
        // and the shared data access patterns don't work well with async execution
        return write_sog_impl(*this, root, iteration, kmeans_iterations);
    }

    PointCloud SplatData::to_point_cloud() const {
        PointCloud pc;

        // Basic attributes
        pc.means = _means.cpu().contiguous();
        pc.normals = torch::zeros_like(pc.means);

        // Gaussian attributes
        pc.sh0 = _sh0.transpose(1, 2).flatten(1).cpu();
        pc.shN = _shN.transpose(1, 2).flatten(1).cpu();
        pc.opacity = _opacity.cpu();
        pc.scaling = _scaling.cpu();

        pc.rotation = torch::nn::functional::normalize(_rotation,
                                                       torch::nn::functional::NormalizeFuncOptions().dim(-1))
                          .cpu()
                          .contiguous();

        // Set attribute names for PLY export
        pc.attribute_names = get_attribute_names();

        return pc;
    }

    std::expected<SplatData, std::string> SplatData::init_model_from_pointcloud(
        const param::TrainingParameters& params,
        torch::Tensor scene_center,
        const PointCloud& pcd) {

        try {
            // Generate positions and colors based on init type
            torch::Tensor positions, colors;
            if (params.optimization.random) {
                const int num_points = params.optimization.init_num_pts;
                const float extent = params.optimization.init_extent;
                const auto f32_cuda = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

                positions = (torch::rand({num_points, 3}, f32_cuda) * 2.0f - 1.0f) * extent;
                colors = torch::rand({num_points, 3}, f32_cuda);
            } else {
                positions = pcd.means;
                colors = pcd.colors / 255.0f; // Normalize directly
            }

            scene_center = scene_center.to(positions.device());
            const torch::Tensor dists = torch::norm(positions - scene_center, 2, 1);
            const auto scene_scale = dists.median().item<float>();

            auto rgb_to_sh = [](const torch::Tensor& rgb) {
                constexpr float kInvSH = 0.28209479177387814f;
                return (rgb - 0.5f) / kInvSH;
            };

            const auto f32 = torch::TensorOptions().dtype(torch::kFloat32);
            const auto f32_cuda = f32.device(torch::kCUDA);

            // 1. means
            torch::Tensor means;
            if (params.optimization.random) {
                // Scale positions before setting requires_grad
                means = (positions * scene_scale).to(torch::kCUDA).set_requires_grad(true);
            } else {
                means = positions.to(torch::kCUDA).set_requires_grad(true);
            }

            // 2. scaling (log(σ))
            auto nn_dist = torch::clamp_min(compute_mean_neighbor_distances(means), 1e-7);
            auto scaling = torch::log(torch::sqrt(nn_dist) * params.optimization.init_scaling)
                               .unsqueeze(-1)
                               .repeat({1, 3})
                               .to(f32_cuda)
                               .set_requires_grad(true);

            // 3. rotation (quaternion, identity) - split into multiple lines to avoid compilation error
            auto rotation = torch::zeros({means.size(0), 4}, f32_cuda);
            rotation.index_put_({torch::indexing::Slice(), 0}, 1);
            rotation = rotation.set_requires_grad(true);

            // 4. opacity (inverse sigmoid of 0.5)
            auto opacity = torch::logit(params.optimization.init_opacity * torch::ones({means.size(0), 1}, f32_cuda))
                               .set_requires_grad(true);

            // 5. shs (SH coefficients)
            auto colors_float = colors.to(torch::kCUDA);
            auto fused_color = rgb_to_sh(colors_float);

            const int64_t feature_shape = static_cast<int64_t>(std::pow(params.optimization.sh_degree + 1, 2));
            auto shs = torch::zeros({fused_color.size(0), 3, feature_shape}, f32_cuda);

            // Set DC coefficients
            shs.index_put_({torch::indexing::Slice(),
                            torch::indexing::Slice(),
                            0},
                           fused_color);

            auto sh0 = shs.index({torch::indexing::Slice(),
                                  torch::indexing::Slice(),
                                  torch::indexing::Slice(0, 1)})
                           .transpose(1, 2)
                           .contiguous()
                           .set_requires_grad(true);

            auto shN = shs.index({torch::indexing::Slice(),
                                  torch::indexing::Slice(),
                                  torch::indexing::Slice(1, torch::indexing::None)})
                           .transpose(1, 2)
                           .contiguous()
                           .set_requires_grad(true);

            std::println("Scene scale: {}", scene_scale);
            std::println("Initialized SplatData with:");
            std::println("  - {} points", means.size(0));
            std::println("  - Max SH degree: {}", params.optimization.sh_degree);
            std::println("  - Total SH coefficients: {}", feature_shape);
            std::cout << std::format("  - sh0 shape: {}\n", tensor_sizes_to_string(sh0.sizes()));
            std::cout << std::format("  - shN shape: {}\n", tensor_sizes_to_string(shN.sizes()));

            return SplatData(
                params.optimization.sh_degree,
                means.contiguous(),
                sh0.contiguous(),
                shN.contiguous(),
                scaling.contiguous(),
                rotation.contiguous(),
                opacity.contiguous(),
                scene_scale);

        } catch (const std::exception& e) {
            return std::unexpected(std::format("Failed to initialize SplatData: {}", e.what()));
        }
    }

    void SplatData::filterByMask(const torch::Tensor& keep_mask) {
        TORCH_CHECK(keep_mask.dim() == 1 && keep_mask.size(0) == _means.size(0),
                    "keep_mask must be 1D and match the number of splats");
        auto km = keep_mask.to(torch::kBool);
        _means = _means.index({km}).contiguous();
        _sh0 = _sh0.index({km}).contiguous();
        _shN = _shN.index({km}).contiguous();
        _scaling = _scaling.index({km}).contiguous();
        _rotation = _rotation.index({km}).contiguous();
        _opacity = _opacity.index({km}).contiguous();
        //_max_radii2D is not required;
    }
    
    SplatData SplatData::crop_by_cropbox(const gs::geometry::BoundingBox& bounding_box) const {
        LOG_TIMER("SplatData::crop_by_cropbox");

        if (_means.size(0) == 0) {
            LOG_WARN("Cannot crop empty SplatData");
            return SplatData(); // Return empty SplatData
        }

        // Get bounding box properties
        const auto bbox_min = bounding_box.getMinBounds();
        const auto bbox_max = bounding_box.getMaxBounds();
        const auto& world2bbox_transform = bounding_box.getworld2BBox();

        const int num_points = _means.size(0);

        LOG_DEBUG("Cropping {} points with bounding box: min({}, {}, {}), max({}, {}, {})",
                  num_points, bbox_min.x, bbox_min.y, bbox_min.z, bbox_max.x, bbox_max.y, bbox_max.z);

        // Get transformation matrix from the EuclideanTransform
        glm::mat4 world_to_bbox_matrix = world2bbox_transform.toMat4();

        // Convert transformation matrix to tensor and move to same device as means
        // we transpose the matrix since gl is colmn major and torch is row major
        auto transform_tensor = torch::tensor({world_to_bbox_matrix[0][0], world_to_bbox_matrix[1][0], world_to_bbox_matrix[2][0], world_to_bbox_matrix[3][0],
                                               world_to_bbox_matrix[0][1], world_to_bbox_matrix[1][1], world_to_bbox_matrix[2][1], world_to_bbox_matrix[3][1],
                                               world_to_bbox_matrix[0][2], world_to_bbox_matrix[1][2], world_to_bbox_matrix[2][2], world_to_bbox_matrix[3][2],
                                               world_to_bbox_matrix[0][3], world_to_bbox_matrix[1][3], world_to_bbox_matrix[2][3], world_to_bbox_matrix[3][3]},
                                              torch::TensorOptions().dtype(torch::kFloat32))
                                    .reshape({4, 4})
                                    .to(_means.device());

        // Convert means to homogeneous coordinates [N, 4]
        auto means_homo = torch::cat({_means, torch::ones({num_points, 1}, _means.options())}, 1);

        // Transform all points: (4x4) @ (Nx4)^T = (4xN), then transpose back to (Nx4)
        auto transformed_points = torch::matmul(transform_tensor, means_homo.t()).t();

        // Extract xyz coordinates (drop homogeneous coordinate)
        auto local_points = transformed_points.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3)});

        // Create bounding box bounds tensors
        auto bbox_min_tensor = torch::tensor({bbox_min.x, bbox_min.y, bbox_min.z},
                                             torch::TensorOptions().dtype(torch::kFloat32).device(_means.device()));
        auto bbox_max_tensor = torch::tensor({bbox_max.x, bbox_max.y, bbox_max.z},
                                             torch::TensorOptions().dtype(torch::kFloat32).device(_means.device()));

        // Check which points are inside the bounding box using tensor operations
        auto inside_min = torch::ge(local_points, bbox_min_tensor.unsqueeze(0)); // [N, 3]
        auto inside_max = torch::le(local_points, bbox_max_tensor.unsqueeze(0)); // [N, 3]

        // Point is inside if all 3 coordinates satisfy both min and max constraints
        auto inside_mask = torch::all(inside_min & inside_max, 1); // [N]

        // Count points inside
        int points_inside = inside_mask.sum().item<int>();

        LOG_DEBUG("Found {} points inside bounding box ({:.1f}%)",
                  points_inside, (float)points_inside / num_points * 100.0f);

        if (points_inside == 0) {
            LOG_WARN("No points found inside bounding box, returning empty SplatData");
            return SplatData();
        }

        // Get indices of points inside the bounding box
        auto indices = torch::nonzero(inside_mask).squeeze(1); // Get 1D tensor of indices

        // Index all tensors using the mask
        auto cropped_means = _means.index({indices}).contiguous();
        auto cropped_sh0 = _sh0.index({indices}).contiguous();
        auto cropped_shN = _shN.index({indices}).contiguous();
        auto cropped_scaling = _scaling.index({indices}).contiguous();
        auto cropped_rotation = _rotation.index({indices}).contiguous();
        auto cropped_opacity = _opacity.index({indices}).contiguous();

        // Recalculate scene scale for the cropped data
        torch::Tensor scene_center = cropped_means.mean(0);
        torch::Tensor dists = torch::norm(cropped_means - scene_center, 2, 1);
        float new_scene_scale = points_inside > 1 ? dists.median().item<float>() : _scene_scale;

        // Create new SplatData with cropped tensors
        SplatData cropped_splat(
            _max_sh_degree,
            std::move(cropped_means),
            std::move(cropped_sh0),
            std::move(cropped_shN),
            std::move(cropped_scaling),
            std::move(cropped_rotation),
            std::move(cropped_opacity),
            new_scene_scale);

        // Copy over the active SH degree
        cropped_splat._active_sh_degree = _active_sh_degree;

        // If densification info exists and has the right size, crop it too
        if (_densification_info.defined() && _densification_info.size(0) == num_points) {
            cropped_splat._densification_info = _densification_info.index({indices}).contiguous();
        }

        LOG_DEBUG("Successfully cropped SplatData: {} -> {} points (scale: {:.4f} -> {:.4f})",
                  num_points, points_inside, _scene_scale, new_scene_scale);

        return cropped_splat;
    }

} // namespace gs
