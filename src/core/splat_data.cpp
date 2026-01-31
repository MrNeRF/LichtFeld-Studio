/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_data.hpp"
#include "core/logger.hpp"
#include "core/parameters.hpp"
#include "core/point_cloud.hpp"
#include "core/tensor/internal/tensor_serialization.hpp"
#include "nanoflann.hpp"

#include <cassert>
#include <cmath>
#include <expected>
#include <format>
#include <vector>

namespace {

    // Point cloud adaptor for nanoflann
    struct PointCloudAdaptor {
        const float* points;
        size_t num_points;

        PointCloudAdaptor(const float* pts, size_t n)
            : points(pts),
              num_points(n) {}

        inline size_t kdtree_get_point_count() const { return num_points; }

        inline float kdtree_get_pt(const size_t idx, const size_t dim) const {
            return points[idx * 3 + dim];
        }

        template <class BBOX>
        bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
    };

    using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, PointCloudAdaptor>,
        PointCloudAdaptor,
        3>;

    /**
     * @brief Compute mean distance to 3 nearest neighbors for each point
     */
    lfs::core::Tensor compute_mean_neighbor_distances(const lfs::core::Tensor& points) {
        auto cpu_points = points.cpu();
        const int num_points = cpu_points.size(0);

        if (cpu_points.ndim() != 2 || cpu_points.size(1) != 3) {
            LOG_ERROR("Input points must have shape [N, 3], got {}", cpu_points.shape().str());
            return lfs::core::Tensor();
        }

        if (cpu_points.dtype() != lfs::core::DataType::Float32) {
            LOG_ERROR("Input points must be float32");
            return lfs::core::Tensor();
        }

        if (num_points <= 1) {
            return lfs::core::Tensor::full({static_cast<size_t>(num_points)}, 0.01f, points.device());
        }

        const float* data = cpu_points.ptr<float>();

        PointCloudAdaptor cloud(data, num_points);
        KDTree index(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        index.buildIndex();

        auto result = lfs::core::Tensor::zeros({static_cast<size_t>(num_points)}, lfs::core::Device::CPU);
        float* result_data = result.ptr<float>();

#pragma omp parallel for if (num_points > 1000)
        for (int i = 0; i < num_points; i++) {
            const float query_pt[3] = {
                data[i * 3 + 0],
                data[i * 3 + 1],
                data[i * 3 + 2]};

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

} // anonymous namespace

namespace lfs::core {

    // ========== CONSTRUCTOR & DESTRUCTOR ==========

    SplatData::SplatData(int sh_degree,
                         Tensor means_,
                         Tensor sh0_,
                         Tensor shN_,
                         Tensor scaling_,
                         Tensor rotation_,
                         Tensor opacity_,
                         float scene_scale_)
        : _max_sh_degree(sh_degree),
          _active_sh_degree(sh_degree), // Set to max degree when loading; training will override this
          _scene_scale(scene_scale_),
          _means(std::move(means_)),
          _sh0(std::move(sh0_)),
          _shN(std::move(shN_)),
          _scaling(std::move(scaling_)),
          _rotation(std::move(rotation_)),
          _opacity(std::move(opacity_)) {
    }

    SplatData::~SplatData() = default;

    // ========== MOVE SEMANTICS ==========

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
          _densification_info(std::move(other._densification_info)),
          _deleted(std::move(other._deleted)) {
        // Reset the moved-from object
        other._active_sh_degree = 0;
        other._max_sh_degree = 0;
        other._scene_scale = 0.0f;
    }

    SplatData& SplatData::operator=(SplatData&& other) noexcept {
        if (this != &other) {
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
            _deleted = std::move(other._deleted);
        }
        return *this;
    }

    // ========== COMPUTED GETTERS ==========

    Tensor SplatData::get_means() const {
        return _means;
    }

    Tensor SplatData::get_opacity() const {
        return _opacity.sigmoid().squeeze(-1);
    }

    Tensor SplatData::get_rotation() const {
        // Normalize quaternions along the last dimension
        // _rotation is [N, 4], we want to normalize each quaternion
        // norm = sqrt(sum(x^2)) along dim=1, keepdim=true to get [N, 1]

        auto squared = _rotation.square();
        auto sum_squared = squared.sum({1}, true);    // [N, 1]
        auto norm = sum_squared.sqrt();               // [N, 1]
        return _rotation.div(norm.clamp_min(1e-12f)); // Avoid division by zero
    }

    Tensor SplatData::get_scaling() const {
        return _scaling.exp();
    }

    Tensor SplatData::get_shs() const {
        // _sh0 is [N, 1, 3], _shN is [N, coeffs, 3]
        // Concatenate along dim 1 (coeffs) to get [N, total_coeffs, 3]
        if (!_shN.is_valid()) {
            return _sh0; // SH degree 0: only DC component
        }
        return _sh0.cat(_shN, 1);
    }

    // ========== UTILITY METHODS ==========

    void SplatData::increment_sh_degree() {
        if (_active_sh_degree < _max_sh_degree) {
            _active_sh_degree++;
        }
    }

    void SplatData::set_active_sh_degree(int sh_degree) {
        if (sh_degree <= _max_sh_degree) {
            _active_sh_degree = sh_degree;
        } else {
            _active_sh_degree = _max_sh_degree;
        }
    }

    void SplatData::reserve_capacity(const size_t capacity) {
        if (_means.is_valid())
            _means.reserve(capacity);
        if (_sh0.is_valid())
            _sh0.reserve(capacity);
        if (_shN.is_valid())
            _shN.reserve(capacity);
        if (_scaling.is_valid())
            _scaling.reserve(capacity);
        if (_rotation.is_valid())
            _rotation.reserve(capacity);
        if (_opacity.is_valid())
            _opacity.reserve(capacity);
    }

    // ========== SOFT DELETION ==========

    unsigned long SplatData::visible_count() const {
        if (!_deleted.is_valid()) {
            return size();
        }
        return size() - static_cast<unsigned long>(_deleted.sum_scalar());
    }

    Tensor SplatData::soft_delete(const Tensor& mask) {
        if (!_means.is_valid() || _means.size(0) == 0) {
            LOG_WARN("soft_delete: invalid or empty SplatData");
            return Tensor();
        }

        const size_t n = size();
        if (mask.size(0) != n) {
            LOG_ERROR("soft_delete: mask size {} != SplatData size {}", mask.size(0), n);
            return Tensor();
        }

        if (!_deleted.is_valid()) {
            _deleted = Tensor::zeros({n}, _means.device(), DataType::Bool);
        }

        Tensor old_deleted = _deleted.clone();
        _deleted = _deleted || mask;
        return old_deleted;
    }

    void SplatData::undelete(const Tensor& mask) {
        if (!_deleted.is_valid()) {
            return;
        }

        const size_t n = size();
        if (mask.size(0) != n) {
            LOG_ERROR("undelete: mask size {} != SplatData size {}", mask.size(0), n);
            return;
        }

        _deleted = _deleted && mask.logical_not();
    }

    void SplatData::clear_deleted() {
        if (_deleted.is_valid()) {
            _deleted.zero_();
        }
    }

    size_t SplatData::apply_deleted() {
        if (!_deleted.is_valid() || !_means.is_valid()) {
            return 0;
        }

        const size_t old_size = size();

        // Validate mask dimensions match data
        if (_deleted.size(0) != old_size) {
            LOG_ERROR("apply_deleted: mask size {} != data size {}, aborting",
                      _deleted.size(0), old_size);
            _deleted = Tensor();
            return 0;
        }

        // Validate mask is boolean type
        if (_deleted.dtype() != DataType::Bool) {
            LOG_ERROR("apply_deleted: mask is not Bool type, aborting");
            _deleted = Tensor();
            return 0;
        }

        // Validate all required tensors have matching sizes
        if (_sh0.size(0) != old_size || _scaling.size(0) != old_size ||
            _rotation.size(0) != old_size || _opacity.size(0) != old_size) {
            LOG_ERROR("apply_deleted: tensor size mismatch, aborting");
            _deleted = Tensor();
            return 0;
        }

        const auto keep_mask = _deleted.logical_not();
        const size_t new_size = static_cast<size_t>(keep_mask.sum_scalar());

        // Nothing to delete
        if (new_size == old_size) {
            _deleted = Tensor();
            return 0;
        }

        // Would delete everything
        if (new_size == 0) {
            LOG_WARN("apply_deleted: would remove all gaussians, aborting");
            return 0;
        }

        LOG_DEBUG("apply_deleted: filtering {} -> {} gaussians", old_size, new_size);

        // Filter all tensors by keep mask
        auto new_means = _means.index_select(0, keep_mask);
        auto new_sh0 = _sh0.index_select(0, keep_mask);
        auto new_scaling = _scaling.index_select(0, keep_mask);
        auto new_rotation = _rotation.index_select(0, keep_mask);
        auto new_opacity = _opacity.index_select(0, keep_mask);

        // Verify new sizes are correct before committing
        if (new_means.size(0) != new_size || new_sh0.size(0) != new_size ||
            new_scaling.size(0) != new_size || new_rotation.size(0) != new_size ||
            new_opacity.size(0) != new_size) {
            LOG_ERROR("apply_deleted: post-filter size mismatch - means:{} sh0:{} scaling:{} rotation:{} opacity:{} expected:{}",
                      new_means.size(0), new_sh0.size(0), new_scaling.size(0),
                      new_rotation.size(0), new_opacity.size(0), new_size);
            return 0;
        }

        // Commit the changes
        _means = std::move(new_means);
        _sh0 = std::move(new_sh0);
        _scaling = std::move(new_scaling);
        _rotation = std::move(new_rotation);
        _opacity = std::move(new_opacity);

        if (_shN.is_valid() && _shN.size(0) == old_size) {
            _shN = _shN.index_select(0, keep_mask);
        }

        // Clear densification info
        _densification_info = Tensor();

        // Clear deletion mask
        _deleted = Tensor();

        const size_t removed = old_size - new_size;
        LOG_INFO("apply_deleted: removed {} gaussians ({} -> {})", removed, old_size, new_size);
        return removed;
    }

    // ========== SERIALIZATION ==========

    namespace {
        constexpr uint32_t SPLAT_DATA_MAGIC = 0x4C465350; // "LFSP"
        constexpr uint32_t SPLAT_DATA_VERSION = 3;
    } // namespace

    void SplatData::serialize(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&SPLAT_DATA_MAGIC), sizeof(SPLAT_DATA_MAGIC));
        os.write(reinterpret_cast<const char*>(&SPLAT_DATA_VERSION), sizeof(SPLAT_DATA_VERSION));
        os.write(reinterpret_cast<const char*>(&_active_sh_degree), sizeof(_active_sh_degree));
        os.write(reinterpret_cast<const char*>(&_max_sh_degree), sizeof(_max_sh_degree));
        os.write(reinterpret_cast<const char*>(&_scene_scale), sizeof(_scene_scale));

        os << _means << _sh0 << _scaling << _rotation << _opacity;

        if (_max_sh_degree > 0) {
            if (!_shN.is_valid()) {
                throw std::runtime_error("shN tensor must be valid when max_sh_degree > 0");
            }
            os << _shN;
        }

        const uint8_t has_deleted = _deleted.is_valid() ? 1 : 0;
        os.write(reinterpret_cast<const char*>(&has_deleted), sizeof(has_deleted));
        if (has_deleted)
            os << _deleted;

        const uint8_t has_densification = _densification_info.is_valid() ? 1 : 0;
        os.write(reinterpret_cast<const char*>(&has_densification), sizeof(has_densification));
        if (has_densification)
            os << _densification_info;

        LOG_DEBUG("Serialized SplatData: {} Gaussians, SH {}/{}", size(), _active_sh_degree, _max_sh_degree);
    }

    void SplatData::deserialize(std::istream& is) {
        uint32_t magic = 0, version = 0;
        is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        is.read(reinterpret_cast<char*>(&version), sizeof(version));

        if (magic != SPLAT_DATA_MAGIC) {
            throw std::runtime_error("Invalid SplatData: wrong magic");
        }
        if (version != SPLAT_DATA_VERSION) {
            throw std::runtime_error("Unsupported SplatData version: " + std::to_string(version));
        }

        int32_t active_sh = 0, max_sh = 0;
        float scene_scale = 0.0f;
        is.read(reinterpret_cast<char*>(&active_sh), sizeof(active_sh));
        is.read(reinterpret_cast<char*>(&max_sh), sizeof(max_sh));
        is.read(reinterpret_cast<char*>(&scene_scale), sizeof(scene_scale));

        Tensor means, sh0, scaling, rotation, opacity;
        is >> means >> sh0 >> scaling >> rotation >> opacity;

        _means = std::move(means).cuda();
        _sh0 = std::move(sh0).cuda();
        _scaling = std::move(scaling).cuda();
        _rotation = std::move(rotation).cuda();
        _opacity = std::move(opacity).cuda();
        _active_sh_degree = active_sh;
        _max_sh_degree = max_sh;
        _scene_scale = scene_scale;

        if (max_sh > 0) {
            Tensor shN;
            is >> shN;
            _shN = std::move(shN).cuda();
        }

        uint8_t has_deleted = 0;
        is.read(reinterpret_cast<char*>(&has_deleted), sizeof(has_deleted));
        if (has_deleted) {
            Tensor deleted;
            is >> deleted;
            _deleted = std::move(deleted).cuda();
        }

        uint8_t has_densification = 0;
        is.read(reinterpret_cast<char*>(&has_densification), sizeof(has_densification));
        if (has_densification) {
            Tensor densification;
            is >> densification;
            _densification_info = std::move(densification).cuda();
        }

        LOG_DEBUG("Deserialized SplatData: {} Gaussians, SH {}/{}", size(), active_sh, max_sh);
    }

    // ========== FREE FUNCTION: FACTORY ==========

    std::expected<SplatData, std::string> init_model_from_pointcloud(
        const param::TrainingParameters& params,
        Tensor scene_center,
        const PointCloud& pcd,
        float scene_scale,
        int capacity) {

        assert(scene_scale > 0.f);

        try {
            LOG_DEBUG("=== init_model_from_pointcloud starting ===");
            LOG_DEBUG("  capacity={}, random={}, sh_degree={}",
                      capacity, params.optimization.random, params.optimization.sh_degree);
            LOG_DEBUG("  scene_center: is_valid={}, device={}, shape={}",
                      scene_center.is_valid(),
                      scene_center.device() == Device::CUDA ? "CUDA" : "CPU",
                      scene_center.shape().str());
            LOG_DEBUG("  pcd.means: is_valid={}, device={}, shape={}, numel={}",
                      pcd.means.is_valid(),
                      pcd.means.device() == Device::CUDA ? "CUDA" : "CPU",
                      pcd.means.shape().str(), pcd.means.numel());
            LOG_DEBUG("  pcd.colors: is_valid={}, device={}, shape={}, numel={}",
                      pcd.colors.is_valid(),
                      pcd.colors.device() == Device::CUDA ? "CUDA" : "CPU",
                      pcd.colors.shape().str(), pcd.colors.numel());

            // Generate positions and colors based on init type
            Tensor positions, colors;

            if (params.optimization.random) {
                const int num_points = params.optimization.init_num_pts;
                const float extent = params.optimization.init_extent;

                LOG_DEBUG("  Using random initialization: num_points={}, extent={}", num_points, extent);
                positions = (Tensor::rand({static_cast<size_t>(num_points), 3}, Device::CUDA)
                                 .mul(2.0f)
                                 .sub(1.0f))
                                .mul(extent);
                colors = Tensor::rand({static_cast<size_t>(num_points), 3}, Device::CUDA);
                LOG_DEBUG("  Random positions created: shape={}, numel={}", positions.shape().str(), positions.numel());
                LOG_DEBUG("  Random colors created: shape={}, numel={}", colors.shape().str(), colors.numel());
            } else {
                LOG_DEBUG("  Using point cloud initialization");
                if (!pcd.means.is_valid() || !pcd.colors.is_valid()) {
                    LOG_ERROR("Point cloud has invalid means or colors: means.is_valid()={}, colors.is_valid()={}",
                              pcd.means.is_valid(), pcd.colors.is_valid());
                    return std::unexpected("Point cloud has invalid means or colors");
                }

                LOG_DEBUG("  Converting pcd.means to CUDA...");
                positions = pcd.means.cuda();
                LOG_DEBUG("  positions after .cuda(): is_valid={}, device={}, ptr={}, shape={}, numel={}",
                          positions.is_valid(),
                          positions.device() == Device::CUDA ? "CUDA" : "CPU",
                          static_cast<void*>(positions.ptr<float>()),
                          positions.shape().str(), positions.numel());

                // Normalize colors from uint8 [0,255] to float32 [0,1] to match old behavior
                LOG_DEBUG("  Converting pcd.colors (dtype={}) to float32...",
                          pcd.colors.dtype() == DataType::UInt8 ? "UInt8" : "Float32");
                colors = pcd.colors.to(DataType::Float32).div(255.0f).cuda();
                LOG_DEBUG("  colors after conversion: is_valid={}, device={}, shape={}, numel={}",
                          colors.is_valid(),
                          colors.device() == Device::CUDA ? "CUDA" : "CPU",
                          colors.shape().str(), colors.numel());
            }

            const auto scene_center_device = scene_center.to(positions.device());

            // RGB to SH conversion (DC component)
            auto rgb_to_sh = [](const Tensor& rgb) {
                constexpr float kInvSH = 0.28209479177387814f;
                return rgb.sub(0.5f).div(kInvSH);
            };

            const size_t num_points = positions.size(0);
            const int64_t feature_shape = static_cast<int64_t>(
                std::pow(params.optimization.sh_degree + 1, 2));

            // Create final tensors first to avoid pool allocations
            Tensor means_, scaling_, rotation_, opacity_, sh0_, shN_;

            if (capacity > 0 && capacity < num_points) {
                LOG_DEBUG("capacity {} was lower than num_points {}.  Matching capacity to points. ", capacity, num_points);
                capacity = num_points;
            }

            if (capacity > 0) {
                LOG_DEBUG("Creating direct tensors with capacity={}", capacity);

                means_ = Tensor::zeros_direct(TensorShape({num_points, 3}), capacity);
                means_.set_name("SplatData.means");
                LOG_DEBUG("  means_ allocated: is_valid={}, ptr={}, shape={}, numel={}",
                          means_.is_valid(), static_cast<void*>(means_.ptr<float>()),
                          means_.shape().str(), means_.numel());

                scaling_ = Tensor::zeros_direct(TensorShape({num_points, 3}), capacity);
                scaling_.set_name("SplatData.scaling");
                LOG_DEBUG("  scaling_ allocated: is_valid={}, ptr={}, shape={}, numel={}",
                          scaling_.is_valid(), static_cast<void*>(scaling_.ptr<float>()),
                          scaling_.shape().str(), scaling_.numel());

                rotation_ = Tensor::zeros_direct(TensorShape({num_points, 4}), capacity);
                rotation_.set_name("SplatData.rotation");
                LOG_DEBUG("  rotation_ allocated: is_valid={}, ptr={}, shape={}, numel={}",
                          rotation_.is_valid(), static_cast<void*>(rotation_.ptr<float>()),
                          rotation_.shape().str(), rotation_.numel());

                opacity_ = Tensor::zeros_direct(TensorShape({num_points, 1}), capacity);
                opacity_.set_name("SplatData.opacity");
                LOG_DEBUG("  opacity_ allocated: is_valid={}, ptr={}, shape={}, numel={}",
                          opacity_.is_valid(), static_cast<void*>(opacity_.ptr<float>()),
                          opacity_.shape().str(), opacity_.numel());

                sh0_ = Tensor::zeros_direct(TensorShape({num_points, 1, 3}), capacity);
                sh0_.set_name("SplatData.sh0");
                LOG_DEBUG("  sh0_ allocated: is_valid={}, ptr={}, shape={}, numel={}",
                          sh0_.is_valid(), static_cast<void*>(sh0_.ptr<float>()),
                          sh0_.shape().str(), sh0_.numel());

                shN_ = Tensor::zeros_direct(TensorShape({num_points, static_cast<size_t>(feature_shape - 1), 3}), capacity);
                shN_.set_name("SplatData.shN");
                LOG_DEBUG("  shN_ allocated: is_valid={}, ptr={}, shape={}, numel={}",
                          shN_.is_valid(), static_cast<void*>(shN_.ptr<float>()),
                          shN_.shape().str(), shN_.numel());

                LOG_DEBUG("Computing and filling values...");
            }

            // Compute parameter values on CPU to avoid pool allocations
            Tensor means_cpu, scaling_cpu, rotation_cpu, opacity_cpu, sh0_cpu, shN_cpu;

            if (capacity > 0) {
                LOG_DEBUG("Computing values on CPU");
                LOG_DEBUG("  positions tensor: is_valid={}, device={}, shape={}, numel={}",
                          positions.is_valid(), positions.device() == Device::CUDA ? "CUDA" : "CPU",
                          positions.shape().str(), positions.numel());

                // Compute means on CPU
                auto positions_cpu = positions.cpu();
                LOG_DEBUG("  positions_cpu after .cpu(): is_valid={}, ptr={}, device={}, shape={}, numel={}",
                          positions_cpu.is_valid(), static_cast<const void*>(positions_cpu.ptr<float>()),
                          positions_cpu.device() == Device::CUDA ? "CUDA" : "CPU",
                          positions_cpu.shape().str(), positions_cpu.numel());

                if (params.optimization.random) {
                    means_cpu = positions_cpu.mul(scene_scale);
                } else {
                    means_cpu = positions_cpu;
                }
                LOG_DEBUG("  means_cpu computed: is_valid={}, ptr={}, device={}, shape={}, numel={}",
                          means_cpu.is_valid(), static_cast<const void*>(means_cpu.ptr<float>()),
                          means_cpu.device() == Device::CUDA ? "CUDA" : "CPU",
                          means_cpu.shape().str(), means_cpu.numel());

                // Compute scaling on CPU
                LOG_DEBUG("  Computing neighbor distances...");
                auto nn_dist = compute_mean_neighbor_distances(means_cpu).clamp_min(1e-7f);
                LOG_DEBUG("  nn_dist computed: is_valid={}, shape={}, numel={}",
                          nn_dist.is_valid(), nn_dist.shape().str(), nn_dist.numel());

                std::vector<int> scale_expand_shape = {static_cast<int>(num_points), 3};
                scaling_cpu = nn_dist.sqrt()
                                  .mul(params.optimization.init_scaling)
                                  .log()
                                  .unsqueeze(-1)
                                  .expand(std::span<const int>(scale_expand_shape));
                LOG_DEBUG("  scaling_cpu computed: is_valid={}, ptr={}, device={}, shape={}, numel={}",
                          scaling_cpu.is_valid(), static_cast<const void*>(scaling_cpu.ptr<float>()),
                          scaling_cpu.device() == Device::CUDA ? "CUDA" : "CPU",
                          scaling_cpu.shape().str(), scaling_cpu.numel());

                // Create identity quaternion rotations on CPU
                LOG_DEBUG("  Creating identity quaternions...");
                rotation_cpu = Tensor::zeros({num_points, 4}, Device::CPU);
                auto rot_acc = rotation_cpu.accessor<float, 2>();
                for (size_t i = 0; i < num_points; i++) {
                    rot_acc(i, 0) = 1.0f;
                }
                LOG_DEBUG("  rotation_cpu created: is_valid={}, ptr={}, shape={}, numel={}",
                          rotation_cpu.is_valid(), static_cast<const void*>(rotation_cpu.ptr<float>()),
                          rotation_cpu.shape().str(), rotation_cpu.numel());

                // Compute opacity on CPU
                LOG_DEBUG("  Computing opacity (init_val={})...", params.optimization.init_opacity);
                auto init_val = params.optimization.init_opacity;
                opacity_cpu = Tensor::full({num_points, 1}, init_val, Device::CPU).logit();
                LOG_DEBUG("  opacity_cpu computed: is_valid={}, ptr={}, shape={}, numel={}",
                          opacity_cpu.is_valid(), static_cast<const void*>(opacity_cpu.ptr<float>()),
                          opacity_cpu.shape().str(), opacity_cpu.numel());

                // Compute SH coefficients on CPU
                LOG_DEBUG("  Computing SH coefficients...");
                LOG_DEBUG("    colors tensor: is_valid={}, device={}, shape={}, numel={}",
                          colors.is_valid(), colors.device() == Device::CUDA ? "CUDA" : "CPU",
                          colors.shape().str(), colors.numel());

                auto colors_cpu = colors.cpu();
                LOG_DEBUG("    colors_cpu: is_valid={}, ptr={}, shape={}, numel={}",
                          colors_cpu.is_valid(), static_cast<const void*>(colors_cpu.ptr<float>()),
                          colors_cpu.shape().str(), colors_cpu.numel());

                auto fused_color = rgb_to_sh(colors_cpu);
                LOG_DEBUG("    fused_color: is_valid={}, shape={}, numel={}",
                          fused_color.is_valid(), fused_color.shape().str(), fused_color.numel());

                // Create SH tensor on CPU
                auto shs_cpu_tensor = Tensor::zeros(
                    {fused_color.size(0), static_cast<size_t>(feature_shape), 3},
                    Device::CPU);
                LOG_DEBUG("    shs_cpu_tensor: is_valid={}, shape={}, numel={}",
                          shs_cpu_tensor.is_valid(), shs_cpu_tensor.shape().str(), shs_cpu_tensor.numel());

                auto shs_acc = shs_cpu_tensor.accessor<float, 3>();
                auto fused_acc = fused_color.accessor<float, 2>();

                for (size_t i = 0; i < fused_color.size(0); ++i) {
                    for (size_t c = 0; c < 3; ++c) {
                        shs_acc(i, 0, c) = fused_acc(i, c); // Set DC coefficient
                    }
                }

                sh0_cpu = shs_cpu_tensor.slice(1, 0, 1).contiguous();
                if (feature_shape > 1) {
                    shN_cpu = shs_cpu_tensor.slice(1, 1, feature_shape).contiguous();
                } else {
                    // sh-degree 0: create empty shN tensor [N, 0, 3]
                    shN_cpu = Tensor::zeros({shs_cpu_tensor.size(0), 0, 3}, Device::CPU);
                }
                LOG_DEBUG("  sh0_cpu: is_valid={}, ptr={}, shape={}, numel={}",
                          sh0_cpu.is_valid(), static_cast<const void*>(sh0_cpu.ptr<float>()),
                          sh0_cpu.shape().str(), sh0_cpu.numel());
                LOG_DEBUG("  shN_cpu: is_valid={}, ptr={}, shape={}, numel={}",
                          shN_cpu.is_valid(), static_cast<const void*>(shN_cpu.ptr<float>()),
                          shN_cpu.shape().str(), shN_cpu.numel());

                // Copy CPU data to direct CUDA tensors
                LOG_DEBUG("Copying CPU values to direct CUDA tensors");
                cudaError_t err;

                // Means copy
                LOG_DEBUG("  Copying means: src_ptr={}, dst_ptr={}, bytes={}",
                          static_cast<const void*>(means_cpu.ptr<float>()),
                          static_cast<void*>(means_.ptr<float>()),
                          means_cpu.numel() * sizeof(float));
                err = cudaMemcpy(means_.ptr<float>(), means_cpu.ptr<float>(),
                                 means_cpu.numel() * sizeof(float), cudaMemcpyHostToDevice);
                if (err != cudaSuccess) {
                    LOG_ERROR("cudaMemcpy failed for means:");
                    LOG_ERROR("  src (CPU): is_valid={}, ptr={}, device={}, numel={}",
                              means_cpu.is_valid(), static_cast<const void*>(means_cpu.ptr<float>()),
                              means_cpu.device() == Device::CPU ? "CPU" : "CUDA", means_cpu.numel());
                    LOG_ERROR("  dst (CUDA): is_valid={}, ptr={}, device={}, numel={}",
                              means_.is_valid(), static_cast<void*>(means_.ptr<float>()),
                              means_.device() == Device::CPU ? "CPU" : "CUDA", means_.numel());
                    throw TensorError("cudaMemcpy failed for means: " + std::string(cudaGetErrorString(err)));
                }
                LOG_DEBUG("  Means copy successful");

                // Scaling copy
                LOG_DEBUG("  Copying scaling: src_ptr={}, dst_ptr={}, bytes={}",
                          static_cast<const void*>(scaling_cpu.ptr<float>()),
                          static_cast<void*>(scaling_.ptr<float>()),
                          scaling_cpu.numel() * sizeof(float));
                err = cudaMemcpy(scaling_.ptr<float>(), scaling_cpu.ptr<float>(),
                                 scaling_cpu.numel() * sizeof(float), cudaMemcpyHostToDevice);
                if (err != cudaSuccess) {
                    LOG_ERROR("cudaMemcpy failed for scaling:");
                    LOG_ERROR("  src (CPU): is_valid={}, ptr={}, numel={}",
                              scaling_cpu.is_valid(), static_cast<const void*>(scaling_cpu.ptr<float>()), scaling_cpu.numel());
                    LOG_ERROR("  dst (CUDA): is_valid={}, ptr={}, numel={}",
                              scaling_.is_valid(), static_cast<void*>(scaling_.ptr<float>()), scaling_.numel());
                    throw TensorError("cudaMemcpy failed for scaling: " + std::string(cudaGetErrorString(err)));
                }
                LOG_DEBUG("  Scaling copy successful");

                // Rotation copy
                LOG_DEBUG("  Copying rotation: src_ptr={}, dst_ptr={}, bytes={}",
                          static_cast<const void*>(rotation_cpu.ptr<float>()),
                          static_cast<void*>(rotation_.ptr<float>()),
                          rotation_cpu.numel() * sizeof(float));
                err = cudaMemcpy(rotation_.ptr<float>(), rotation_cpu.ptr<float>(),
                                 rotation_cpu.numel() * sizeof(float), cudaMemcpyHostToDevice);
                if (err != cudaSuccess) {
                    LOG_ERROR("cudaMemcpy failed for rotation:");
                    LOG_ERROR("  src (CPU): is_valid={}, ptr={}, numel={}",
                              rotation_cpu.is_valid(), static_cast<const void*>(rotation_cpu.ptr<float>()), rotation_cpu.numel());
                    LOG_ERROR("  dst (CUDA): is_valid={}, ptr={}, numel={}",
                              rotation_.is_valid(), static_cast<void*>(rotation_.ptr<float>()), rotation_.numel());
                    throw TensorError("cudaMemcpy failed for rotation: " + std::string(cudaGetErrorString(err)));
                }
                LOG_DEBUG("  Rotation copy successful");

                // Opacity copy
                LOG_DEBUG("  Copying opacity: src_ptr={}, dst_ptr={}, bytes={}",
                          static_cast<const void*>(opacity_cpu.ptr<float>()),
                          static_cast<void*>(opacity_.ptr<float>()),
                          opacity_cpu.numel() * sizeof(float));
                err = cudaMemcpy(opacity_.ptr<float>(), opacity_cpu.ptr<float>(),
                                 opacity_cpu.numel() * sizeof(float), cudaMemcpyHostToDevice);
                if (err != cudaSuccess) {
                    LOG_ERROR("cudaMemcpy failed for opacity:");
                    LOG_ERROR("  src (CPU): is_valid={}, ptr={}, numel={}",
                              opacity_cpu.is_valid(), static_cast<const void*>(opacity_cpu.ptr<float>()), opacity_cpu.numel());
                    LOG_ERROR("  dst (CUDA): is_valid={}, ptr={}, numel={}",
                              opacity_.is_valid(), static_cast<void*>(opacity_.ptr<float>()), opacity_.numel());
                    throw TensorError("cudaMemcpy failed for opacity: " + std::string(cudaGetErrorString(err)));
                }
                LOG_DEBUG("  Opacity copy successful");

                // SH0 copy
                LOG_DEBUG("  Copying sh0: src_ptr={}, dst_ptr={}, bytes={}",
                          static_cast<const void*>(sh0_cpu.ptr<float>()),
                          static_cast<void*>(sh0_.ptr<float>()),
                          sh0_cpu.numel() * sizeof(float));
                err = cudaMemcpy(sh0_.ptr<float>(), sh0_cpu.ptr<float>(),
                                 sh0_cpu.numel() * sizeof(float), cudaMemcpyHostToDevice);
                if (err != cudaSuccess) {
                    LOG_ERROR("cudaMemcpy failed for sh0:");
                    LOG_ERROR("  src (CPU): is_valid={}, ptr={}, numel={}",
                              sh0_cpu.is_valid(), static_cast<const void*>(sh0_cpu.ptr<float>()), sh0_cpu.numel());
                    LOG_ERROR("  dst (CUDA): is_valid={}, ptr={}, numel={}",
                              sh0_.is_valid(), static_cast<void*>(sh0_.ptr<float>()), sh0_.numel());
                    throw TensorError("cudaMemcpy failed for sh0: " + std::string(cudaGetErrorString(err)));
                }
                LOG_DEBUG("  SH0 copy successful");

                // SHN copy
                LOG_DEBUG("  Copying shN: src_ptr={}, dst_ptr={}, bytes={}",
                          static_cast<const void*>(shN_cpu.ptr<float>()),
                          static_cast<void*>(shN_.ptr<float>()),
                          shN_cpu.numel() * sizeof(float));
                err = cudaMemcpy(shN_.ptr<float>(), shN_cpu.ptr<float>(),
                                 shN_cpu.numel() * sizeof(float), cudaMemcpyHostToDevice);
                if (err != cudaSuccess) {
                    LOG_ERROR("cudaMemcpy failed for shN:");
                    LOG_ERROR("  src (CPU): is_valid={}, ptr={}, numel={}",
                              shN_cpu.is_valid(), static_cast<const void*>(shN_cpu.ptr<float>()), shN_cpu.numel());
                    LOG_ERROR("  dst (CUDA): is_valid={}, ptr={}, numel={}",
                              shN_.is_valid(), static_cast<void*>(shN_.ptr<float>()), shN_.numel());
                    throw TensorError("cudaMemcpy failed for shN: " + std::string(cudaGetErrorString(err)));
                }
                LOG_DEBUG("  SHN copy successful");

                LOG_DEBUG("All CPU to CUDA copies completed successfully");
            } else {
                // No capacity specified - use pool
                Tensor means_temp;
                if (params.optimization.random) {
                    means_temp = positions.mul(scene_scale).cuda();
                } else {
                    means_temp = positions.cuda();
                }

                auto nn_dist = compute_mean_neighbor_distances(means_temp).clamp_min(1e-7f);
                std::vector<int> scale_expand_shape = {static_cast<int>(num_points), 3};
                auto scaling_temp = nn_dist.sqrt()
                                        .mul(params.optimization.init_scaling)
                                        .log()
                                        .unsqueeze(-1)
                                        .expand(std::span<const int>(scale_expand_shape))
                                        .cuda();

                auto ones_col = Tensor::ones({num_points, 1}, Device::CUDA);
                auto zeros_cols = Tensor::zeros({num_points, 3}, Device::CUDA);
                auto rotation_temp = ones_col.cat(zeros_cols, 1);

                auto opacity_temp = Tensor::full({num_points, 1}, params.optimization.init_opacity, Device::CUDA).logit();

                auto colors_device = colors.cuda();
                auto fused_color = rgb_to_sh(colors_device);

                auto shs = Tensor::zeros({fused_color.size(0), static_cast<size_t>(feature_shape), 3}, Device::CUDA);
                auto shs_cpu_tmp = shs.cpu();
                auto fused_cpu_tmp = fused_color.cpu();

                auto shs_acc = shs_cpu_tmp.accessor<float, 3>();
                auto fused_acc = fused_cpu_tmp.accessor<float, 2>();

                for (size_t i = 0; i < fused_color.size(0); ++i) {
                    for (size_t c = 0; c < 3; ++c) {
                        shs_acc(i, 0, c) = fused_acc(i, c);
                    }
                }

                shs = shs_cpu_tmp.cuda();
                auto sh0_temp = shs.slice(1, 0, 1).contiguous();
                Tensor shN_temp;
                if (feature_shape > 1) {
                    shN_temp = shs.slice(1, 1, feature_shape).contiguous();
                } else {
                    // sh-degree 0: create empty shN tensor [N, 0, 3]
                    shN_temp = Tensor::zeros({shs.size(0), 0, 3}, Device::CUDA);
                }

                means_ = means_temp;
                scaling_ = scaling_temp;
                rotation_ = rotation_temp;
                opacity_ = opacity_temp;
                sh0_ = sh0_temp;
                shN_ = shN_temp;
            }

            LOG_INFO("Scene scale: {}", scene_scale);
            LOG_INFO("Initialized SplatData: {} points, max SH degree: {}, SH coefficients: {}, sh0 shape: {}, shN shape: {}",
                     num_points, params.optimization.sh_degree, feature_shape, sh0_.shape().str(), shN_.shape().str());

            auto result = SplatData(
                params.optimization.sh_degree,
                std::move(means_),
                std::move(sh0_),
                std::move(shN_),
                std::move(scaling_),
                std::move(rotation_),
                std::move(opacity_),
                scene_scale);

            return result;

        } catch (const std::exception& e) {
            return std::unexpected(
                std::format("Failed to initialize SplatData: {}", e.what()));
        }
    }

} // namespace lfs::core
