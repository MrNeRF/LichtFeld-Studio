/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_data.hpp"
#include "core/cuda/sh_layout.cuh"
#include "core/logger.hpp"
#include "core/parameters.hpp"
#include "core/point_cloud.hpp"
#include "core/tensor/internal/tensor_serialization.hpp"
#include "nanoflann.hpp"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <expected>
#include <format>
#include <vector>

namespace {
    constexpr int MAX_SUPPORTED_SH_DEGREE = 3;
    constexpr size_t SH_CHANNELS = 3;

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

    lfs::core::Tensor compute_mrnf_knn_log_scales(const lfs::core::Tensor& points) {
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

        // Match MRNF: if there are too few points, use log_scale=0.
        if (num_points < 3) {
            auto zeros = lfs::core::Tensor::zeros(
                {static_cast<size_t>(num_points), 3},
                lfs::core::Device::CPU);
            return zeros.to(points.device());
        }

        const float* data = cpu_points.ptr<float>();

        constexpr float percentile = 0.75f;
        std::vector<float> x_vals;
        std::vector<float> y_vals;
        std::vector<float> z_vals;
        x_vals.reserve(num_points);
        y_vals.reserve(num_points);
        z_vals.reserve(num_points);
        for (int i = 0; i < num_points; ++i) {
            const float x = data[i * 3 + 0];
            const float y = data[i * 3 + 1];
            const float z = data[i * 3 + 2];
            if (std::isfinite(x))
                x_vals.push_back(x);
            if (std::isfinite(y))
                y_vals.push_back(y);
            if (std::isfinite(z))
                z_vals.push_back(z);
        }

        if (x_vals.empty() || y_vals.empty() || z_vals.empty()) {
            auto zeros = lfs::core::Tensor::zeros(
                {static_cast<size_t>(num_points), 3},
                lfs::core::Device::CPU);
            return zeros.to(points.device());
        }

        std::sort(x_vals.begin(), x_vals.end());
        std::sort(y_vals.begin(), y_vals.end());
        std::sort(z_vals.begin(), z_vals.end());

        const auto idx_pair = [percentile](const size_t len) {
            const size_t lower_idx = static_cast<size_t>(((1.0f - percentile) / 2.0f) * static_cast<float>(len));
            const size_t upper_idx =
                std::min(len - 1, static_cast<size_t>(((1.0f + percentile) / 2.0f) * static_cast<float>(len)));
            return std::pair<size_t, size_t>{lower_idx, upper_idx};
        };

        const auto [lx, ux] = idx_pair(x_vals.size());
        const auto [ly, uy] = idx_pair(y_vals.size());
        const auto [lz, uz] = idx_pair(z_vals.size());

        const float ex = (x_vals[ux] - x_vals[lx]) * 0.5f;
        const float ey = (y_vals[uy] - y_vals[ly]) * 0.5f;
        const float ez = (z_vals[uz] - z_vals[lz]) * 0.5f;

        float sorted_extents[3] = {ex, ey, ez};
        std::sort(sorted_extents, sorted_extents + 3);
        const float median_size = std::max(sorted_extents[1] * 2.0f, 0.01f);
        const float max_scale = median_size * 0.1f;

        PointCloudAdaptor cloud(data, num_points);
        KDTree index(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        index.buildIndex();

        auto result = lfs::core::Tensor::zeros(
            {static_cast<size_t>(num_points), 3},
            lfs::core::Device::CPU);
        float* result_data = result.ptr<float>();

#pragma omp parallel for if (num_points > 1000)
        for (int i = 0; i < num_points; i++) {
            const float query_pt[3] = {
                data[i * 3 + 0],
                data[i * 3 + 1],
                data[i * 3 + 2]};

            constexpr size_t num_results = 3; // self + 2 nearest neighbors
            std::vector<size_t> ret_indices(num_results);
            std::vector<float> out_dists_sqr(num_results);

            nanoflann::KNNResultSet<float> result_set(num_results);
            result_set.init(&ret_indices[0], &out_dists_sqr[0]);
            index.findNeighbors(result_set, &query_pt[0], nanoflann::SearchParameters(10));

            const float a1 = std::sqrt(std::max(out_dists_sqr[1], 0.0f));
            const float a2 = std::sqrt(std::max(out_dists_sqr[2], 0.0f));
            const float dist = (a1 + a2) * 0.25f;
            const float log_scale = std::log(std::clamp(dist, 1e-3f, max_scale));

            result_data[i * 3 + 0] = log_scale;
            result_data[i * 3 + 1] = log_scale;
            result_data[i * 3 + 2] = log_scale;
        }

        return result.to(points.device());
    }

    // Allocate a 1D swizzled-layout shN tensor sized for `n` primitives with `capacity`
    // primitive-row slots reserved. Zero-initialised so dead lanes & inactive coefficient
    // slots contribute nothing in Adam updates.
    //
    // The swizzled layout is intrinsically CUDA-only: every reader / writer
    // (reorder_sh_to_swizzled, undo_reorder_sh_from_swizzled, shN_swizzled_gather_self,
    // the rasterizer's load_shN_coeffs, the fused Adam path) is a CUDA kernel. There is no
    // CPU swizzle path, so this buffer always lives on Device::CUDA regardless of where
    // the other SplatData tensors live.
    lfs::core::Tensor allocate_param_tensor(const lfs::core::TensorShape& shape,
                                            size_t capacity,
                                            const lfs::core::SplatTensorAllocator& allocator,
                                            std::string_view name) {
        using namespace lfs::core;
        Tensor tensor = allocator
                            ? allocator(shape, capacity, DataType::Float32, name)
                            : Tensor::zeros_direct(shape, capacity, Device::CUDA);
        tensor.set_name(std::string{name});
        return tensor;
    }

    [[nodiscard]] uint32_t infer_swizzled_rest_coefficients(size_t n, size_t numel) {
        using namespace lfs::core;
        const size_t blocks = sh_swizzled_block_count(n);
        if (blocks == 0 || numel == 0) {
            return 0;
        }
        const size_t denom = blocks * kShReorderSize * 4u;
        const size_t slots = denom > 0 ? numel / denom : 0;
        if (slots == 0) {
            return 0;
        }
        if (slots <= 3) {
            return sh_rest_coefficients_for_degree(1);
        }
        if (slots <= 6) {
            return sh_rest_coefficients_for_degree(2);
        }
        return sh_rest_coefficients_for_degree(3);
    }

    lfs::core::Tensor allocate_swizzled_shN(size_t n, size_t capacity, uint32_t layout_coeffs_rest) {
        using namespace lfs::core;
        const size_t cap = std::max(capacity, n);
        const size_t logical_floats = sh_swizzled_float_count(n, layout_coeffs_rest);
        const size_t capacity_floats = sh_swizzled_float_count(cap, layout_coeffs_rest);
        if (capacity_floats == 0) {
            return Tensor::zeros({0}, Device::CUDA);
        }
        return Tensor::zeros_direct(TensorShape({logical_floats}), capacity_floats, Device::CUDA);
    }

    lfs::core::Tensor allocate_swizzled_shN(size_t n,
                                            size_t capacity,
                                            uint32_t layout_coeffs_rest,
                                            const lfs::core::SplatTensorAllocator& allocator,
                                            std::string_view name) {
        using namespace lfs::core;
        const size_t cap = std::max(capacity, n);
        const size_t logical_floats = sh_swizzled_float_count(n, layout_coeffs_rest);
        const size_t capacity_floats = sh_swizzled_float_count(cap, layout_coeffs_rest);
        if (capacity_floats == 0) {
            Tensor tensor = Tensor::zeros({0}, Device::CUDA);
            tensor.set_name(std::string{name});
            return tensor;
        }
        if (allocator) {
            return allocate_param_tensor(TensorShape({logical_floats}),
                                         capacity_floats,
                                         allocator,
                                         name);
        }
        Tensor tensor = Tensor::zeros_direct(TensorShape({logical_floats}),
                                             capacity_floats,
                                             Device::CUDA);
        tensor.set_name(std::string{name});
        return tensor;
    }

    [[nodiscard]] uint32_t canonical_rest_coefficients(const lfs::core::Tensor& canonical) {
        if (!canonical.is_valid() || canonical.numel() == 0) {
            return 0;
        }
        if (canonical.ndim() == 3) {
            return static_cast<uint32_t>(std::min<size_t>(
                canonical.size(1), lfs::core::kShMaxCoeffsRest));
        }
        if (canonical.ndim() == 2) {
            return static_cast<uint32_t>(std::min<size_t>(
                canonical.size(1) / SH_CHANNELS, lfs::core::kShMaxCoeffsRest));
        }
        return 0;
    }

    // Reorder a canonical-layout shN tensor into the swizzled `dst` buffer.
    // canonical may be [N, K, 3] or [N, K*3]. K may be smaller than the resident
    // layout; missing coefficients are zero-filled.
    void reorder_canonical_into_swizzled(
        const lfs::core::Tensor& canonical,
        float* dst_swizzled,
        size_t n_primitives,
        uint32_t src_coeffs_rest,
        uint32_t layout_coeffs_rest) {
        using namespace lfs::core;
        if (n_primitives == 0) {
            return;
        }
        if (src_coeffs_rest == 0 || layout_coeffs_rest == 0) {
            return;
        }
        Tensor src = canonical;
        Tensor truncated;
        if (canonical.ndim() == 3 &&
            canonical.size(1) > src_coeffs_rest) {
            truncated = canonical.slice(1, 0, static_cast<int>(src_coeffs_rest)).contiguous();
            src = truncated;
        } else if (canonical.ndim() == 2 &&
                   canonical.size(1) > static_cast<size_t>(src_coeffs_rest) * SH_CHANNELS) {
            truncated = canonical.slice(
                                      1,
                                      0,
                                      static_cast<int>(src_coeffs_rest * SH_CHANNELS))
                            .contiguous();
            src = truncated;
        }
        Tensor src_cuda = src.device() == Device::CUDA ? src : src.cuda();
        if (!src_cuda.is_contiguous()) {
            src_cuda = src_cuda.contiguous();
        }
        reorder_sh_to_swizzled(src_cuda.ptr<float>(),
                               dst_swizzled,
                               n_primitives,
                               src_coeffs_rest,
                               layout_coeffs_rest);
    }

    [[nodiscard]] bool swizzled_storage_matches(const lfs::core::Tensor& shN,
                                                size_t n,
                                                size_t capacity,
                                                uint32_t layout_coeffs_rest) {
        using namespace lfs::core;
        const size_t cap = std::max(capacity, n);
        return shN.is_valid() &&
               shN.ndim() == 1 &&
               static_cast<size_t>(shN.shape()[0]) == sh_swizzled_float_count(n, layout_coeffs_rest) &&
               shN.capacity() >= sh_swizzled_float_count(cap, layout_coeffs_rest);
    }

    void resize_swizzled_storage_preserving(lfs::core::Tensor& shN,
                                            size_t n,
                                            size_t capacity,
                                            uint32_t layout_coeffs_rest) {
        using namespace lfs::core;
        const size_t cap = std::max(capacity, n);
        const uint32_t old_layout_rest =
            (shN.is_valid() && shN.ndim() == 1 && n > 0)
                ? infer_swizzled_rest_coefficients(n, static_cast<size_t>(shN.numel()))
                : 0u;

        Tensor old_canonical;
        if (shN.is_valid() && shN.numel() > 0 && n > 0 && old_layout_rest > 0) {
            old_canonical = Tensor::empty({n, static_cast<size_t>(old_layout_rest), SH_CHANNELS}, shN.device());
            undo_reorder_sh_from_swizzled(shN.ptr<float>(),
                                          old_canonical.ptr<float>(),
                                          n,
                                          old_layout_rest,
                                          old_layout_rest);
        }

        Tensor resized = allocate_swizzled_shN(n, cap, layout_coeffs_rest);
        const auto copy_rest = std::min(old_layout_rest, layout_coeffs_rest);
        if (copy_rest > 0 && old_canonical.is_valid() && old_canonical.numel() > 0) {
            reorder_canonical_into_swizzled(old_canonical,
                                            resized.ptr<float>(),
                                            n,
                                            copy_rest,
                                            layout_coeffs_rest);
        }
        shN = std::move(resized);
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
                         float scene_scale_,
                         ShNLayout shN_layout)
        : _max_sh_degree(sh_degree),
          _active_sh_degree(sh_degree), // Set to max degree when loading; training will override this
          _scene_scale(scene_scale_),
          _means(std::move(means_)),
          _sh0(std::move(sh0_)),
          _scaling(std::move(scaling_)),
          _rotation(std::move(rotation_)),
          _opacity(std::move(opacity_)) {
        const size_t n = _means.is_valid() ? static_cast<size_t>(_means.shape()[0]) : 0;
        const size_t capacity = _means.is_valid() ? std::max<size_t>(_means.capacity(), n) : n;
        uint32_t layout_coeffs_rest =
            static_cast<uint32_t>(sh_rest_coefficients_for_degree(_max_sh_degree));
        if (shN_layout == ShNLayout::Swizzled) {
            _shN = std::move(shN_);
            if (_shN.is_valid() && _shN.ndim() == 1 && n > 0) {
                const auto stored_rest = infer_swizzled_rest_coefficients(n, static_cast<size_t>(_shN.numel()));
                const size_t expected_for_requested_degree = sh_swizzled_float_count(n, layout_coeffs_rest);
                if (stored_rest > 0 && stored_rest != layout_coeffs_rest &&
                    static_cast<size_t>(_shN.numel()) == sh_swizzled_float_count(n, stored_rest)) {
                    Tensor old_canonical = Tensor::empty({n, static_cast<size_t>(stored_rest), SH_CHANNELS}, _shN.device());
                    undo_reorder_sh_from_swizzled(_shN.ptr<float>(),
                                                  old_canonical.ptr<float>(),
                                                  n,
                                                  stored_rest,
                                                  stored_rest);
                    Tensor resized = allocate_swizzled_shN(n, capacity, layout_coeffs_rest);
                    const auto copy_rest = std::min(stored_rest, layout_coeffs_rest);
                    reorder_canonical_into_swizzled(old_canonical,
                                                    resized.ptr<float>(),
                                                    n,
                                                    copy_rest,
                                                    layout_coeffs_rest);
                    _shN = std::move(resized);
                } else if (stored_rest > 0 &&
                           static_cast<size_t>(_shN.numel()) < expected_for_requested_degree) {
                    _max_sh_degree = sh_degree_for_rest_coefficients(stored_rest);
                    _active_sh_degree = std::min(_active_sh_degree, _max_sh_degree);
                    layout_coeffs_rest =
                        static_cast<uint32_t>(sh_rest_coefficients_for_degree(_max_sh_degree));
                }
            }
            const size_t expected_floats = sh_swizzled_float_count(n, layout_coeffs_rest);
            if (!_shN.is_valid() || _shN.ndim() != 1 ||
                static_cast<size_t>(_shN.shape()[0]) != expected_floats) {
                _shN = allocate_swizzled_shN(n, capacity, layout_coeffs_rest);
            }
            return;
        }

        // Convert the canonical-layout shN argument into swizzled storage.
        _shN = allocate_swizzled_shN(n, capacity, layout_coeffs_rest);
        const auto src_rest = std::min(canonical_rest_coefficients(shN_), layout_coeffs_rest);
        if (shN_.is_valid() && shN_.numel() > 0 && n > 0 && src_rest > 0 && layout_coeffs_rest > 0) {
            reorder_canonical_into_swizzled(shN_, _shN.ptr<float>(), n, src_rest, layout_coeffs_rest);
        }
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
        // _sh0 is [N, 1, 3]; _shN is the swizzled flat buffer. Deswizzle on demand
        // and concatenate to produce [N, K_total, 3].
        const size_t active_rest = active_sh_coeffs_rest();
        if (active_rest == 0) {
            return _sh0;
        }
        auto shN = shN_canonical();
        if (shN.ndim() == 3 && shN.size(1) > active_rest) {
            shN = shN.slice(1, 0, static_cast<int>(active_rest)).contiguous();
        }
        if (shN.device() != _sh0.device()) {
            shN = shN.to(_sh0.device());
        }
        return _sh0.cat(shN, 1);
    }

    Tensor SplatData::shN_canonical() const {
        const size_t n = static_cast<size_t>(size());
        const size_t k = max_sh_coeffs_rest();
        // The swizzled buffer is CUDA-only (see allocate_swizzled_shN); align the canonical
        // output device with where the source data actually lives.
        const Device dst_device = _shN.is_valid() ? _shN.device() : Device::CUDA;
        if (n == 0 || k == 0) {
            return Tensor::zeros({n, k, SH_CHANNELS}, dst_device);
        }
        Tensor out = Tensor::empty({n, k, SH_CHANNELS}, dst_device);
        undo_reorder_sh_from_swizzled(_shN.ptr<float>(),
                                      out.ptr<float>(),
                                      n,
                                      static_cast<uint32_t>(k),
                                      static_cast<uint32_t>(k));
        return out;
    }

    Tensor SplatData::shN_canonical_cpu() const {
        const size_t n = static_cast<size_t>(size());
        const size_t k = max_sh_coeffs_rest();
        if (n == 0 || k == 0) {
            return Tensor::zeros({n, k, SH_CHANNELS}, Device::CPU);
        }

        Tensor out = Tensor::empty({n, k, SH_CHANNELS}, Device::CPU, DataType::Float32);
        if (!_shN.is_valid() || _shN.numel() == 0) {
            out.zero_();
            return out;
        }

        const Tensor shN_cpu = _shN.cpu().contiguous();
        const auto* const src = shN_cpu.ptr<float>();
        auto* const dst = out.ptr<float>();
        const size_t src_floats = shN_cpu.numel();
        const size_t active_floats = k * SH_CHANNELS;

        for (size_t p = 0; p < n; ++p) {
            float* const dst_row = dst + p * active_floats;
            for (size_t offset = 0; offset < active_floats; ++offset) {
                const auto slot = static_cast<std::uint32_t>(offset / 4u);
                const auto component = static_cast<std::uint32_t>(offset % 4u);
                const size_t src_offset =
                    static_cast<size_t>(sh_swizzled_index(static_cast<std::uint32_t>(p), slot, static_cast<uint32_t>(k))) * 4u +
                    component;
                dst_row[offset] = src_offset < src_floats ? src[src_offset] : 0.0f;
            }
        }

        return out;
    }

    void SplatData::shN_set_from_canonical(const Tensor& canonical, size_t capacity) {
        const size_t n = static_cast<size_t>(size());
        const size_t cap = std::max<size_t>(capacity, n);
        const uint32_t layout_rest = static_cast<uint32_t>(max_sh_coeffs_rest());
        const size_t needed_floats = sh_swizzled_float_count(n, layout_rest);
        const size_t needed_capacity = sh_swizzled_float_count(cap, layout_rest);

        // Adjust _shN's logical size to match the new N without losing reserved capacity
        // when possible. Reallocating drops the pre-alloc buffer and can break async
        // pointer aliasing for downstream kernels; we only do it when capacity is short.
        if (_shN.is_valid() && _shN.capacity() >= needed_capacity) {
            if (_shN.numel() < needed_floats) {
                _shN.append_zeros(needed_floats - _shN.numel());
            } else if (_shN.numel() > needed_floats) {
                // N shrank (e.g. random_choose, crop, prune-then-compact). The Tensor lib
                // doesn't have a "shrink logical size" op other than reassigning shape.
                // Allocate a smaller buffer in this case — it's a one-shot edit operation.
                _shN = allocate_swizzled_shN(n, cap, layout_rest);
            }
            // else: numel() == needed_floats, nothing to do.
        } else {
            _shN = allocate_swizzled_shN(n, cap, layout_rest);
        }

        const auto src_rest = std::min(canonical_rest_coefficients(canonical), layout_rest);
        if (n > 0 && layout_rest > 0 && src_rest == 0 && _shN.is_valid() && _shN.numel() > 0) {
            _shN.zero_();
        }
        if (canonical.is_valid() && canonical.numel() > 0 && n > 0 && src_rest > 0 && layout_rest > 0) {
            reorder_canonical_into_swizzled(canonical, _shN.ptr<float>(), n, src_rest, layout_rest);
        }
    }

    size_t SplatData::active_sh_coeffs_rest() const {
        return sh_rest_coefficients_for_degree(_active_sh_degree);
    }

    size_t SplatData::max_sh_coeffs_rest() const {
        return sh_rest_coefficients_for_degree(_max_sh_degree);
    }

    // ========== UTILITY METHODS ==========

    void SplatData::increment_sh_degree() {
        if (_active_sh_degree < _max_sh_degree) {
            set_active_sh_degree(_active_sh_degree + 1);
        }
    }

    void SplatData::set_active_sh_degree(int sh_degree) {
        const int target_degree = std::clamp(sh_degree, 0, _max_sh_degree);
        const size_t n = static_cast<size_t>(size());
        const size_t cap = _means.is_valid() ? std::max<size_t>(_means.capacity(), n) : n;
        const auto layout_rest = static_cast<uint32_t>(max_sh_coeffs_rest());
        if (!swizzled_storage_matches(_shN, n, cap, layout_rest)) {
            resize_swizzled_storage_preserving(_shN, n, cap, layout_rest);
        }
        _active_sh_degree = target_degree;
    }

    void SplatData::set_max_sh_degree(int sh_degree) {
        const int target_degree = std::clamp(sh_degree, 0, MAX_SUPPORTED_SH_DEGREE);
        if (_max_sh_degree == target_degree) {
            if (_active_sh_degree > _max_sh_degree) {
                _active_sh_degree = _max_sh_degree;
            }
            const size_t n = static_cast<size_t>(size());
            const size_t cap = _means.is_valid() ? std::max<size_t>(_means.capacity(), n) : n;
            const auto layout_rest = static_cast<uint32_t>(max_sh_coeffs_rest());
            if (!swizzled_storage_matches(_shN, n, cap, layout_rest)) {
                resize_swizzled_storage_preserving(_shN, n, cap, layout_rest);
            }
            return;
        }

        _max_sh_degree = target_degree;
        if (_active_sh_degree > _max_sh_degree) {
            _active_sh_degree = _max_sh_degree;
        }

        const size_t n = static_cast<size_t>(size());
        const size_t cap = _means.is_valid() ? std::max<size_t>(_means.capacity(), n) : n;
        resize_swizzled_storage_preserving(_shN,
                                           n,
                                           cap,
                                           static_cast<uint32_t>(max_sh_coeffs_rest()));
    }

    bool SplatData::set_sh_degree(const int sh_degree) {
        assert(_means.is_valid());

        const int target_degree = std::clamp(sh_degree, 0, MAX_SUPPORTED_SH_DEGREE);
        bool changed = _max_sh_degree != target_degree || _active_sh_degree != target_degree;

        set_max_sh_degree(target_degree);
        _active_sh_degree = target_degree;
        return changed;
    }

    void SplatData::reserve_capacity(const size_t capacity) {
        if (_means.is_valid())
            _means.reserve(capacity);
        if (_sh0.is_valid())
            _sh0.reserve(capacity);
        if (_shN.is_valid()) {
            // shN is 1D swizzled — reserve in float count.
            _shN.reserve(sh_swizzled_float_count(capacity, static_cast<uint32_t>(max_sh_coeffs_rest())));
        }
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

        // shN is in swizzled layout — block-aware gather of kept primitives.
        const auto layout_rest = static_cast<uint32_t>(max_sh_coeffs_rest());
        if (_shN.is_valid() && _shN.numel() > 0 && layout_rest > 0) {
            // nonzero() returns [num_kept, 1] int tensor of true indices.
            auto kept_indices_2d = keep_mask.nonzero();
            const auto kept_numel = static_cast<int>(kept_indices_2d.numel());
            Tensor kept_indices = kept_indices_2d.ndim() > 1
                                      ? kept_indices_2d.reshape({kept_numel})
                                      : kept_indices_2d;
            kept_indices = kept_indices.to(DataType::Int32);

            auto new_shN = allocate_swizzled_shN(new_size, new_size, layout_rest);
            shN_swizzled_gather_self(_shN.ptr<float>(), new_shN.ptr<float>(),
                                     kept_indices.ptr<int>(), new_size, 0, layout_rest);
            _shN = std::move(new_shN);
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
            // On-disk format is canonical [N, K, 3]; deswizzle before writing for
            // forward compatibility and to keep the format identical to pre-swizzle builds.
            Tensor shN_canon = shN_canonical_cpu();
            os << shN_canon;
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
        _max_sh_degree = std::clamp(max_sh, 0, MAX_SUPPORTED_SH_DEGREE);
        _active_sh_degree = std::clamp(active_sh, 0, _max_sh_degree);
        _scene_scale = scene_scale;

        if (_max_sh_degree > 0) {
            Tensor shN_canon;
            is >> shN_canon;
            // shN_canon is canonical [N, K, 3]; reorder into swizzled storage.
            shN_set_from_canonical(shN_canon.cuda(), static_cast<size_t>(_means.capacity()));
        } else {
            // Allocate an empty swizzled tensor so _shN is valid even at SH degree 0.
            const size_t n = static_cast<size_t>(size());
            const size_t cap = std::max<size_t>(_means.capacity(), n);
            _shN = allocate_swizzled_shN(n, cap, static_cast<uint32_t>(max_sh_coeffs_rest()));
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
        int capacity,
        SplatTensorAllocator tensor_allocator) {

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

            auto scene_center_device = scene_center.to(positions.device());
            const Tensor dists = positions.sub(scene_center_device).norm(2.0f, {1}, false);

            // Get median distance for scene scale
            auto sorted_dists = dists.sort(0, false);
            const float scene_scale = sorted_dists.first[dists.size(0) / 2].item();

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
                LOG_DEBUG("Creating resident tensors with capacity={}", capacity);

                means_ = allocate_param_tensor(TensorShape({num_points, 3}),
                                               static_cast<size_t>(capacity),
                                               tensor_allocator,
                                               "SplatData.means");
                LOG_DEBUG("  means_ allocated: is_valid={}, ptr={}, shape={}, numel={}",
                          means_.is_valid(), static_cast<void*>(means_.ptr<float>()),
                          means_.shape().str(), means_.numel());

                scaling_ = allocate_param_tensor(TensorShape({num_points, 3}),
                                                 static_cast<size_t>(capacity),
                                                 tensor_allocator,
                                                 "SplatData.scaling");
                LOG_DEBUG("  scaling_ allocated: is_valid={}, ptr={}, shape={}, numel={}",
                          scaling_.is_valid(), static_cast<void*>(scaling_.ptr<float>()),
                          scaling_.shape().str(), scaling_.numel());

                rotation_ = allocate_param_tensor(TensorShape({num_points, 4}),
                                                 static_cast<size_t>(capacity),
                                                 tensor_allocator,
                                                 "SplatData.rotation");
                LOG_DEBUG("  rotation_ allocated: is_valid={}, ptr={}, shape={}, numel={}",
                          rotation_.is_valid(), static_cast<void*>(rotation_.ptr<float>()),
                          rotation_.shape().str(), rotation_.numel());

                opacity_ = allocate_param_tensor(TensorShape({num_points, 1}),
                                                static_cast<size_t>(capacity),
                                                tensor_allocator,
                                                "SplatData.opacity");
                LOG_DEBUG("  opacity_ allocated: is_valid={}, ptr={}, shape={}, numel={}",
                          opacity_.is_valid(), static_cast<void*>(opacity_.ptr<float>()),
                          opacity_.shape().str(), opacity_.numel());

                sh0_ = allocate_param_tensor(TensorShape({num_points, 1, 3}),
                                             static_cast<size_t>(capacity),
                                             tensor_allocator,
                                             "SplatData.sh0");
                LOG_DEBUG("  sh0_ allocated: is_valid={}, ptr={}, shape={}, numel={}",
                          sh0_.is_valid(), static_cast<void*>(sh0_.ptr<float>()),
                          sh0_.shape().str(), sh0_.numel());

                // Build SH-rest directly in the resident vksplat-swizzled layout.
                // The old path allocated a canonical CUDA tensor and then the final
                // swizzled tensor, briefly holding both. At SH3/max-cap that transient
                // is large enough to show up in the VRAM profile.
                shN_ = allocate_swizzled_shN(num_points,
                                             static_cast<size_t>(capacity),
                                             static_cast<uint32_t>(feature_shape - 1),
                                             tensor_allocator,
                                             "SplatData.shN");
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
                if (lfs::core::param::is_mrnf_strategy(params.optimization.strategy)) {
                    scaling_cpu = compute_mrnf_knn_log_scales(means_cpu);
                } else {
                    auto nn_dist = compute_mean_neighbor_distances(means_cpu).clamp_min(1e-7f);
                    LOG_DEBUG("  nn_dist computed: is_valid={}, shape={}, numel={}",
                              nn_dist.is_valid(), nn_dist.shape().str(), nn_dist.numel());

                    std::vector<int> scale_expand_shape = {static_cast<int>(num_points), 3};
                    scaling_cpu = nn_dist.sqrt()
                                      .mul(params.optimization.init_scaling)
                                      .log()
                                      .unsqueeze(-1)
                                      .expand(std::span<const int>(scale_expand_shape));
                }
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

                // SHN swizzle
                LOG_DEBUG("  Swizzling shN: src_ptr={}, dst_ptr={}, src_bytes={}",
                          static_cast<const void*>(shN_cpu.ptr<float>()),
                          static_cast<void*>(shN_.ptr<float>()),
                          shN_cpu.numel() * sizeof(float));
                reorder_canonical_into_swizzled(
                    shN_cpu,
                    shN_.ptr<float>(),
                    num_points,
                    static_cast<uint32_t>(feature_shape - 1),
                    static_cast<uint32_t>(feature_shape - 1));
                err = cudaGetLastError();
                if (err != cudaSuccess) {
                    LOG_ERROR("SH swizzle failed for shN:");
                    LOG_ERROR("  src (CPU): is_valid={}, ptr={}, numel={}",
                              shN_cpu.is_valid(), static_cast<const void*>(shN_cpu.ptr<float>()), shN_cpu.numel());
                    LOG_ERROR("  dst (CUDA): is_valid={}, ptr={}, numel={}",
                              shN_.is_valid(), static_cast<void*>(shN_.ptr<float>()), shN_.numel());
                    throw TensorError("SH swizzle failed for shN: " + std::string(cudaGetErrorString(err)));
                }
                LOG_DEBUG("  SHN swizzle successful");

                LOG_DEBUG("All CPU to CUDA copies completed successfully");
            } else {
                // No capacity specified - use pool
                Tensor means_temp;
                if (params.optimization.random) {
                    means_temp = positions.mul(scene_scale).cuda();
                } else {
                    means_temp = positions.cuda();
                }

                Tensor scaling_temp;
                if (lfs::core::param::is_mrnf_strategy(params.optimization.strategy)) {
                    scaling_temp = compute_mrnf_knn_log_scales(means_temp).cuda();
                } else {
                    auto nn_dist = compute_mean_neighbor_distances(means_temp).clamp_min(1e-7f);
                    std::vector<int> scale_expand_shape = {static_cast<int>(num_points), 3};
                    scaling_temp = nn_dist.sqrt()
                                       .mul(params.optimization.init_scaling)
                                       .log()
                                       .unsqueeze(-1)
                                       .expand(std::span<const int>(scale_expand_shape))
                                       .cuda();
                }

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
                scene_scale,
                capacity > 0 ? SplatData::ShNLayout::Swizzled
                             : SplatData::ShNLayout::Canonical);

            return result;

        } catch (const std::exception& e) {
            return std::unexpected(
                std::format("Failed to initialize SplatData: {}", e.what()));
        }
    }

} // namespace lfs::core
