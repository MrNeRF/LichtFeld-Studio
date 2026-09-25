/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Operations not yet ported to Metal. They fail like the Vulkan backend's
// unsupported cases, so callers and the backend corpus report them uniformly.

#include "metal_backend_ops.hpp"

#include "../../internal/tensor_impl.hpp"
#include "core/tensor_export.hpp"

#include <format>
#include <string_view>

namespace lfs::core::internal {

    namespace {
        [[noreturn]] void not_ported(const std::string_view operation) {
            throw TensorError(std::format("Metal backend: {} is not implemented yet", operation));
        }
    } // namespace

    void MetalBackendOps::compiled_expression(const ExpressionLaunch&, ExecContext) {
        not_ported("compiled_expression");
    }

    void MetalBackendOps::sort_1d(StorageRef, StorageRef, size_t, const SortProgram&, ExecContext) {
        not_ported("sort_1d");
    }

    void MetalBackendOps::sort_2d(StorageRef, StorageRef, const SortProgram&, ExecContext) {
        not_ported("sort_2d");
    }

    void MetalBackendOps::project_points(StorageRef, StorageRef, size_t, const PointProjection&, const StorageRef*, size_t, const StorageRef*, const StorageRef*, size_t, ExecContext) {
        not_ported("project_points");
    }

    void MetalBackendOps::radius_neighbors(StorageRef, StorageRef, StorageRef, StorageRef, StorageRef, size_t, size_t, float, ExecContext) {
        not_ported("radius_neighbors");
    }

    void MetalBackendOps::mark_points_2d(StorageRef, StorageRef, size_t, const PointRegion2D&, const StorageRef*, size_t, ExecContext) {
        not_ported("mark_points_2d");
    }

    void MetalBackendOps::filter_points(StorageRef, const PointFilterProgram&, ExecContext) {
        not_ported("filter_points");
    }

    void MetalBackendOps::update_labels(StorageRef, StorageRef, const LabelUpdateProgram&, ExecContext) {
        not_ported("update_labels");
    }

    void MetalBackendOps::ppisp_apply(StorageRef, StorageRef, int, int, const PpispParams&, ExecContext) {
        not_ported("ppisp_apply");
    }

    void MetalBackendOps::environment_composite(StorageRef, StorageRef, StorageRef, StorageRef, const EnvironmentCompositeParams&, ExecContext) {
        not_ported("environment_composite");
    }

    Tensor MetalBackendOps::image_undistort(const Tensor&, const UndistortParams&, bool, ExecContext) {
        not_ported("image_undistort");
    }

    Tensor MetalBackendOps::image_resize_prior(const Tensor&, int, int, bool, ExecContext) {
        not_ported("image_resize_prior");
    }

    void MetalBackendOps::histogram_u8(StorageRef, StorageRef, size_t, ExecContext) {
        not_ported("histogram_u8");
    }

    void MetalBackendOps::affine_splat_geometry(StorageRef, StorageRef, StorageRef, StorageRef, const splat_transform::LinearTransform&, size_t, ExecContext) {
        not_ported("affine_splat_geometry");
    }

    void MetalBackendOps::sh_codec(StorageRef, StorageRef, const ShCodecProgram&, ExecContext) {
        not_ported("sh_codec");
    }

    Tensor MetalBackendOps::morton_sort(const Tensor&, Tensor*, ExecContext) {
        not_ported("morton_sort");
    }

    std::tuple<Tensor, Tensor> MetalBackendOps::kmeans_sh(const Tensor&, int, int, int, int, bool, ExecContext) {
        not_ported("kmeans_sh");
    }

    void MetalBackendOps::assign_sh3(const Tensor&, const Tensor&, const Tensor&, Tensor&, bool, bool, ExecContext) {
        not_ported("assign_sh3");
    }

    void MetalBackendOps::decimate_candidates(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, int, std::vector<uint32_t>&, std::vector<float>&, ExecContext) {
        not_ported("decimate_candidates");
    }

    DecimateMerge MetalBackendOps::decimate_merge(const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, const Tensor&, int, const std::vector<int>&, const std::vector<uint32_t>&, const std::vector<uint32_t>&, const std::vector<uint32_t>&, size_t, ExecContext) {
        not_ported("decimate_merge");
    }

    void MetalBackendOps::uniform(StorageRef, const RandomProgram&, ExecContext) {
        not_ported("uniform");
    }

    void MetalBackendOps::bernoulli(StorageRef, const RandomProgram&, ExecContext) {
        not_ported("bernoulli");
    }

    void MetalBackendOps::randint(StorageRef, const RandomProgram&, ExecContext) {
        not_ported("randint");
    }

    void MetalBackendOps::multinomial(StorageRef, StorageRef, const RandomProgram&, ExecContext) {
        not_ported("multinomial");
    }

    void MetalBackendOps::normal(StorageRef, StorageRef, const RandomProgram&, ExecContext) {
        not_ported("normal");
    }

} // namespace lfs::core::internal
