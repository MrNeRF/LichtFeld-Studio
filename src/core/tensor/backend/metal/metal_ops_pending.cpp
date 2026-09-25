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

    void MetalBackendOps::broadcast_binary(const PointwiseProgram&, StorageRef, const StridedLayout&, StorageRef, const StridedLayout&, StorageRef, const StridedLayout&, ExecContext) {
        not_ported("broadcast_binary");
    }

    void MetalBackendOps::clamp_scalar(StorageRef, ScalarOperand, ScalarOperand, size_t, ExecContext) {
        not_ported("clamp_scalar");
    }

    void MetalBackendOps::clamp_fused(StorageRef, StorageRef, ScalarOperand, ScalarOperand, size_t, ExecContext) {
        not_ported("clamp_fused");
    }

    void MetalBackendOps::clamp_scalar_int(StorageRef, ScalarOperand, ScalarOperand, size_t, ExecContext) {
        not_ported("clamp_scalar_int");
    }

    void MetalBackendOps::index_cast(StorageRef, StorageRef, size_t, size_t, ExecContext) {
        not_ported("index_cast");
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

    void MetalBackendOps::gather(StorageRef, StorageRef, StorageRef, const StridedLayout&, const StridedLayout&, const IndexProgram&, ExecContext) {
        not_ported("gather");
    }

    void MetalBackendOps::gather_fused_unary(StorageRef, StorageRef, StorageRef, PointwiseOp, const IndexProgram&, ExecContext) {
        not_ported("gather_fused_unary");
    }

    void MetalBackendOps::take(StorageRef, StorageRef, StorageRef, const IndexProgram&, ExecContext) {
        not_ported("take");
    }

    void MetalBackendOps::index_select(StorageRef, StorageRef, StorageRef, const StridedLayout&, const IndexProgram&, ExecContext) {
        not_ported("index_select");
    }

    void MetalBackendOps::scatter(StorageRef, StorageRef, StorageRef, const StridedLayout&, const StridedLayout&, const IndexProgram&, ExecContext) {
        not_ported("scatter");
    }

    void MetalBackendOps::index_copy(StorageRef, StorageRef, StorageRef, const StridedLayout&, const IndexProgram&, ExecContext) {
        not_ported("index_copy");
    }

    void MetalBackendOps::index_add(StorageRef, StorageRef, StorageRef, const StridedLayout&, const IndexProgram&, ExecContext) {
        not_ported("index_add");
    }

    void MetalBackendOps::index_fill(StorageRef, StorageRef, const StridedLayout&, const IndexProgram&, ScalarOperand, ExecContext) {
        not_ported("index_fill");
    }

    void MetalBackendOps::index_put(StorageRef, StorageRef, StorageRef, const IndexProgram&, ExecContext) {
        not_ported("index_put");
    }

    void MetalBackendOps::masked_fill(StorageRef, StorageRef, const MaskProgram&, ExecContext) {
        not_ported("masked_fill");
    }

    size_t MetalBackendOps::masked_select(StorageRef, StorageRef, StorageRef, const MaskProgram&, ExecContext) {
        not_ported("masked_select");
    }

    void MetalBackendOps::masked_scatter(StorageRef, StorageRef, StorageRef, const MaskProgram&, ExecContext) {
        not_ported("masked_scatter");
    }

    void MetalBackendOps::and_live(StorageRef, StorageRef, const MaskProgram&, ExecContext) {
        not_ported("and_live");
    }

    size_t MetalBackendOps::nonzero(StorageRef, StorageRef, const MaskProgram&, ExecContext) {
        not_ported("nonzero");
    }

    size_t MetalBackendOps::nonzero_bool(StorageRef, StorageRef, const MaskProgram&, ExecContext) {
        not_ported("nonzero_bool");
    }

    void MetalBackendOps::cat_last_dim(StorageRef, std::span<const StorageRef>, std::span<const StridedLayout>, size_t, size_t, size_t, ExecContext) {
        not_ported("cat_last_dim");
    }

    void MetalBackendOps::cat_middle_dim(StorageRef, std::span<const StorageRef>, std::span<const StridedLayout>, size_t, size_t, int, size_t, ExecContext) {
        not_ported("cat_middle_dim");
    }

    void MetalBackendOps::pad(StorageRef, StorageRef, const StridedLayout&, const StridedLayout&, const std::array<size_t, MAX_TENSOR_RANK>&, ExecContext) {
        not_ported("pad");
    }

} // namespace lfs::core::internal
