/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/internal/private_access.hpp"

#include "../gpu_backend_ops.hpp"

#include <span>

namespace lfs::core::internal {

    class VulkanBackendOps final : public GpuBackendOps {
    public:
        void compiled_expression(const ExpressionLaunch& launch, ExecContext context) override;
        ~VulkanBackendOps() override = default;

        void unary(const PointwiseProgram&, StorageRef, StorageRef, size_t, ExecContext) override;
        void binary(const PointwiseProgram&, StorageRef, StorageRef, StorageRef, size_t, ExecContext) override;
        void scalar(const PointwiseProgram&, StorageRef, StorageRef, size_t, ExecContext) override;
        void broadcast_binary(const PointwiseProgram&, StorageRef, const StridedLayout&,
                              StorageRef, const StridedLayout&, StorageRef,
                              const StridedLayout&, ExecContext) override;
        void fused_pointwise_chain(StorageRef, StorageRef, size_t,
                                   const tensor_ops::FusedPointwiseOpChain&,
                                   std::span<const StorageRef> rhs_storages,
                                   ExecContext) override;
        void clamp_scalar(StorageRef, ScalarOperand, ScalarOperand, size_t, ExecContext) override;
        void clamp_fused(StorageRef, StorageRef, ScalarOperand, ScalarOperand, size_t, ExecContext) override;
        void clamp_scalar_int(StorageRef, ScalarOperand, ScalarOperand, size_t, ExecContext) override;
        void index_cast(StorageRef, StorageRef, size_t, size_t, ExecContext) override;
        void convert_type(StorageRef, StorageRef, size_t, ExecContext) override;
        void fill_strided(StorageRef, const StridedLayout&, ScalarOperand, ExecContext) override;
        void load_fill(StorageRef, size_t, ScalarOperand, ExecContext) override;
        void load_arange(StorageRef, size_t, ScalarOperand, ScalarOperand, ExecContext) override;

        float sum_scalar(StorageRef, size_t, ExecContext) override;
        float mean_scalar(StorageRef, size_t, ExecContext) override;
        float max_scalar(StorageRef, size_t, ExecContext) override;
        float min_scalar(StorageRef, size_t, ExecContext) override;
        void reduce(StorageRef, StorageRef, const StridedLayout&, const ReduceProgram&, ExecContext) override;
        void column_reduce(StorageRef, StorageRef, size_t, size_t, const ReduceProgram&, ExecContext) override;
        void strided_reduce(StorageRef, StorageRef, size_t, size_t, size_t, const ReduceProgram&, ExecContext) override;
        void fused_transform_reduce(StorageRef, StorageRef, size_t,
                                    const tensor_ops::FusedPointwiseOpChain&,
                                    const ReduceProgram&, std::span<const StorageRef>,
                                    ExecContext) override;
        void fused_segmented_transform_reduce(
            StorageRef, StorageRef, size_t, size_t,
            const tensor_ops::FusedPointwiseOpChain&, const ReduceProgram&,
            std::span<const StorageRef>, ExecContext) override;
        size_t count_nonzero_bool(StorageRef, size_t, ExecContext) override;
        size_t count_nonzero_float(StorageRef, size_t, ExecContext) override;
        bool has_nan(StorageRef, size_t, ExecContext) override;
        bool has_inf(StorageRef, size_t, ExecContext) override;
        void cumsum(StorageRef, const StridedLayout&, int, ExecContext) override;
        void sort_1d(StorageRef, StorageRef, size_t, const SortProgram&, ExecContext) override;
        void sort_2d(StorageRef, StorageRef, const SortProgram&, ExecContext) override;
        void sgemm(StorageRef, StorageRef, StorageRef, const GemmProgram&, ExecContext) override;
        void sgemm_tn(StorageRef, StorageRef, StorageRef, const GemmProgram&, ExecContext) override;
        void sgemm_batched(StorageRef, StorageRef, StorageRef, const GemmProgram&, ExecContext) override;
        void sgemm_bias_relu(StorageRef, StorageRef, StorageRef, StorageRef,
                             const GemmProgram&, ExecContext) override;
        void dot_product(StorageRef, StorageRef, StorageRef, size_t, ExecContext) override;
        void diag(StorageRef, StorageRef, size_t, ExecContext) override;
        void eye(StorageRef, size_t, size_t, ExecContext) override;
        void cdist(StorageRef, StorageRef, StorageRef, size_t, size_t, size_t,
                   float, ExecContext) override;
        void project_points(StorageRef, StorageRef, size_t, const PointProjection&,
                            const StorageRef*, size_t, const StorageRef*,
                            const StorageRef*, size_t, ExecContext) override;
        void radius_neighbors(StorageRef, StorageRef, StorageRef, StorageRef, StorageRef,
                              size_t, size_t, float, ExecContext) override;
        void mark_points_2d(StorageRef, StorageRef, size_t, const PointRegion2D&,
                            const StorageRef*, size_t, ExecContext) override;
        void filter_points(StorageRef mask, const PointFilterProgram& program, ExecContext context) override;
        void update_labels(StorageRef, StorageRef, const LabelUpdateProgram&, ExecContext) override;
        void ppisp_apply(StorageRef input, StorageRef output, int width, int height, const PpispParams& params, ExecContext context) override;
        void environment_composite(StorageRef rgb, StorageRef alpha, StorageRef environment, StorageRef output, const EnvironmentCompositeParams& params, ExecContext context) override;
        Tensor image_undistort(const Tensor& input, const UndistortParams& params, bool mask, ExecContext context) override;
        Tensor image_resize_prior(const Tensor& input, int height, int width, bool normal, ExecContext context) override;
        void histogram_u8(StorageRef, StorageRef, size_t, ExecContext) override;
        void affine_splat_geometry(StorageRef, StorageRef, StorageRef, StorageRef, const splat_transform::LinearTransform&, size_t, ExecContext) override;
        void sh_codec(StorageRef, StorageRef, const ShCodecProgram&, ExecContext) override;
        Tensor morton_sort(const Tensor& positions, Tensor* sorted_keys, ExecContext context) override;
        std::tuple<Tensor, Tensor> kmeans_sh(const Tensor& shN, int n_points, int sh_coeffs, int k,
                                             int iterations, bool fast, ExecContext context) override;
        void assign_sh3(const Tensor& shN, const Tensor& centroids, const Tensor& norms, Tensor& labels,
                        bool fast, bool have_labels, ExecContext context) override;
        void decimate_candidates(const Tensor& position, const Tensor& rotation, const Tensor& scale,
                                 const Tensor& opacity, const Tensor& dc, const Tensor& sh, int rest,
                                 std::vector<uint32_t>& idx, std::vector<float>& cost, ExecContext context) override;
        DecimateMerge decimate_merge(const Tensor& position, const Tensor& rotation, const Tensor& scale,
                                     const Tensor& opacity, const Tensor& dc, const Tensor& sh, int rest,
                                     const std::vector<int>& member_group, const std::vector<uint32_t>& minimum,
                                     const std::vector<uint32_t>& members, const std::vector<uint32_t>& offsets,
                                     size_t removed, ExecContext context) override;
        void max_pool2d(StorageRef, StorageRef, const PoolProgram&, ExecContext) override;
        void adaptive_avg_pool2d(StorageRef, StorageRef, const PoolProgram&, ExecContext) override;
        void inference(StorageRef, StorageRef, const InferenceProgram&, ExecContext) override;
        bool nn_kernels() const override { return false; }
        void nn_linear(StorageRef, StorageRef, std::optional<StorageRef>, std::optional<StorageRef>,
                       std::optional<StorageRef>, StorageRef, const LinearProgram&, ExecContext) override;
        void nn_attention(StorageRef, StorageRef, StorageRef, std::optional<StorageRef>, StorageRef,
                          const AttentionProgram&, ExecContext) override;
        void nn_norm(StorageRef, StorageRef, std::optional<StorageRef>, StorageRef, const NormProgram&,
                     ExecContext) override;
        void bias_add(StorageRef, StorageRef, StorageRef, int, int, int, ExecContext) override;
        void bias_relu(StorageRef, StorageRef, StorageRef, int, int, int, ExecContext) override;
        void relu(StorageRef, StorageRef, int, ExecContext) override;
        void uniform(StorageRef, const RandomProgram&, ExecContext) override;
        void bernoulli(StorageRef, const RandomProgram&, ExecContext) override;
        void randint(StorageRef, const RandomProgram&, ExecContext) override;
        void multinomial(StorageRef, StorageRef, const RandomProgram&, ExecContext) override;
        void normal(StorageRef, StorageRef, const RandomProgram&, ExecContext) override;

        void gather(StorageRef, StorageRef, StorageRef, const StridedLayout&,
                    const StridedLayout&, const IndexProgram&, ExecContext) override;
        void gather_fused_unary(StorageRef, StorageRef, StorageRef, PointwiseOp,
                                const IndexProgram&, ExecContext) override;
        void take(StorageRef, StorageRef, StorageRef, const IndexProgram&, ExecContext) override;
        void index_select(StorageRef, StorageRef, StorageRef, const StridedLayout&,
                          const IndexProgram&, ExecContext) override;
        void scatter(StorageRef, StorageRef, StorageRef, const StridedLayout&,
                     const StridedLayout&, const IndexProgram&, ExecContext) override;
        void index_copy(StorageRef, StorageRef, StorageRef, const StridedLayout&,
                        const IndexProgram&, ExecContext) override;
        void index_add(StorageRef, StorageRef, StorageRef, const StridedLayout&,
                       const IndexProgram&, ExecContext) override;
        void index_fill(StorageRef, StorageRef, const StridedLayout&,
                        const IndexProgram&, ScalarOperand, ExecContext) override;
        void index_put(StorageRef, StorageRef, StorageRef, const IndexProgram&, ExecContext) override;
        void strided_scatter(StorageRef, StorageRef, const StridedLayout&, ExecContext) override;
        void strided_scatter_immediate(StorageRef, StorageRef, const StridedLayout&, ExecContext) override;
        void strided_scatter_int32_to_float32(StorageRef, StorageRef,
                                              const StridedLayout&, ExecContext) override;
        void masked_fill(StorageRef, StorageRef, const MaskProgram&, ExecContext) override;
        size_t masked_select(StorageRef, StorageRef, StorageRef, const MaskProgram&, ExecContext) override;
        void masked_scatter(StorageRef, StorageRef, StorageRef, const MaskProgram&, ExecContext) override;
        void and_live(StorageRef, StorageRef, const MaskProgram&, ExecContext) override;
        void where(StorageRef, StorageRef, StorageRef, StorageRef,
                   const StridedLayout&, const StridedLayout&, const StridedLayout&,
                   const StridedLayout&, ExecContext) override;
        size_t nonzero(StorageRef, StorageRef, const MaskProgram&, ExecContext) override;
        size_t nonzero_bool(StorageRef, StorageRef, const MaskProgram&, ExecContext) override;
        void strided_copy(StorageRef, StorageRef, const StridedLayout&, ExecContext) override;
        void strided_copy_immediate(StorageRef, StorageRef, const StridedLayout&, ExecContext) override;
        void strided_upload(StorageRef, StorageRef, const StridedLayout&, ExecContext) override;
        void cat_last_dim(StorageRef, std::span<const StorageRef>,
                          std::span<const StridedLayout>, size_t, size_t, size_t,
                          ExecContext) override;
        void cat_middle_dim(StorageRef, std::span<const StorageRef>,
                            std::span<const StridedLayout>, size_t, size_t, int,
                            size_t, ExecContext) override;
        void pad(StorageRef, StorageRef, const StridedLayout&, const StridedLayout&,
                 const std::array<size_t, MAX_TENSOR_RANK>&, ExecContext) override;

        StorageRef allocate(size_t, size_t, ExecContext) override;
        void deallocate(StorageRef, ExecContext) noexcept override;
        void record_stream(StorageRef, ExecContext) override;
        void release_stream(ExecContext) override;
        void rehome_stream(StorageRef, ExecContext) override;
        void trim() override;
        void trim_if_reserved_unused_exceeds(size_t) override;
        MemoryInfo stats() override;
        void shutdown() override;
        void set_allocation_iteration(int) override;
        void record_tensor_allocation(StorageRef, const StridedLayout&, size_t) override;
        void copy_host_to_device(const CopyRequest&) override;
        void copy_device_to_host(const CopyRequest&) override;
        void copy_device_to_device(const CopyRequest&) override;
        void memset(const FillRequest&) override;

        std::unique_ptr<ReadbackBuffer> create_readback_buffer() override;
        void synchronize_stream(ExecContext) override;
        void synchronize_device() override;
        void device_barrier() override;
        void wait_for(SyncToken) override;
        SyncToken bridge(ExecContext, ExecContext) override;
        PointerClass classify_pointer(const void*) override;
        bool stream_is_capturing(ExecContext) override;
    };

} // namespace lfs::core::internal
