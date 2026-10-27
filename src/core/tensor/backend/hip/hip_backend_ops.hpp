/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "device_identity_api.h"
#include "runtime_api.h"
#include <core/detail/gpu_backend_ops.hpp>
#include <core/gpu_device_info.hpp>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>
namespace lfs::core::internal {
    // Preserve native failure information across Core's SDK-free boundary.
    class HipBackendError : public std::runtime_error {
    public:
        int native_code;
        HipBackendError(int code, const std::string& message)
            : std::runtime_error(message), native_code(code) {}
        bool checked_execution_error() const noexcept {
            return native_code == LFS_HIP_ERROR_INVALID_INDEX || native_code == LFS_HIP_ERROR_MASK_COUNT;
        }
    };
    struct ExpressionCacheStats;
    bool hip_backend_is_live() noexcept;
    int select_hip_device(const std::filesystem::path& module, int requested);
    // Hardware inventory loads only the versioned module; no Tensor allocator
    // or execution context is constructed during preferences discovery.
    LFS_CORE_API int hip_device_count(const std::filesystem::path& module);
    LFS_CORE_API LfsHipDevice hip_device_info(const std::filesystem::path& module, int ordinal);
    LFS_CORE_API LfsHipDeviceIdentity hip_device_identity(const std::filesystem::path& module, int ordinal);
    std::optional<GpuDeviceInfo> hip_gpu_device_info(const std::filesystem::path&, int ordinal);
    // Opt-in native Tensor adapter. SDK types stay behind the private C ABI.
    class HipBackendOps final : public GpuBackendOps {
    public:
        LFS_CORE_API HipBackendOps(const std::filesystem::path& module, int device, size_t max_cached_bytes = 64 * 1024 * 1024);
        ~HipBackendOps() override;
        LFS_CORE_API void ensure_initialized();
        std::unique_ptr<BackendUploadBuffer> create_upload_buffer() override;
        std::unique_ptr<BackendFence> create_fence() override;
        std::unique_ptr<BackendTimestamps> create_timestamps(size_t count) override;
        std::unique_ptr<BackendReadbackSlot> create_readback_slot(size_t bytes) override;
        LFS_CORE_API BackendQueueLease execution_lease(ExecContext) override;
        bool drain_queue_for_destruction() noexcept override;
        ExpressionCacheStats expression_cache_stats() override;
        LFS_CORE_API uint64_t storage_address(StorageRef, size_t required_bytes);

        // Non-owning view of a live Tensor allocation. The arena owner must
        // outlive the returned view, as for the existing raw-view APIs.
        LFS_CORE_API StorageRef borrow_native_range(void* pointer, size_t bytes) override;
        LFS_CORE_API bool queue_ready(ExecContext) override;
        void push_range(const char*) override;
        void pop_range() override;
        LFS_CORE_API void shared_image(const LfsHipSharedImage&, StorageRef source, StorageRef destination, StorageRef auxiliary, ExecContext);
        void enqueue_host_callback(void (*callback)(void*), void* user, ExecContext) override;
        LFS_CORE_API Tensor shared_image_resize(const Tensor&, int height, int width, unsigned mode, int kernel_size, ExecContext);
        Tensor image_warp(const Tensor&, const UndistortParams&, int mode, bool inverse, Tensor* validity, ExecContext) override;
        void compiled_expression(const ExpressionLaunch& launch, ExecContext context) override;
        void unary(const PointwiseProgram& program, StorageRef input, StorageRef output, size_t count, ExecContext context) override;
        void binary(const PointwiseProgram& program, StorageRef lhs, StorageRef rhs, StorageRef output, size_t count, ExecContext context) override;
        void scalar(const PointwiseProgram& program, StorageRef input, StorageRef output, size_t count, ExecContext context) override;
        void broadcast_binary(const PointwiseProgram& program, StorageRef lhs, const StridedLayout& lhs_layout, StorageRef rhs, const StridedLayout& rhs_layout, StorageRef output, const StridedLayout& output_layout, ExecContext context) override;
        void fused_pointwise_chain(StorageRef input, StorageRef output, size_t count, const tensor_ops::FusedPointwiseOpChain& chain, std::span<const StorageRef> rhs_storages, ExecContext context) override;
        void clamp_scalar(StorageRef data, ScalarOperand minimum, ScalarOperand maximum, size_t count, ExecContext context) override;
        void clamp_fused(StorageRef input, StorageRef output, ScalarOperand minimum, ScalarOperand maximum, size_t count, ExecContext context) override;
        void clamp_scalar_int(StorageRef data, ScalarOperand minimum, ScalarOperand maximum, size_t count, ExecContext context) override;
        void index_cast(StorageRef input, StorageRef output, size_t count, size_t extent, ExecContext context) override;
        void convert_type(StorageRef input, StorageRef output, size_t count, ExecContext context) override;
        void fill_strided(StorageRef output, const StridedLayout& layout, ScalarOperand value, ExecContext context) override;
        void load_fill(StorageRef output, size_t count, ScalarOperand value, ExecContext context) override;
        void load_arange(StorageRef output, size_t count, ScalarOperand start, ScalarOperand step, ExecContext context) override;
        float sum_scalar(StorageRef input, size_t count, ExecContext context) override;
        float mean_scalar(StorageRef input, size_t count, ExecContext context) override;
        float max_scalar(StorageRef input, size_t count, ExecContext context) override;
        float min_scalar(StorageRef input, size_t count, ExecContext context) override;
        void reduce(StorageRef input, StorageRef output, const StridedLayout& input_layout, const ReduceProgram& program, ExecContext context) override;
        void column_reduce(StorageRef input, StorageRef output, size_t rows, size_t columns, const ReduceProgram& program, ExecContext context) override;
        void strided_reduce(StorageRef input, StorageRef output, size_t outer_size, size_t reduce_size, size_t inner_size, const ReduceProgram& program, ExecContext context) override;
        void fused_transform_reduce(StorageRef input, StorageRef output, size_t count, const tensor_ops::FusedPointwiseOpChain& chain, const ReduceProgram& program, std::span<const StorageRef> rhs_storages, ExecContext context) override;
        void fused_segmented_transform_reduce(StorageRef input, StorageRef output, size_t segment_count, size_t segment_size, const tensor_ops::FusedPointwiseOpChain& chain, const ReduceProgram& program, std::span<const StorageRef> rhs_storages, ExecContext context) override;
        size_t count_nonzero_bool(StorageRef input, size_t count, ExecContext context) override;
        size_t count_nonzero_float(StorageRef input, size_t count, ExecContext context) override;
        bool has_nan(StorageRef input, size_t count, ExecContext context) override;
        bool has_inf(StorageRef input, size_t count, ExecContext context) override;
        bool arg_extreme(StorageRef input, StorageRef values, StorageRef indices, const ArgExtremeProgram& program, ExecContext context) override;
        void cumsum(StorageRef data, const StridedLayout& layout, int dim, ExecContext context) override;
        void sort_1d(StorageRef values, StorageRef indices, size_t count, const SortProgram& program, ExecContext context) override;
        void sort_2d(StorageRef values, StorageRef indices, const SortProgram& program, ExecContext context) override;
        void sgemm(StorageRef lhs, StorageRef rhs, StorageRef output, const GemmProgram& program, ExecContext context) override;
        void sgemm_tn(StorageRef lhs, StorageRef rhs, StorageRef output, const GemmProgram& program, ExecContext context) override;
        void sgemm_batched(StorageRef lhs, StorageRef rhs, StorageRef output, const GemmProgram& program, ExecContext context) override;
        void sgemm_bias_relu(StorageRef lhs, StorageRef rhs, StorageRef bias, StorageRef output, const GemmProgram& program, ExecContext context) override;
        void dot_product(StorageRef lhs, StorageRef rhs, StorageRef output, size_t count, ExecContext context) override;
        void diag(StorageRef diagonal, StorageRef output, size_t count, ExecContext context) override;
        void eye(StorageRef output, size_t rows, size_t columns, ExecContext context) override;
        void cdist(StorageRef lhs, StorageRef rhs, StorageRef output, size_t lhs_rows, size_t rhs_rows, size_t columns, float p, ExecContext context) override;
        void nearest_point_indices(StorageRef, StorageRef, StorageRef, StorageRef, StorageRef, size_t, size_t, size_t, float, ExecContext) override;
        void camera_frustum_counts(StorageRef, StorageRef, StorageRef, size_t, size_t, float, ExecContext) override;
        void radius_neighbors(StorageRef points, StorageRef references, StorageRef heads, StorageRef next, StorageRef output, size_t count, size_t buckets, float radius, bool exclude_self, std::optional<StorageRef> queries, ExecContext context) override;
        void project_points(StorageRef points, StorageRef output, size_t count, const PointProjection& projection, const StorageRef* transforms, size_t transform_count, const StorageRef* indices, const StorageRef* visibility, size_t visibility_count, ExecContext context) override;
        void radius_neighbor_counts(StorageRef points, StorageRef references, StorageRef heads, StorageRef next, StorageRef output, size_t count, size_t buckets, float radius, int32_t max_count, std::optional<StorageRef> queries, ExecContext context) override;
        void radius_neighbor_min(StorageRef points, StorageRef values, StorageRef references, StorageRef heads, StorageRef next, StorageRef output, size_t count, size_t buckets, float radius, std::optional<StorageRef> radii, ExecContext context) override;
        void point_neighbor_spacing(StorageRef points, StorageRef references, StorageRef heads, StorageRef next, StorageRef output, size_t count, size_t buckets, float cell_width, ExecContext context) override;
        void rasterize_points(const PointRasterProgram& program, ExecContext context) override;
        void mark_points_2d(StorageRef mask, StorageRef points, size_t count, const PointRegion2D& region, const StorageRef* geometry, size_t geometry_count, ExecContext context) override;
        void filter_points(StorageRef mask, const PointFilterProgram& program, ExecContext context) override;
        void update_labels(StorageRef output, StorageRef selected, const LabelUpdateProgram& program, ExecContext context) override;
        void ppisp_apply(StorageRef input, StorageRef output, int width, int height, const PpispParams& params, ExecContext context) override;
        void environment_composite(StorageRef rgb, StorageRef alpha, StorageRef environment, StorageRef output, const EnvironmentCompositeParams& params, ExecContext context) override;
        Tensor image_undistort(const Tensor& input, const UndistortParams& params, bool mask, ExecContext context) override;
        Tensor image_resize_prior(const Tensor& input, int height, int width, bool normal, ExecContext context) override;
        void affine_splat_geometry(StorageRef, StorageRef, StorageRef, StorageRef, const splat_transform::LinearTransform&, size_t, std::optional<StorageRef>, ExecContext) override;
        void sh_codec(StorageRef, StorageRef, const ShCodecProgram&, ExecContext) override;
        Tensor morton_sort(const Tensor& positions, Tensor* sorted_keys, ExecContext context) override;
        std::tuple<Tensor, Tensor> kmeans_sh(const Tensor& shN, int n_points, int sh_coeffs, int k, int iterations, bool fast, ExecContext context) override;
        void assign_sh3(const Tensor& shN, const Tensor& centroids, const Tensor& norms, Tensor& labels, bool fast, bool have_labels, ExecContext context) override;
        void decimate_candidates(const Tensor& position, const Tensor& rotation, const Tensor& scale, const Tensor& opacity, const Tensor& dc, const Tensor& sh, int rest, std::vector<uint32_t>& idx, std::vector<float>& cost, ExecContext context) override;
        DecimateMerge decimate_merge(const Tensor& position, const Tensor& rotation, const Tensor& scale, const Tensor& opacity, const Tensor& dc, const Tensor& sh, int rest, const std::vector<int>& member_group, const std::vector<uint32_t>& minimum, const std::vector<uint32_t>& members, const std::vector<uint32_t>& offsets, size_t removed, ExecContext context) override;
        void histogram_u8(StorageRef values, StorageRef counts, size_t size, ExecContext context) override;
        void max_pool2d(StorageRef input, StorageRef output, const PoolProgram& program, ExecContext context) override;
        void adaptive_avg_pool2d(StorageRef input, StorageRef output, const PoolProgram& program, ExecContext context) override;
        void inference(StorageRef input, StorageRef output, const InferenceProgram& program, ExecContext context) override;
        bool nn_kernels() const override;
        void nn_linear(StorageRef a, StorageRef w, std::optional<StorageRef> bias, std::optional<StorageRef> scale, std::optional<StorageRef> residual, StorageRef output, const LinearProgram& program, ExecContext context) override;
        void nn_attention(StorageRef q, StorageRef k, StorageRef v, std::optional<StorageRef> mask, StorageRef output, const AttentionProgram& program, ExecContext context) override;
        void nn_norm(StorageRef input, StorageRef weight, std::optional<StorageRef> bias, StorageRef output, const NormProgram& program, ExecContext context) override;
        void nn_conv2d(StorageRef input, StorageRef weight, std::optional<StorageRef> bias, StorageRef output, const ConvProgram& program, ExecContext context) override;
        void bias_add(StorageRef input, StorageRef bias, StorageRef output, int count, int channels, int spatial_size, ExecContext context) override;
        void bias_relu(StorageRef input, StorageRef bias, StorageRef output, int count, int channels, int spatial_size, ExecContext context) override;
        void relu(StorageRef input, StorageRef output, int count, ExecContext context) override;
        void uniform(StorageRef output, const RandomProgram& program, ExecContext context) override;
        void bernoulli(StorageRef output, const RandomProgram& program, ExecContext context) override;
        void randint(StorageRef output, const RandomProgram& program, ExecContext context) override;
        void multinomial(StorageRef weights, StorageRef output, const RandomProgram& program, ExecContext context) override;
        void normal(StorageRef output, StorageRef odd_count_scratch, const RandomProgram& program, ExecContext context) override;
        void gather(StorageRef input, StorageRef indices, StorageRef output, const StridedLayout& input_layout, const StridedLayout& index_layout, const IndexProgram& program, ExecContext context) override;
        void gather_fused_unary(StorageRef input, StorageRef indices, StorageRef output, PointwiseOp unary, const IndexProgram& program, ExecContext context) override;
        void index_select(StorageRef input, StorageRef indices, StorageRef output, const StridedLayout& input_layout, const IndexProgram& program, ExecContext context) override;
        void scatter(StorageRef output, StorageRef indices, StorageRef source, const StridedLayout& output_layout, const StridedLayout& index_layout, const IndexProgram& program, ExecContext context) override;
        void index_copy(StorageRef output, StorageRef indices, StorageRef source, const StridedLayout& output_layout, const IndexProgram& program, ExecContext context) override;
        void index_add(StorageRef output, StorageRef indices, StorageRef source, const StridedLayout& output_layout, const IndexProgram& program, ExecContext context) override;
        void index_fill(StorageRef output, StorageRef indices, const StridedLayout& output_layout, const IndexProgram& program, ScalarOperand value, ExecContext context) override;
        void index_put(StorageRef output, StorageRef indices, StorageRef values, const IndexProgram& program, ExecContext context) override;
        void strided_scatter(StorageRef input, StorageRef output, const StridedLayout& output_layout, ExecContext context) override;
        void strided_scatter_immediate(StorageRef input, StorageRef output, const StridedLayout& output_layout, ExecContext context) override;
        void strided_scatter_int32_to_float32(StorageRef input, StorageRef output, const StridedLayout& output_layout, ExecContext context) override;
        void masked_fill(StorageRef output, StorageRef mask, const MaskProgram& program, ExecContext context) override;
        size_t masked_select(StorageRef input, StorageRef mask, StorageRef output, const MaskProgram& program, ExecContext context) override;
        void masked_scatter(StorageRef output, StorageRef mask, StorageRef source, const MaskProgram& program, ExecContext context) override;
        void and_live(StorageRef mask, StorageRef live_mask, const MaskProgram& program, ExecContext context) override;
        void where(StorageRef condition, StorageRef x, StorageRef y, StorageRef output, const StridedLayout& condition_layout, const StridedLayout& x_layout, const StridedLayout& y_layout, const StridedLayout& output_layout, ExecContext context) override;
        size_t nonzero(StorageRef input, StorageRef output, const MaskProgram& program, ExecContext context) override;
        size_t nonzero_bool(StorageRef input, StorageRef output, const MaskProgram& program, ExecContext context) override;
        void strided_copy(StorageRef input, StorageRef output, const StridedLayout& input_layout, ExecContext context) override;
        void strided_copy_immediate(StorageRef input, StorageRef output, const StridedLayout& input_layout, ExecContext context) override;
        void strided_upload(StorageRef host_input, StorageRef output, const StridedLayout& input_layout, ExecContext context) override;
        void cat_last_dim(StorageRef output, std::span<const StorageRef> inputs, std::span<const StridedLayout> layouts, size_t num_rows, size_t row_size, size_t element_size, ExecContext context) override;
        void cat_middle_dim(StorageRef output, std::span<const StorageRef> inputs, std::span<const StridedLayout> layouts, size_t outer_size, size_t inner_size, int dim, size_t element_size, ExecContext context) override;
        void pad(StorageRef input, StorageRef output, const StridedLayout& input_layout, const StridedLayout& output_layout, const std::array<size_t, MAX_TENSOR_RANK>& pad_before, ExecContext context) override;
        StorageRef allocate(size_t bytes, size_t alignment, ExecContext context) override;
        void deallocate(StorageRef storage, ExecContext context) noexcept override;
        void record_stream(StorageRef storage, ExecContext context) override;
        void release_stream(ExecContext context) override;
        void rehome_stream(StorageRef storage, ExecContext context) override;
        void trim() override;
        void trim_if_reserved_unused_exceeds(size_t threshold_bytes) override;
        MemoryInfo stats() override;
        void shutdown() override;
        void set_allocation_iteration(int iteration) override;
        void record_tensor_allocation(StorageRef storage, const StridedLayout& layout, size_t bytes) override;
        void copy_host_to_device(const CopyRequest& request) override;
        void copy_device_to_host(const CopyRequest& request) override;
        void copy_device_to_device(const CopyRequest& request) override;
        void memset(const FillRequest& request) override;
        std::unique_ptr<ReadbackBuffer> create_readback_buffer() override;
        void synchronize_stream(ExecContext context) override;
        void synchronize_device() override;
        void device_barrier() override;
        void wait_for(SyncToken token) override;
        SyncToken bridge(ExecContext producer, ExecContext consumer) override;
        PointerClass classify_pointer(const void* pointer) override;
        bool stream_is_capturing(ExecContext context) override;

        void where_scalar(StorageRef mask, StorageRef source, StorageRef output, size_t count, float value, ExecContext) override;
        bool radius_connected_components(StorageRef, StorageRef, StorageRef, StorageRef, StorageRef, size_t, size_t, float, ExecContext) override;
        bool point_tree_counts(StorageRef, StorageRef, StorageRef, StorageRef, StorageRef, std::optional<StorageRef>, StorageRef, const PointTreeProgram&, ExecContext) override;
        bool point_tree_components(StorageRef, StorageRef, StorageRef, StorageRef, StorageRef, StorageRef, StorageRef, StorageRef, const PointTreeProgram&, ExecContext) override;
        bool simplify_merge(const std::array<StorageRef, 5>&, StorageRef, StorageRef, const std::array<StorageRef, 5>&, const SimplifyMergeProgram&, ExecContext) override;
        bool triangle_tree_parity(StorageRef, StorageRef, StorageRef, StorageRef, StorageRef, const PointTreeProgram&, ExecContext) override;
        bool point_tree_spacing(StorageRef, StorageRef, StorageRef, StorageRef, StorageRef, const PointTreeProgram&, ExecContext) override;

    private:
        friend class HipExportKernels;
        void launch_export(const char*, uint32_t, std::span<const std::byte>, std::span<const StorageRef>, std::span<const StorageRef>, size_t);
        void launch_point_tree(StorageRef, StorageRef, StorageRef, StorageRef, std::optional<StorageRef>, std::optional<StorageRef>, StorageRef, const PointTreeProgram&, bool, ExecContext);
        void launch_shared_resize(const LfsHipSharedResize&, StorageRef source, StorageRef destination, StorageRef coefficients_x, StorageRef coefficients_y, ExecContext);
        void launch_inverse_lens(StorageRef, const UndistortParams&, ExecContext);
        void launch_shared_warp(StorageRef, StorageRef, const UndistortParams&, unsigned channels, unsigned mode, ExecContext);
        void launch_inverse_warp(StorageRef, StorageRef, StorageRef, const UndistortParams&, unsigned channels, unsigned mode, ExecContext);
        void launch_gemm(StorageRef, StorageRef, StorageRef, StorageRef, const GemmProgram&, bool transpose, bool relu, ExecContext);
        void launch_rgb_warp(StorageRef, StorageRef, const UndistortParams&, ExecContext);
        void launch_radius(StorageRef, StorageRef, StorageRef, StorageRef, StorageRef, size_t, size_t,
                           float, unsigned, int32_t, bool, std::optional<StorageRef>,
                           std::optional<StorageRef>, std::optional<StorageRef>, ExecContext);
        struct Impl;
        std::shared_ptr<Impl> impl_;
    };
} // namespace lfs::core::internal
