/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "mrnf.hpp"
#include "core/alloc_counter.hpp"
#include "core/assert.hpp"
#include "core/camera.hpp"
#include "core/cuda/sh_layout.cuh"
#include "core/cuda_error.hpp"
#include "core/logger.hpp"
#include "core/sh_value_quant.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "edge_rasterizer.hpp"
#include "kernels/densification_kernels.hpp"
#include "kernels/image_kernels.hpp"
#include "kernels/mrnf_kernels.hpp"
#include "lfs/training/mean_step_scale.cuh"
#include "lfs/training/morton_reorder.hpp"
#include "lfs/training/perf_bench.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include "strategy_utils.hpp"
#include "training/dataset.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lfs::training {

    namespace {

        constexpr int MRNF_RATIO_RANK_LOG_INTERVAL = 25;

        constexpr float MRNF_EDGE_SCORE_WEIGHT = 0.25f;
        constexpr int MRNF_EDGE_MIN_VIEW_SAMPLES = 10;
        constexpr int MRNF_BOUNDS_RECOMPUTE_INTERVAL_REFINES = 5;
        constexpr float MRNF_RAW_OPACITY_PRUNE_THRESHOLD = -5.54126358f; // logit(1 / 255)
        constexpr float MRNF_LOG_MIN_SCALE_THRESHOLD = -23.0258509f;     // log(1e-10)
        constexpr float MRNF_SH_C0 = 0.28209479177387814f;
        constexpr float MRNF_EXPLORE_SCORE_FLOOR = 0.05f;
        constexpr float MRNF_PROJECT_NEAR = 0.01f;

        [[nodiscard]] lfs::core::Tensor squeeze_leading_ones(lfs::core::Tensor tensor) {
            while (tensor.is_valid() && tensor.ndim() > 1 && tensor.shape()[0] == 1) {
                tensor = tensor.squeeze(0);
            }
            return tensor;
        }

        [[nodiscard]] bool is_cuda_image(const lfs::core::Tensor& tensor) {
            return tensor.is_valid() &&
                   tensor.device() == lfs::core::Device::CUDA &&
                   (tensor.dtype() == lfs::core::DataType::Float32 ||
                    tensor.dtype() == lfs::core::DataType::UInt8);
        }

        [[nodiscard]] lfs::core::Tensor to_float01_cuda(const lfs::core::Tensor& tensor) {
            auto value = squeeze_leading_ones(tensor);
            if (value.dtype() == lfs::core::DataType::UInt8) {
                value = value.to(lfs::core::DataType::Float32) / 255.0f;
            }
            return value;
        }

        [[nodiscard]] bool same_spatial(const lfs::core::Tensor& a, const lfs::core::Tensor& b) {
            if (!a.is_valid() || !b.is_valid() || a.ndim() < 2 || b.ndim() < 2) {
                return false;
            }
            return a.shape()[a.ndim() - 2] == b.shape()[b.ndim() - 2] &&
                   a.shape()[a.ndim() - 1] == b.shape()[b.ndim() - 1];
        }

        void ensure_cuda_float(lfs::core::Tensor& tensor, const lfs::core::TensorShape& shape) {
            if (tensor.is_valid() &&
                tensor.device() == lfs::core::Device::CUDA &&
                tensor.dtype() == lfs::core::DataType::Float32 &&
                tensor.shape() == shape) {
                return;
            }
            tensor = lfs::core::Tensor::empty(shape, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        }

        void copy_into_cached(lfs::core::Tensor& dst, const lfs::core::Tensor& src) {
            auto value = to_float01_cuda(src).contiguous();
            ensure_cuda_float(dst, value.shape());
            if (dst.data_ptr() != value.data_ptr() && value.bytes() > 0) {
                LFS_CUDA_CHECK_MSG(
                    cudaMemcpy(dst.data_ptr(), value.data_ptr(), value.bytes(),
                               cudaMemcpyDeviceToDevice),
                    "MRNF cache seed view copy");
            }
        }

        [[nodiscard]] bool fill_mean_abs_error_hw(
            const lfs::core::Tensor& image,
            const lfs::core::Tensor& target,
            lfs::core::Tensor& out_hw) {
            auto pred = to_float01_cuda(image).contiguous();
            auto gt = to_float01_cuda(target).contiguous();
            if (pred.ndim() != 3 || gt.ndim() != 3) {
                return false;
            }
            const size_t channels = std::min(pred.shape()[0], gt.shape()[0]);
            const size_t height = pred.shape()[1];
            const size_t width = pred.shape()[2];
            if (channels == 0 || height == 0 || width == 0 ||
                gt.shape()[1] != height || gt.shape()[2] != width) {
                return false;
            }
            if (pred.shape()[0] != channels) {
                pred = pred.slice(0, 0, channels).contiguous();
            }
            if (gt.shape()[0] != channels) {
                gt = gt.slice(0, 0, channels).contiguous();
            }
            ensure_cuda_float(out_hw, lfs::core::TensorShape({height, width}));
            const int channels_i = static_cast<int>(channels);
            const int height_i = static_cast<int>(height);
            const int width_i = static_cast<int>(width);
            mrnf_strategy::launch_mean_abs_error_hw(
                pred.ptr<float>(),
                gt.ptr<float>(),
                channels_i,
                height_i,
                width_i,
                out_hw.ptr<float>());
            return true;
        }

        [[nodiscard]] lfs::core::Tensor flatten_hw(const lfs::core::Tensor& tensor) {
            auto value = squeeze_leading_ones(tensor);
            while (value.is_valid() && value.ndim() > 2 && value.shape()[value.ndim() - 1] == 1) {
                value = value.squeeze(-1);
            }
            if (value.ndim() == 3 && value.shape()[0] == 1) {
                value = value.squeeze(0);
            }
            if (value.ndim() != 2) {
                return {};
            }
            const size_t numel = value.numel();
            return value.reshape(lfs::core::TensorShape({numel}));
        }

        [[nodiscard]] float logit_clamped(const float p) {
            const float q = std::min(std::max(p, 1e-6f), 1.0f - 1e-6f);
            return std::log(q / (1.0f - q));
        }

        [[nodiscard]] std::size_t tensor_vram_required_bytes(
            const lfs::core::Tensor& tensor) noexcept {
            return tensor.is_valid() && tensor.device() == lfs::core::Device::CUDA
                       ? tensor.bytes()
                       : 0;
        }

        [[nodiscard]] std::size_t tensor_vram_allocated_bytes(
            const lfs::core::Tensor& tensor) noexcept {
            if (!tensor.is_valid() || tensor.device() != lfs::core::Device::CUDA) {
                return 0;
            }
            if (tensor.capacity() == 0 || tensor.ndim() == 0) {
                return tensor.bytes();
            }
            std::size_t row_elements = 1;
            for (std::size_t dim = 1; dim < tensor.ndim(); ++dim) {
                row_elements *= tensor.shape()[dim];
            }
            return tensor.capacity() * row_elements * lfs::core::dtype_size(tensor.dtype());
        }

        void publish_required_allocated_pair(
            lfs::diagnostics::VramProfiler& profiler,
            const std::string_view name,
            const std::size_t required_bytes,
            const std::size_t allocated_bytes) {
            const std::string prefix = std::string("vram.audit.mrnf.") + std::string(name);
            profiler.setGauge(prefix + ".required_bytes", static_cast<double>(required_bytes));
            profiler.setGauge(prefix + ".allocated_bytes", static_cast<double>(allocated_bytes));
        }

        [[nodiscard]] double compute_decay_gamma(const double start, const double end, const size_t steps) {
            if (steps == 0 || start <= 0.0 || end <= 0.0) {
                return 1.0;
            }
            return std::pow(end / start, 1.0 / static_cast<double>(steps));
        }

        void reset_vector_buffer(
            lfs::core::Tensor& tensor,
            const size_t size,
            const lfs::core::Device device,
            const size_t reserve_capacity = 0) {
            const size_t desired_capacity = reserve_capacity > 0 ? std::max(reserve_capacity, size) : size;
            const bool needs_new_tensor = !tensor.is_valid() ||
                                          tensor.ndim() != 1 ||
                                          tensor.device() != device ||
                                          tensor.dtype() != lfs::core::DataType::Float32;
            const auto make_fresh = [&]() {
                if (desired_capacity > size) {
                    tensor = lfs::core::Tensor::zeros_direct(lfs::core::TensorShape({size}), desired_capacity, device);
                } else {
                    tensor = lfs::core::Tensor::zeros({size}, device);
                }
            };

            if (needs_new_tensor) {
                make_fresh();
                return;
            }

            const size_t current_size = tensor.numel();
            if (current_size == 0) {
                if (size == 0) {
                    if (desired_capacity > tensor.capacity()) {
                        make_fresh();
                    } else {
                        tensor.zero_();
                    }
                } else if (tensor.capacity() >= desired_capacity) {
                    tensor.append_zeros(size);
                } else {
                    make_fresh();
                }
                return;
            }

            if (current_size == size) {
                if (desired_capacity > size && tensor.capacity() < desired_capacity) {
                    tensor.reserve(desired_capacity);
                }
                tensor.zero_();
                return;
            }

            if (current_size < size) {
                if (tensor.capacity() < desired_capacity) {
                    tensor.reserve(desired_capacity);
                }
                tensor.append_zeros(size - current_size);
                tensor.zero_();
                return;
            }

            make_fresh();
        }

        [[nodiscard]] bool has_zero_dimension(const lfs::core::TensorShape& shape) {
            for (size_t i = 0; i < shape.rank(); ++i) {
                if (shape[i] == 0) {
                    return true;
                }
            }
            return false;
        }

        void reset_optimizer_state_at_indices(
            AdamOptimizer& optimizer,
            const ParamType param_type,
            const lfs::core::Tensor& indices,
            const uint32_t shN_layout_rest = 0) {
            if (!indices.is_valid() || indices.numel() == 0) {
                return;
            }

            auto* state = optimizer.get_state_mutable(param_type);
            if (!state) {
                return;
            }

            // Joint codec: zero packed moment rows via optimizer GPU path (uint8
            // index_put_ is not supported). Grad zero still runs below for non-SH.
            if (state->is_joint()) {
                // Host vector of indices for AdamOptimizer::reset_state_at_indices
                auto idx_cpu = indices.cpu();
                std::vector<int64_t> host_idx;
                host_idx.reserve(indices.numel());
                if (idx_cpu.dtype() == lfs::core::DataType::Int64) {
                    const auto* p = idx_cpu.ptr<int64_t>();
                    host_idx.assign(p, p + indices.numel());
                } else if (idx_cpu.dtype() == lfs::core::DataType::Int32) {
                    const auto* p = idx_cpu.ptr<int32_t>();
                    for (size_t i = 0; i < indices.numel(); ++i)
                        host_idx.push_back(static_cast<int64_t>(p[i]));
                }
                if (!host_idx.empty()) {
                    optimizer.reset_state_at_indices(param_type, host_idx);
                }
                if (param_type == ParamType::ShN) {
                    const auto layout_rest = shN_layout_rest;
                    if (layout_rest != 0 && state->grad.is_valid() && state->grad.numel() > 0) {
                        auto idx_i32 = indices.dtype() == lfs::core::DataType::Int32
                                           ? indices
                                           : indices.to(lfs::core::DataType::Int32);
                        lfs::core::shN_swizzled_zero_at_indices(
                            state->grad.ptr<float>(), idx_i32.ptr<int>(), idx_i32.numel(), layout_rest);
                    }
                    return;
                }
                // continue to grad zero for non-SH
            }

            if (state->grad.is_valid() && state->grad.numel() > 0) {
                const auto& shape = state->grad.shape();
                if (has_zero_dimension(shape)) {
                    return;
                }
                std::vector<size_t> dims = {indices.numel()};
                for (size_t i = 1; i < shape.rank(); ++i) {
                    dims.push_back(shape[i]);
                }
                auto zeros = lfs::core::Tensor::zeros(lfs::core::TensorShape(dims), state->grad.device());
                state->grad.index_put_(indices, zeros);
            }
        }

        [[nodiscard]] size_t deleted_mask_capacity(
            const lfs::core::SplatData& splat_data,
            const lfs::core::Tensor& free_mask) {
            return free_mask.is_valid() ? static_cast<size_t>(free_mask.numel())
                                        : static_cast<size_t>(splat_data.size());
        }

        void copy_deleted_mask_prefix(lfs::core::Tensor& destination,
                                      const lfs::core::Tensor& source,
                                      const size_t rows) {
            if (rows == 0) {
                return;
            }
            // Keep the source allocation alive until the asynchronous copy is
            // complete. Today zeros_direct uses the legacy stream, but this
            // remains correct if either tensor gains an explicit stream later.
            const lfs::core::Tensor source_keepalive = source;
            const cudaStream_t copy_stream = destination.stream();
            LFS_CUDA_CHECK(cudaMemcpyAsync(
                destination.ptr<uint8_t>(),
                source_keepalive.ptr<uint8_t>(),
                rows * sizeof(uint8_t),
                cudaMemcpyDeviceToDevice,
                copy_stream));
            LFS_CUDA_CHECK(cudaStreamSynchronize(copy_stream));
        }

        void ensure_deleted_mask_size(
            lfs::core::SplatData& splat_data,
            const lfs::core::Tensor& free_mask) {
            const size_t current_size = static_cast<size_t>(splat_data.size());
            const size_t desired_capacity = deleted_mask_capacity(splat_data, free_mask);
            auto& deleted = splat_data.deleted();
            if (!deleted.is_valid() || deleted.ndim() != 1 || deleted.numel() != current_size) {
                if (desired_capacity > current_size) {
                    deleted = lfs::core::Tensor::zeros_direct(
                        lfs::core::TensorShape({current_size}),
                        desired_capacity,
                        splat_data.means().device(),
                        lfs::core::DataType::Bool);
                } else {
                    deleted = lfs::core::Tensor::zeros_bool({current_size}, splat_data.means().device());
                }
                deleted.set_name("splat.deleted_mask");
                splat_data.notify_deleted_mask_changed();
                return;
            }
            if (deleted.capacity() >= desired_capacity) {
                return;
            }
            // Growing cuda.direct (from prior zeros_direct/reserve) cannot use reserve().
            auto fresh = lfs::core::Tensor::zeros_direct(
                lfs::core::TensorShape({current_size}),
                desired_capacity,
                deleted.device(),
                lfs::core::DataType::Bool);
            copy_deleted_mask_prefix(fresh, deleted, current_size);
            deleted = std::move(fresh);
            splat_data.notify_deleted_mask_changed();
        }

        void sync_deleted_mask_from_free_mask(
            lfs::core::SplatData& splat_data,
            const lfs::core::Tensor& free_mask) {
            const size_t current_size = static_cast<size_t>(splat_data.size());
            const size_t desired_capacity = deleted_mask_capacity(splat_data, free_mask);

            if (!free_mask.is_valid()) {
                splat_data.deleted() = lfs::core::Tensor::zeros_bool({current_size}, splat_data.means().device());
                splat_data.deleted().reserve(desired_capacity);
                splat_data.notify_deleted_mask_changed();
                return;
            }

            splat_data.deleted() = free_mask.slice(0, 0, current_size).clone();
            splat_data.deleted().reserve(desired_capacity);
            splat_data.notify_deleted_mask_changed();
        }

        void set_deleted_mask_rows(
            lfs::core::SplatData& splat_data,
            const lfs::core::Tensor& free_mask,
            const lfs::core::Tensor& indices,
            const bool deleted) {
            if (!indices.is_valid() || indices.numel() == 0) {
                return;
            }

            ensure_deleted_mask_size(splat_data, free_mask);
            auto values = deleted
                              ? lfs::core::Tensor::ones_bool({static_cast<size_t>(indices.numel())}, indices.device())
                              : lfs::core::Tensor::zeros_bool({static_cast<size_t>(indices.numel())}, indices.device());
            splat_data.deleted().index_put_(indices, values);
            splat_data.notify_deleted_mask_changed();
        }

        void append_live_deleted_rows(
            lfs::core::SplatData& splat_data,
            const lfs::core::Tensor& free_mask,
            const size_t n_rows) {
            // Keep deleted.numel() == size(). Safe to call before or after param
            // growth: pad with live(false) rows, never leave a stale
            // mask that violates the VkSplat packer contract.
            if (n_rows == 0 && splat_data.deleted_mask_matches_size()) {
                return;
            }

            const size_t target_size = static_cast<size_t>(splat_data.size());
            auto& deleted = splat_data.deleted();
            const size_t desired_capacity = std::max(
                deleted_mask_capacity(splat_data, free_mask),
                target_size);

            if (!deleted.is_valid() || deleted.ndim() != 1) {
                if (target_size == 0) {
                    return;
                }
                if (desired_capacity > target_size) {
                    deleted = lfs::core::Tensor::zeros_direct(
                        lfs::core::TensorShape({target_size}),
                        desired_capacity,
                        splat_data.means().device(),
                        lfs::core::DataType::Bool);
                } else {
                    deleted = lfs::core::Tensor::zeros_bool(
                        {target_size}, splat_data.means().device());
                }
                deleted.set_name("splat.deleted_mask");
                splat_data.notify_deleted_mask_changed();
                return;
            }

            const size_t cur = static_cast<size_t>(deleted.numel());
            if (cur == target_size) {
                if (deleted.capacity() < desired_capacity &&
                    !deleted.is_external_storage()) {
                    // Capacity-only growth; rebuild with headroom.
                    auto fresh = lfs::core::Tensor::zeros_direct(
                        lfs::core::TensorShape({target_size}),
                        desired_capacity,
                        deleted.device(),
                        lfs::core::DataType::Bool);
                    copy_deleted_mask_prefix(fresh, deleted, target_size);
                    deleted = std::move(fresh);
                    splat_data.notify_deleted_mask_changed();
                }
                return;
            }

            if (cur < target_size) {
                const size_t pad = target_size - cur;
                const size_t grow_cap = std::max(
                    desired_capacity,
                    deleted.capacity() > 0
                        ? static_cast<size_t>(deleted.capacity() * 3 / 2)
                        : target_size);
                if (deleted.capacity() >= target_size) {
                    deleted.append_zeros(pad);
                } else {
                    auto fresh = lfs::core::Tensor::zeros_direct(
                        lfs::core::TensorShape({cur}),
                        grow_cap,
                        deleted.device(),
                        lfs::core::DataType::Bool);
                    copy_deleted_mask_prefix(fresh, deleted, cur);
                    fresh.append_zeros(pad);
                    deleted = std::move(fresh);
                }
                splat_data.notify_deleted_mask_changed();
                return;
            }

            // cur > target_size: oversized/stale mask after a shrink path.
            splat_data.reconcile_deleted_mask();
        }

        void ensure_canny_workspace(lfs::core::Tensor& nms_output, const int height, const int width) {
            if (!nms_output.is_valid() ||
                nms_output.ndim() != 2 ||
                height != static_cast<int>(nms_output.shape()[0]) ||
                width != static_cast<int>(nms_output.shape()[1])) {
                nms_output = lfs::core::Tensor::zeros(
                    {static_cast<size_t>(height), static_cast<size_t>(width)},
                    lfs::core::Device::CUDA,
                    lfs::core::DataType::Float32);
            }
        }

        void apply_canny_filter(const lfs::core::Tensor& input_data, lfs::core::Tensor& nms_output) {
            assert(input_data.dtype() == lfs::core::DataType::Float32 ||
                   input_data.dtype() == lfs::core::DataType::UInt8);
            assert(input_data.device() == lfs::core::Device::CUDA);
            assert(input_data.ndim() == 3);
            assert(input_data.shape()[0] >= 3);

            const int width = static_cast<int>(input_data.shape()[2]);
            const int height = static_cast<int>(input_data.shape()[1]);

            ensure_canny_workspace(nms_output, height, width);
            auto input_contig = input_data.contiguous();
            if (input_contig.dtype() == lfs::core::DataType::UInt8) {
                kernels::launch_fused_canny_edge_filter_chw(
                    input_contig.ptr<uint8_t>(),
                    nms_output.ptr<float>(),
                    height,
                    width);
            } else {
                kernels::launch_fused_canny_edge_filter_chw(
                    input_contig.ptr<float>(),
                    nms_output.ptr<float>(),
                    height,
                    width);
            }
        }

        void normalize_by_positive_median_inplace(
            lfs::core::Tensor& tensor,
            PositiveMedianScratch* scratch = nullptr) {
            if (tensor.device() == lfs::core::Device::CUDA &&
                tensor.dtype() == lfs::core::DataType::Float32 &&
                tensor.is_valid() && tensor.numel() > 0) {
                if (scratch) {
                    scratch->ensure_n(tensor.numel(), tensor.device());
                }
                kernels::launch_normalize_by_positive_median(
                    tensor.ptr<float>(), tensor.numel(), nullptr, scratch);
                return;
            }
            // CPU fallback (tests / rare).
            tensor.masked_fill_(tensor.isnan(), 0.0f);
            auto valid = tensor.masked_select(tensor > 0.0f);
            if (valid.numel() == 0) {
                tensor.zero_();
                return;
            }
            auto [sorted, _] = valid.sort();
            const float median = sorted[valid.numel() / 2].item_as<float>();
            tensor.div_(std::max(median, 1e-9f));
        }

        [[nodiscard]] lfs::core::Tensor normalized_by_positive_median(
            const lfs::core::Tensor& tensor,
            PositiveMedianScratch* scratch = nullptr) {
            auto normalized = tensor.clone();
            normalize_by_positive_median_inplace(normalized, scratch);
            return normalized;
        }
    } // namespace

    MRNF::MRNF(lfs::core::SplatData& splat_data) : _splat_data(&splat_data) {}

    void MRNF::initialize(const lfs::core::param::OptimizationParameters& optimParams) {
        using namespace lfs::core;

        _strategy_required_peak_bytes = 0;
        _strategy_allocated_peak_bytes = 0;
        _densify_n_required_peak_bytes = 0;
        _densify_n_allocated_peak_bytes = 0;
        _densify_child_required_peak_bytes = 0;
        _densify_child_allocated_peak_bytes = 0;
        _params = std::make_unique<const lfs::core::param::OptimizationParameters>(optimParams);

        if (_params->max_cap > 0) {
            const size_t capacity = static_cast<size_t>(_params->max_cap);
            const size_t current_size = _splat_data->size();
            LOG_INFO("MRNF: pre-allocating capacity for {} Gaussians (current: {}, utilization: {:.1f}%)",
                     capacity, current_size, 100.0f * current_size / capacity);

            // When init_model_from_pointcloud was called with capacity = max_cap, every param
            // is already direct-allocated at that capacity. Re-allocating would briefly hold
            // both old and new buffers (≈2× peak) before the cuda caching allocator releases the
            // freed chunk — so only replace if the param's capacity is actually below the target.
            // Only skip Vulkan/exportable interop storage. zeros_direct marks
            // external_kind="cuda.direct", which is_external_storage() also reports
            // true for — those must still grow to max_cap here.
            // "splat.exportable" is the CUDA-only view of the same packed
            // block (post-grow pre-Vulkan-reimport). Stealing either kind onto a
            // private max_cap buffer orphans the zero-copy layout and, for q16 SH,
            // mis-sizes float topology capacity against pad-dropped cells.
            auto is_interop_external = [](const Tensor& t) {
                if (!t.is_external_storage())
                    return false;
                const auto kind = t.external_storage_kind();
                return kind == "vulkan_external_buffer" || kind == "splat.exportable";
            };

            auto ensure_capacity_direct = [capacity, &is_interop_external](Tensor& param) {
                if (param.capacity() >= capacity)
                    return;
                // GUI exportable / Vulkan-external tensors grow with live N via
                // SplatExportableStorage; do not steal them onto a
                // private zeros_direct max_cap buffer.
                if (is_interop_external(param))
                    return;
                auto new_param = Tensor::zeros_direct(param.shape(), capacity);
                cudaMemcpy(new_param.ptr<float>(), param.ptr<float>(),
                           param.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
                param = std::move(new_param);
            };

            // shN capacity units depend on resident layout: float4-swizzle floats
            // (fp32 / IEEE f16) or pad-dropped q16 u16 cells. Never treat q16 cell
            // capacity as float-slot capacity.
            const auto layout_rest = static_cast<uint32_t>(_splat_data->max_sh_coeffs_rest());
            auto ensure_shN_capacity_direct = [capacity, layout_rest, &is_interop_external,
                                               this](Tensor& param) {
                if (is_interop_external(param))
                    return;
                const bool q16 = _splat_data->shN_value_quantized();
                const size_t need_cap =
                    q16 ? lfs::core::sh_value_quant::sh_value_u16_count(capacity, layout_rest)
                        : lfs::core::sh_swizzled_float_count(capacity, layout_rest);
                if (param.capacity() >= need_cap)
                    return;
                // Refuse to grow quantized codes with a float memcpy — expand via
                // ensure_shN_fp32 / commit instead.
                if (param.dtype() != DataType::Float32) {
                    LOG_DEBUG("MRNF: skip pre-alloc grow of non-float shN (dtype={})",
                              static_cast<int>(param.dtype()));
                    return;
                }
                auto new_param = Tensor::zeros_direct(param.shape(), need_cap);
                cudaMemcpy(new_param.ptr<float>(), param.ptr<float>(),
                           param.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
                param = std::move(new_param);
            };

            ensure_capacity_direct(_splat_data->means());
            ensure_capacity_direct(_splat_data->sh0());
            if (layout_rest > 0 && _splat_data->shN().is_valid() && _splat_data->shN().numel() > 0) {
                ensure_shN_capacity_direct(_splat_data->shN());
            }
            ensure_capacity_direct(_splat_data->scaling_raw());
            ensure_capacity_direct(_splat_data->rotation_raw());
            ensure_capacity_direct(_splat_data->opacity_raw());
        }

        // Convert shN to pad-dropped u16 after reserving float capacity.
        lfs::training::sh_value::apply_shN_value_quant(*_splat_data);

        _optimizer = create_optimizer(*_splat_data, *_params);
        _optimizer->allocate_gradients(_params->max_cap > 0 ? static_cast<size_t>(_params->max_cap) : 0);
        _scheduler = create_scheduler(*_params, *_optimizer);
        _mean_lr_unscaled = _params->means_lr;
        _scale_lr_current = _params->scaling_lr;
        _mean_lr_gamma = compute_decay_gamma(_params->means_lr, _params->means_lr_end, _params->iterations);
        _scale_lr_gamma = compute_decay_gamma(_params->scaling_lr, _params->scaling_lr_end, _params->iterations);

        ensure_densification_info_shape();

        const size_t capacity = _params->max_cap > 0 ? static_cast<size_t>(_params->max_cap)
                                                     : static_cast<size_t>(_splat_data->size());
        _free_mask = Tensor::zeros_bool({capacity}, _splat_data->means().device());
        sync_deleted_mask_from_free_mask(*_splat_data, _free_mask);

        // Pre-size densification scratch to max_cap so increasing live N does not
        // allocate from the driver during refinement.
        if (capacity > 0) {
            const auto device = _splat_data->means().device();
            _densify_n_scratch.ensure_n(capacity, device);
            _densify_n_scratch.ensure_k(std::max(capacity / 20, size_t{1024}), device);
            _gumbel_scratch.ensure_n(capacity, device);
            _median_scratch.ensure_n(capacity, device);
            if (lfs::training::PerfBenchCollector::enabled()) {
                lfs::training::PerfBenchCollector::instance().set_densify_workspace_bytes(
                    _densify_n_scratch.resident_bytes());
            }
        }

        const size_t n = static_cast<size_t>(_splat_data->size());
        _initial_sfm_point_count = n;
        const size_t tracking_capacity = _params->max_cap > 0 ? static_cast<size_t>(_params->max_cap) : 0;
        reset_vector_buffer(_refine_weight_max, n, _splat_data->means().device(), tracking_capacity);
        reset_vector_buffer(_vis_count, n, _splat_data->means().device(), tracking_capacity);
        if (cfg_ratio_rank_on()) {
            reset_vector_buffer(_refine_ratio_max, n, _splat_data->means().device(), tracking_capacity);
        }

        publish_vram_attribution();
        compute_bounds();
        refresh_camera_hull();
        ensure_mean_step_far_mask();

        LOG_INFO("MRNF strategy initialized with {} Gaussians", n);
    }

    void MRNF::set_training_dataset(std::shared_ptr<CameraDataset> views) {
        _views = std::move(views);
        refresh_camera_hull();
        if (_optimizer) {
            ensure_mean_step_far_mask();
        }
    }

    void MRNF::pre_step(int iter, RenderOutput& render_output) {
        publish_vram_attribution();
        _precomputed_edge_scores = lfs::core::Tensor();
        _edge_precompute_valid = false;

        if (!_params || !_params->use_edge_map || iter >= static_cast<int>(_params->stop_refine)) {
            reset_edge_accumulator();
            publish_vram_attribution();
            return;
        }

        if (should_accumulate_edge_sample(iter)) {
            accumulate_edge_sample(iter, render_output);
        }

        if (!is_refining(iter)) {
            return;
        }

        if (_edge_sample_count <= 0 ||
            !_edge_score_sum.is_valid() ||
            _edge_score_sum.ndim() != 1 ||
            _edge_score_sum.numel() != static_cast<size_t>(_splat_data->size())) {
            reset_edge_accumulator();
            publish_vram_attribution();
            return;
        }

        _precomputed_edge_scores = _edge_score_sum.clone();
        _precomputed_edge_scores.div_(static_cast<float>(_edge_sample_count));
        zero_frozen_scores_inplace(*_splat_data, _precomputed_edge_scores);
        _edge_precompute_valid = true;
        // Capture the short, real overlap before score_sum is released.
        publish_vram_attribution();
        reset_edge_accumulator();
        publish_vram_attribution();
    }

    size_t MRNF::densification_row_count() const {
        return 2;
    }

    void MRNF::ensure_densification_info_shape() {
        const size_t n = static_cast<size_t>(_splat_data->size());
        ensure_densification_info_shape_inplace(
            _splat_data->_densification_info,
            n,
            _splat_data->means().device(),
            densification_row_count());
    }

    int MRNF::edge_target_samples_per_refine_window() const {
        const int refine_window = _params ? static_cast<int>(_params->refine_every) : 1;
        if (!_views || _views->size() == 0) {
            return std::max(1, std::min(refine_window, MRNF_EDGE_MIN_VIEW_SAMPLES));
        }

        const int num_cam_dataset = static_cast<int>(_views->size());
        const int requested_samples = num_cam_dataset < MRNF_EDGE_MIN_VIEW_SAMPLES
                                          ? num_cam_dataset
                                          : std::max(
                                                MRNF_EDGE_MIN_VIEW_SAMPLES,
                                                static_cast<int>(0.08f * static_cast<float>(num_cam_dataset)));
        return std::max(1, std::min(refine_window, requested_samples));
    }

    bool MRNF::should_accumulate_view_sample(int iter) const {
        if (!_params ||
            iter <= static_cast<int>(_params->start_refine) ||
            iter >= static_cast<int>(_params->stop_refine) ||
            _splat_data->size() == 0) {
            return false;
        }

        if (is_refining(iter)) {
            return true;
        }

        const int target_samples = edge_target_samples_per_refine_window();
        const int refine_every = std::max(1, static_cast<int>(_params->refine_every));
        const int stride = std::max(1, refine_every / target_samples);
        return (iter % stride) == 0;
    }

    bool MRNF::should_accumulate_edge_sample(int iter) const {
        return _params && _params->use_edge_map && should_accumulate_view_sample(iter);
    }

    bool MRNF::should_accumulate_explore_sample(int iter) const {
        return background_improvements_enabled() && far_operators_active() &&
               should_accumulate_view_sample(iter);
    }

    void MRNF::reset_edge_accumulator() {
        _edge_score_sum = lfs::core::Tensor();
        _edge_sample_count = 0;
        _edge_last_sample_iter = -1;
    }

    void MRNF::reset_explore_accumulator() {
        _explore_score_sum = lfs::core::Tensor();
        _explore_sample_count = 0;
        _explore_last_sample_iter = -1;
    }

    void MRNF::publish_vram_attribution() noexcept {
        try {
            auto& profiler = lfs::diagnostics::VramProfiler::instance();
            const bool publish_live = profiler.enabled();
            const bool publish_bench = PerfBenchCollector::enabled();
            if (!publish_live && !publish_bench) {
                return;
            }

            std::size_t strategy_required = 0;
            std::size_t strategy_allocated = 0;
            const auto account_tensor = [&](const std::string_view name,
                                            const lfs::core::Tensor& tensor) {
                const auto required = tensor_vram_required_bytes(tensor);
                const auto allocated = tensor_vram_allocated_bytes(tensor);
                strategy_required += required;
                strategy_allocated += allocated;
                if (publish_live) {
                    publish_required_allocated_pair(profiler, name, required, allocated);
                }
            };

            account_tensor("refine_weight_max", _refine_weight_max);
            account_tensor("refine_ratio_max", _refine_ratio_max);
            account_tensor("vis_count", _vis_count);
            account_tensor("free_mask", _free_mask);
            account_tensor("refine_counts_device", _refine_counts_dev);
            account_tensor("edge.precomputed_scores", _precomputed_edge_scores);
            account_tensor("edge.score_sum", _edge_score_sum);
            account_tensor("edge.canny_nms", _edge_canny_nms_output);
            account_tensor("explore.score_sum", _explore_score_sum);
            account_tensor("explore.error_hw", _explore_error_hw);
            account_tensor("explore.view_scores", _explore_view_scores);
            account_tensor("explore.means2d", _explore_means2d);
            account_tensor("explore.radii", _explore_radii);
            account_tensor("explore.far_field_mask", _far_field_mask);
            account_tensor("explore.cached_image", _cached_seed_image);
            account_tensor("explore.cached_target", _cached_seed_target);
            account_tensor("explore.cached_alpha", _cached_seed_alpha);
            account_tensor("explore.cached_depth", _cached_seed_depth);

            _strategy_required_peak_bytes =
                std::max(_strategy_required_peak_bytes, strategy_required);
            _strategy_allocated_peak_bytes =
                std::max(_strategy_allocated_peak_bytes, strategy_allocated);

            _densify_n_required_peak_bytes = std::max(
                _densify_n_required_peak_bytes, _densify_n_scratch.required_bytes());
            _densify_n_allocated_peak_bytes = std::max(
                _densify_n_allocated_peak_bytes, _densify_n_scratch.resident_bytes());
            _densify_child_required_peak_bytes = std::max(
                _densify_child_required_peak_bytes, _densify_ws.required_bytes());
            _densify_child_allocated_peak_bytes = std::max(
                _densify_child_allocated_peak_bytes, _densify_ws.resident_bytes());

            if (publish_live) {
                publish_required_allocated_pair(
                    profiler, "strategy_peak", _strategy_required_peak_bytes,
                    _strategy_allocated_peak_bytes);
                publish_required_allocated_pair(
                    profiler, "densify_n_scratch", _densify_n_required_peak_bytes,
                    _densify_n_allocated_peak_bytes);
                publish_required_allocated_pair(
                    profiler, "densify_child", _densify_child_required_peak_bytes,
                    _densify_child_allocated_peak_bytes);
            }

            if (publish_bench) {
                auto& bench = PerfBenchCollector::instance();
                bench.set_mrnf_strategy_bytes(_strategy_required_peak_bytes,
                                              _strategy_allocated_peak_bytes);
                bench.set_mrnf_densify_n_bytes(_densify_n_required_peak_bytes,
                                               _densify_n_allocated_peak_bytes);
                bench.set_mrnf_densify_child_bytes(_densify_child_required_peak_bytes,
                                                   _densify_child_allocated_peak_bytes);
            }
        } catch (...) {
            // Attribution must never alter the training control path.
        }
    }

    void MRNF::accumulate_edge_sample(int iter, const RenderOutput& render_output) {
        using namespace lfs::core;

        if (_edge_last_sample_iter == iter) {
            return;
        }
        if (!render_output.camera ||
            !render_output.target_image.is_valid() ||
            render_output.target_image.device() != Device::CUDA ||
            (render_output.target_image.dtype() != DataType::Float32 &&
             render_output.target_image.dtype() != DataType::UInt8) ||
            render_output.target_image.ndim() != 3 ||
            render_output.target_image.shape()[0] < 3) {
            return;
        }

        const size_t n = static_cast<size_t>(_splat_data->size());
        if (!_edge_score_sum.is_valid() ||
            _edge_score_sum.ndim() != 1 ||
            _edge_score_sum.numel() != n) {
            _edge_score_sum = Tensor::zeros({n}, _splat_data->means().device());
            _edge_sample_count = 0;
        }

        apply_canny_filter(render_output.target_image, _edge_canny_nms_output);
        normalize_by_positive_median_inplace(_edge_canny_nms_output, &_median_scratch);

        auto score_render = edge_rasterize(
            *render_output.camera,
            this->get_model(),
            _edge_canny_nms_output);
        normalize_by_positive_median_inplace(score_render.edges_score, &_median_scratch);
        _edge_score_sum.add_(zero_frozen_scores(*_splat_data, score_render.edges_score));
        ++_edge_sample_count;
        _edge_last_sample_iter = iter;
        publish_vram_attribution();
    }

    bool MRNF::should_cache_seed_view(int iter) const {
        if (!background_improvements_enabled()) {
            return false;
        }
        const int next_iter = iter + 1;
        return is_refining(next_iter) &&
               next_iter < effective_grow_until_iter();
    }

    void MRNF::cache_seed_view(int iter, const RenderOutput& render_output) {
        using namespace lfs::core;
        if (!should_cache_seed_view(iter) || !render_output.camera) {
            return;
        }
        LOG_TIMER("MRNF::cache_seed_view");
        if (!is_cuda_image(render_output.image) ||
            !is_cuda_image(render_output.target_image) ||
            !render_output.alpha.is_valid() ||
            render_output.alpha.device() != Device::CUDA ||
            !same_spatial(render_output.image, render_output.target_image)) {
            return;
        }

        copy_into_cached(_cached_seed_image, render_output.image);
        copy_into_cached(_cached_seed_target, render_output.target_image);
        copy_into_cached(_cached_seed_alpha, render_output.alpha);
        if (render_output.depth.is_valid() && render_output.depth.device() == Device::CUDA) {
            copy_into_cached(_cached_seed_depth, render_output.depth);
        } else {
            _cached_seed_depth = Tensor();
        }
        _cached_seed_camera = render_output.camera;
        _cached_seed_width = render_output.width > 0
                                 ? render_output.width
                                 : static_cast<int>(_cached_seed_image.shape()[_cached_seed_image.ndim() - 1]);
        _cached_seed_height = render_output.height > 0
                                  ? render_output.height
                                  : static_cast<int>(_cached_seed_image.shape()[_cached_seed_image.ndim() - 2]);
        _cached_seed_valid = true;
    }

    void MRNF::accumulate_explore_sample(int iter, const RenderOutput& render_output) {
        using namespace lfs::core;

        if (_explore_last_sample_iter == iter) {
            return;
        }
        if (!should_accumulate_explore_sample(iter)) {
            return;
        }
        LOG_TIMER("MRNF::accumulate_explore_sample");
        if (!is_cuda_image(render_output.image) ||
            !is_cuda_image(render_output.target_image) ||
            render_output.image.ndim() < 3 ||
            render_output.target_image.ndim() < 3 ||
            !same_spatial(render_output.image, render_output.target_image)) {
            return;
        }

        if (!fill_mean_abs_error_hw(render_output.image, render_output.target_image, _explore_error_hw) ||
            !_explore_error_hw.is_valid() || _explore_error_hw.ndim() != 2 ||
            _explore_error_hw.device() != Device::CUDA ||
            _explore_error_hw.dtype() != DataType::Float32) {
            return;
        }

        const int height = static_cast<int>(_explore_error_hw.shape()[0]);
        const int width = static_cast<int>(_explore_error_hw.shape()[1]);
        if (height <= 0 || width <= 0) {
            return;
        }

        const size_t n = static_cast<size_t>(_splat_data->size());
        if (n == 0) {
            return;
        }

        Tensor means2d;
        Tensor radii;
        bool used_render_buffers = false;
        if (render_output.means2d.is_valid() && render_output.radii.is_valid() &&
            render_output.means2d.device() == Device::CUDA &&
            render_output.radii.device() == Device::CUDA &&
            render_output.means2d.numel() > 0 &&
            render_output.radii.numel() > 0) {
            means2d = squeeze_leading_ones(render_output.means2d);
            radii = squeeze_leading_ones(render_output.radii);
            if (means2d.ndim() == 3 && means2d.shape()[0] == 1) {
                means2d = means2d.squeeze(0);
            }
            if (radii.ndim() == 2 && radii.shape()[0] == 1) {
                radii = radii.squeeze(0);
            }
            if (means2d.ndim() == 2 && means2d.shape()[0] == n && means2d.shape()[1] == 2 &&
                radii.ndim() == 1 && radii.numel() == n) {
                used_render_buffers = true;
            }
        }

        if (!used_render_buffers) {
            if (!render_output.camera) {
                return;
            }
            const auto [fx, fy, cx, cy] = render_output.camera->get_intrinsics();
            const float* w2c = render_output.camera->world_view_transform_ptr();
            if (w2c == nullptr || !(fx > 0.0f) || !(fy > 0.0f)) {
                return;
            }
            ensure_cuda_float(_explore_means2d, TensorShape({n, 2}));
            ensure_cuda_float(_explore_radii, TensorShape({n}));
            means2d = _explore_means2d;
            radii = _explore_radii;
            mrnf_strategy::launch_project_visible_centers(
                _splat_data->means().ptr<float>(),
                w2c,
                fx,
                fy,
                cx,
                cy,
                width,
                height,
                MRNF_PROJECT_NEAR,
                means2d.ptr<float>(),
                radii.ptr<float>(),
                n);
        }

        if (!_explore_score_sum.is_valid() ||
            _explore_score_sum.ndim() != 1 ||
            _explore_score_sum.numel() != n) {
            _explore_score_sum = Tensor::zeros({n}, _splat_data->means().device());
            _explore_sample_count = 0;
        }

        ensure_cuda_float(_explore_view_scores, TensorShape({n}));
        assert(means2d.is_valid());
        assert(means2d.ndim() == 2);
        assert(means2d.shape()[0] == n);
        assert(means2d.shape()[1] == 2);
        assert(radii.is_valid());
        assert(radii.ndim() == 1);
        assert(radii.numel() == n);
        mrnf_strategy::launch_gather_center_error(
            means2d.ptr<float>(),
            radii.ptr<float>(),
            _explore_error_hw.ptr<float>(),
            width,
            height,
            _explore_view_scores.ptr<float>(),
            n);
        normalize_by_positive_median_inplace(_explore_view_scores, &_median_scratch);
        zero_frozen_scores_inplace(*_splat_data, _explore_view_scores);
        _explore_score_sum.add_(_explore_view_scores);
        ++_explore_sample_count;
        _explore_last_sample_iter = iter;
        publish_vram_attribution();
    }

    void MRNF::post_render(int iter, RenderOutput& render_output) {
        accumulate_explore_sample(iter, render_output);
        cache_seed_view(iter, render_output);
    }

    void MRNF::post_backward(int iter, RenderOutput& render_output) {
        LOG_TIMER("MRNF::post_backward");
        using namespace lfs::core;

        // The only degree-bump site is post-commit when refining,
        // otherwise immediately. Same cadence as before — bump when
        // iter % sh_degree_interval == 0 — but never duplicated across branches.
        // On refining steps, bump after refine() so densify/commit sees the prior
        // degree and the next forward first samples rest SH on a stable model.
        const bool refining_this_iter = is_refining(iter);
        const bool degree_bump_due = (iter % _params->sh_degree_interval == 0);

        if (iter == static_cast<int>(_params->stop_refine)) {
            _splat_data->_densification_info = Tensor::empty({0});
            _precomputed_edge_scores = Tensor();
            _edge_precompute_valid = false;
            reset_edge_accumulator();
            reset_explore_accumulator();
            _cached_seed_valid = false;
            // Topology freeze safety net: re-encode if the stop_refine step still
            // holds float SH (every regular refine already commits). is_refining()
            // classifies this boundary as an exclusive mutation step even when it
            // is off cadence, so Trainer holds render_mutex_ across this commit.
            if (lfs::core::sh_value_quant::enabled() &&
                _splat_data->shN().is_valid() &&
                _splat_data->shN().dtype() == lfs::core::DataType::Float32) {
                lfs::training::sh_value::commit_shN_after_mutation(*_splat_data);
            }
        }

        if (iter >= static_cast<int>(_params->stop_refine)) {
            // Still honor a late degree schedule after topology freeze (no-op
            // once at max). Metadata-only on q16.
            if (degree_bump_due) {
                _splat_data->increment_sh_degree();
            }
            publish_vram_attribution();
            return;
        }

        ensure_densification_info_shape();

        const size_t n = static_cast<size_t>(_splat_data->size());
        const auto& info = _splat_data->_densification_info;

        assert(info.is_valid());
        assert(info.ndim() == 2);
        assert(info.shape()[0] >= densification_row_count());
        assert(info.shape()[1] == n);

        if (_refine_weight_max.numel() == n) {
            float* ratio_max_ptr = nullptr;
            if (cfg_ratio_rank_on() && _refine_ratio_max.numel() == n) {
                ratio_max_ptr = _refine_ratio_max.ptr<float>();
            }
            mrnf_strategy::launch_fold_densification_and_zero(
                _vis_count.ptr<float>(),
                _refine_weight_max.ptr<float>(),
                _splat_data->_densification_info.ptr<float>(),
                n,
                nullptr,
                densification_row_count(),
                ratio_max_ptr,
                cfg_ratio_pow());
            zero_frozen_scores_inplace(*_splat_data, _refine_weight_max);
            if (ratio_max_ptr != nullptr) {
                zero_frozen_scores_inplace(*_splat_data, _refine_ratio_max);
            }
            zero_frozen_scores_inplace(*_splat_data, _vis_count);
        } else if (info.is_valid() && info.numel() > 0) {
            _splat_data->_densification_info.zero_();
        }

        if (_bounds_valid) {
            inject_noise(iter);
        }

        accumulate_explore_sample(iter, render_output);
        cache_seed_view(iter, render_output);

        if (refining_this_iter) {
            refine(iter, render_output);
            _precomputed_edge_scores = Tensor();
            _edge_precompute_valid = false;
        }
        if (degree_bump_due) {
            _splat_data->increment_sh_degree();
        }
        publish_vram_attribution();
    }

    void MRNF::permute_gaussian_rows(const lfs::core::Tensor& perm) {
        morton::permute_row_tensor(_refine_weight_max, perm);
        morton::permute_row_tensor(_refine_ratio_max, perm);
        morton::permute_row_tensor(_vis_count, perm);
        morton::permute_row_tensor(_precomputed_edge_scores, perm);
        morton::permute_row_tensor(_edge_score_sum, perm);
        morton::permute_row_tensor(_explore_score_sum, perm);
        morton::permute_row_tensor(_free_mask, perm);
        morton::permute_row_tensor(_far_field_mask, perm);
        if (_far_growth.outside_mask.is_valid()) {
            _far_growth.outside_mask = _far_field_mask;
        }
    }

    bool MRNF::is_refining(int iter) const {
        const int stop_refine = static_cast<int>(_params->stop_refine);
        if (iter == stop_refine) {
            return true;
        }
        return (iter < stop_refine &&
                iter > static_cast<int>(_params->start_refine) &&
                iter % _params->refine_every == 0);
    }

    int MRNF::effective_grow_until_iter() const {
        int value = static_cast<int>(_params->grow_until_iter);
        const int fill_iter = cfg_fill_target_iter();
        if (fill_iter > 0) {
            value = std::max(value, fill_iter);
        }
        return value;
    }

    void MRNF::refine(int iter, RenderOutput& render_output) {
        lfs::core::alloc_counter::ScopedSite densify_site("densify");
        LOG_TIMER("MRNF::refine");
        LFS_VRAM_SCOPE("MRNF::refine");
        using namespace lfs::core;
        (void)lfs::training::sh_value::ensure_shN_fp32_for_mutation(*_splat_data);

        ++_refine_windows_since_bounds;
        if (!_bounds_valid || _refine_windows_since_bounds >= MRNF_BOUNDS_RECOMPUTE_INTERVAL_REFINES) {
            compute_bounds();
        }

        const size_t n = static_cast<size_t>(_splat_data->size());

        auto raw_opacities = _splat_data->opacity_raw();
        if (raw_opacities.ndim() == 2 && raw_opacities.shape()[1] == 1)
            raw_opacities = raw_opacities.squeeze(-1);
        const auto& log_scales = _splat_data->scaling_raw();
        const auto& means = _splat_data->means();
        assert(raw_opacities.numel() == n);
        assert(log_scales.shape()[0] == n && log_scales.shape()[1] == 3);
        assert(means.shape()[0] == n && means.shape()[1] == 3);

        auto scale_min = log_scales.min(1);
        auto scale_max = log_scales.max(1);

        auto prune_mask = (raw_opacities < MRNF_RAW_OPACITY_PRUNE_THRESHOLD) |
                          compute_near_zero_rotation_mask(_splat_data->rotation_raw()) |
                          (scale_min < MRNF_LOG_MIN_SCALE_THRESHOLD);

        // Bounds-dependent pruning is unsafe for one-point or colocated models:
        // log(0) would classify every finite scale as oversized. Keep the
        // bounds-independent safety checks active until a real scene extent exists.
        if (_bounds_valid) {
            const float max_allowed =
                _bounds.max_extent <= std::numeric_limits<float>::max() / 100.0f
                    ? _bounds.max_extent * 100.0f
                    : std::numeric_limits<float>::max();
            const float log_max_allowed = std::log(max_allowed);
            auto center = Tensor::from_vector(
                {_bounds.center[0], _bounds.center[1], _bounds.center[2]},
                TensorShape({1, 3}), Device::CUDA);
            auto dist_from_center = (means - center).abs().max(1);
            prune_mask = prune_mask |
                         (scale_max > log_max_allowed) |
                         (dist_from_center > max_allowed);
        }

        if (_free_mask.is_valid() && n > 0) {
            auto active_mask = _free_mask.slice(0, 0, n).logical_not();
            prune_mask = prune_mask.logical_and(active_mask);
        }
        prune_mask = exclude_frozen_from_mask(*_splat_data, prune_mask);

        if (!_refine_counts_dev.is_valid() || _refine_counts_dev.numel() < 4) {
            _refine_counts_dev = Tensor::zeros({4}, Device::CUDA, DataType::Int64);
        }
        kernels::launch_packed_refine_counts(
            prune_mask.ptr<bool>(), n,
            nullptr, 0,
            nullptr, 0,
            nullptr, 0,
            _refine_counts_dev.ptr<int64_t>());
        int64_t host_counts[4] = {0, 0, 0, 0};
        LFS_CUDA_CHECK_MSG(
            cudaMemcpy(host_counts, _refine_counts_dev.ptr<int64_t>(),
                       4 * sizeof(int64_t), cudaMemcpyDeviceToHost),
            "MRNF refine prune-count D2H");
        const int pruned_count = static_cast<int>(host_counts[0]);

        if (pruned_count > 0) {
            auto prune_indices = prune_mask.nonzero().squeeze(-1);
            mark_as_free(prune_indices);
            set_deleted_mask_rows(*_splat_data, _free_mask, prune_indices, true);

            // Zero quaternion so deleted rows exit early in preprocessing.
            auto zero_rotation = Tensor::zeros({static_cast<size_t>(pruned_count), 4}, _splat_data->rotation_raw().device());
            _splat_data->rotation_raw().index_put_(prune_indices, zero_rotation);

            const auto layout_rest = static_cast<uint32_t>(_splat_data->max_sh_coeffs_rest());
            reset_optimizer_state_at_indices(*_optimizer, ParamType::Means, prune_indices);
            reset_optimizer_state_at_indices(*_optimizer, ParamType::Sh0, prune_indices);
            reset_optimizer_state_at_indices(*_optimizer, ParamType::ShN, prune_indices, layout_rest);
            reset_optimizer_state_at_indices(*_optimizer, ParamType::Scaling, prune_indices);
            reset_optimizer_state_at_indices(*_optimizer, ParamType::Rotation, prune_indices);
            reset_optimizer_state_at_indices(*_optimizer, ParamType::Opacity, prune_indices);

            LOG_DEBUG("MRNF: soft-pruned {} splats at iter {} (active: {}, total slots: {})",
                      pruned_count, iter, active_count(), _splat_data->size());
            LFS_COUNTER_ADD("strategy.mrnf.pruned", pruned_count);
        }

        const bool growing = iter < effective_grow_until_iter();
        const bool seed_far = growing && background_improvements_enabled();
        int reserved_seeds = seed_far ? starved_cadence_count(kExploreSeeds) : 0;
        if (seed_far && cfg_seed_dose() > 0)
            reserved_seeds = std::max(reserved_seeds, cfg_seed_dose());
        begin_far_growth_window(n, reserved_seeds);

        grow_and_split(iter, pruned_count);
        if (seed_far) {
            seed_from_view(iter, render_output);
        }
        reset_explore_accumulator();

        int live_far = 0;
        const size_t live_n = static_cast<size_t>(_splat_data->size());
        refresh_far_field_mask(live_n);
        if (_camera_hull_valid && _far_field_mask.is_valid() &&
            _far_field_mask.numel() == live_n) {
            auto live_mask = _far_field_mask;
            if (_free_mask.is_valid() && _free_mask.numel() >= live_mask.numel()) {
                live_mask = live_mask.logical_and(_free_mask.slice(0, 0, live_mask.numel()).logical_not());
            }
            live_far = static_cast<int>(live_mask.to(DataType::Int32).sum().item<int>());
        }
        LFS_GAUGE("model.gaussians.live_far_field", live_far);

        enforce_max_cap();
        apply_decay(iter);
        ensure_mean_step_far_mask();

        const size_t new_n = static_cast<size_t>(_splat_data->size());
        const size_t tracking_capacity = _params->max_cap > 0 ? static_cast<size_t>(_params->max_cap) : 0;
        reset_vector_buffer(_refine_weight_max, new_n, _splat_data->means().device(), tracking_capacity);
        reset_vector_buffer(_vis_count, new_n, _splat_data->means().device(), tracking_capacity);
        if (cfg_ratio_rank_on()) {
            reset_vector_buffer(_refine_ratio_max, new_n, _splat_data->means().device(), tracking_capacity);
        }
        ensure_densification_info_shape();
        _splat_data->_densification_info.zero_();

        // Always-commit q16 after every refine (exportable + headless). The
        // multi-iter exportable float densify window is deleted: Scene cache
        // rebuild and preview share Trainer::render_mutex_ with commit+trim, so
        // the float workspace cannot be read/decommitted concurrently.
        if (lfs::core::sh_value_quant::enabled() &&
            _splat_data->shN().is_valid() &&
            _splat_data->shN().dtype() == lfs::core::DataType::Float32) {
            lfs::training::sh_value::commit_shN_after_mutation(*_splat_data);
        }

        publish_vram_attribution();

        // MRNF trim_memory_pool parity with MCMC/IGS+ after refine.
        // Epoch-pinned release: trim runs under the same render_mutex_ exclusive
        // that bars Scene rebuild / preview from holding the float workspace
        // (trainer acquires exclusive for refining steps around post_backward).
        lfs::core::Tensor::trim_memory_pool();
    }

    bool MRNF::cfg_ratio_rank_on() const {
        return background_improvements_enabled() && _params->growth_ratio_rank;
    }

    float MRNF::cfg_ratio_pow() const {
        return cfg_ratio_rank_on() ? _params->growth_ratio_pow : 0.0f;
    }

    int MRNF::cfg_fill_target_iter() const {
        return (background_improvements_enabled() && _params->fill_pacing_iter > 0)
                   ? static_cast<int>(_params->fill_pacing_iter)
                   : 0;
    }

    int MRNF::cfg_seed_dose() const {
        return background_improvements_enabled() ? static_cast<int>(_params->far_seed_dose) : 0;
    }

    bool MRNF::background_improvements_enabled() const {
        return _params && _params->background_improvements;
    }

    bool MRNF::far_operators_active() const {
        return _scene_has_far_field || explore_starvation_weighting_enabled();
    }

    bool MRNF::explore_starvation_weighting_enabled() const {
        return background_improvements_enabled() && _params->explore_starvation_weighting;
    }

    void MRNF::refresh_camera_hull() {
        _camera_hull_valid = false;
        _cam_centroid[0] = 0.0f;
        _cam_centroid[1] = 0.0f;
        _cam_centroid[2] = 0.0f;
        _orbit_radius = 0.0f;

        if (!background_improvements_enabled()) {
            _scene_has_far_field = false;
            _far_field_mask = {};
            update_far_starvation();
            return;
        }

        const size_t n_cam = _views ? _views->size() : 0;
        if (n_cam < 2) {
            update_far_starvation();
            if (!_logged_degenerate_hull && background_improvements_enabled()) {
                LOG_INFO("MRNF: camera hull unavailable (need >= 2 training cameras); far-field guard is inert");
                _logged_degenerate_hull = true;
            }
            return;
        }

        float sum_x = 0.0f;
        float sum_y = 0.0f;
        float sum_z = 0.0f;
        std::vector<float> positions;
        positions.reserve(n_cam * 3);
        size_t counted = 0;
        for (size_t i = 0; i < n_cam; ++i) {
            lfs::core::Camera* cam = _views->get_camera(i);
            if (!cam) {
                continue;
            }
            auto pos = cam->cam_position().cpu().contiguous();
            if (!pos.is_valid() || pos.numel() < 3) {
                continue;
            }
            const float* p = pos.ptr<float>();
            if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2])) {
                continue;
            }
            sum_x += p[0];
            sum_y += p[1];
            sum_z += p[2];
            positions.push_back(p[0]);
            positions.push_back(p[1]);
            positions.push_back(p[2]);
            ++counted;
        }

        if (counted < 2) {
            update_far_starvation();
            if (!_logged_degenerate_hull && background_improvements_enabled()) {
                LOG_INFO("MRNF: camera hull unavailable (need >= 2 valid cameras); far-field guard is inert");
                _logged_degenerate_hull = true;
            }
            return;
        }

        const float inv = 1.0f / static_cast<float>(counted);
        const float cx = sum_x * inv;
        const float cy = sum_y * inv;
        const float cz = sum_z * inv;
        float radius = 0.0f;
        for (size_t i = 0; i < counted; ++i) {
            const float dx = positions[i * 3 + 0] - cx;
            const float dy = positions[i * 3 + 1] - cy;
            const float dz = positions[i * 3 + 2] - cz;
            radius = std::max(radius, std::sqrt(dx * dx + dy * dy + dz * dz));
        }

        const float centroid_norm = std::sqrt(cx * cx + cy * cy + cz * cz);
        const float radius_eps =
            32.0f * std::numeric_limits<float>::epsilon() * std::max(centroid_norm, 1.0f);
        if (!std::isfinite(radius) || radius <= radius_eps) {
            update_far_starvation();
            if (!_logged_degenerate_hull && background_improvements_enabled()) {
                LOG_INFO("MRNF: camera hull degenerate (orbit radius {:.6g}); far-field guard is inert",
                         radius);
                _logged_degenerate_hull = true;
            }
            return;
        }

        _cam_centroid[0] = cx;
        _cam_centroid[1] = cy;
        _cam_centroid[2] = cz;
        _orbit_radius = radius;
        _camera_hull_valid = true;
        _logged_degenerate_hull = false;

        constexpr float kDeepFarRadiusOrbits = 8.0f;
        const float far_scene_min_fraction = _params->far_scene_min_fraction;
        const size_t n_now = _splat_data ? static_cast<size_t>(_splat_data->size()) : 0;
        _scene_has_far_field = false;
        if (n_now > 0) {
            if (far_scene_min_fraction <= 0.0f) {
                _scene_has_far_field = (n_now > 0);
                LOG_INFO("MRNF: deep-far census unconditional at threshold 0 (n={}, far features {})",
                         n_now, _scene_has_far_field ? "active" : "inert");
            } else {
                if (!_far_field_mask.is_valid() || _far_field_mask.numel() != n_now) {
                    _far_field_mask = lfs::core::Tensor::zeros_bool({n_now}, lfs::core::Device::CUDA);
                }
                mrnf_strategy::launch_far_field_mask(
                    _splat_data->means().ptr<float>(),
                    _cam_centroid[0], _cam_centroid[1], _cam_centroid[2],
                    kDeepFarRadiusOrbits * _orbit_radius,
                    _far_field_mask.ptr<bool>(), n_now);
                const float far_frac =
                    static_cast<float>(_far_field_mask.to(lfs::core::DataType::Int32).sum().item<int>()) /
                    static_cast<float>(n_now);
                _scene_has_far_field = far_frac >= far_scene_min_fraction;
                LOG_INFO("MRNF: deep-far fraction {:.4f} at {}x orbit (threshold {:.2f}, far features {})",
                         far_frac, kDeepFarRadiusOrbits, far_scene_min_fraction,
                         _scene_has_far_field ? "active" : "inert");
            }

            if (!far_operators_active()) {
                _camera_hull_valid = false;
                _far_field_mask = lfs::core::Tensor();
            }
            update_far_starvation();
            if (far_operators_active()) {
                refresh_far_field_mask(n_now);
            }
        } else {
            update_far_starvation();
        }
    }

    int MRNF::cadence_scaled(const int count) const {
        const int refine_every = _params
                                     ? static_cast<int>(std::max<size_t>(_params->refine_every, size_t{1}))
                                     : 100;
        return static_cast<int>(std::lround(static_cast<double>(count) *
                                            static_cast<double>(refine_every) / 100.0));
    }

    int MRNF::starved_cadence_count(const int count) const {
        if (explore_starvation_weighting_enabled()) {
            return static_cast<int>(std::lround(static_cast<double>(cadence_scaled(count)) *
                                                static_cast<double>(kExploreStarvDose)));
        }
        return static_cast<int>(std::lround(static_cast<double>(cadence_scaled(count)) *
                                            static_cast<double>(_far_starvation)));
    }

    float MRNF::effective_far_growth_cap() const {
        if (!background_improvements_enabled()) {
            return 1.0f;
        }
        return 1.0f - _far_starvation * (1.0f - kFarGrowthCap);
    }

    float MRNF::effective_far_decay_scale() const {
        if (!background_improvements_enabled()) {
            return 1.0f;
        }
        return 1.0f - _far_starvation * (1.0f - kFarDecayScale);
    }

    float MRNF::effective_mean_step_ratio_max() const {
        if (!background_improvements_enabled()) {
            return 1.0f;
        }
        return 1.0f + _far_starvation * (kPerSplatMeanStepRatioMax - 1.0f);
    }

    float MRNF::far_starvation_factor(const float ratio, const float full, const float rich) {
        if (!std::isfinite(ratio) || !std::isfinite(full) || !std::isfinite(rich) ||
            !(full > 0.0f) || !(rich > full)) {
            return 0.0f;
        }
        return std::clamp((rich - ratio) / (rich - full), 0.0f, 1.0f);
    }

    float MRNF::explore_starvation_multiplier(const float vis_i, const float median_vis) {
        if (vis_i == 0.0f) {
            return 0.0f;
        }
        const float denom = std::max(median_vis, std::numeric_limits<float>::epsilon());
        const float starved = std::clamp(1.0f - vis_i / denom, 0.0f, 1.0f);
        const float term = std::pow(starved, kStarvGamma);
        return kStarvEps + term;
    }

    void MRNF::update_far_starvation() {
        // Params and P are bound at initialize(); skip the pre-init hull/setter pass.
        if (!_params || _initial_sfm_point_count == 0) {
            return;
        }
        // Census-active scenes keep s = 1 at any capacity for all far features.
        // Census-inert scenes anneal mask-gated features with the cap/points ramp
        // (uncapped = rich / 0). Exploration dose is independent of s.
        const bool uncapped = _params->max_cap == 0;
        const bool census_forces_full = _scene_has_far_field;
        float s = 0.0f;
        float ratio = 0.0f;
        if (census_forces_full) {
            s = 1.0f;
        } else if (!uncapped) {
            const float max_cap = static_cast<float>(_params->max_cap);
            const float points = static_cast<float>(_initial_sfm_point_count);
            ratio = max_cap / points;
            s = far_starvation_factor(ratio, kFarCapRatioFull, kFarCapRatioRich);
        }
        _far_starvation = s;
        if (std::abs(s - _logged_far_starvation) > 0.1f) {
            if (census_forces_full) {
                LOG_INFO("MRNF: far starvation 1.00 (far-field scene)");
            } else if (uncapped) {
                LOG_INFO("MRNF: far starvation 0.00 (uncapped)");
            } else {
                LOG_INFO("MRNF: far starvation {:.2f} (cap/points {:.2f})", s, ratio);
            }
            _logged_far_starvation = s;
        }
        if (_optimizer && _optimizer->per_splat_mean_step()) {
            _optimizer->set_per_splat_mean_step(
                true, _median_splat_extent, kPerSplatMeanStepRatioMin,
                effective_mean_step_ratio_max());
        }
    }

    void MRNF::refresh_far_field_mask(const size_t n) {
        using namespace lfs::core;
        if (!_camera_hull_valid || n == 0) {
            _far_field_mask = Tensor();
            update_far_starvation();
            publish_mean_step_far_mask();
            return;
        }
        if (!_far_field_mask.is_valid() ||
            _far_field_mask.device() != Device::CUDA ||
            _far_field_mask.dtype() != DataType::Bool ||
            _far_field_mask.numel() != n) {
            _far_field_mask = Tensor::zeros_bool({n}, Device::CUDA);
        }
        mrnf_strategy::launch_far_field_mask(
            _splat_data->means().ptr<float>(),
            _cam_centroid[0],
            _cam_centroid[1],
            _cam_centroid[2],
            kFarMaskOrbits * _orbit_radius,
            _far_field_mask.ptr<bool>(),
            n);
        publish_mean_step_far_mask();
    }

    void MRNF::publish_mean_step_far_mask() {
        if (!_optimizer) {
            return;
        }
        if (!background_improvements_enabled()) {
            _optimizer->set_mean_step_far_mask(nullptr, 0);
            return;
        }
        const size_t n = _splat_data ? static_cast<size_t>(_splat_data->size()) : 0;
        if (!_camera_hull_valid || n == 0 || !_far_field_mask.is_valid() ||
            _far_field_mask.numel() != n) {
            _optimizer->set_mean_step_far_mask(nullptr, 0);
            return;
        }
        _optimizer->set_mean_step_far_mask(_far_field_mask.ptr<bool>(), static_cast<int>(n));
    }

    void MRNF::ensure_mean_step_far_mask() {
        if (!background_improvements_enabled()) {
            publish_mean_step_far_mask();
            return;
        }
        const size_t n = _splat_data ? static_cast<size_t>(_splat_data->size()) : 0;
        refresh_far_field_mask(n);
    }

    void MRNF::begin_far_growth_window(const size_t n, const int reserved_seeds) {
        using namespace lfs::core;
        _far_growth = FarGrowthState{};
        _far_growth.cap = effective_far_growth_cap();
        _far_growth.reserved_for_seeds = std::max(0, reserved_seeds);
        if (!_camera_hull_valid || !_params || _far_growth.cap >= 1.0f || n == 0) {
            _far_growth.active = false;
            if (_camera_hull_valid && n > 0) {
                refresh_far_field_mask(n);
                _far_growth.outside_mask = _far_field_mask;
            }
            return;
        }

        _far_growth.active = true;
        refresh_far_field_mask(n);
        _far_growth.outside_mask = _far_field_mask;
    }

    lfs::core::Tensor MRNF::sample_gumbel_with_far_guard(
        const lfs::core::Tensor& weights,
        const int k,
        const uint64_t seed,
        const size_t known_nnz) {
        using namespace lfs::core;
        if (k <= 0 || !weights.is_valid() || weights.numel() == 0) {
            return {};
        }

        const size_t n = weights.numel();
        _gumbel_scratch.ensure_n(n, Device::CUDA);
        if (!_far_growth.active || !_far_growth.outside_mask.is_valid() ||
            _far_growth.outside_mask.numel() != n) {
            auto indices = Tensor::empty({static_cast<size_t>(k)}, Device::CUDA, DataType::Int64);
            mrnf_strategy::launch_gumbel_topk(
                weights.ptr<float>(), n, static_cast<size_t>(k), seed, indices.ptr<int64_t>(),
                nullptr, true, &_gumbel_scratch, known_nnz);
            _far_growth.allocated += k;
            return indices;
        }

        if (!_refine_counts_dev.is_valid() || _refine_counts_dev.numel() < 4) {
            _refine_counts_dev = Tensor::zeros({4}, Device::CUDA, DataType::Int64);
        }

        auto weights_out = weights.masked_fill(_far_growth.outside_mask.logical_not(), 0.0f);
        auto weights_in = weights.masked_fill(_far_growth.outside_mask, 0.0f);

        kernels::launch_packed_refine_counts(
            nullptr, 0, nullptr, 0,
            weights_out.ptr<float>(), n,
            weights_in.ptr<float>(), n,
            _refine_counts_dev.ptr<int64_t>());
        int64_t host_counts[4] = {0, 0, 0, 0};
        LFS_CUDA_CHECK_MSG(
            cudaMemcpy(host_counts, _refine_counts_dev.ptr<int64_t>(),
                       4 * sizeof(int64_t), cudaMemcpyDeviceToHost),
            "MRNF far-guard pool nnz D2H");
        const int selectable_out = static_cast<int>(host_counts[2]);
        const int selectable_in = static_cast<int>(host_counts[3]);

        const int prospective_total = _far_growth.allocated + k + _far_growth.reserved_for_seeds;
        const int window_out_budget = static_cast<int>(
            std::lround(static_cast<double>(_far_growth.cap) * static_cast<double>(prospective_total)));
        const int remaining_window_out = std::max(
            0, window_out_budget - _far_growth.outside_used - _far_growth.reserved_for_seeds);
        const int stage_out = static_cast<int>(
            std::lround(static_cast<double>(_far_growth.cap) * static_cast<double>(k)));
        int k_out = std::min({k, remaining_window_out, selectable_out, std::max(0, stage_out)});
        int k_in = k - k_out;
        if (k_in > selectable_in) {
            k_in = selectable_in;
        }

        Tensor out_inds;
        Tensor in_inds;
        if (k_out > 0) {
            out_inds = Tensor::empty({static_cast<size_t>(k_out)}, Device::CUDA, DataType::Int64);
            mrnf_strategy::launch_gumbel_topk(
                weights_out.ptr<float>(), n, static_cast<size_t>(k_out), seed,
                out_inds.ptr<int64_t>(), nullptr, true, &_gumbel_scratch,
                static_cast<size_t>(selectable_out));
        }
        if (k_in > 0) {
            in_inds = Tensor::empty({static_cast<size_t>(k_in)}, Device::CUDA, DataType::Int64);
            mrnf_strategy::launch_gumbel_topk(
                weights_in.ptr<float>(), n, static_cast<size_t>(k_in), seed + 17,
                in_inds.ptr<int64_t>(), nullptr, true, &_gumbel_scratch,
                static_cast<size_t>(selectable_in));
        }

        Tensor selected;
        if (out_inds.is_valid() && out_inds.numel() > 0 && in_inds.is_valid() && in_inds.numel() > 0) {
            selected = Tensor::cat({out_inds, in_inds}, 0);
        } else if (out_inds.is_valid() && out_inds.numel() > 0) {
            selected = out_inds;
        } else if (in_inds.is_valid() && in_inds.numel() > 0) {
            selected = in_inds;
        }

        _far_growth.allocated += static_cast<int>(selected.is_valid() ? selected.numel() : 0);
        _far_growth.outside_used += k_out;
        return selected;
    }

    void MRNF::apply_explore_starvation_weights(lfs::core::Tensor& weights, const size_t n) {
        using namespace lfs::core;
        if (!weights.is_valid() || weights.numel() != n ||
            !_vis_count.is_valid() || _vis_count.numel() != n) {
            return;
        }

        float median_vis = 0.0f;
        Tensor live_vis = _vis_count;
        if (_free_mask.is_valid() && _free_mask.numel() >= n) {
            live_vis = _vis_count.masked_select(_free_mask.slice(0, 0, n).logical_not());
        }
        if (live_vis.is_valid() && live_vis.numel() > 0) {
            mrnf_strategy::launch_sorted_median(
                live_vis.ptr<float>(), live_vis.numel(), &median_vis, &_median_scratch);
        }
        mrnf_strategy::launch_apply_explore_starvation_weights(
            weights.ptr<float>(), _vis_count.ptr<float>(), n, median_vis);
    }

    lfs::core::Tensor MRNF::build_explore_split_weights(
        const size_t n,
        const lfs::core::Tensor& active_mask,
        const lfs::core::Tensor& trainable_mask,
        const lfs::core::Tensor& replace_mask,
        const lfs::core::Tensor& growth_inds) {
        using namespace lfs::core;
        auto log_scales = _splat_data->scaling_raw();
        auto score_mean = Tensor::zeros({n}, Device::CUDA);
        if (_explore_score_sum.is_valid() &&
            _explore_score_sum.ndim() == 1 &&
            _explore_score_sum.numel() == n) {
            const float denom = static_cast<float>(std::max(_explore_sample_count, 1));
            score_mean = _explore_score_sum / denom;
        }
        auto logw = log_scales.sum(1) + (score_mean + MRNF_EXPLORE_SCORE_FLOOR).log();
        auto explore_weights = (logw - logw.max()).exp();
        if (explore_starvation_weighting_enabled()) {
            apply_explore_starvation_weights(explore_weights, n);
        }
        if (active_mask.is_valid()) {
            explore_weights = explore_weights * active_mask;
        }
        if (trainable_mask.is_valid()) {
            explore_weights = explore_weights * trainable_mask;
        }
        explore_weights = apply_crop_damping_to_scores(*_optimizer, explore_weights);
        if (replace_mask.is_valid()) {
            explore_weights = explore_weights.masked_fill(replace_mask, 0.0f);
        }
        if (growth_inds.is_valid() && growth_inds.numel() > 0) {
            auto growth_mask = Tensor::zeros_bool({n}, Device::CUDA);
            auto true_vals = Tensor::ones_bool({growth_inds.numel()}, Device::CUDA);
            growth_mask.index_put_(growth_inds, true_vals);
            explore_weights = explore_weights.masked_fill(growth_mask, 0.0f);
        }
        return explore_weights;
    }

    void MRNF::grow_and_split(int iter, int pruned_count) {
        LOG_TIMER("MRNF::grow_and_split");
        LFS_VRAM_SCOPE("MRNF::grow_and_split");
        using namespace lfs::core;
        // Expand q16 → float if needed. Do NOT re-encode here: refine() owns the single
        // post-growth commit so bounds/codes always match the final N. Tests that call
        // grow_and_split alone must commit themselves or force quant OFF (Legacy guard).
        (void)lfs::training::sh_value::ensure_shN_fp32_for_mutation(*_splat_data);

        const size_t n = static_cast<size_t>(_splat_data->size());
        if (!_far_growth.outside_mask.is_valid() || _far_growth.outside_mask.numel() != n) {
            begin_far_growth_window(n, 0);
        }
        const size_t current_active = active_count();
        // densify N-scratch pre-sized to max_cap; views avoid per-refine
        // driver allocs (post-refine trim_memory_pool would otherwise force
        // pool misses on every growing exact size).
        _densify_n_scratch.ensure_n(n, Device::CUDA);
        if (PerfBenchCollector::enabled()) {
            PerfBenchCollector::instance().set_densify_workspace_bytes(
                _densify_n_scratch.resident_bytes());
        }
        publish_vram_attribution();

        lfs::core::Tensor active_mask;
        if (_free_mask.is_valid() && n > 0) {
            active_mask = _free_mask.slice(0, 0, n).logical_not();
        }
        lfs::core::Tensor trainable_mask = make_trainable_mask(*_splat_data, n, _splat_data->means().device());
        auto refine_candidates = compute_refine_candidates();
        if (active_mask.is_valid()) {
            refine_candidates = refine_candidates.logical_and(active_mask);
        }
        if (trainable_mask.is_valid()) {
            refine_candidates = refine_candidates.logical_and(trainable_mask);
        }

        int budget = (_params->max_cap > 0)
                         ? std::max(0, _params->max_cap - static_cast<int>(current_active))
                         : INT_MAX;
        const int requested_replace = std::min(pruned_count, budget);
        int n_grow = 0;
        lfs::core::Tensor above_threshold;

        auto seed = static_cast<uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());

        const auto edge_guidance = edge_guidance_factor();

        Tensor split_indices;
        Tensor replace_inds;
        Tensor growth_inds;
        Tensor replace_mask;
        int actual_replace = 0;

        // Build replacement weights first so the candidate sum and nonzero count
        // share one D2H transfer. Store the final weights in grow-only scratch.
        Tensor replace_weights;
        if (requested_replace > 0) {
            replace_weights = build_replace_parent_weights(n, active_mask, trainable_mask, edge_guidance);
        }

        if (!_refine_counts_dev.is_valid() || _refine_counts_dev.numel() < 4) {
            _refine_counts_dev = Tensor::zeros({4}, Device::CUDA, DataType::Int64);
        }
        kernels::launch_packed_refine_counts(
            refine_candidates.ptr<bool>(), n,
            nullptr, 0,
            (replace_weights.is_valid() ? replace_weights.ptr<float>() : nullptr),
            (replace_weights.is_valid() ? n : 0),
            nullptr, 0,
            _refine_counts_dev.ptr<int64_t>());
        int64_t host_counts[4] = {0, 0, 0, 0};
        LFS_CUDA_CHECK_MSG(
            cudaMemcpy(host_counts, _refine_counts_dev.ptr<int64_t>(),
                       4 * sizeof(int64_t), cudaMemcpyDeviceToHost),
            "MRNF grow packed counts D2H");
        int desired_total = static_cast<int>(
            std::round(static_cast<float>(host_counts[0]) * _params->grow_fraction));
        const int candidate_count = static_cast<int>(host_counts[0]);

        int pacing_windows_left = 0;
        if (cfg_fill_target_iter() > 0 && _params->max_cap > 0) {
            const int refine_every = std::max(1, static_cast<int>(_params->refine_every));
            pacing_windows_left = std::max(1, (cfg_fill_target_iter() - iter) / refine_every);
            const long long remaining = std::max<long long>(
                0,
                static_cast<long long>(_params->max_cap) - static_cast<long long>(current_active));
            const long long paced =
                (remaining + static_cast<long long>(pacing_windows_left) - 1) /
                static_cast<long long>(pacing_windows_left); // ceil(remaining / windows_left)

            if (desired_total > 0 && paced < static_cast<long long>(desired_total)) {
                desired_total = static_cast<int>(std::min<long long>(paced, INT_MAX));
            }
        }
        const int selectable_replace = static_cast<int>(host_counts[2]);

        if (requested_replace > 0) {
            actual_replace = std::min(requested_replace, selectable_replace);
            if (actual_replace > 0) {
                // grow-only index/mask scratch (no per-refine driver alloc).
                _densify_n_scratch.ensure_n(n, Device::CUDA);
                replace_inds = sample_gumbel_with_far_guard(
                    replace_weights, actual_replace, seed,
                    static_cast<size_t>(selectable_replace));
                actual_replace = replace_inds.is_valid() ? static_cast<int>(replace_inds.numel()) : 0;
                if (actual_replace > 0) {
                    replace_mask = _densify_n_scratch.bool_a_view(n);
                    replace_mask.zero_();
                    auto true_vals = Tensor::ones_bool({static_cast<size_t>(actual_replace)}, Device::CUDA);
                    replace_mask.index_put_(replace_inds, true_vals);
                }
            }
        }

        if (iter < effective_grow_until_iter()) {
            above_threshold = refine_candidates;
            n_grow = std::max(0, desired_total - actual_replace);
            n_grow = std::min(n_grow, budget - actual_replace);
        }

        if (n_grow > 0) {
            const bool use_ratio_rank = cfg_ratio_rank_on() &&
                                        _refine_ratio_max.is_valid() &&
                                        _refine_ratio_max.numel() == n;
            Tensor growth_weights =
                above_threshold * (use_ratio_rank ? _refine_ratio_max : _refine_weight_max);
            if (edge_guidance.is_valid()) {
                growth_weights = growth_weights * edge_guidance;
            }

            if (replace_mask.is_valid()) {
                // Keep replacement and growth disjoint on device instead of
                // deduplicating sampled indices on the host.
                growth_weights = growth_weights.masked_fill(replace_mask, 0.0f);
            }

            // Growth nnz is data-dependent on replace_mask — second packed slot.
            kernels::launch_packed_refine_counts(
                nullptr, 0, nullptr, 0,
                growth_weights.ptr<float>(), n,
                nullptr, 0,
                _refine_counts_dev.ptr<int64_t>());
            LFS_CUDA_CHECK_MSG(
                cudaMemcpy(host_counts, _refine_counts_dev.ptr<int64_t>(),
                           4 * sizeof(int64_t), cudaMemcpyDeviceToHost),
                "MRNF growth nnz D2H");
            const int selectable_growth = static_cast<int>(host_counts[2]);
            if (selectable_growth > 0) {
                const int growth_budget = std::min(n_grow, selectable_growth);
                growth_inds = sample_gumbel_with_far_guard(
                    growth_weights, growth_budget, seed + 1,
                    static_cast<size_t>(selectable_growth));
            }
        }

        ++_growth_window_count;

        if (cfg_ratio_rank_on() &&
            (_growth_window_count % MRNF_RATIO_RANK_LOG_INTERVAL) == 1) {
            const int grow_sampled =
                growth_inds.is_valid() ? static_cast<int>(growth_inds.numel()) : 0;
            LOG_INFO("MRNF ratio-rank iter={} candidates={} grow={}",
                     iter, candidate_count, grow_sampled);
        }
        if (cfg_fill_target_iter() > 0 &&
            (_growth_window_count % MRNF_RATIO_RANK_LOG_INTERVAL) == 1) {
            LOG_INFO("MRNF pacing iter={} active={} desired={} windows_left={}",
                     iter, current_active, desired_total, pacing_windows_left);
        }
        Tensor explore_inds;
        if (iter < effective_grow_until_iter() && background_improvements_enabled() &&
            far_operators_active()) {
            const int growth_count = (growth_inds.is_valid() ? static_cast<int>(growth_inds.numel()) : 0);
            const int remaining_budget = std::max(0, budget - actual_replace - growth_count);
            int n_explore = std::min(starved_cadence_count(kExploreSplits), remaining_budget);
            if (n_explore > 0) {
                auto log_scales = _splat_data->scaling_raw();
                if (log_scales.ndim() != 2 || log_scales.shape()[0] != n || log_scales.shape()[1] != 3) {
                    n_explore = 0;
                } else {
                    auto explore_weights = build_explore_split_weights(
                        n, active_mask, trainable_mask, replace_mask, growth_inds);
                    if (!explore_weights.is_valid() || explore_weights.numel() != n) {
                        n_explore = 0;
                    } else {
                        kernels::launch_packed_refine_counts(
                            nullptr, 0, nullptr, 0,
                            explore_weights.ptr<float>(), n,
                            nullptr, 0,
                            _refine_counts_dev.ptr<int64_t>());
                        LFS_CUDA_CHECK_MSG(
                            cudaMemcpy(host_counts, _refine_counts_dev.ptr<int64_t>(),
                                       4 * sizeof(int64_t), cudaMemcpyDeviceToHost),
                            "MRNF explore nnz D2H");
                        const int selectable_explore = static_cast<int>(host_counts[2]);
                        n_explore = std::min(n_explore, selectable_explore);
                        if (n_explore > 0) {
                            explore_inds = sample_gumbel_with_far_guard(
                                explore_weights, n_explore, seed + 2,
                                static_cast<size_t>(selectable_explore));
                            n_explore = explore_inds.is_valid() ? static_cast<int>(explore_inds.numel()) : 0;
                            LFS_COUNTER_ADD("strategy.mrnf.explore_split", n_explore);
                        }
                    }
                }
            }
        }

        std::vector<Tensor> split_parts;
        if (replace_inds.is_valid() && replace_inds.numel() > 0) {
            split_parts.push_back(replace_inds);
        }
        if (growth_inds.is_valid() && growth_inds.numel() > 0) {
            split_parts.push_back(growth_inds);
        }
        if (explore_inds.is_valid() && explore_inds.numel() > 0) {
            split_parts.push_back(explore_inds);
        }
        if (split_parts.size() == 1) {
            split_indices = split_parts[0];
        } else if (split_parts.size() > 1) {
            split_indices = Tensor::cat(split_parts, 0);
        }

        if (!split_indices.is_valid() || split_indices.numel() == 0) {
            publish_vram_attribution();
            return;
        }

        assert(_params->max_cap <= 0 ||
               current_active + split_indices.numel() <= static_cast<size_t>(_params->max_cap));

        const size_t K = split_indices.numel();
        const size_t sh_rest = _splat_data->max_sh_coeffs_rest();
        const auto layout_rest = static_cast<uint32_t>(sh_rest);
        const bool use_shN = layout_rest > 0 &&
                             _splat_data->shN().is_valid() &&
                             _splat_data->shN().numel() > 0;

        _densify_ws.ensure(K, sh_rest, use_shN, /*sh0_flat_layout=*/false, Device::CUDA);
        publish_vram_attribution();
        auto child_means = _densify_ws.means_view(K);
        auto child_log_scales = _densify_ws.scales_view(K);
        auto child_raw_opacities = _densify_ws.opacities_view(K);
        auto child_rotations = _densify_ws.rotations_view(K);
        auto child_sh0 = _densify_ws.sh0_view(K);
        Tensor child_shN = use_shN ? _densify_ws.shN_view(K) : Tensor();

        // The LAS kernel only needs linear shN to copy child rows. shN itself is unchanged
        // for the parent rows, so keep the resident swizzled buffer in place and gather the
        // selected child rows below.
        kernels::launch_long_axis_split_gaussians_inplace(
            _splat_data->means().ptr<float>(),
            _splat_data->rotation_raw().ptr<float>(),
            _splat_data->scaling_raw().ptr<float>(),
            _splat_data->sh0().ptr<float>(),
            nullptr,
            _splat_data->opacity_raw().ptr<float>(),
            child_means.ptr<float>(),
            child_rotations.ptr<float>(),
            child_log_scales.ptr<float>(),
            child_sh0.ptr<float>(),
            nullptr,
            child_raw_opacities.ptr<float>(),
            split_indices.ptr<int64_t>(),
            static_cast<int>(K),
            0,
            nullptr);

        if (use_shN) {
            shN_swizzled_gather_to_linear_i64(
                _splat_data->shN().ptr<float>(),
                split_indices.ptr<int64_t>(),
                child_shN.ptr<float>(),
                K,
                layout_rest,
                layout_rest);
        }

        reset_optimizer_state_at_indices(*_optimizer, ParamType::Means, split_indices);
        reset_optimizer_state_at_indices(*_optimizer, ParamType::Sh0, split_indices);
        reset_optimizer_state_at_indices(*_optimizer, ParamType::ShN, split_indices, layout_rest);
        reset_optimizer_state_at_indices(*_optimizer, ParamType::Scaling, split_indices);
        reset_optimizer_state_at_indices(*_optimizer, ParamType::Rotation, split_indices);
        reset_optimizer_state_at_indices(*_optimizer, ParamType::Opacity, split_indices);

        size_t append_start = 0;
        if (free_count() > 0) {
            auto [filled_indices, remaining_after_fill] = fill_free_slots_with_data(
                child_means,
                child_rotations,
                child_log_scales,
                child_sh0,
                child_shN,
                child_raw_opacities,
                static_cast<int64_t>(K));
            append_start = K - static_cast<size_t>(remaining_after_fill);
        }

        const size_t n_append = append_child_rows(
            child_means,
            child_rotations,
            child_log_scales,
            child_sh0,
            child_shN,
            child_raw_opacities,
            append_start,
            K);

        LOG_DEBUG("MRNF: split {} splats at iter {} (reused: {}, appended: {}, active: {}, total slots: {})",
                  K, iter, append_start, n_append, active_count(), _splat_data->size());
        LFS_COUNTER_ADD("strategy.mrnf.split", K);
        LFS_COUNTER_ADD("strategy.mrnf.appended", n_append);
        LFS_GAUGE("model.gaussians.live", active_count());
        LFS_GAUGE("model.gaussians.capacity", static_cast<double>(_splat_data->size()));
        publish_vram_attribution();
    }

    lfs::core::Tensor MRNF::build_replace_parent_weights(
        const size_t n,
        const lfs::core::Tensor& active_mask,
        const lfs::core::Tensor& trainable_mask,
        const lfs::core::Tensor& edge_guidance) const {
        using namespace lfs::core;

        Tensor replace_weights;
        {
            auto opacities = _splat_data->get_opacity();
            if (opacities.ndim() == 2 && opacities.shape()[1] == 1)
                opacities = opacities.squeeze(-1);
            replace_weights = opacities * (_vis_count > 0.0f);
        }
        if (active_mask.is_valid()) {
            replace_weights = replace_weights * active_mask;
        }
        if (trainable_mask.is_valid()) {
            replace_weights = replace_weights * trainable_mask;
        }
        if (edge_guidance.is_valid()) {
            replace_weights = replace_weights * edge_guidance;
        }

        // Stash into scratch so subsequent growth_weights path can reuse
        // the same physical buffer after replace stage is done.
        auto w_view = _densify_n_scratch.f32_a_view(n);
        w_view.copy_(replace_weights);
        return w_view;
    }

    lfs::core::Tensor MRNF::compute_refine_candidates() const {
        using namespace lfs::core;
        auto candidate_weights = apply_crop_damping_to_scores(*_optimizer, _refine_weight_max);
        return (candidate_weights > _params->growth_grad_threshold) && (_vis_count > 0.0f);
    }

    void MRNF::compact_splats(const lfs::core::Tensor& keep_mask) {
        LOG_TIMER("MRNF::compact_splats");
        // Float-native gather. Expand q16 if needed; refine() (or remove_gaussians) owns re-encode.
        (void)lfs::training::sh_value::ensure_shN_fp32_for_mutation(*_splat_data);
        using namespace lfs::core;

        const size_t old_size = static_cast<size_t>(_splat_data->size());
        Tensor valid_indices = keep_mask.nonzero().squeeze(-1);
        const size_t new_size = valid_indices.numel();
        const size_t cap = _params->max_cap > 0 ? static_cast<size_t>(_params->max_cap) : 0;

        // Gather kept rows into one max-capacity destination so compaction keeps
        // at most the source and destination allocations live concurrently.
        auto compact = [&](Tensor& t) {
            if (!t.is_valid() || t.numel() == 0)
                return;
            auto dims = t.shape().dims();
            dims[0] = new_size;
            const size_t dest_cap = cap > 0 ? cap : new_size;
            Tensor dest = Tensor::zeros_direct(
                TensorShape(dims), dest_cap, t.device(), t.dtype());
            t.index_select_into(dest, 0, valid_indices, BoundaryMode::Assert);
            t = std::move(dest);
        };

        // shN is swizzled — compact via block-aware gather.
        const auto layout_rest_u32 = static_cast<uint32_t>(_splat_data->max_sh_coeffs_rest());
        auto compact_shN_swizzled = [&](Tensor& t, size_t cap_rows, int uint8_fill = -1) {
            if (!t.is_valid() || t.numel() == 0)
                return;
            if (layout_rest_u32 == 0)
                return;
            auto idx_i32 = valid_indices.dtype() == lfs::core::DataType::Int32
                               ? valid_indices
                               : valid_indices.to(lfs::core::DataType::Int32);
            const size_t cap_floats = cap_rows > 0 ? lfs::core::sh_swizzled_float_count(cap_rows, layout_rest_u32)
                                                   : lfs::core::sh_swizzled_float_count(new_size, layout_rest_u32);
            const size_t logical_floats = lfs::core::sh_swizzled_float_count(new_size, layout_rest_u32);
            auto fresh = Tensor::zeros_direct(TensorShape({logical_floats}), cap_floats, t.device(), t.dtype());
            if (t.dtype() == DataType::Float32) {
                lfs::core::shN_swizzled_gather_self(
                    t.ptr<float>(), fresh.ptr<float>(),
                    idx_i32.ptr<int>(), new_size, 0, layout_rest_u32);
            } else if (t.dtype() == DataType::UInt8 || t.dtype() == DataType::Bool) {
                if (uint8_fill >= 0 && cap_floats > 0) {
                    const cudaError_t err = cudaMemsetAsync(
                        fresh.ptr<uint8_t>(),
                        static_cast<unsigned char>(uint8_fill),
                        cap_floats * sizeof(uint8_t),
                        fresh.stream());
                    if (err != cudaSuccess) {
                        throw std::runtime_error(
                            std::string("MRNF::compact_splats: cudaMemsetAsync failed: ") +
                            cudaGetErrorString(err));
                    }
                }
                lfs::core::shN_swizzled_gather_self_u8(
                    t.ptr<uint8_t>(), fresh.ptr<uint8_t>(),
                    idx_i32.ptr<int>(), new_size, 0, layout_rest_u32);
            } else {
                throw std::runtime_error("MRNF::compact_splats: unsupported swizzled shN dtype");
            }
            t = std::move(fresh);
        };

        compact(_splat_data->means());
        compact(_splat_data->sh0());
        if (_splat_data->shN().is_valid() && _splat_data->shN().numel() > 0)
            compact_shN_swizzled(_splat_data->shN(), cap);
        compact(_splat_data->scaling_raw());
        compact(_splat_data->rotation_raw());
        compact(_splat_data->opacity_raw());

        static constexpr ParamType ALL_PARAMS[] = {
            ParamType::Means, ParamType::Sh0, ParamType::ShN,
            ParamType::Scaling, ParamType::Rotation, ParamType::Opacity};

        for (auto pt : ALL_PARAMS) {
            auto* state = _optimizer->get_state_mutable(pt);
            if (!state)
                continue;
            if (state->is_joint()) {
                // Compact packed rows + rebuild zero bounds → free-zero moments
                // (decode under zero bounds is (m,v)=(0,0) regardless of codes).
                if (pt == ParamType::ShN) {
                    // 1D packed [n_floats * bpc]: allocate correct size (zero free-init).
                    // Swizzled gather of multi-byte cells is not needed when bounds are
                    // zeroed — moments restart clean after compact.
                    const int bpc = state->joint_bits == 16 ? 4 : 2;
                    const size_t logical_floats =
                        lfs::core::sh_swizzled_float_count(new_size, layout_rest_u32);
                    const size_t cap_floats = cap > 0
                                                  ? lfs::core::sh_swizzled_float_count(cap, layout_rest_u32)
                                                  : logical_floats;
                    const size_t logical_bytes = logical_floats * static_cast<size_t>(bpc);
                    const size_t cap_bytes = cap_floats * static_cast<size_t>(bpc);
                    state->exp_avg = Tensor::zeros_direct(
                        TensorShape({logical_bytes}), cap_bytes, Device::CUDA, DataType::UInt8);
                    state->size = logical_floats;
                    state->capacity = cap_floats;
                } else {
                    compact(state->exp_avg);
                    state->size = new_size;
                    state->capacity = cap;
                }
                // grow-only zero bounds (free-zero moments after compact).
                ensure_joint_bounds_capacity(state->joint_bounds, new_size, cap,
                                             Device::CUDA, /*zero_all=*/true);
            } else if (pt == ParamType::ShN) {
                compact_shN_swizzled(state->exp_avg, cap, 128);
                state->size = lfs::core::sh_swizzled_float_count(new_size, layout_rest_u32);
                state->capacity = cap > 0 ? lfs::core::sh_swizzled_float_count(cap, layout_rest_u32)
                                          : lfs::core::sh_swizzled_float_count(new_size, layout_rest_u32);
            } else {
                compact(state->exp_avg);
                state->size = new_size;
                state->capacity = cap;
            }
            // Grad buffers match param dtype/shape (fp32), not joint packed bytes.
            if (pt == ParamType::ShN) {
                if (layout_rest_u32 > 0 && state->capacity > 0) {
                    const size_t logical_floats =
                        lfs::core::sh_swizzled_float_count(new_size, layout_rest_u32);
                    state->grad = Tensor::zeros_direct(
                        TensorShape({logical_floats}), state->capacity, Device::CUDA);
                } else {
                    state->grad = {};
                }
            } else if (state->exp_avg.is_valid()) {
                // Param shape for this type (means/sh0/etc.)
                Tensor* param_t = nullptr;
                switch (pt) {
                case ParamType::Means:
                    param_t = &_splat_data->means();
                    break;
                case ParamType::Sh0:
                    param_t = &_splat_data->sh0();
                    break;
                case ParamType::Scaling:
                    param_t = &_splat_data->scaling_raw();
                    break;
                case ParamType::Rotation:
                    param_t = &_splat_data->rotation_raw();
                    break;
                case ParamType::Opacity:
                    param_t = &_splat_data->opacity_raw();
                    break;
                default:
                    break;
                }
                if (param_t && param_t->is_valid()) {
                    if (cap > 0) {
                        state->grad = Tensor::zeros_direct(
                            param_t->shape(), cap, param_t->device());
                    } else {
                        state->grad = Tensor::zeros(param_t->shape(), param_t->device());
                    }
                } else if (!state->is_joint()) {
                    // Legacy fallback: moments share param shape
                    if (cap > 0) {
                        state->grad = Tensor::zeros_direct(
                            state->exp_avg.shape(), cap, state->exp_avg.device());
                    } else {
                        state->grad = Tensor::zeros(state->exp_avg.shape(), state->exp_avg.device());
                    }
                }
            }
            // Capacity invariant after compact: capacity >= size.
            LFS_DEBUG_ASSERT_MSG(state->capacity >= state->size,
                                 "MRNF::compact_splats: state.capacity < state.size");
        }

        const auto& info = _splat_data->_densification_info;
        if (info.is_valid() && info.ndim() == 2 && info.shape()[1] == old_size) {
            // densification_info is [2, N] — capacity is along dim 0, so allocate exact
            // gather into a fresh [2, new_size] (no max_cap reserve on this aux).
            auto dims = info.shape().dims();
            dims[1] = new_size;
            Tensor dest = Tensor::zeros(TensorShape(dims), info.device(), info.dtype());
            info.index_select_into(dest, 1, valid_indices, BoundaryMode::Assert);
            _splat_data->_densification_info = std::move(dest);
        }
        // deleted mask must track the new live N for VkSplat. Compact
        // when sized to the pre-compact N; otherwise rebuild/clear so a stale
        // pre-compact mask cannot freeze the viewport after training.
        if (_splat_data->has_deleted_mask()) {
            if (_splat_data->deleted().numel() == old_size) {
                compact(_splat_data->deleted());
                _splat_data->notify_deleted_mask_changed();
                _splat_data->refresh_deleted_count();
            } else {
                LOG_WARN("MRNF::compact_splats: deleted mask numel {} != old size {}; reconciling",
                         _splat_data->deleted().numel(), old_size);
                _splat_data->reconcile_deleted_mask();
            }
        }
        if (_free_mask.is_valid() && old_size > 0) {
            // Compact the live prefix of free_mask, then extend to max_cap with free=false tail.
            auto live = _free_mask.slice(0, 0, old_size);
            auto dims = live.shape().dims();
            dims[0] = new_size;
            const size_t dest_cap = cap > 0 ? cap : new_size;
            Tensor dest = Tensor::zeros_direct(
                TensorShape(dims), dest_cap, live.device(), live.dtype());
            live.index_select_into(dest, 0, valid_indices, BoundaryMode::Assert);
            if (cap > new_size) {
                // Tail beyond new_size is already zeroed by zeros_direct → free=false.
                // Grow logical size to cap so free_mask covers the reserved range.
                dest.append_zeros(cap - new_size);
            }
            _free_mask = std::move(dest);
        }
        if (_refine_weight_max.is_valid() && _refine_weight_max.numel() > new_size)
            compact(_refine_weight_max);
        if (_refine_ratio_max.is_valid() && _refine_ratio_max.numel() > new_size)
            compact(_refine_ratio_max);
        if (_vis_count.is_valid() && _vis_count.numel() > new_size)
            compact(_vis_count);
        if (_precomputed_edge_scores.is_valid() && _precomputed_edge_scores.numel() > new_size)
            compact(_precomputed_edge_scores);
        if (_explore_score_sum.is_valid() && _explore_score_sum.numel() > new_size)
            compact(_explore_score_sum);

        remap_frozen_ranges_after_compaction(*_splat_data, valid_indices, old_size);
        apply_frozen_ranges_to_optimizer(*_splat_data, *_optimizer);
    }

    void MRNF::inject_noise(int /*iter*/) {
        const size_t n = static_cast<size_t>(_splat_data->size());
        if (n == 0)
            return;

        const float lr_mean = static_cast<float>(_optimizer->get_param_lr(ParamType::Means));

        auto seed = static_cast<uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
        const auto frozen_mask = make_frozen_mask(*_splat_data, n, _splat_data->means().device());

        mrnf_strategy::launch_mrnf_noise_injection(
            _splat_data->means().ptr<float>(),
            _splat_data->opacity_raw().ptr<float>(),
            _vis_count.ptr<float>(),
            frozen_mask.is_valid() ? frozen_mask.ptr<bool>() : nullptr,
            frozen_mask.is_valid() ? frozen_mask.numel() : 0,
            lr_mean,
            _params->means_noise_weight,
            _bounds.median_size,
            n, seed);
    }

    void MRNF::apply_decay(int iter) {
        const size_t n = static_cast<size_t>(_splat_data->size());
        if (n == 0)
            return;

        const float train_t = static_cast<float>(iter) / static_cast<float>(_params->iterations);
        const auto frozen_mask = make_frozen_mask(*_splat_data, n, _splat_data->means().device());
        const bool scale_far = background_improvements_enabled() &&
                               _camera_hull_valid;
        if (scale_far) {
            refresh_far_field_mask(n);
        }

        mrnf_strategy::launch_mrnf_decay(
            _splat_data->opacity_raw().ptr<float>(),
            _splat_data->scaling_raw().ptr<float>(),
            frozen_mask.is_valid() ? frozen_mask.ptr<bool>() : nullptr,
            frozen_mask.is_valid() ? frozen_mask.numel() : 0,
            scale_far && _far_field_mask.is_valid() ? _far_field_mask.ptr<bool>() : nullptr,
            scale_far && _far_field_mask.is_valid() ? _far_field_mask.numel() : 0,
            _params->opacity_decay,
            _params->scale_decay,
            scale_far ? effective_far_decay_scale() : 1.0f,
            train_t,
            n);
    }

    void MRNF::enforce_max_cap() {
        if (_params->max_cap <= 0)
            return;

        using namespace lfs::core;

        const size_t n = _splat_data->size();
        const size_t cap = static_cast<size_t>(_params->max_cap);
        if (n <= cap)
            return;

        LOG_INFO("MRNF: count {} exceeds max_cap {}, pruning excess", n, cap);

        auto opacities = _splat_data->get_opacity();
        if (opacities.ndim() == 2 && opacities.shape()[1] == 1)
            opacities = opacities.squeeze(-1);
        opacities = apply_crop_damping_to_scores(*_optimizer, opacities);

        auto seed = static_cast<uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());

        auto keep_mask = Tensor::zeros_bool({n}, opacities.device());
        const auto frozen_mask = make_frozen_mask(*_splat_data, n, opacities.device());
        size_t keep_budget = cap;
        if (frozen_mask.is_valid()) {
            const size_t frozen_count = frozen_row_count(*_splat_data, n);
            if (frozen_count > cap) {
                LOG_WARN("MRNF: {} frozen splats exceed max_cap {}; preserving frozen rows", frozen_count, cap);
                return;
            }
            keep_mask = frozen_mask.clone();
            keep_budget = cap - frozen_count;
            opacities = opacities.masked_fill(frozen_mask, 0.0f);
        }

        if (keep_budget > 0) {
            if (!_refine_counts_dev.is_valid() || _refine_counts_dev.numel() < 4) {
                _refine_counts_dev = Tensor::zeros({4}, Device::CUDA, DataType::Int64);
            }
            kernels::launch_packed_refine_counts(
                nullptr, 0, nullptr, 0,
                opacities.ptr<float>(), n,
                nullptr, 0,
                _refine_counts_dev.ptr<int64_t>());
            int64_t host_counts[4] = {0, 0, 0, 0};
            LFS_CUDA_CHECK_MSG(
                cudaMemcpy(host_counts, _refine_counts_dev.ptr<int64_t>(),
                           4 * sizeof(int64_t), cudaMemcpyDeviceToHost),
                "MRNF enforce_max_cap nnz D2H");
            auto keep_indices = Tensor::empty({keep_budget}, Device::CUDA, DataType::Int64);
            _gumbel_scratch.ensure_n(n, Device::CUDA);
            mrnf_strategy::launch_gumbel_topk(
                opacities.ptr<float>(), n, keep_budget, seed,
                keep_indices.ptr<int64_t>(), nullptr, true, &_gumbel_scratch,
                static_cast<size_t>(host_counts[2]));

            auto true_vals = Tensor::ones_bool({keep_budget}, opacities.device());
            keep_mask.index_put_(keep_indices, true_vals);
        }
        compact_splats(keep_mask);

        assert(_splat_data->size() <= cap);
    }

    size_t MRNF::active_count() const {
        if (!_free_mask.is_valid()) {
            return static_cast<size_t>(_splat_data->size());
        }

        const size_t current_size = static_cast<size_t>(_splat_data->size());
        if (current_size == 0)
            return 0;

        auto active_region = _free_mask.slice(0, 0, current_size);
        const size_t free_count_val = static_cast<size_t>(active_region.sum_scalar());
        return current_size - free_count_val;
    }

    size_t MRNF::free_count() const {
        if (!_free_mask.is_valid()) {
            return 0;
        }

        const size_t current_size = static_cast<size_t>(_splat_data->size());
        if (current_size == 0)
            return 0;

        auto active_region = _free_mask.slice(0, 0, current_size);
        return static_cast<size_t>(active_region.sum_scalar());
    }

    lfs::core::Tensor MRNF::get_active_indices() const {
        const size_t current_size = static_cast<size_t>(_splat_data->size());
        if (current_size == 0) {
            return {};
        }

        if (!_free_mask.is_valid() || free_count() == 0) {
            auto all_active = lfs::core::Tensor::ones_bool({current_size}, _splat_data->means().device());
            return all_active.nonzero().squeeze(-1);
        }

        auto active_region = _free_mask.slice(0, 0, current_size);
        auto is_active = active_region.logical_not();
        return is_active.nonzero().squeeze(-1);
    }

    void MRNF::mark_as_free(const lfs::core::Tensor& indices) {
        if (!_free_mask.is_valid() || indices.numel() == 0) {
            return;
        }

        auto target_indices = indices;
        if (auto frozen_mask = make_frozen_mask(*_splat_data, _splat_data->size(), indices.device());
            frozen_mask.is_valid()) {
            auto trainable = frozen_mask.index_select(0, indices).logical_not();
            target_indices = indices.index_select(0, trainable.nonzero().squeeze(-1));
            if (target_indices.numel() == 0) {
                return;
            }
        }

        auto true_vals = lfs::core::Tensor::ones_bool({static_cast<size_t>(target_indices.numel())}, target_indices.device());
        _free_mask.index_put_(target_indices, true_vals);
    }

    std::pair<lfs::core::Tensor, int64_t> MRNF::fill_free_slots_with_data(
        const lfs::core::Tensor& positions,
        const lfs::core::Tensor& rotations,
        const lfs::core::Tensor& scales,
        const lfs::core::Tensor& sh0,
        const lfs::core::Tensor& shN,
        const lfs::core::Tensor& opacities,
        int64_t count) {

        using namespace lfs::core;

        if (!_free_mask.is_valid() || count == 0) {
            return {Tensor(), count};
        }

        const size_t current_size = static_cast<size_t>(_splat_data->size());
        auto active_region = _free_mask.slice(0, 0, current_size);
        auto free_indices = active_region.nonzero().squeeze(-1);
        if (auto frozen_mask = make_frozen_mask(*_splat_data, current_size, free_indices.device());
            frozen_mask.is_valid() && free_indices.numel() > 0) {
            auto trainable = frozen_mask.index_select(0, free_indices).logical_not();
            free_indices = free_indices.index_select(0, trainable.nonzero().squeeze(-1));
        }
        const int64_t num_free = free_indices.numel();

        if (num_free == 0) {
            return {Tensor(), count};
        }

        const int64_t slots_to_fill = std::min(count, num_free);
        auto target_indices = free_indices.slice(0, 0, slots_to_fill);

        // one fused kernel writes all attrs + zeros Adam scales + clears free mask.
        float* adam_ptrs[12] = {};
        const int n_adam = collect_adam_scale_ptrs(*_optimizer, adam_ptrs);
        const int opacity_dim = (_splat_data->opacity_raw().ndim() == 2) ? 1 : 0;
        auto pos_slice = positions.slice(0, 0, slots_to_fill);
        auto rot_slice = rotations.slice(0, 0, slots_to_fill);
        auto scale_slice = scales.slice(0, 0, slots_to_fill);
        auto sh0_slice = sh0.slice(0, 0, slots_to_fill);
        auto opac_slice = opacities.slice(0, 0, slots_to_fill);

        kernels::launch_fill_free_slots_fused(
            target_indices.ptr<int64_t>(),
            static_cast<size_t>(slots_to_fill),
            pos_slice.ptr<float>(),
            rot_slice.ptr<float>(),
            scale_slice.ptr<float>(),
            sh0_slice.ptr<float>(),
            opac_slice.ptr<float>(),
            _splat_data->means().ptr<float>(),
            _splat_data->rotation_raw().ptr<float>(),
            _splat_data->scaling_raw().ptr<float>(),
            _splat_data->sh0().ptr<float>(),
            _splat_data->opacity_raw().ptr<float>(),
            opacity_dim,
            adam_ptrs,
            n_adam,
            _free_mask.ptr<bool>(),
            current_size);

        const auto layout_rest = static_cast<uint32_t>(_splat_data->max_sh_coeffs_rest());
        if (layout_rest > 0 && shN.is_valid() && shN.numel() > 0 &&
            _splat_data->shN().is_valid() && _splat_data->shN().numel() > 0) {
            auto target_i32 = target_indices.dtype() == DataType::Int32
                                  ? target_indices
                                  : target_indices.to(DataType::Int32);
            auto shN_slice = shN.slice(0, 0, slots_to_fill);
            shN_swizzled_scatter_linear(
                _splat_data->shN().ptr<float>(),
                target_i32.ptr<int>(),
                shN_slice.ptr<float>(),
                static_cast<size_t>(slots_to_fill),
                layout_rest,
                layout_rest);
        }

        // Zero residual grads so the post-densify Adam step does not use
        // previous-occupant / pre-split gradients on rewritten rows.
        zero_adam_grads_at_indices(*_optimizer, target_indices, layout_rest);

        set_deleted_mask_rows(*_splat_data, _free_mask, target_indices, false);

        return {target_indices, count - slots_to_fill};
    }

    size_t MRNF::append_child_rows(
        const lfs::core::Tensor& child_means,
        const lfs::core::Tensor& child_rotations,
        const lfs::core::Tensor& child_log_scales,
        const lfs::core::Tensor& child_sh0,
        const lfs::core::Tensor& child_shN,
        const lfs::core::Tensor& child_raw_opacities,
        const size_t append_start,
        const size_t K) {
        using namespace lfs::core;
        if (K <= append_start) {
            return 0;
        }

        const size_t n_append = K - append_start;
        const size_t old_size = static_cast<size_t>(_splat_data->size());
        const auto layout_rest = static_cast<uint32_t>(_splat_data->max_sh_coeffs_rest());
        const bool use_shN = layout_rest > 0 &&
                             _splat_data->shN().is_valid() &&
                             _splat_data->shN().numel() > 0 &&
                             child_shN.is_valid() &&
                             child_shN.numel() > 0;

        // capacity-ensure MUST succeed before free_mask or
        // any param mutates. A mid-commit throw left torn Means/Sh0 vs Scaling
        // and free_mask past size() → loss spikes then abort.
        if (!_optimizer->preflight_grow_capacity(n_append)) {
            LOG_ERROR(
                "MRNF densify aborted: capacity-ensure failed for {} -> {} rows "
                "(no params mutated)",
                old_size, old_size + n_append);
            return 0;
        }

        // Grow free_mask bookkeeping first, then params (size()), then the
        // deleted mask — never leave deleted.numel() != size() mid-grow
        // (viewer packer rejects stale masks and freezes the viewport).
        if (_free_mask.is_valid() && _free_mask.numel() < old_size + n_append) {
            _free_mask.reserve(old_size + n_append);
            _free_mask.append_zeros(n_append);
        }

        auto append_means = child_means.slice(0, append_start, K);
        auto append_sh0 = child_sh0.slice(0, append_start, K);
        Tensor append_shN;
        if (use_shN) {
            append_shN = child_shN.slice(0, append_start, K);
        }
        auto append_scaling = child_log_scales.slice(0, append_start, K);
        auto append_rotation = child_rotations.slice(0, append_start, K);
        auto append_opacity = child_raw_opacities.slice(0, append_start, K);
        if (_splat_data->opacity_raw().ndim() == 2) {
            append_opacity = append_opacity.unsqueeze(-1);
        }

        _optimizer->add_new_params(ParamType::Means, append_means, true);
        _optimizer->add_new_params(ParamType::Sh0, append_sh0, true);

        if (use_shN && append_shN.is_valid() && append_shN.numel() > 0) {
            const size_t new_size = old_size + n_append;
            const size_t needed_floats = sh_swizzled_float_count(new_size, layout_rest);
            auto& shN_buf = _splat_data->shN();
            // Grow capacity to means/max_cap headroom so post-densify re-encode is not exact-N.
            const size_t means_cap = _splat_data->means().is_valid()
                                         ? std::max(_splat_data->means().capacity(), new_size)
                                         : new_size;
            const size_t cap_floats = sh_swizzled_float_count(means_cap, layout_rest);
            if (shN_buf.capacity() < needed_floats) {
                const size_t dest_cap = std::max(needed_floats, cap_floats);
                auto grown = Tensor::zeros_direct(
                    TensorShape({static_cast<size_t>(shN_buf.numel())}), dest_cap,
                    shN_buf.device(), shN_buf.dtype());
                if (shN_buf.numel() > 0) {
                    // Sync copy before move-free of source (async UAF → illegal address).
                    LFS_CUDA_CHECK(cudaMemcpy(grown.data_ptr(), shN_buf.data_ptr(),
                                              shN_buf.bytes(), cudaMemcpyDeviceToDevice));
                }
                grown.set_name(shN_buf.name().empty() ? "splat.shN" : shN_buf.name());
                shN_buf = std::move(grown);
            }
            if (shN_buf.numel() < needed_floats) {
                shN_buf.append_zeros(needed_floats - shN_buf.numel());
            }
        }

        if (use_shN && append_shN.is_valid() && append_shN.numel() > 0) {
            shN_swizzled_gather_from_linear(
                _splat_data->shN().ptr<float>(),
                old_size,
                append_shN.ptr<float>(),
                n_append,
                layout_rest,
                layout_rest);
            _optimizer->extend_state_for_new_params(ParamType::ShN, n_append);
        } else {
            _optimizer->extend_state_for_new_params(ParamType::ShN, n_append);
        }
        _optimizer->add_new_params(ParamType::Scaling, append_scaling, true);
        _optimizer->add_new_params(ParamType::Rotation, append_rotation, true);
        _optimizer->add_new_params(ParamType::Opacity, append_opacity, true);
        // Append live (false) deleted rows now that size() has advanced.
        append_live_deleted_rows(*_splat_data, _free_mask, n_append);
        if (_splat_data->has_deleted_mask() &&
            !_splat_data->deleted_mask_matches_size()) {
            _splat_data->reconcile_deleted_mask();
        }
        if (_splat_data->has_frozen_ranges()) {
            apply_frozen_ranges_to_optimizer(*_splat_data, *_optimizer);
        }

        return n_append;
    }

    void MRNF::seed_from_view(int iter, const RenderOutput& render_output) {
        using namespace lfs::core;
        if (!background_improvements_enabled() ||
            iter >= effective_grow_until_iter() ||
            !_camera_hull_valid || !_bounds_valid) {
            return;
        }
        LOG_TIMER("MRNF::seed_from_view");
        (void)lfs::training::sh_value::ensure_shN_fp32_for_mutation(*_splat_data);

        Tensor image = render_output.image;
        Tensor target = render_output.target_image;
        Tensor alpha = render_output.alpha;
        Tensor depth = render_output.depth;
        Camera* camera = render_output.camera;
        int width = render_output.width;
        int height = render_output.height;

        const bool live_valid = camera &&
                                is_cuda_image(image) &&
                                is_cuda_image(target) &&
                                alpha.is_valid() &&
                                alpha.device() == Device::CUDA &&
                                same_spatial(image, target);
        if (!live_valid) {
            if (_cached_seed_valid && _cached_seed_camera) {
                image = _cached_seed_image;
                target = _cached_seed_target;
                alpha = _cached_seed_alpha;
                depth = _cached_seed_depth;
                camera = _cached_seed_camera;
                width = _cached_seed_width;
                height = _cached_seed_height;
            } else {
                return;
            }
        }

        image = to_float01_cuda(image);
        target = to_float01_cuda(target);
        alpha = to_float01_cuda(alpha);
        if (image.ndim() != 3 || target.ndim() != 3) {
            return;
        }
        if (width <= 0 || height <= 0) {
            height = static_cast<int>(image.shape()[1]);
            width = static_cast<int>(image.shape()[2]);
        }
        if (width <= 0 || height <= 0) {
            return;
        }

        if (!fill_mean_abs_error_hw(image, target, _explore_error_hw)) {
            return;
        }
        auto alpha_flat = flatten_hw(alpha);
        if (!_explore_error_hw.is_valid() || !alpha_flat.is_valid() ||
            _explore_error_hw.numel() != alpha_flat.numel()) {
            return;
        }
        const size_t error_n = _explore_error_hw.numel();
        auto error_flat = _explore_error_hw.reshape(TensorShape({error_n}));
        mrnf_strategy::launch_seed_weights_from_error_alpha(
            error_flat.ptr<float>(),
            alpha_flat.ptr<float>(),
            error_flat.ptr<float>(),
            error_n);
        auto seed_weights = error_flat;

        const size_t hw = seed_weights.numel();
        const int remaining_budget = (_params->max_cap > 0)
                                         ? std::max(0, _params->max_cap - static_cast<int>(active_count()))
                                         : static_cast<int>(std::min(hw, static_cast<size_t>(INT_MAX)));
        const int dose_seeds = cfg_seed_dose();
        const int requested_cadence =
            dose_seeds > 0 ? dose_seeds : starved_cadence_count(kExploreSeeds);
        int n_seed = std::min({requested_cadence, remaining_budget,
                               static_cast<int>(hw)});
        if (n_seed <= 0) {
            return;
        }

        if (!_refine_counts_dev.is_valid() || _refine_counts_dev.numel() < 4) {
            _refine_counts_dev = Tensor::zeros({4}, Device::CUDA, DataType::Int64);
        }
        kernels::launch_packed_refine_counts(
            nullptr, 0, nullptr, 0,
            seed_weights.ptr<float>(), hw,
            nullptr, 0,
            _refine_counts_dev.ptr<int64_t>());
        int64_t host_counts[4] = {0, 0, 0, 0};
        LFS_CUDA_CHECK_MSG(
            cudaMemcpy(host_counts, _refine_counts_dev.ptr<int64_t>(),
                       4 * sizeof(int64_t), cudaMemcpyDeviceToHost),
            "MRNF seed nnz D2H");
        n_seed = std::min(n_seed, static_cast<int>(host_counts[2]));
        if (n_seed <= 0) {
            return;
        }

        auto seed = static_cast<uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
        auto pixel_inds = Tensor::empty({static_cast<size_t>(n_seed)}, Device::CUDA, DataType::Int64);
        _gumbel_scratch.ensure_n(hw, Device::CUDA);
        mrnf_strategy::launch_gumbel_topk(
            seed_weights.ptr<float>(), hw, static_cast<size_t>(n_seed), seed,
            pixel_inds.ptr<int64_t>(), nullptr, false, &_gumbel_scratch, hw);

        Tensor depth_flat;
        if (depth.is_valid() && depth.device() == Device::CUDA && depth.numel() > 0) {
            auto depth_f = to_float01_cuda(depth);
            depth_flat = flatten_hw(depth_f);
        }
        if (!depth_flat.is_valid() || depth_flat.numel() != hw) {
            depth_flat = Tensor::zeros({hw}, Device::CUDA);
        }

        auto rgb = Tensor::empty({static_cast<size_t>(n_seed), 3}, Device::CUDA, DataType::Float32);
        auto seed_alpha = Tensor::empty({static_cast<size_t>(n_seed)}, Device::CUDA, DataType::Float32);
        auto seed_depth = Tensor::empty({static_cast<size_t>(n_seed)}, Device::CUDA, DataType::Float32);
        const int channels = static_cast<int>(target.shape()[0]);
        mrnf_strategy::launch_gather_seed_payloads(
            pixel_inds.ptr<int64_t>(),
            static_cast<size_t>(n_seed),
            hw,
            target.ptr<float>(),
            channels,
            alpha_flat.ptr<float>(),
            depth_flat.ptr<float>(),
            rgb.ptr<float>(),
            seed_alpha.ptr<float>(),
            seed_depth.ptr<float>());

        auto pix_cpu = pixel_inds.cpu();
        auto rgb_cpu = rgb.cpu();
        auto alpha_cpu = seed_alpha.cpu();
        auto depth_cpu = seed_depth.cpu();
        const auto* pix_ptr = pix_cpu.ptr<int64_t>();
        const auto* rgb_ptr = rgb_cpu.ptr<float>();
        const auto* a_ptr = alpha_cpu.ptr<float>();
        const auto* d_ptr = depth_cpu.ptr<float>();

        auto pos_cpu = camera->cam_position().cpu().contiguous();
        auto R_cpu = camera->R().cpu().contiguous();
        const float* cam_pos = pos_cpu.ptr<float>();
        const float* R = R_cpu.ptr<float>();
        const auto [fx, fy, cx, cy] = camera->get_intrinsics();
        if (!(fx > 0.0f) || R_cpu.ndim() != 2 || R_cpu.shape()[0] != 3 || R_cpu.shape()[1] != 3) {
            return;
        }

        const float d_hi = std::sqrt(
                               (cam_pos[0] - _cam_centroid[0]) * (cam_pos[0] - _cam_centroid[0]) +
                               (cam_pos[1] - _cam_centroid[1]) * (cam_pos[1] - _cam_centroid[1]) +
                               (cam_pos[2] - _cam_centroid[2]) * (cam_pos[2] - _cam_centroid[2])) +
                           kSeedDepthOrbits * _orbit_radius;

        const float raw_opacity = logit_clamped(kSeedOpacity);
        const size_t sh_rest = _splat_data->max_sh_coeffs_rest();
        std::mt19937 rng{static_cast<unsigned int>(seed ^ 0x9E3779B9u)};
        std::uniform_real_distribution<float> unit(0.0f, 1.0f);

        std::vector<float> means_h;
        std::vector<float> rotations_h;
        std::vector<float> scales_h;
        std::vector<float> sh0_h;
        std::vector<float> opac_h;
        means_h.reserve(static_cast<size_t>(n_seed) * 3);
        rotations_h.reserve(static_cast<size_t>(n_seed) * 4);
        scales_h.reserve(static_cast<size_t>(n_seed) * 3);
        sh0_h.reserve(static_cast<size_t>(n_seed) * 3);
        opac_h.reserve(static_cast<size_t>(n_seed));

        int kept = 0;
        int outside_kept = 0;
        const int remaining_out = _far_growth.active
                                      ? std::max(0, _far_growth.reserved_for_seeds)
                                      : n_seed;

        for (int i = 0; i < n_seed; ++i) {
            const int64_t pix = pix_ptr[i];
            if (pix < 0 || static_cast<size_t>(pix) >= hw) {
                continue;
            }
            const int x = static_cast<int>(pix % static_cast<int64_t>(width));
            const int y = static_cast<int>(pix / static_cast<int64_t>(width));
            const float u = (static_cast<float>(x) + 0.5f - cx) / fx;
            const float v = (static_cast<float>(y) + 0.5f - cy) / fy;
            const float a_p = a_ptr[i];
            float d_lo = 0.25f * _bounds.median_size;
            if (a_p > 0.5f) {
                const float depth_acc = d_ptr[i];
                d_lo = (a_p > 1e-6f) ? (depth_acc / a_p) : depth_acc;
            } else if (cfg_seed_dose() > 0 && _orbit_radius > 0.0f) {
                // far seeding active: raise the transparent-pixel depth floor to the far shell
                d_lo = std::max(d_lo, kFarMaskOrbits * _orbit_radius);
            }
            if (!(d_lo > 0.0f) || !(d_hi > d_lo) || !std::isfinite(d_lo) ||
                !std::isfinite(d_hi)) {
                continue;
            }

            const float inv_hi = 1.0f / d_hi;
            const float inv_lo = 1.0f / d_lo;
            const float inv_t = inv_hi + unit(rng) * (inv_lo - inv_hi);
            if (!(inv_t > 0.0f) || !std::isfinite(inv_t)) {
                continue;
            }
            const float t = 1.0f / inv_t;
            const float pcam_x = u * t;
            const float pcam_y = v * t;
            const float pcam_z = t;
            const float wx = R[0] * pcam_x + R[3] * pcam_y + R[6] * pcam_z + cam_pos[0];
            const float wy = R[1] * pcam_x + R[4] * pcam_y + R[7] * pcam_z + cam_pos[1];
            const float wz = R[2] * pcam_x + R[5] * pcam_y + R[8] * pcam_z + cam_pos[2];
            if (!std::isfinite(wx) || !std::isfinite(wy) || !std::isfinite(wz)) {
                continue;
            }

            const float far_dx = wx - _cam_centroid[0];
            const float far_dy = wy - _cam_centroid[1];
            const float far_dz = wz - _cam_centroid[2];
            const float far_limit = kFarMaskOrbits * _orbit_radius;
            const bool outside =
                (far_dx * far_dx + far_dy * far_dy + far_dz * far_dz) > (far_limit * far_limit);
            if (_far_growth.active && outside && outside_kept >= remaining_out) {
                continue;
            }

            const float log_s = std::log(std::max(t * 4.0f / fx, 1e-6f));
            means_h.push_back(wx);
            means_h.push_back(wy);
            means_h.push_back(wz);
            rotations_h.push_back(1.0f);
            rotations_h.push_back(0.0f);
            rotations_h.push_back(0.0f);
            rotations_h.push_back(0.0f);
            scales_h.push_back(log_s);
            scales_h.push_back(log_s);
            scales_h.push_back(log_s);
            sh0_h.push_back((rgb_ptr[i * 3 + 0] - 0.5f) / MRNF_SH_C0);
            sh0_h.push_back((rgb_ptr[i * 3 + 1] - 0.5f) / MRNF_SH_C0);
            sh0_h.push_back((rgb_ptr[i * 3 + 2] - 0.5f) / MRNF_SH_C0);
            opac_h.push_back(raw_opacity);
            ++kept;
            if (outside) {
                ++outside_kept;
            }
        }

        if (kept == 0) {
            return;
        }

        const size_t kept_n = static_cast<size_t>(kept);
        auto pos = Tensor::from_vector(means_h, TensorShape({kept_n, 3}), Device::CUDA);
        auto rot = Tensor::from_vector(rotations_h, TensorShape({kept_n, 4}), Device::CUDA);
        auto scl = Tensor::from_vector(scales_h, TensorShape({kept_n, 3}), Device::CUDA);
        auto sh0 = Tensor::from_vector(sh0_h, TensorShape({kept_n, 1, 3}), Device::CUDA);
        auto opa = Tensor::from_vector(opac_h, TensorShape({kept_n}), Device::CUDA);
        Tensor shN;
        if (sh_rest > 0) {
            shN = Tensor::zeros({kept_n, sh_rest, 3}, Device::CUDA);
        }

        auto [filled, remaining_after_fill] = fill_free_slots_with_data(
            pos, rot, scl, sh0, shN, opa, static_cast<int64_t>(kept_n));
        const size_t append_start = kept_n - static_cast<size_t>(remaining_after_fill);
        append_child_rows(pos, rot, scl, sh0, shN, opa, append_start, kept_n);

        _far_growth.allocated += kept;
        _far_growth.outside_used += outside_kept;
        _far_growth.reserved_for_seeds = 0;
        LFS_COUNTER_ADD("strategy.mrnf.explore_seed", kept);
    }

    void MRNF::compute_bounds() {
        const size_t current_size = static_cast<size_t>(_splat_data->size());
        lfs::core::Tensor active_indices;
        lfs::core::Tensor active_means = _splat_data->means();
        size_t n = active_count();

        if (_free_mask.is_valid() && free_count() > 0) {
            active_indices = get_active_indices();
        }
        if (auto frozen_mask = make_frozen_mask(*_splat_data, current_size, _splat_data->means().device());
            frozen_mask.is_valid()) {
            if (!active_indices.is_valid()) {
                active_indices = get_active_indices();
            }
            if (active_indices.numel() > 0) {
                auto trainable = frozen_mask.index_select(0, active_indices).logical_not();
                active_indices = active_indices.index_select(0, trainable.nonzero().squeeze(-1));
            }
        }
        if (active_indices.is_valid()) {
            n = active_indices.numel();
            if (n > 0) {
                active_means = _splat_data->means().index_select(0, active_indices).contiguous();
            }
        }

        if (n == 0) {
            _bounds_valid = false;
            _median_splat_extent = 0.0f;
            _median_splat_extent_valid = false;
            if (_optimizer) {
                _optimizer->set_per_splat_mean_step(
                    false, 0.0f, kPerSplatMeanStepRatioMin,
                    effective_mean_step_ratio_max());
            }
            return;
        }

        mrnf_strategy::MRNFBounds candidate{};
        mrnf_strategy::launch_percentile_bounds(
            active_means.ptr<float>(),
            n,
            _params->bounds_percentile,
            &candidate);

        float coordinate_scale = 1.0f;
        bool finite_bounds = std::isfinite(candidate.max_extent) && candidate.max_extent >= 0.0f;
        for (int axis = 0; axis < 3; ++axis) {
            finite_bounds = finite_bounds &&
                            std::isfinite(candidate.center[axis]) &&
                            std::isfinite(candidate.extent[axis]) &&
                            candidate.extent[axis] >= 0.0f;
            coordinate_scale = std::max(coordinate_scale, std::abs(candidate.center[axis]));
        }
        const float extent_epsilon =
            32.0f * std::numeric_limits<float>::epsilon() * coordinate_scale;
        if (!finite_bounds || candidate.max_extent <= extent_epsilon) {
            if (!_bounds_valid) {
                LOG_WARN("MRNF: spatial bounds unavailable for a degenerate active model; "
                         "skipping bounds-dependent noise and pruning");
            }
            return;
        }

        if (!std::isfinite(candidate.median_size) || candidate.median_size <= extent_epsilon) {
            // A line-like model has a useful spatial extent but a zero median
            // axis. Use its full largest-axis extent to keep the mean LR finite.
            candidate.median_size =
                candidate.max_extent <= std::numeric_limits<float>::max() / 2.0f
                    ? candidate.max_extent * 2.0f
                    : candidate.max_extent;
        }

        _bounds = candidate;

        _bounds_valid = true;
        _refine_windows_since_bounds = 0;

        lfs::core::Tensor active_scales = _splat_data->scaling_raw();
        if (active_indices.is_valid()) {
            active_scales = active_scales.index_select(0, active_indices).contiguous();
        }
        float median_extent = 0.0f;
        bool median_ok = false;
        if (active_scales.is_valid() &&
            active_scales.numel() >= n * 3) {
            mrnf_strategy::launch_median_geomean_extent(
                active_scales.ptr<float>(),
                n,
                &median_extent,
                &median_ok);
        }
        _median_splat_extent = median_extent;
        _median_splat_extent_valid =
            median_ok && std::isfinite(median_extent) && median_extent > 0.0f;

        sync_mean_learning_rate();
    }

    void MRNF::sync_mean_learning_rate() {
        if (!_optimizer || !_bounds_valid)
            return;
        _optimizer->set_param_lr(ParamType::Means, _mean_lr_unscaled * _bounds.median_size);
        if (background_improvements_enabled() && _median_splat_extent_valid) {
            _optimizer->set_per_splat_mean_step(
                true, _median_splat_extent, kPerSplatMeanStepRatioMin,
                effective_mean_step_ratio_max());
            const size_t n = static_cast<size_t>(_splat_data->size());
            if (!_far_field_mask.is_valid() || _far_field_mask.numel() != n) {
                if (_camera_hull_valid && n > 0) {
                    refresh_far_field_mask(n);
                } else {
                    publish_mean_step_far_mask();
                }
            } else {
                publish_mean_step_far_mask();
            }
        } else {
            _optimizer->set_per_splat_mean_step(
                false, 0.0f, kPerSplatMeanStepRatioMin,
                effective_mean_step_ratio_max());
        }
    }

    void MRNF::step(int iter) {
        LOG_TIMER("MRNF::step");
        if (iter < _params->iterations) {
            _optimizer->step(iter);
            _optimizer->zero_grad(iter);

            _mean_lr_unscaled *= _mean_lr_gamma;
            _scale_lr_current *= _scale_lr_gamma;
            _optimizer->set_param_lr(ParamType::Scaling, _scale_lr_current);
            sync_mean_learning_rate();
        }
    }

    void MRNF::remove_gaussians(const lfs::core::Tensor& mask) {
        using namespace lfs::core;

        const Tensor prune_mask = exclude_frozen_from_mask(*_splat_data, mask);
        Tensor keep_mask = prune_mask.logical_not();
        const size_t old_size = static_cast<size_t>(_splat_data->size());
        const int n_remove = static_cast<int>(old_size - keep_mask.to(DataType::Int32).sum().template item<int>());

        LOG_INFO("MRNF::remove_gaussians: mask size={}, n_remove={}, current size={}",
                 mask.numel(), n_remove, _splat_data->size());

        if (n_remove == 0)
            return;

        compact_splats(keep_mask);
        // compact expands q16 → float; re-encode when quant is on.
        if (lfs::core::sh_value_quant::enabled() &&
            _splat_data->shN().is_valid() &&
            _splat_data->shN().dtype() == lfs::core::DataType::Float32) {
            lfs::training::sh_value::commit_shN_after_mutation(*_splat_data);
        }

        if (_splat_data->size() == 0) {
            _bounds_valid = false;
        } else if (_bounds_valid) {
            compute_bounds();
        }
    }

    lfs::core::Tensor MRNF::edge_guidance_factor() {
        if (!_params || !_params->use_edge_map || !_edge_precompute_valid) {
            return {};
        }

        const size_t n = static_cast<size_t>(_splat_data->size());
        if (!_precomputed_edge_scores.is_valid() ||
            _precomputed_edge_scores.ndim() != 1 ||
            _precomputed_edge_scores.numel() != n) {
            return {};
        }

        auto normalized_edge = normalized_by_positive_median(_precomputed_edge_scores, &_median_scratch);
        return normalized_edge.mul(MRNF_EDGE_SCORE_WEIGHT).add(1.0f);
    }

    namespace {
        constexpr uint32_t LFS_MAGIC = 0x4C464252; // "LFBR"
        constexpr uint32_t LFS_VERSION = 3;
    } // namespace

    void MRNF::serialize(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&LFS_MAGIC), sizeof(LFS_MAGIC));
        os.write(reinterpret_cast<const char*>(&LFS_VERSION), sizeof(LFS_VERSION));

        if (_optimizer) {
            uint8_t has_optimizer = 1;
            os.write(reinterpret_cast<const char*>(&has_optimizer), sizeof(has_optimizer));
            _optimizer->serialize(os);
        } else {
            uint8_t has_optimizer = 0;
            os.write(reinterpret_cast<const char*>(&has_optimizer), sizeof(has_optimizer));
        }

        if (_scheduler) {
            uint8_t has_scheduler = 1;
            os.write(reinterpret_cast<const char*>(&has_scheduler), sizeof(has_scheduler));
            _scheduler->serialize(os);
        } else {
            uint8_t has_scheduler = 0;
            os.write(reinterpret_cast<const char*>(&has_scheduler), sizeof(has_scheduler));
        }

        const uint8_t has_free_mask = _free_mask.is_valid() ? 1 : 0;
        os.write(reinterpret_cast<const char*>(&has_free_mask), sizeof(has_free_mask));
        if (has_free_mask) {
            os << _free_mask;
        }

        os.write(reinterpret_cast<const char*>(&_mean_lr_unscaled), sizeof(_mean_lr_unscaled));
        os.write(reinterpret_cast<const char*>(&_scale_lr_current), sizeof(_scale_lr_current));
    }

    void MRNF::deserialize(std::istream& is) {
        uint32_t magic = 0, version = 0;
        lfs::core::serialization_detail::read_exact(is, &magic, sizeof(magic), "MRNF magic");
        lfs::core::serialization_detail::read_exact(is, &version, sizeof(version), "MRNF version");

        if (magic != LFS_MAGIC)
            throw std::runtime_error("Invalid MRNF checkpoint: wrong magic");
        if (version == 0 || version > LFS_VERSION)
            throw std::runtime_error("Unsupported MRNF checkpoint version: " + std::to_string(version));

        uint8_t has_optimizer = 0;
        lfs::core::serialization_detail::read_exact(
            is, &has_optimizer, sizeof(has_optimizer), "MRNF optimizer flag");
        if (has_optimizer > 1 || (has_optimizer && !_optimizer))
            throw std::runtime_error("Invalid MRNF checkpoint: optimizer flag/state mismatch");
        if (has_optimizer)
            _optimizer->deserialize(is);

        uint8_t has_scheduler = 0;
        lfs::core::serialization_detail::read_exact(
            is, &has_scheduler, sizeof(has_scheduler), "MRNF scheduler flag");
        if (has_scheduler > 1 || (has_scheduler && !_scheduler))
            throw std::runtime_error("Invalid MRNF checkpoint: scheduler flag/state mismatch");
        if (has_scheduler)
            _scheduler->deserialize(is);

        const double optimizer_mean_lr = _optimizer ? _optimizer->get_param_lr(ParamType::Means) : 0.0;
        const double optimizer_scaling_lr = _optimizer ? _optimizer->get_param_lr(ParamType::Scaling) : 0.0;

        if (version >= 2) {
            uint8_t has_free_mask = 0;
            lfs::core::serialization_detail::read_exact(
                is, &has_free_mask, sizeof(has_free_mask), "MRNF free-mask flag");
            if (has_free_mask > 1)
                throw std::runtime_error("Invalid MRNF checkpoint: free-mask flag must be boolean");
            if (has_free_mask) {
                lfs::core::Tensor free_mask;
                is >> free_mask;
                const size_t model_size = static_cast<size_t>(_splat_data->size());
                const size_t max_capacity = _params && _params->max_cap > 0
                                                ? static_cast<size_t>(_params->max_cap)
                                                : model_size;
                if (!free_mask.is_valid() || !lfs::core::is_bool_like(free_mask.dtype()) ||
                    free_mask.ndim() != 1 || free_mask.numel() < model_size ||
                    free_mask.numel() > max_capacity) {
                    throw std::runtime_error("Invalid MRNF checkpoint: free mask has incompatible schema");
                }
                if (free_mask.device() != lfs::core::Device::CUDA)
                    free_mask = free_mask.cuda();
                _free_mask = std::move(free_mask);
            }
        }
        if (version >= 3) {
            double mean_lr_unscaled = 0.0;
            double scale_lr_current = 0.0;
            lfs::core::serialization_detail::read_exact(
                is, &mean_lr_unscaled, sizeof(mean_lr_unscaled), "MRNF mean learning rate");
            lfs::core::serialization_detail::read_exact(
                is, &scale_lr_current, sizeof(scale_lr_current), "MRNF scale learning rate");
            if (!std::isfinite(mean_lr_unscaled) || mean_lr_unscaled < 0.0 ||
                !std::isfinite(scale_lr_current) || scale_lr_current < 0.0) {
                throw std::runtime_error("Invalid MRNF checkpoint: learning-rate state is invalid");
            }
            _mean_lr_unscaled = mean_lr_unscaled;
            _scale_lr_current = scale_lr_current;
        } else {
            _mean_lr_unscaled = _params ? _params->means_lr : _mean_lr_unscaled;
            _scale_lr_current = optimizer_scaling_lr > 0.0
                                    ? optimizer_scaling_lr
                                    : (_params ? _params->scaling_lr : _scale_lr_current);
        }

        if (!_free_mask.is_valid()) {
            const size_t capacity = (_params && _params->max_cap > 0)
                                        ? static_cast<size_t>(_params->max_cap)
                                        : static_cast<size_t>(_splat_data->size());
            _free_mask = lfs::core::Tensor::zeros_bool({capacity}, _splat_data->means().device());
        }
        sync_deleted_mask_from_free_mask(*_splat_data, _free_mask);

        const size_t n = static_cast<size_t>(_splat_data->size());
        const size_t tracking_capacity = (_params && _params->max_cap > 0)
                                             ? static_cast<size_t>(_params->max_cap)
                                             : 0;
        reset_vector_buffer(_refine_weight_max, n, _splat_data->means().device(), tracking_capacity);
        reset_vector_buffer(_vis_count, n, _splat_data->means().device(), tracking_capacity);
        reset_vector_buffer(_explore_score_sum, n, _splat_data->means().device(), tracking_capacity);
        if (cfg_ratio_rank_on()) {
            reset_vector_buffer(_refine_ratio_max, n, _splat_data->means().device(), tracking_capacity);
        }
        _explore_sample_count = 0;
        _explore_last_sample_iter = -1;
        ensure_densification_info_shape();
        _precomputed_edge_scores = lfs::core::Tensor();
        _edge_precompute_valid = false;

        if (_splat_data->size() == 0 || active_count() == 0) {
            _bounds_valid = false;
        } else {
            compute_bounds();
        }

        if (version < 3 && _bounds_valid && optimizer_mean_lr > 0.0 && _bounds.median_size > 0.0f) {
            _mean_lr_unscaled = optimizer_mean_lr / static_cast<double>(_bounds.median_size);
        }

        refresh_decay_schedule_from_current_state();

        if (_optimizer) {
            _optimizer->set_param_lr(ParamType::Scaling, _scale_lr_current);
            sync_mean_learning_rate();
        }
        publish_vram_attribution();
    }

    bool MRNF::can_adopt_checkpoint_state(const IStrategy& loaded) const noexcept {
        const auto* source = dynamic_cast<const MRNF*>(&loaded);
        return source && static_cast<bool>(_optimizer) == static_cast<bool>(source->_optimizer) &&
               static_cast<bool>(_scheduler) == static_cast<bool>(source->_scheduler);
    }

    void MRNF::adopt_checkpoint_state(IStrategy& loaded) noexcept {
        auto& source = checked_checkpoint_source<MRNF>(loaded);
        if (_optimizer)
            _optimizer->adopt_checkpoint_state(*source._optimizer);
        if (_scheduler)
            _scheduler->adopt_checkpoint_state(*source._scheduler);
        _params.swap(source._params);
        std::swap(_refine_weight_max, source._refine_weight_max);
        std::swap(_refine_ratio_max, source._refine_ratio_max);
        std::swap(_vis_count, source._vis_count);
        std::swap(_precomputed_edge_scores, source._precomputed_edge_scores);
        std::swap(_edge_precompute_valid, source._edge_precompute_valid);
        std::swap(_edge_score_sum, source._edge_score_sum);
        std::swap(_edge_canny_nms_output, source._edge_canny_nms_output);
        std::swap(_edge_sample_count, source._edge_sample_count);
        std::swap(_edge_last_sample_iter, source._edge_last_sample_iter);
        std::swap(_explore_score_sum, source._explore_score_sum);
        std::swap(_explore_error_hw, source._explore_error_hw);
        std::swap(_explore_view_scores, source._explore_view_scores);
        std::swap(_explore_means2d, source._explore_means2d);
        std::swap(_explore_radii, source._explore_radii);
        std::swap(_explore_sample_count, source._explore_sample_count);
        std::swap(_explore_last_sample_iter, source._explore_last_sample_iter);
        std::swap(_cached_seed_image, source._cached_seed_image);
        std::swap(_cached_seed_target, source._cached_seed_target);
        std::swap(_cached_seed_alpha, source._cached_seed_alpha);
        std::swap(_cached_seed_depth, source._cached_seed_depth);
        std::swap(_cached_seed_camera, source._cached_seed_camera);
        std::swap(_cached_seed_width, source._cached_seed_width);
        std::swap(_cached_seed_height, source._cached_seed_height);
        std::swap(_cached_seed_valid, source._cached_seed_valid);
        std::swap(_free_mask, source._free_mask);
        std::swap(_far_growth, source._far_growth);
        std::swap(_far_field_mask, source._far_field_mask);
        std::swap(_cam_centroid, source._cam_centroid);
        std::swap(_orbit_radius, source._orbit_radius);
        std::swap(_camera_hull_valid, source._camera_hull_valid);
        std::swap(_scene_has_far_field, source._scene_has_far_field);
        std::swap(_logged_degenerate_hull, source._logged_degenerate_hull);
        std::swap(_initial_sfm_point_count, source._initial_sfm_point_count);
        std::swap(_far_starvation, source._far_starvation);
        std::swap(_logged_far_starvation, source._logged_far_starvation);
        std::swap(_growth_window_count, source._growth_window_count);
        std::swap(_bounds, source._bounds);
        std::swap(_bounds_valid, source._bounds_valid);
        std::swap(_refine_windows_since_bounds, source._refine_windows_since_bounds);
        std::swap(_median_splat_extent, source._median_splat_extent);
        std::swap(_median_splat_extent_valid, source._median_splat_extent_valid);
        publish_mean_step_far_mask();
        std::swap(_mean_lr_unscaled, source._mean_lr_unscaled);
        std::swap(_scale_lr_current, source._scale_lr_current);
        std::swap(_mean_lr_gamma, source._mean_lr_gamma);
        std::swap(_scale_lr_gamma, source._scale_lr_gamma);
        publish_vram_attribution();
    }

    void MRNF::reserve_optimizer_capacity(size_t capacity) {
        if (_optimizer) {
            _optimizer->reserve_capacity(capacity);
            LOG_INFO("MRNF: reserved optimizer capacity for {} Gaussians", capacity);
        }
    }

    void MRNF::set_optimization_params(const lfs::core::param::OptimizationParameters& params) {
        _params = std::make_unique<const lfs::core::param::OptimizationParameters>(params);

        if (_mean_lr_unscaled <= 0.0) {
            _mean_lr_unscaled = params.means_lr;
        }
        if (_scale_lr_current <= 0.0) {
            _scale_lr_current = params.scaling_lr;
        }

        refresh_decay_schedule_from_current_state();
        if (_splat_data) {
            ensure_densification_info_shape();
        }

        if (_optimizer) {
            _optimizer->set_param_lr(ParamType::Scaling, _scale_lr_current);
            sync_mean_learning_rate();
        }
        update_far_starvation();
    }

    void MRNF::refresh_decay_schedule_from_current_state() {
        const int64_t completed_steps = _optimizer ? std::max<int64_t>(0, _optimizer->get_step_count(ParamType::Means))
                                                   : 0;
        const size_t remaining_steps =
            (_params && _params->iterations > static_cast<size_t>(completed_steps))
                ? (_params->iterations - static_cast<size_t>(completed_steps))
                : 0;

        const double mean_lr_end = _params ? _params->means_lr_end : _mean_lr_unscaled;
        const double scaling_lr_end = _params ? _params->scaling_lr_end : _scale_lr_current;
        _mean_lr_gamma = compute_decay_gamma(_mean_lr_unscaled, mean_lr_end, remaining_steps);
        _scale_lr_gamma = compute_decay_gamma(_scale_lr_current, scaling_lr_end, remaining_steps);
    }

} // namespace lfs::training
