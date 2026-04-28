/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/memory_arena.hpp"
#include "forward.h"
#include "rasterization_api_tensor.h"
#include "rasterization_config.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>

namespace lfs::rendering {

    namespace {

        [[noreturn]] void throw_cuda_runtime_error(const char* operation, const cudaError_t error) {
            throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(error));
        }

        class CachedDualForwardExecutor {
        public:
            using Task = std::function<void(cudaStream_t)>;

            static CachedDualForwardExecutor& instance() {
                static CachedDualForwardExecutor executor;
                return executor;
            }

            void run(std::array<Task, 2> tasks, const int device) {
                std::lock_guard<std::mutex> run_lock(run_mutex_);

                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    pending_tasks_ = tasks.size();
                    for (size_t i = 0; i < tasks.size(); ++i) {
                        states_[i].task = std::move(tasks[i]);
                        states_[i].device = device;
                        states_[i].error = nullptr;
                        states_[i].has_task = true;
                    }
                }

                task_cv_.notify_all();

                std::unique_lock<std::mutex> lock(mutex_);
                done_cv_.wait(lock, [&]() { return pending_tasks_ == 0; });

                std::exception_ptr first_error;
                for (const auto& state : states_) {
                    if (state.error) {
                        first_error = state.error;
                        break;
                    }
                }
                lock.unlock();

                if (first_error) {
                    std::rethrow_exception(first_error);
                }
            }

        private:
            struct WorkerState {
                Task task;
                std::exception_ptr error;
                cudaStream_t stream = nullptr;
                int stream_device = -1;
                int device = -1;
                bool has_task = false;
            };

            CachedDualForwardExecutor() {
                for (size_t i = 0; i < workers_.size(); ++i) {
                    workers_[i] = std::thread([this, i]() { worker_loop(i); });
                }
            }

            ~CachedDualForwardExecutor() {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    stop_ = true;
                }
                task_cv_.notify_all();

                for (auto& worker : workers_) {
                    if (worker.joinable()) {
                        worker.join();
                    }
                }
            }

            void worker_loop(const size_t index) {
                for (;;) {
                    Task task;
                    int device = -1;

                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        task_cv_.wait(lock, [&]() {
                            return stop_ || states_[index].has_task;
                        });

                        if (stop_ && !states_[index].has_task) {
                            break;
                        }

                        task = std::move(states_[index].task);
                        device = states_[index].device;
                        states_[index].has_task = false;
                        states_[index].error = nullptr;
                    }

                    std::exception_ptr error;
                    try {
                        ensure_stream(index, device);
                        task(states_[index].stream);
                    } catch (...) {
                        error = std::current_exception();
                    }

                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        states_[index].error = error;
                        if (pending_tasks_ > 0) {
                            --pending_tasks_;
                            if (pending_tasks_ == 0) {
                                done_cv_.notify_one();
                            }
                        }
                    }
                }

                destroy_stream(index);
            }

            void ensure_stream(const size_t index, const int device) {
                if (const cudaError_t set_device_err = cudaSetDevice(device); set_device_err != cudaSuccess) {
                    throw_cuda_runtime_error("cudaSetDevice", set_device_err);
                }

                auto& state = states_[index];
                if (state.stream != nullptr && state.stream_device == device) {
                    return;
                }

                if (state.stream != nullptr) {
                    if (state.stream_device >= 0) {
                        cudaSetDevice(state.stream_device);
                    }
                    cudaStreamDestroy(state.stream);
                    state.stream = nullptr;
                    state.stream_device = -1;

                    if (const cudaError_t reset_device_err = cudaSetDevice(device); reset_device_err != cudaSuccess) {
                        throw_cuda_runtime_error("cudaSetDevice", reset_device_err);
                    }
                }

                cudaStream_t stream = nullptr;
                if (const cudaError_t create_err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
                    create_err != cudaSuccess) {
                    throw_cuda_runtime_error("cudaStreamCreateWithFlags", create_err);
                }

                state.stream = stream;
                state.stream_device = device;
            }

            void destroy_stream(const size_t index) noexcept {
                auto& state = states_[index];
                if (state.stream == nullptr) {
                    return;
                }

                if (state.stream_device >= 0) {
                    cudaSetDevice(state.stream_device);
                }
                cudaStreamDestroy(state.stream);
                state.stream = nullptr;
                state.stream_device = -1;
            }

            std::mutex run_mutex_;
            std::mutex mutex_;
            std::condition_variable task_cv_;
            std::condition_variable done_cv_;
            std::array<WorkerState, 2> states_;
            std::array<std::thread, 2> workers_;
            size_t pending_tasks_ = 0;
            bool stop_ = false;
        };

    } // namespace

    inline std::function<char*(size_t)> resize_function_wrapper_tensor(Tensor& t) {
        return [&t](size_t N) -> char* {
            if (N == 0) {
                t = Tensor::empty({0}, lfs::core::Device::CUDA, lfs::core::DataType::UInt8);
                return nullptr;
            }
            t = Tensor::empty({N}, lfs::core::Device::CUDA, lfs::core::DataType::UInt8);
            return reinterpret_cast<char*>(t.ptr<uint8_t>());
        };
    }

    inline void check_tensor_input(bool debug, const Tensor& tensor, const char* name) {
        if (debug) {
            if (!tensor.is_valid() || tensor.device() != lfs::core::Device::CUDA ||
                tensor.dtype() != lfs::core::DataType::Float32 || !tensor.is_contiguous()) {
                throw std::runtime_error("Input tensor '" + std::string(name) +
                                         "' must be a contiguous CUDA float tensor.");
            }
        }
    }

    struct GpuBoolMask {
        Tensor tensor;
        const bool* ptr = nullptr;
        int count = 0;

        GpuBoolMask() = default;
        explicit GpuBoolMask(const std::vector<bool>& mask) : count(static_cast<int>(mask.size())) {
            if (count > 0) {
                std::vector<uint8_t> data(count);
                std::transform(mask.begin(), mask.end(), data.begin(), [](const bool b) -> uint8_t { return b ? 1 : 0; });
                tensor = Tensor::from_blob(data.data(), {static_cast<size_t>(count)},
                                           lfs::core::Device::CPU, lfs::core::DataType::UInt8)
                             .cuda();
                ptr = reinterpret_cast<const bool*>(tensor.ptr<uint8_t>());
            }
        }
    };

    struct VisibilityPredicate {
        const int* transform_indices;
        const bool* node_visibility_mask;
        int num_nodes;

        __host__ __device__ bool operator()(int gaussian_idx) const {
            if (transform_indices == nullptr || node_visibility_mask == nullptr || num_nodes <= 0) {
                return true;
            }
            const int node_idx = transform_indices[gaussian_idx];
            if (node_idx < 0 || node_idx >= num_nodes) {
                return true;
            }
            return node_visibility_mask[node_idx];
        }
    };

    struct ComputedVisibleIndices {
        Tensor tensor;
        size_t count = 0;

        static ComputedVisibleIndices compute(
            int n_gaussians,
            const Tensor* transform_indices,
            const std::vector<bool>& node_visibility_mask_cpu,
            const GpuBoolMask& node_visibility_mask_gpu,
            cudaStream_t stream = nullptr) {

            ComputedVisibleIndices result;

            if (transform_indices == nullptr || !transform_indices->is_valid() ||
                node_visibility_mask_gpu.ptr == nullptr || node_visibility_mask_gpu.count == 0) {
                return result;
            }

            bool all_visible = true;
            for (size_t i = 0; i < node_visibility_mask_cpu.size() && all_visible; ++i) {
                if (!node_visibility_mask_cpu[i]) {
                    all_visible = false;
                }
            }
            if (all_visible) {
                return result;
            }

            VisibilityPredicate predicate{
                transform_indices->ptr<int>(),
                node_visibility_mask_gpu.ptr,
                node_visibility_mask_gpu.count};

            result.tensor = Tensor::empty({static_cast<size_t>(n_gaussians)},
                                          lfs::core::Device::CUDA, lfs::core::DataType::Int32);

            thrust::counting_iterator<int> counting(0);
            auto out_ptr = thrust::device_pointer_cast(result.tensor.ptr<int>());

            auto end_it = thrust::copy_if(
                thrust::cuda::par.on(stream),
                counting, counting + n_gaussians, out_ptr, predicate);

            result.count = static_cast<size_t>(end_it - out_ptr);
            return result;
        }
    };

    struct PreparedModelTransforms {
        Tensor contig;
        const float* ptr = nullptr;
        int count = 0;

        static PreparedModelTransforms from(const Tensor* model_transforms) {
            PreparedModelTransforms result;
            if (model_transforms == nullptr || !model_transforms->is_valid() || model_transforms->numel() == 0) {
                return result;
            }
            result.contig = model_transforms->is_contiguous() ? *model_transforms : model_transforms->contiguous();
            if (result.contig.numel() % 16 != 0) {
                throw std::runtime_error("model_transforms tensor must contain a multiple of 16 float values (N x 4 x 4).");
            }
            result.ptr = result.contig.ptr<float>();
            result.count = static_cast<int>(result.contig.numel() / 16);
            return result;
        }
    };

    struct PreparedForwardSharedInputs {
        PreparedModelTransforms prepared_transforms;
        const float* model_transforms_ptr = nullptr;
        int num_transforms = 0;

        Tensor transform_indices_contig;
        const int* transform_indices_ptr = nullptr;

        Tensor selection_mask_contig;
        const uint8_t* selection_mask_ptr = nullptr;

        bool* preview_selection_ptr = nullptr;

        Tensor crop_box_transform_contig;
        Tensor crop_box_min_contig;
        Tensor crop_box_max_contig;
        const float* crop_box_transform_ptr = nullptr;
        const float3* crop_box_min_ptr = nullptr;
        const float3* crop_box_max_ptr = nullptr;

        Tensor ellipsoid_transform_contig;
        Tensor ellipsoid_radii_contig;
        const float* ellipsoid_transform_ptr = nullptr;
        const float3* ellipsoid_radii_ptr = nullptr;

        Tensor view_volume_transform_contig;
        Tensor view_volume_min_contig;
        Tensor view_volume_max_contig;
        const float* view_volume_transform_ptr = nullptr;
        const float3* view_volume_min_ptr = nullptr;
        const float3* view_volume_max_ptr = nullptr;

        Tensor deleted_mask_contig;
        const bool* deleted_mask_ptr = nullptr;

        Tensor emphasized_node_mask_tensor;
        const bool* emphasized_node_mask_ptr = nullptr;
        int num_selected_nodes = 0;

        GpuBoolMask visibility_mask;
        ComputedVisibleIndices computed_visible;
        const int* visible_indices_ptr = nullptr;
        int actual_visible_count = 0;
    };

    [[nodiscard]] PreparedForwardSharedInputs prepare_forward_shared_inputs(
        const int n_primitives,
        const Tensor* model_transforms,
        const Tensor* transform_indices,
        const Tensor* selection_mask,
        Tensor* preview_selection_out,
        const Tensor* crop_box_transform,
        const Tensor* crop_box_min,
        const Tensor* crop_box_max,
        const Tensor* ellipsoid_transform,
        const Tensor* ellipsoid_radii,
        const Tensor* view_volume_transform,
        const Tensor* view_volume_min,
        const Tensor* view_volume_max,
        const Tensor* deleted_mask,
        const std::vector<bool>& emphasized_node_mask,
        const std::vector<bool>& node_visibility_mask,
        cudaStream_t stream = nullptr) {

        PreparedForwardSharedInputs prepared;

        prepared.prepared_transforms = PreparedModelTransforms::from(model_transforms);
        prepared.model_transforms_ptr = prepared.prepared_transforms.ptr;
        prepared.num_transforms = prepared.prepared_transforms.count;

        if (transform_indices != nullptr && transform_indices->is_valid() && transform_indices->numel() > 0) {
            prepared.transform_indices_contig = transform_indices->is_contiguous() ? *transform_indices : transform_indices->contiguous();
            prepared.transform_indices_ptr = prepared.transform_indices_contig.ptr<int>();
        }

        if (selection_mask != nullptr && selection_mask->is_valid() && selection_mask->numel() > 0) {
            prepared.selection_mask_contig = selection_mask->is_contiguous() ? *selection_mask : selection_mask->contiguous();
            prepared.selection_mask_ptr = prepared.selection_mask_contig.ptr<uint8_t>();
        }

        prepared.preview_selection_ptr = (preview_selection_out && preview_selection_out->is_valid())
                                             ? preview_selection_out->ptr<bool>()
                                             : nullptr;

        if (crop_box_transform != nullptr && crop_box_transform->is_valid() &&
            crop_box_min != nullptr && crop_box_min->is_valid() &&
            crop_box_max != nullptr && crop_box_max->is_valid()) {
            prepared.crop_box_transform_contig = crop_box_transform->is_contiguous() ? *crop_box_transform : crop_box_transform->contiguous();
            prepared.crop_box_min_contig = crop_box_min->is_contiguous() ? *crop_box_min : crop_box_min->contiguous();
            prepared.crop_box_max_contig = crop_box_max->is_contiguous() ? *crop_box_max : crop_box_max->contiguous();
            prepared.crop_box_transform_ptr = prepared.crop_box_transform_contig.ptr<float>();
            prepared.crop_box_min_ptr = reinterpret_cast<const float3*>(prepared.crop_box_min_contig.ptr<float>());
            prepared.crop_box_max_ptr = reinterpret_cast<const float3*>(prepared.crop_box_max_contig.ptr<float>());
        }

        if (ellipsoid_transform != nullptr && ellipsoid_transform->is_valid() &&
            ellipsoid_radii != nullptr && ellipsoid_radii->is_valid()) {
            prepared.ellipsoid_transform_contig = ellipsoid_transform->is_contiguous() ? *ellipsoid_transform : ellipsoid_transform->contiguous();
            prepared.ellipsoid_radii_contig = ellipsoid_radii->is_contiguous() ? *ellipsoid_radii : ellipsoid_radii->contiguous();
            prepared.ellipsoid_transform_ptr = prepared.ellipsoid_transform_contig.ptr<float>();
            prepared.ellipsoid_radii_ptr = reinterpret_cast<const float3*>(prepared.ellipsoid_radii_contig.ptr<float>());
        }

        if (view_volume_transform != nullptr && view_volume_transform->is_valid() &&
            view_volume_min != nullptr && view_volume_min->is_valid() &&
            view_volume_max != nullptr && view_volume_max->is_valid()) {
            prepared.view_volume_transform_contig = view_volume_transform->is_contiguous() ? *view_volume_transform : view_volume_transform->contiguous();
            prepared.view_volume_min_contig = view_volume_min->is_contiguous() ? *view_volume_min : view_volume_min->contiguous();
            prepared.view_volume_max_contig = view_volume_max->is_contiguous() ? *view_volume_max : view_volume_max->contiguous();
            prepared.view_volume_transform_ptr = prepared.view_volume_transform_contig.ptr<float>();
            prepared.view_volume_min_ptr = reinterpret_cast<const float3*>(prepared.view_volume_min_contig.ptr<float>());
            prepared.view_volume_max_ptr = reinterpret_cast<const float3*>(prepared.view_volume_max_contig.ptr<float>());
        }

        if (deleted_mask != nullptr && deleted_mask->is_valid() && deleted_mask->numel() > 0) {
            prepared.deleted_mask_contig = deleted_mask->is_contiguous() ? *deleted_mask : deleted_mask->contiguous();
            prepared.deleted_mask_ptr = prepared.deleted_mask_contig.ptr<bool>();
        }

        prepared.num_selected_nodes = static_cast<int>(emphasized_node_mask.size());
        if (prepared.num_selected_nodes > 0) {
            std::vector<uint8_t> mask_data(static_cast<size_t>(prepared.num_selected_nodes));
            std::transform(emphasized_node_mask.begin(), emphasized_node_mask.end(),
                           mask_data.begin(), [](const bool value) -> uint8_t { return value ? 1 : 0; });
            prepared.emphasized_node_mask_tensor = Tensor::from_blob(
                                                       mask_data.data(),
                                                       {static_cast<size_t>(prepared.num_selected_nodes)},
                                                       lfs::core::Device::CPU,
                                                       lfs::core::DataType::UInt8)
                                                       .cuda();
            prepared.emphasized_node_mask_ptr =
                reinterpret_cast<const bool*>(prepared.emphasized_node_mask_tensor.ptr<uint8_t>());
        }

        prepared.visibility_mask = GpuBoolMask(node_visibility_mask);
        prepared.computed_visible = ComputedVisibleIndices::compute(
            n_primitives,
            prepared.transform_indices_ptr ? &prepared.transform_indices_contig : nullptr,
            node_visibility_mask,
            prepared.visibility_mask,
            stream);
        prepared.visible_indices_ptr = prepared.computed_visible.count > 0
                                           ? prepared.computed_visible.tensor.ptr<int>()
                                           : nullptr;
        prepared.actual_visible_count = static_cast<int>(prepared.computed_visible.count);

        return prepared;
    }

    std::tuple<Tensor, Tensor, Tensor>
    forward_wrapper_tensor(
        const Tensor& means,
        const Tensor& scales_raw,
        const Tensor& rotations_raw,
        const Tensor& opacities_raw,
        const Tensor& sh_coefficients_0,
        const Tensor& sh_coefficients_rest,
        const Tensor& w2c,
        const Tensor& cam_position,
        const int active_sh_bases,
        const int width,
        const int height,
        const float focal_x,
        const float focal_y,
        const float center_x,
        const float center_y,
        const float near_plane,
        const float far_plane,
        const bool show_rings,
        const float ring_width,
        const Tensor* model_transforms,
        const Tensor* transform_indices,
        const Tensor* selection_mask,
        Tensor* screen_positions_out,
        bool cursor_active,
        float cursor_x,
        float cursor_y,
        float cursor_radius,
        bool preview_selection_add_mode,
        Tensor* preview_selection_out,
        bool cursor_saturation_preview,
        float cursor_saturation_amount,
        bool show_center_markers,
        const Tensor* crop_box_transform,
        const Tensor* crop_box_min,
        const Tensor* crop_box_max,
        bool crop_inverse,
        bool crop_desaturate,
        int crop_parent_node_index,
        const Tensor* ellipsoid_transform,
        const Tensor* ellipsoid_radii,
        bool ellipsoid_inverse,
        bool ellipsoid_desaturate,
        int ellipsoid_parent_node_index,
        const Tensor* view_volume_transform,
        const Tensor* view_volume_min,
        const Tensor* view_volume_max,
        bool view_volume_cull,
        const Tensor* deleted_mask,
        unsigned long long* hovered_depth_id,
        int focused_gaussian_id,
        const std::vector<bool>& emphasized_node_mask,
        bool dim_non_emphasized,
        const std::vector<bool>& node_visibility_mask,
        float emphasis_flash_intensity,
        bool orthographic,
        float ortho_scale,
        bool mip_filter) {

        check_tensor_input(config::debug, means, "means");
        check_tensor_input(config::debug, scales_raw, "scales_raw");
        check_tensor_input(config::debug, rotations_raw, "rotations_raw");
        check_tensor_input(config::debug, opacities_raw, "opacities_raw");
        check_tensor_input(config::debug, sh_coefficients_0, "sh_coefficients_0");
        check_tensor_input(config::debug, sh_coefficients_rest, "sh_coefficients_rest");

        const int n_primitives = static_cast<int>(means.size(0));
        const int total_bases_sh_rest = static_cast<int>(sh_coefficients_rest.size(1));

        Tensor image = Tensor::empty({3, static_cast<size_t>(height), static_cast<size_t>(width)},
                                     lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        Tensor alpha = Tensor::empty({1, static_cast<size_t>(height), static_cast<size_t>(width)},
                                     lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        Tensor depth = Tensor::empty({1, static_cast<size_t>(height), static_cast<size_t>(width)},
                                     lfs::core::Device::CUDA, lfs::core::DataType::Float32);

        // Coordinate with training: wait for training, use shared arena
        auto& arena = lfs::core::GlobalArenaManager::instance().get_arena();
        arena.set_rendering_active(true);
        bool frame_started = false;
        uint64_t frame_id = 0;

        try {
            frame_id = arena.begin_frame(true); // true = from_rendering
            frame_started = true;
            auto arena_allocator = arena.get_allocator(frame_id);

            const std::function<char*(size_t)> per_primitive_buffers_func = arena_allocator;
            const std::function<char*(size_t)> per_tile_buffers_func = arena_allocator;
            const std::function<char*(size_t)> per_instance_buffers_func = arena_allocator;

            Tensor w2c_contig = w2c.is_contiguous() ? w2c : w2c.contiguous();
            Tensor cam_pos_contig = cam_position.is_contiguous() ? cam_position : cam_position.contiguous();

            const auto prepared_inputs = prepare_forward_shared_inputs(
                n_primitives,
                model_transforms,
                transform_indices,
                selection_mask,
                preview_selection_out,
                crop_box_transform,
                crop_box_min,
                crop_box_max,
                ellipsoid_transform,
                ellipsoid_radii,
                view_volume_transform,
                view_volume_min,
                view_volume_max,
                deleted_mask,
                emphasized_node_mask,
                node_visibility_mask);

            // Prepare screen positions output buffer if requested
            float2* screen_positions_ptr = nullptr;
            if (screen_positions_out != nullptr) {
                *screen_positions_out = Tensor::empty({static_cast<size_t>(n_primitives), 2},
                                                      lfs::core::Device::CUDA, lfs::core::DataType::Float32);
                screen_positions_ptr = reinterpret_cast<float2*>(screen_positions_out->ptr<float>());
            }

            forward(
                per_primitive_buffers_func,
                per_tile_buffers_func,
                per_instance_buffers_func,
                reinterpret_cast<const float3*>(means.ptr<float>()),
                reinterpret_cast<const float3*>(scales_raw.ptr<float>()),
                reinterpret_cast<const float4*>(rotations_raw.ptr<float>()),
                opacities_raw.ptr<float>(),
                reinterpret_cast<const float3*>(sh_coefficients_0.ptr<float>()),
                reinterpret_cast<const float3*>(sh_coefficients_rest.ptr<float>()),
                reinterpret_cast<const float4*>(w2c_contig.ptr<float>()),
                reinterpret_cast<const float3*>(cam_pos_contig.ptr<float>()),
                image.ptr<float>(),
                alpha.ptr<float>(),
                depth.ptr<float>(),
                n_primitives,
                active_sh_bases,
                total_bases_sh_rest,
                width,
                height,
                focal_x,
                focal_y,
                center_x,
                center_y,
                near_plane,
                far_plane,
                show_rings,
                ring_width,
                prepared_inputs.model_transforms_ptr,
                prepared_inputs.transform_indices_ptr,
                prepared_inputs.num_transforms,
                prepared_inputs.selection_mask_ptr,
                screen_positions_ptr,
                cursor_active,
                cursor_x,
                cursor_y,
                cursor_radius,
                preview_selection_add_mode,
                prepared_inputs.preview_selection_ptr,
                cursor_saturation_preview,
                cursor_saturation_amount,
                show_center_markers,
                prepared_inputs.crop_box_transform_ptr,
                prepared_inputs.crop_box_min_ptr,
                prepared_inputs.crop_box_max_ptr,
                crop_inverse,
                crop_desaturate,
                crop_parent_node_index,
                prepared_inputs.ellipsoid_transform_ptr,
                prepared_inputs.ellipsoid_radii_ptr,
                ellipsoid_inverse,
                ellipsoid_desaturate,
                ellipsoid_parent_node_index,
                prepared_inputs.view_volume_transform_ptr,
                prepared_inputs.view_volume_min_ptr,
                prepared_inputs.view_volume_max_ptr,
                view_volume_cull,
                prepared_inputs.deleted_mask_ptr,
                hovered_depth_id,
                focused_gaussian_id,
                prepared_inputs.emphasized_node_mask_ptr,
                prepared_inputs.num_selected_nodes,
                dim_non_emphasized,
                prepared_inputs.visibility_mask.ptr,
                prepared_inputs.visibility_mask.count,
                emphasis_flash_intensity,
                orthographic,
                ortho_scale,
                mip_filter,
                prepared_inputs.visible_indices_ptr,
                prepared_inputs.actual_visible_count);

            arena.end_frame(frame_id, true); // true = from_rendering
            frame_started = false;
            arena.set_rendering_active(false);

            return {std::move(image), std::move(alpha), std::move(depth)};
        } catch (...) {
            if (frame_started) {
                arena.end_frame(frame_id, true);
            }
            arena.set_rendering_active(false);
            throw;
        }
    }

    std::array<ForwardWrapperTensorResult, 2> forward_wrapper_tensor_dual(
        const Tensor& means,
        const Tensor& scales_raw,
        const Tensor& rotations_raw,
        const Tensor& opacities_raw,
        const Tensor& sh_coefficients_0,
        const Tensor& sh_coefficients_rest,
        const std::array<ForwardWrapperTensorViewState, 2>& views,
        const ForwardWrapperTensorSharedParams& shared) {

        check_tensor_input(config::debug, means, "means");
        check_tensor_input(config::debug, scales_raw, "scales_raw");
        check_tensor_input(config::debug, rotations_raw, "rotations_raw");
        check_tensor_input(config::debug, opacities_raw, "opacities_raw");
        check_tensor_input(config::debug, sh_coefficients_0, "sh_coefficients_0");
        check_tensor_input(config::debug, sh_coefficients_rest, "sh_coefficients_rest");

        const int n_primitives = static_cast<int>(means.size(0));
        const int total_bases_sh_rest = static_cast<int>(sh_coefficients_rest.size(1));
        const std::vector<bool> empty_mask;
        const std::vector<bool>& emphasized_node_mask =
            shared.emphasized_node_mask ? *shared.emphasized_node_mask : empty_mask;
        const std::vector<bool>& node_visibility_mask =
            shared.node_visibility_mask ? *shared.node_visibility_mask : empty_mask;

        std::array<ForwardWrapperTensorResult, 2> outputs;
        for (size_t i = 0; i < outputs.size(); ++i) {
            outputs[i].image = Tensor::empty(
                {3, static_cast<size_t>(views[i].height), static_cast<size_t>(views[i].width)},
                lfs::core::Device::CUDA, lfs::core::DataType::Float32);
            outputs[i].alpha = Tensor::empty(
                {1, static_cast<size_t>(views[i].height), static_cast<size_t>(views[i].width)},
                lfs::core::Device::CUDA, lfs::core::DataType::Float32);
            outputs[i].depth = Tensor::empty(
                {1, static_cast<size_t>(views[i].height), static_cast<size_t>(views[i].width)},
                lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        }

        auto& arena = lfs::core::GlobalArenaManager::instance().get_arena();
        arena.set_rendering_active(true);
        bool frame_started = false;
        uint64_t frame_id = 0;
        try {
            frame_id = arena.begin_frame(true);
            frame_started = true;
            auto arena_allocator = arena.get_allocator(frame_id);

            const auto cleanup = [&]() {
                arena.end_frame(frame_id, true);
                frame_started = false;
                arena.set_rendering_active(false);
            };

            const auto prepared_inputs = prepare_forward_shared_inputs(
                n_primitives,
                shared.model_transforms,
                shared.transform_indices,
                shared.selection_mask,
                shared.preview_selection_out,
                shared.crop_box_transform,
                shared.crop_box_min,
                shared.crop_box_max,
                shared.ellipsoid_transform,
                shared.ellipsoid_radii,
                shared.view_volume_transform,
                shared.view_volume_min,
                shared.view_volume_max,
                shared.deleted_mask,
                emphasized_node_mask,
                node_visibility_mask);

            std::array<Tensor, 2> w2c_contig;
            std::array<Tensor, 2> cam_pos_contig;

            int device = 0;
            if (const cudaError_t get_device_err = cudaGetDevice(&device); get_device_err != cudaSuccess) {
                throw_cuda_runtime_error("cudaGetDevice", get_device_err);
            }

            for (size_t i = 0; i < views.size(); ++i) {
                w2c_contig[i] = views[i].w2c.is_contiguous() ? views[i].w2c : views[i].w2c.contiguous();
                cam_pos_contig[i] = views[i].cam_position.is_contiguous()
                                        ? views[i].cam_position
                                        : views[i].cam_position.contiguous();
            }

            assert(
                prepared_inputs.preview_selection_ptr == nullptr ||
                !(views[0].cursor_active && views[1].cursor_active));

            std::array<CachedDualForwardExecutor::Task, 2> tasks;
            for (size_t i = 0; i < views.size(); ++i) {
                tasks[i] = [&, i](cudaStream_t stream) {
                    forward(
                        arena_allocator,
                        arena_allocator,
                        arena_allocator,
                        reinterpret_cast<const float3*>(means.ptr<float>()),
                        reinterpret_cast<const float3*>(scales_raw.ptr<float>()),
                        reinterpret_cast<const float4*>(rotations_raw.ptr<float>()),
                        opacities_raw.ptr<float>(),
                        reinterpret_cast<const float3*>(sh_coefficients_0.ptr<float>()),
                        reinterpret_cast<const float3*>(sh_coefficients_rest.ptr<float>()),
                        reinterpret_cast<const float4*>(w2c_contig[i].ptr<float>()),
                        reinterpret_cast<const float3*>(cam_pos_contig[i].ptr<float>()),
                        outputs[i].image.ptr<float>(),
                        outputs[i].alpha.ptr<float>(),
                        outputs[i].depth.ptr<float>(),
                        n_primitives,
                        shared.active_sh_bases,
                        total_bases_sh_rest,
                        views[i].width,
                        views[i].height,
                        views[i].focal_x,
                        views[i].focal_y,
                        views[i].center_x,
                        views[i].center_y,
                        shared.near_plane,
                        shared.far_plane,
                        shared.show_rings,
                        shared.ring_width,
                        prepared_inputs.model_transforms_ptr,
                        prepared_inputs.transform_indices_ptr,
                        prepared_inputs.num_transforms,
                        prepared_inputs.selection_mask_ptr,
                        nullptr,
                        views[i].cursor_active,
                        views[i].cursor_x,
                        views[i].cursor_y,
                        views[i].cursor_radius,
                        shared.preview_selection_add_mode,
                        prepared_inputs.preview_selection_ptr,
                        views[i].cursor_saturation_preview,
                        views[i].cursor_saturation_amount,
                        shared.show_center_markers,
                        prepared_inputs.crop_box_transform_ptr,
                        prepared_inputs.crop_box_min_ptr,
                        prepared_inputs.crop_box_max_ptr,
                        shared.crop_inverse,
                        shared.crop_desaturate,
                        shared.crop_parent_node_index,
                        prepared_inputs.ellipsoid_transform_ptr,
                        prepared_inputs.ellipsoid_radii_ptr,
                        shared.ellipsoid_inverse,
                        shared.ellipsoid_desaturate,
                        shared.ellipsoid_parent_node_index,
                        prepared_inputs.view_volume_transform_ptr,
                        prepared_inputs.view_volume_min_ptr,
                        prepared_inputs.view_volume_max_ptr,
                        shared.view_volume_cull,
                        prepared_inputs.deleted_mask_ptr,
                        views[i].hovered_depth_id,
                        views[i].focused_gaussian_id,
                        prepared_inputs.emphasized_node_mask_ptr,
                        prepared_inputs.num_selected_nodes,
                        shared.dim_non_emphasized,
                        prepared_inputs.visibility_mask.ptr,
                        prepared_inputs.visibility_mask.count,
                        shared.emphasis_flash_intensity,
                        shared.orthographic,
                        shared.ortho_scale,
                        shared.mip_filter,
                        prepared_inputs.visible_indices_ptr,
                        prepared_inputs.actual_visible_count,
                        stream);
                };
            }

            CachedDualForwardExecutor::instance().run(std::move(tasks), device);
            cleanup();
            return outputs;
        } catch (...) {
            if (frame_started) {
                arena.end_frame(frame_id, true);
            }
            arena.set_rendering_active(false);
            throw;
        }
    }

    void brush_select_tensor(
        const Tensor& screen_positions,
        float mouse_x,
        float mouse_y,
        float radius,
        Tensor& selection_out) {

        if (!screen_positions.is_valid() || screen_positions.size(0) == 0)
            return;

        int n_primitives = static_cast<int>(screen_positions.size(0));

        brush_select(
            reinterpret_cast<const float2*>(screen_positions.ptr<float>()),
            mouse_x,
            mouse_y,
            radius,
            selection_out.ptr<uint8_t>(),
            n_primitives);
    }

    void polygon_select_tensor(
        const Tensor& positions,
        const Tensor& polygon,
        Tensor& selection) {
        if (!positions.is_valid() || positions.size(0) == 0)
            return;
        if (!polygon.is_valid() || polygon.size(0) < 3)
            return;

        polygon_select(
            reinterpret_cast<const float2*>(positions.ptr<float>()),
            reinterpret_cast<const float2*>(polygon.ptr<float>()),
            static_cast<int>(polygon.size(0)),
            selection.ptr<bool>(),
            static_cast<int>(positions.size(0)));
    }

    void rect_select_tensor(
        const Tensor& positions,
        const float x0, const float y0, const float x1, const float y1,
        Tensor& selection) {
        if (!positions.is_valid() || positions.size(0) == 0)
            return;

        rect_select(
            reinterpret_cast<const float2*>(positions.ptr<float>()),
            x0, y0, x1, y1,
            selection.ptr<bool>(),
            static_cast<int>(positions.size(0)));
    }

    void rect_select_mode_tensor(
        const Tensor& positions,
        const float x0, const float y0, const float x1, const float y1,
        Tensor& selection,
        const bool add_mode) {
        if (!positions.is_valid() || positions.size(0) == 0)
            return;

        rect_select_mode(
            reinterpret_cast<const float2*>(positions.ptr<float>()),
            x0, y0, x1, y1,
            selection.ptr<bool>(),
            static_cast<int>(positions.size(0)),
            add_mode);
    }

    void polygon_select_mode_tensor(
        const Tensor& positions,
        const Tensor& polygon,
        Tensor& selection,
        const bool add_mode) {
        if (!positions.is_valid() || positions.size(0) == 0)
            return;
        if (!polygon.is_valid() || polygon.size(0) < 3)
            return;

        polygon_select_mode(
            reinterpret_cast<const float2*>(positions.ptr<float>()),
            reinterpret_cast<const float2*>(polygon.ptr<float>()),
            static_cast<int>(polygon.size(0)),
            selection.ptr<bool>(),
            static_cast<int>(positions.size(0)),
            add_mode);
    }

    __global__ void apply_selection_group_kernel(
        const bool* __restrict__ cumulative,
        const uint8_t* __restrict__ existing,
        uint8_t* __restrict__ output,
        const int n,
        const uint8_t group_id,
        const uint32_t* __restrict__ locked_groups,
        const bool add_mode,
        const int* __restrict__ node_indices,
        const int target_node) {

        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n)
            return;

        const uint8_t existing_group = existing ? existing[idx] : 0;
        const bool selected = cumulative[idx];

        if (node_indices && target_node >= 0 && node_indices[idx] != target_node) {
            output[idx] = existing_group;
            return;
        }

        if (add_mode) {
            if (selected) {
                // Check if existing group is locked (bit test)
                const bool is_locked = existing_group != 0 &&
                                       existing_group != group_id &&
                                       locked_groups &&
                                       (locked_groups[existing_group / 32] & (1u << (existing_group % 32)));
                output[idx] = is_locked ? existing_group : group_id;
            } else {
                output[idx] = existing_group;
            }
        } else {
            // Remove mode: only clear if selected AND belongs to this group
            output[idx] = (selected && existing_group == group_id) ? 0 : existing_group;
        }
    }

    void apply_selection_group_tensor(
        const Tensor& cumulative_selection,
        const Tensor& existing_mask,
        Tensor& output_mask,
        const uint8_t group_id,
        const uint32_t* locked_groups,
        const bool add_mode,
        const Tensor* transform_indices,
        const int target_node_index) {

        if (!cumulative_selection.is_valid() || cumulative_selection.size(0) == 0)
            return;

        const int n = static_cast<int>(cumulative_selection.size(0));
        constexpr int BLOCK_SIZE = 256;
        const int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        const uint8_t* existing_ptr = (existing_mask.is_valid() && existing_mask.numel() == static_cast<size_t>(n))
                                          ? existing_mask.ptr<uint8_t>()
                                          : nullptr;

        const int* node_indices_ptr = (transform_indices && transform_indices->is_valid() &&
                                       transform_indices->numel() == static_cast<size_t>(n))
                                          ? transform_indices->ptr<int>()
                                          : nullptr;

        apply_selection_group_kernel<<<grid_size, BLOCK_SIZE>>>(
            cumulative_selection.ptr<bool>(),
            existing_ptr,
            output_mask.ptr<uint8_t>(),
            n,
            group_id,
            locked_groups,
            add_mode,
            node_indices_ptr,
            target_node_index);
    }

    namespace {
        constexpr int KERNEL_BLOCK_SIZE = 256;

        // Upload bool vector to GPU (small data, typically < 100 nodes)
        inline Tensor upload_bool_mask(const std::vector<bool>& mask) {
            const size_t n = mask.size();
            auto tensor = Tensor::empty({n}, lfs::core::Device::CPU, lfs::core::DataType::UInt8);
            auto* const ptr = tensor.ptr<uint8_t>();
            for (size_t i = 0; i < n; ++i) {
                ptr[i] = mask[i] ? 1 : 0;
            }
            return tensor.cuda();
        }
    } // namespace

    __global__ void apply_selection_group_mask_kernel(
        const bool* __restrict__ cumulative,
        const uint8_t* __restrict__ existing,
        uint8_t* __restrict__ output,
        const int n,
        const uint8_t group_id,
        const uint32_t* __restrict__ locked_groups,
        const bool add_mode,
        const int* __restrict__ node_indices,
        const bool* __restrict__ valid_nodes,
        const int num_nodes,
        const bool replace_mode) {

        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n)
            return;

        const uint8_t existing_group = existing ? existing[idx] : 0;

        // Skip if node not in valid set
        if (node_indices && valid_nodes) {
            const int node_idx = node_indices[idx];
            if (node_idx < 0 || node_idx >= num_nodes || !valid_nodes[node_idx]) {
                output[idx] = existing_group;
                return;
            }
        }

        const bool selected = cumulative[idx];

        // Check if existing group is locked (can't be modified)
        const bool is_other_locked = existing_group != 0 &&
                                     existing_group != group_id &&
                                     locked_groups &&
                                     (locked_groups[existing_group / 32] & (1u << (existing_group % 32)));

        if (replace_mode) {
            // Replace: clear active group, apply new selection
            if (selected) {
                output[idx] = is_other_locked ? existing_group : group_id;
            } else if (existing_group == group_id) {
                output[idx] = 0;
            } else {
                output[idx] = existing_group;
            }
        } else if (add_mode) {
            // Add: set selected to group_id
            output[idx] = (selected && !is_other_locked) ? group_id : existing_group;
        } else {
            // Remove: clear from active group only
            output[idx] = (selected && existing_group == group_id) ? 0 : existing_group;
        }
    }

    void apply_selection_group_tensor_mask(
        const Tensor& cumulative_selection,
        const Tensor& existing_mask,
        Tensor& output_mask,
        const uint8_t group_id,
        const uint32_t* locked_groups,
        const bool add_mode,
        const Tensor* transform_indices,
        const std::vector<bool>& valid_nodes,
        const bool replace_mode) {

        if (!cumulative_selection.is_valid() || cumulative_selection.size(0) == 0)
            return;

        const int n = static_cast<int>(cumulative_selection.size(0));

        // If valid_nodes is empty, treat all nodes as valid
        std::vector<bool> effective_valid_nodes = valid_nodes;
        if (effective_valid_nodes.empty()) {
            effective_valid_nodes.resize(1, true);
        }
        const int num_nodes = static_cast<int>(effective_valid_nodes.size());

        const uint8_t* const existing_ptr = (existing_mask.is_valid() &&
                                             existing_mask.numel() == static_cast<size_t>(n))
                                                ? existing_mask.ptr<uint8_t>()
                                                : nullptr;

        const int* const node_indices_ptr = (transform_indices && transform_indices->is_valid() &&
                                             transform_indices->numel() == static_cast<size_t>(n))
                                                ? transform_indices->ptr<int>()
                                                : nullptr;

        const Tensor valid_nodes_gpu = upload_bool_mask(effective_valid_nodes);
        const int grid_size = (n + KERNEL_BLOCK_SIZE - 1) / KERNEL_BLOCK_SIZE;

        apply_selection_group_mask_kernel<<<grid_size, KERNEL_BLOCK_SIZE>>>(
            cumulative_selection.ptr<bool>(),
            existing_ptr,
            output_mask.ptr<uint8_t>(),
            n,
            group_id,
            locked_groups,
            add_mode,
            node_indices_ptr,
            reinterpret_cast<const bool*>(valid_nodes_gpu.ptr<uint8_t>()),
            num_nodes,
            replace_mode);
    }

    __global__ void filter_selection_by_node_kernel(
        bool* __restrict__ selection,
        const int* __restrict__ node_indices,
        const int n,
        const int target_node) {

        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n)
            return;

        if (node_indices[idx] != target_node) {
            selection[idx] = false;
        }
    }

    void filter_selection_by_node(
        Tensor& selection,
        const Tensor& transform_indices,
        const int target_node_index) {

        if (!selection.is_valid() || !transform_indices.is_valid())
            return;
        if (target_node_index < 0)
            return;

        const int n = static_cast<int>(selection.size(0));
        if (transform_indices.numel() != static_cast<size_t>(n))
            return;

        constexpr int BLOCK_SIZE = 256;
        const int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        filter_selection_by_node_kernel<<<grid_size, BLOCK_SIZE>>>(
            selection.ptr<bool>(),
            transform_indices.ptr<int>(),
            n,
            target_node_index);
    }

    __global__ void filter_selection_by_node_mask_kernel(
        bool* __restrict__ selection,
        const int* __restrict__ node_indices,
        const bool* __restrict__ valid_nodes,
        const int n,
        const int num_nodes) {

        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n)
            return;

        const int node_idx = node_indices[idx];
        if (node_idx < 0 || node_idx >= num_nodes || !valid_nodes[node_idx]) {
            selection[idx] = false;
        }
    }

    void filter_selection_by_node_mask(
        Tensor& selection,
        const Tensor& transform_indices,
        const std::vector<bool>& valid_nodes) {

        if (!selection.is_valid() || !transform_indices.is_valid())
            return;
        if (valid_nodes.empty())
            return;

        const int n = static_cast<int>(selection.size(0));
        if (transform_indices.numel() != static_cast<size_t>(n))
            return;

        const int num_nodes = static_cast<int>(valid_nodes.size());
        const Tensor valid_nodes_gpu = upload_bool_mask(valid_nodes);
        const int grid_size = (n + KERNEL_BLOCK_SIZE - 1) / KERNEL_BLOCK_SIZE;

        filter_selection_by_node_mask_kernel<<<grid_size, KERNEL_BLOCK_SIZE>>>(
            selection.ptr<bool>(),
            transform_indices.ptr<int>(),
            reinterpret_cast<const bool*>(valid_nodes_gpu.ptr<uint8_t>()),
            n,
            num_nodes);
    }

    __global__ void filter_selection_by_crop_kernel(
        bool* __restrict__ selection,
        const float3* __restrict__ means,
        const float* __restrict__ crop_transform,
        const float3* crop_min,
        const float3* crop_max,
        const bool crop_inverse,
        const float* __restrict__ ellipsoid_transform,
        const float3* ellipsoid_radii,
        const bool ellipsoid_inverse,
        const float* __restrict__ model_transforms,
        const int* __restrict__ transform_indices,
        const int num_model_transforms,
        const int n) {

        const int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= n || !selection[idx])
            return;

        float3 pos = means[idx];

        if (model_transforms != nullptr && num_model_transforms > 0) {
            const int transform_idx = transform_indices != nullptr
                                          ? min(max(transform_indices[idx], 0), num_model_transforms - 1)
                                          : 0;
            const float* const m = model_transforms + transform_idx * 16;
            pos = make_float3(
                m[0] * pos.x + m[1] * pos.y + m[2] * pos.z + m[3],
                m[4] * pos.x + m[5] * pos.y + m[6] * pos.z + m[7],
                m[8] * pos.x + m[9] * pos.y + m[10] * pos.z + m[11]);
        }

        if (crop_transform && crop_min && crop_max) {
            const float* const c = crop_transform;
            const float lx = c[0] * pos.x + c[4] * pos.y + c[8] * pos.z + c[12];
            const float ly = c[1] * pos.x + c[5] * pos.y + c[9] * pos.z + c[13];
            const float lz = c[2] * pos.x + c[6] * pos.y + c[10] * pos.z + c[14];

            const float3 bmin = *crop_min;
            const float3 bmax = *crop_max;
            const bool inside = lx >= bmin.x && lx <= bmax.x &&
                                ly >= bmin.y && ly <= bmax.y &&
                                lz >= bmin.z && lz <= bmax.z;

            if (inside == crop_inverse) {
                selection[idx] = false;
                return;
            }
        }

        if (ellipsoid_transform && ellipsoid_radii) {
            const float* const e = ellipsoid_transform;
            const float lx = e[0] * pos.x + e[4] * pos.y + e[8] * pos.z + e[12];
            const float ly = e[1] * pos.x + e[5] * pos.y + e[9] * pos.z + e[13];
            const float lz = e[2] * pos.x + e[6] * pos.y + e[10] * pos.z + e[14];

            const float3 r = *ellipsoid_radii;
            const float norm = (lx * lx) / (r.x * r.x) + (ly * ly) / (r.y * r.y) + (lz * lz) / (r.z * r.z);

            if ((norm <= 1.0f) == ellipsoid_inverse) {
                selection[idx] = false;
            }
        }
    }

    void filter_selection_by_crop(
        Tensor& selection,
        const Tensor& means,
        const Tensor* crop_box_transform,
        const Tensor* crop_box_min,
        const Tensor* crop_box_max,
        const bool crop_inverse,
        const Tensor* ellipsoid_transform,
        const Tensor* ellipsoid_radii,
        const bool ellipsoid_inverse,
        const Tensor* model_transforms,
        const Tensor* transform_indices) {

        if (!selection.is_valid() || !means.is_valid())
            return;

        const int n = static_cast<int>(selection.size(0));
        if (means.size(0) != static_cast<size_t>(n))
            return;

        const float* crop_t_ptr = nullptr;
        const float3* crop_min_ptr = nullptr;
        const float3* crop_max_ptr = nullptr;
        if (crop_box_transform && crop_box_transform->is_valid() &&
            crop_box_min && crop_box_min->is_valid() &&
            crop_box_max && crop_box_max->is_valid()) {
            crop_t_ptr = crop_box_transform->ptr<float>();
            crop_min_ptr = reinterpret_cast<const float3*>(crop_box_min->ptr<float>());
            crop_max_ptr = reinterpret_cast<const float3*>(crop_box_max->ptr<float>());
        }

        const float* ellip_t_ptr = nullptr;
        const float3* ellip_radii_ptr = nullptr;
        if (ellipsoid_transform && ellipsoid_transform->is_valid() &&
            ellipsoid_radii && ellipsoid_radii->is_valid()) {
            ellip_t_ptr = ellipsoid_transform->ptr<float>();
            ellip_radii_ptr = reinterpret_cast<const float3*>(ellipsoid_radii->ptr<float>());
        }

        if (!crop_t_ptr && !ellip_t_ptr)
            return;

        const auto prepared_transforms = PreparedModelTransforms::from(model_transforms);
        const float* const model_transforms_ptr = prepared_transforms.ptr;
        const int num_model_transforms = prepared_transforms.count;

        Tensor transform_indices_contig;
        const int* transform_indices_ptr = nullptr;
        if (transform_indices != nullptr && transform_indices->is_valid() &&
            transform_indices->numel() == static_cast<size_t>(n)) {
            transform_indices_contig = transform_indices->is_contiguous() ? *transform_indices : transform_indices->contiguous();
            transform_indices_ptr = transform_indices_contig.ptr<int>();
        }

        const int grid_size = (n + KERNEL_BLOCK_SIZE - 1) / KERNEL_BLOCK_SIZE;
        filter_selection_by_crop_kernel<<<grid_size, KERNEL_BLOCK_SIZE>>>(
            selection.ptr<bool>(),
            reinterpret_cast<const float3*>(means.ptr<float>()),
            crop_t_ptr, crop_min_ptr, crop_max_ptr, crop_inverse,
            ellip_t_ptr, ellip_radii_ptr, ellipsoid_inverse,
            model_transforms_ptr, transform_indices_ptr, num_model_transforms,
            n);
    }

    std::tuple<Tensor, Tensor, Tensor>
    forward_gut_tensor(
        const Tensor& means,
        const Tensor& scales_raw,
        const Tensor& rotations_raw,
        const Tensor& opacities_raw,
        const Tensor& sh0,
        const Tensor& sh_rest,
        const Tensor& w2c,
        const Tensor& K,
        const int sh_degree,
        const int width,
        const int height,
        const GutCameraModel camera_model,
        const Tensor* radial_coeffs,
        const Tensor* tangential_coeffs,
        const Tensor* background,
        const Tensor* model_transforms,
        const Tensor* transform_indices,
        const std::vector<bool>& node_visibility_mask) {

        constexpr float QUAT_NORM_EPS = 1e-8f;

        check_tensor_input(config::debug, means, "means");
        check_tensor_input(config::debug, scales_raw, "scales_raw");
        check_tensor_input(config::debug, rotations_raw, "rotations_raw");
        check_tensor_input(config::debug, opacities_raw, "opacities_raw");
        check_tensor_input(config::debug, sh0, "sh0");
        check_tensor_input(config::debug, sh_rest, "sh_rest");

        const int N_total = static_cast<int>(means.size(0));

        // Compute visible_indices from transform_indices + node_visibility_mask on GPU
        const GpuBoolMask visibility_mask(node_visibility_mask);
        auto computed_visible = ComputedVisibleIndices::compute(
            N_total, transform_indices, node_visibility_mask, visibility_mask);
        const bool use_visibility_filter = computed_visible.count > 0;

        const size_t H = static_cast<size_t>(height);
        const size_t W = static_cast<size_t>(width);
        const int num_sh_coeffs = 1 + static_cast<int>(sh_rest.size(1));

        // Output tensors
        Tensor image = Tensor::empty({3, H, W}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        Tensor alpha = Tensor::empty({1, H, W}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        Tensor depth = Tensor::empty({1, H, W}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);

        // Activate parameters on N-sized data (runs on all gaussians, but avoids expensive index_select copies)
        const Tensor scales = scales_raw.exp();
        const Tensor rotations = rotations_raw / rotations_raw.norm(2, -1, true).clamp_min(QUAT_NORM_EPS);
        const Tensor opacities = opacities_raw.sigmoid().squeeze(-1);

        // Concatenate SH coefficients [N_total, K, 3] - N-sized, accessed via visible_indices in kernel
        const Tensor sh_coeffs = (sh_rest.numel() > 0 && num_sh_coeffs > 1)
                                     ? Tensor::cat({sh0, sh_rest}, 1).contiguous()
                                     : sh0.contiguous();

        // Contiguous copies (N-sized)
        const Tensor means_c = means.contiguous();
        const Tensor scales_c = scales.contiguous();
        const Tensor rotations_c = rotations.contiguous();
        const Tensor opacities_c = opacities.contiguous();
        const Tensor w2c_c = w2c.contiguous();
        const Tensor K_c = K.contiguous();

        const float* const radial_ptr = (radial_coeffs && radial_coeffs->is_valid()) ? radial_coeffs->ptr<float>() : nullptr;
        const float* const tangential_ptr = (tangential_coeffs && tangential_coeffs->is_valid()) ? tangential_coeffs->ptr<float>() : nullptr;
        const float* const bg_ptr = (background && background->is_valid()) ? background->ptr<float>() : nullptr;

        const auto prepared_transforms = PreparedModelTransforms::from(model_transforms);
        const float* model_transforms_ptr = prepared_transforms.ptr;
        const int num_transforms = prepared_transforms.count;

        // Transform indices (N-sized, not filtered)
        const int* transform_indices_ptr = (transform_indices && transform_indices->is_valid())
                                               ? transform_indices->ptr<int>()
                                               : nullptr;

        // visible_indices for kernel-level indirect indexing
        const int* visible_indices_ptr = use_visibility_filter ? computed_visible.tensor.ptr<int>() : nullptr;
        const uint32_t visible_count = use_visibility_filter ? static_cast<uint32_t>(computed_visible.count) : 0;

        // Render buffers in HWC format (gsplat output format)
        Tensor render_hwc = Tensor::empty({H, W, 3}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        Tensor alpha_hw = Tensor::empty({H, W}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);

        // Pass N-sized arrays with visible_indices for kernel-level indirect indexing
        gsplat_forward_gut(
            means_c.ptr<float>(),
            rotations_c.ptr<float>(),
            scales_c.ptr<float>(),
            opacities_c.ptr<float>(),
            sh_coeffs.ptr<float>(),
            static_cast<uint32_t>(sh_degree),
            static_cast<uint32_t>(N_total), // N_total - full array size
            static_cast<uint32_t>(num_sh_coeffs),
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height),
            w2c_c.ptr<float>(),
            K_c.ptr<float>(),
            camera_model,
            radial_ptr,
            tangential_ptr,
            bg_ptr,
            GutRenderMode::RGB,
            1.0f,
            model_transforms_ptr,
            transform_indices_ptr,
            num_transforms,
            visibility_mask.ptr,
            visibility_mask.count,
            visible_indices_ptr, // Kernel uses this for indirect indexing
            visible_count,       // M (0 = use N)
            render_hwc.ptr<float>(),
            alpha_hw.ptr<float>(),
            depth.ptr<float>(),
            nullptr);

        // Convert HWC to CHW
        image = render_hwc.permute({2, 0, 1}).contiguous().clamp(0.0f, 1.0f);
        alpha = alpha_hw.unsqueeze(0);

        return {image, alpha, depth};
    }

} // namespace lfs::rendering
