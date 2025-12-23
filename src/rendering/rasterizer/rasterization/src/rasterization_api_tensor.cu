/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/memory_arena.hpp"
#include "forward.h"
#include "rasterization_api_tensor.h"
#include "rasterization_config.h"
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <tuple>

namespace lfs::rendering {

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
        bool brush_active,
        float brush_x,
        float brush_y,
        float brush_radius,
        bool brush_add_mode,
        Tensor* brush_selection_out,
        bool brush_saturation_mode,
        float brush_saturation_amount,
        bool selection_mode_rings,
        bool show_center_markers,
        const Tensor* crop_box_transform,
        const Tensor* crop_box_min,
        const Tensor* crop_box_max,
        bool crop_inverse,
        bool crop_desaturate,
        const Tensor* depth_filter_transform,
        const Tensor* depth_filter_min,
        const Tensor* depth_filter_max,
        const Tensor* deleted_mask,
        unsigned long long* hovered_depth_id,
        int highlight_gaussian_id,
        const std::vector<bool>& selected_node_mask,
        bool desaturate_unselected,
        float selection_flash_intensity,
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
        arena.wait_for_training();
        uint64_t frame_id = arena.begin_frame(true); // true = from_rendering
        auto arena_allocator = arena.get_allocator(frame_id);

        const std::function<char*(size_t)> per_primitive_buffers_func = arena_allocator;
        const std::function<char*(size_t)> per_tile_buffers_func = arena_allocator;
        const std::function<char*(size_t)> per_instance_buffers_func = arena_allocator;

        Tensor w2c_contig = w2c.is_contiguous() ? w2c : w2c.contiguous();
        Tensor cam_pos_contig = cam_position.is_contiguous() ? cam_position : cam_position.contiguous();

        // Prepare model transforms array pointer
        const float* model_transforms_ptr = nullptr;
        Tensor model_transforms_contig;
        int num_transforms = 0;
        if (model_transforms != nullptr && model_transforms->is_valid() && model_transforms->numel() > 0) {
            model_transforms_contig = model_transforms->is_contiguous() ? *model_transforms : model_transforms->contiguous();
            model_transforms_ptr = model_transforms_contig.ptr<float>();
            // Transforms are stored as [num_transforms, 4, 4] or flat [num_transforms * 16]
            num_transforms = static_cast<int>(model_transforms_contig.numel() / 16);
        }

        // Prepare transform indices pointer
        const int* transform_indices_ptr = nullptr;
        Tensor transform_indices_contig;
        if (transform_indices != nullptr && transform_indices->is_valid() && transform_indices->numel() > 0) {
            transform_indices_contig = transform_indices->is_contiguous() ? *transform_indices : transform_indices->contiguous();
            transform_indices_ptr = transform_indices_contig.ptr<int>();
        }

        // Prepare selection mask pointer
        const uint8_t* selection_mask_ptr = nullptr;
        Tensor selection_mask_contig;
        if (selection_mask != nullptr && selection_mask->is_valid() && selection_mask->numel() > 0) {
            selection_mask_contig = selection_mask->is_contiguous() ? *selection_mask : selection_mask->contiguous();
            selection_mask_ptr = selection_mask_contig.ptr<uint8_t>();
        }

        // Prepare screen positions output buffer if requested
        float2* screen_positions_ptr = nullptr;
        if (screen_positions_out != nullptr) {
            *screen_positions_out = Tensor::empty({static_cast<size_t>(n_primitives), 2},
                                                  lfs::core::Device::CUDA, lfs::core::DataType::Float32);
            screen_positions_ptr = reinterpret_cast<float2*>(screen_positions_out->ptr<float>());
        }

        // Preview selection tensor (used by rectangle/lasso/polygon modes regardless of brush_active)
        bool* const brush_selection_ptr = (brush_selection_out && brush_selection_out->is_valid())
                                              ? brush_selection_out->ptr<bool>()
                                              : nullptr;

        // Prepare crop box parameters
        const float* crop_box_transform_ptr = nullptr;
        const float3* crop_box_min_ptr = nullptr;
        const float3* crop_box_max_ptr = nullptr;
        Tensor crop_box_transform_contig, crop_box_min_contig, crop_box_max_contig;
        if (crop_box_transform != nullptr && crop_box_transform->is_valid() &&
            crop_box_min != nullptr && crop_box_min->is_valid() &&
            crop_box_max != nullptr && crop_box_max->is_valid()) {
            crop_box_transform_contig = crop_box_transform->is_contiguous() ? *crop_box_transform : crop_box_transform->contiguous();
            crop_box_min_contig = crop_box_min->is_contiguous() ? *crop_box_min : crop_box_min->contiguous();
            crop_box_max_contig = crop_box_max->is_contiguous() ? *crop_box_max : crop_box_max->contiguous();
            crop_box_transform_ptr = crop_box_transform_contig.ptr<float>();
            crop_box_min_ptr = reinterpret_cast<const float3*>(crop_box_min_contig.ptr<float>());
            crop_box_max_ptr = reinterpret_cast<const float3*>(crop_box_max_contig.ptr<float>());
        }

        // Prepare depth filter parameters (Selection tool - separate from crop box)
        const float* depth_filter_transform_ptr = nullptr;
        const float3* depth_filter_min_ptr = nullptr;
        const float3* depth_filter_max_ptr = nullptr;
        Tensor depth_filter_transform_contig, depth_filter_min_contig, depth_filter_max_contig;
        if (depth_filter_transform != nullptr && depth_filter_transform->is_valid() &&
            depth_filter_min != nullptr && depth_filter_min->is_valid() &&
            depth_filter_max != nullptr && depth_filter_max->is_valid()) {
            depth_filter_transform_contig = depth_filter_transform->is_contiguous() ? *depth_filter_transform : depth_filter_transform->contiguous();
            depth_filter_min_contig = depth_filter_min->is_contiguous() ? *depth_filter_min : depth_filter_min->contiguous();
            depth_filter_max_contig = depth_filter_max->is_contiguous() ? *depth_filter_max : depth_filter_max->contiguous();
            depth_filter_transform_ptr = depth_filter_transform_contig.ptr<float>();
            depth_filter_min_ptr = reinterpret_cast<const float3*>(depth_filter_min_contig.ptr<float>());
            depth_filter_max_ptr = reinterpret_cast<const float3*>(depth_filter_max_contig.ptr<float>());
        }

        // Prepare deleted mask pointer
        const bool* deleted_mask_ptr = nullptr;
        Tensor deleted_mask_contig;
        if (deleted_mask != nullptr && deleted_mask->is_valid() && deleted_mask->numel() > 0) {
            deleted_mask_contig = deleted_mask->is_contiguous() ? *deleted_mask : deleted_mask->contiguous();
            deleted_mask_ptr = deleted_mask_contig.ptr<bool>();
        }

        // Selected node mask (small, typically < 20 nodes)
        const bool* selected_node_mask_ptr = nullptr;
        Tensor selected_node_mask_tensor;
        const int num_selected_nodes = static_cast<int>(selected_node_mask.size());
        if (num_selected_nodes > 0) {
            // vector<bool> is not contiguous, convert to uint8_t
            std::vector<uint8_t> mask_data(num_selected_nodes);
            std::transform(selected_node_mask.begin(), selected_node_mask.end(),
                           mask_data.begin(), [](bool b) { return b ? 1 : 0; });
            selected_node_mask_tensor = Tensor::from_blob(
                                            mask_data.data(), {static_cast<size_t>(num_selected_nodes)},
                                            lfs::core::Device::CPU, lfs::core::DataType::UInt8)
                                            .cuda();
            selected_node_mask_ptr = reinterpret_cast<const bool*>(selected_node_mask_tensor.ptr<uint8_t>());
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
            model_transforms_ptr,
            transform_indices_ptr,
            num_transforms,
            selection_mask_ptr,
            screen_positions_ptr,
            brush_active,
            brush_x,
            brush_y,
            brush_radius,
            brush_add_mode,
            brush_selection_ptr,
            brush_saturation_mode,
            brush_saturation_amount,
            selection_mode_rings,
            show_center_markers,
            crop_box_transform_ptr,
            crop_box_min_ptr,
            crop_box_max_ptr,
            crop_inverse,
            crop_desaturate,
            depth_filter_transform_ptr,
            depth_filter_min_ptr,
            depth_filter_max_ptr,
            deleted_mask_ptr,
            hovered_depth_id,
            highlight_gaussian_id,
            selected_node_mask_ptr,
            num_selected_nodes,
            desaturate_unselected,
            selection_flash_intensity,
            orthographic,
            ortho_scale,
            mip_filter);

        arena.end_frame(frame_id, true); // true = from_rendering
        arena.set_rendering_active(false);

        return {std::move(image), std::move(alpha), std::move(depth)};
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
        if (valid_nodes.empty())
            return;

        const int n = static_cast<int>(cumulative_selection.size(0));
        const int num_nodes = static_cast<int>(valid_nodes.size());

        const uint8_t* const existing_ptr = (existing_mask.is_valid() &&
                                             existing_mask.numel() == static_cast<size_t>(n))
                                                ? existing_mask.ptr<uint8_t>()
                                                : nullptr;

        const int* const node_indices_ptr = (transform_indices && transform_indices->is_valid() &&
                                             transform_indices->numel() == static_cast<size_t>(n))
                                                ? transform_indices->ptr<int>()
                                                : nullptr;

        const Tensor valid_nodes_gpu = upload_bool_mask(valid_nodes);
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
        const Tensor* background) {

        constexpr float QUAT_NORM_EPS = 1e-8f;

        check_tensor_input(config::debug, means, "means");
        check_tensor_input(config::debug, scales_raw, "scales_raw");
        check_tensor_input(config::debug, rotations_raw, "rotations_raw");
        check_tensor_input(config::debug, opacities_raw, "opacities_raw");
        check_tensor_input(config::debug, sh0, "sh0");
        check_tensor_input(config::debug, sh_rest, "sh_rest");

        const int N = static_cast<int>(means.size(0));
        const size_t H = static_cast<size_t>(height);
        const size_t W = static_cast<size_t>(width);
        const int num_sh_coeffs = 1 + static_cast<int>(sh_rest.size(1));

        // Output tensors
        Tensor image = Tensor::empty({3, H, W}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        Tensor alpha = Tensor::empty({1, H, W}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        Tensor depth = Tensor::empty({1, H, W}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);

        // Activate parameters
        const Tensor scales = scales_raw.exp();
        const Tensor rotations = rotations_raw / rotations_raw.norm(2, -1, true).clamp_min(QUAT_NORM_EPS);
        const Tensor opacities = opacities_raw.sigmoid().squeeze(-1);

        // Concatenate SH coefficients [N, K, 3]
        const Tensor sh_coeffs = (sh_rest.numel() > 0 && num_sh_coeffs > 1)
                                     ? Tensor::cat({sh0, sh_rest}, 1).contiguous()
                                     : sh0.contiguous();

        // Contiguous copies
        const Tensor means_c = means.contiguous();
        const Tensor scales_c = scales.contiguous();
        const Tensor rotations_c = rotations.contiguous();
        const Tensor opacities_c = opacities.contiguous();
        const Tensor w2c_c = w2c.contiguous();
        const Tensor K_c = K.contiguous();

        const float* const radial_ptr = (radial_coeffs && radial_coeffs->is_valid()) ? radial_coeffs->ptr<float>() : nullptr;
        const float* const tangential_ptr = (tangential_coeffs && tangential_coeffs->is_valid()) ? tangential_coeffs->ptr<float>() : nullptr;
        const float* const bg_ptr = (background && background->is_valid()) ? background->ptr<float>() : nullptr;

        // Render buffers in HWC format (gsplat output format)
        Tensor render_hwc = Tensor::empty({H, W, 3}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        Tensor alpha_hw = Tensor::empty({H, W}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);

        gsplat_forward_gut(
            means_c.ptr<float>(),
            rotations_c.ptr<float>(),
            scales_c.ptr<float>(),
            opacities_c.ptr<float>(),
            sh_coeffs.ptr<float>(),
            static_cast<uint32_t>(sh_degree),
            static_cast<uint32_t>(N),
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
