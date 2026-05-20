/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "selection_ops.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace lfs::rendering {
    namespace {
        constexpr float kInvalidScreenPositionThreshold = -1000.0f;
        constexpr int kBlockSize = 256;

        [[nodiscard]] int checkedToInt(const std::size_t value, const char* const message) {
            if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error(message);
            }
            return static_cast<int>(value);
        }

        struct PreparedModelTransforms {
            Tensor contig;
            const float* ptr = nullptr;
            int count = 0;

            static PreparedModelTransforms from(const Tensor* const model_transforms) {
                PreparedModelTransforms result;
                if (model_transforms == nullptr || !model_transforms->is_valid() ||
                    model_transforms->numel() == 0) {
                    return result;
                }
                result.contig = model_transforms->is_contiguous()
                                    ? *model_transforms
                                    : model_transforms->contiguous();
                if (result.contig.numel() % 16 != 0) {
                    throw std::runtime_error(
                        "model_transforms tensor must contain a multiple of 16 float values (N x 4 x 4).");
                }
                result.ptr = result.contig.ptr<float>();
                result.count = checkedToInt(result.contig.numel() / 16,
                                            "model transform count exceeds int range");
                return result;
            }
        };

        [[nodiscard]] Tensor uploadBoolMask(const std::vector<bool>& mask) {
            auto tensor = Tensor::empty({mask.size()}, lfs::core::Device::CPU, lfs::core::DataType::UInt8);
            auto* const ptr = tensor.ptr<uint8_t>();
            for (std::size_t i = 0; i < mask.size(); ++i) {
                ptr[i] = mask[i] ? 1 : 0;
            }
            return tensor.cuda();
        }

        __device__ __forceinline__ bool pointInPolygon(
            const float px,
            const float py,
            const float2* __restrict__ poly,
            const int n) {
            bool inside = false;
            for (int i = 0, j = n - 1; i < n; j = i++) {
                const float yi = poly[i].y;
                const float yj = poly[j].y;
                if ((yi > py) != (yj > py)) {
                    const float xi = poly[i].x;
                    const float xj = poly[j].x;
                    if (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
                        inside = !inside;
                    }
                }
            }
            return inside;
        }

        __global__ void brushSelectKernel(
            const float2* __restrict__ screen_positions,
            const float mouse_x,
            const float mouse_y,
            const float radius_sq,
            uint8_t* __restrict__ selection_out,
            const int n_primitives) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n_primitives) {
                return;
            }

            const float2 pos = screen_positions[idx];
            if (pos.x < kInvalidScreenPositionThreshold || pos.y < kInvalidScreenPositionThreshold) {
                return;
            }

            const float dx = pos.x - mouse_x;
            const float dy = pos.y - mouse_y;
            if (dx * dx + dy * dy <= radius_sq) {
                selection_out[idx] = 1;
            }
        }

        __global__ void rectSelectKernel(
            const float2* __restrict__ positions,
            const float x0,
            const float y0,
            const float x1,
            const float y1,
            bool* __restrict__ selection,
            const int n) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) {
                return;
            }

            const float2 pos = positions[idx];
            if (pos.x < kInvalidScreenPositionThreshold) {
                return;
            }
            if (pos.x >= x0 && pos.x <= x1 && pos.y >= y0 && pos.y <= y1) {
                selection[idx] = true;
            }
        }

        __global__ void rectSelectModeKernel(
            const float2* __restrict__ positions,
            const float x0,
            const float y0,
            const float x1,
            const float y1,
            bool* __restrict__ selection,
            const int n,
            const bool add_mode) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) {
                return;
            }

            const float2 pos = positions[idx];
            if (pos.x < kInvalidScreenPositionThreshold) {
                return;
            }
            if (pos.x >= x0 && pos.x <= x1 && pos.y >= y0 && pos.y <= y1) {
                selection[idx] = add_mode;
            }
        }

        __global__ void polygonSelectKernel(
            const float2* __restrict__ positions,
            const float2* __restrict__ polygon,
            const int num_verts,
            bool* __restrict__ selection,
            const int n) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) {
                return;
            }

            const float2 pos = positions[idx];
            if (pos.x < kInvalidScreenPositionThreshold) {
                return;
            }
            if (pointInPolygon(pos.x, pos.y, polygon, num_verts)) {
                selection[idx] = true;
            }
        }

        __global__ void polygonSelectModeKernel(
            const float2* __restrict__ positions,
            const float2* __restrict__ polygon,
            const int num_verts,
            bool* __restrict__ selection,
            const int n,
            const bool add_mode) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) {
                return;
            }

            const float2 pos = positions[idx];
            if (pos.x < kInvalidScreenPositionThreshold) {
                return;
            }
            if (pointInPolygon(pos.x, pos.y, polygon, num_verts)) {
                selection[idx] = add_mode;
            }
        }

        __global__ void applySelectionGroupKernel(
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
            if (idx >= n) {
                return;
            }

            const uint8_t existing_group = existing ? existing[idx] : 0;
            const bool selected = cumulative[idx];

            if (node_indices && target_node >= 0 && node_indices[idx] != target_node) {
                output[idx] = existing_group;
                return;
            }

            if (add_mode) {
                if (selected) {
                    const bool is_locked = existing_group != 0 &&
                                           existing_group != group_id &&
                                           locked_groups &&
                                           (locked_groups[existing_group / 32] &
                                            (1u << (existing_group % 32)));
                    output[idx] = is_locked ? existing_group : group_id;
                } else {
                    output[idx] = existing_group;
                }
            } else {
                output[idx] = (selected && existing_group == group_id) ? 0 : existing_group;
            }
        }

        __global__ void applySelectionGroupMaskKernel(
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
            if (idx >= n) {
                return;
            }

            const uint8_t existing_group = existing ? existing[idx] : 0;
            if (node_indices && valid_nodes) {
                const int node_idx = node_indices[idx];
                if (node_idx < 0 || node_idx >= num_nodes || !valid_nodes[node_idx]) {
                    output[idx] = existing_group;
                    return;
                }
            }

            const bool selected = cumulative[idx];
            const bool is_other_locked = existing_group != 0 &&
                                         existing_group != group_id &&
                                         locked_groups &&
                                         (locked_groups[existing_group / 32] &
                                          (1u << (existing_group % 32)));

            if (replace_mode) {
                if (selected) {
                    output[idx] = is_other_locked ? existing_group : group_id;
                } else if (existing_group == group_id) {
                    output[idx] = 0;
                } else {
                    output[idx] = existing_group;
                }
            } else if (add_mode) {
                output[idx] = (selected && !is_other_locked) ? group_id : existing_group;
            } else {
                output[idx] = (selected && existing_group == group_id) ? 0 : existing_group;
            }
        }

        __global__ void filterSelectionByNodeKernel(
            bool* __restrict__ selection,
            const int* __restrict__ node_indices,
            const int n,
            const int target_node) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) {
                return;
            }
            if (node_indices[idx] != target_node) {
                selection[idx] = false;
            }
        }

        __global__ void filterSelectionByNodeMaskKernel(
            bool* __restrict__ selection,
            const int* __restrict__ node_indices,
            const bool* __restrict__ valid_nodes,
            const int n,
            const int num_nodes) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) {
                return;
            }
            const int node_idx = node_indices[idx];
            if (node_idx < 0 || node_idx >= num_nodes || !valid_nodes[node_idx]) {
                selection[idx] = false;
            }
        }

        __global__ void filterSelectionByCropKernel(
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
            if (idx >= n || !selection[idx]) {
                return;
            }

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
                const float norm = (lx * lx) / (r.x * r.x) +
                                   (ly * ly) / (r.y * r.y) +
                                   (lz * lz) / (r.z * r.z);

                if ((norm <= 1.0f) == ellipsoid_inverse) {
                    selection[idx] = false;
                }
            }
        }
    } // namespace

    void brush_select(
        const float2* const screen_positions,
        const float mouse_x,
        const float mouse_y,
        const float radius,
        uint8_t* const selection_out,
        const int n_primitives) {
        if (n_primitives <= 0) {
            return;
        }
        const int grid_size = (n_primitives + kBlockSize - 1) / kBlockSize;
        brushSelectKernel<<<grid_size, kBlockSize>>>(
            screen_positions, mouse_x, mouse_y, radius * radius, selection_out, n_primitives);
    }

    void rect_select(
        const float2* const positions,
        const float x0,
        const float y0,
        const float x1,
        const float y1,
        bool* const selection,
        const int n_primitives) {
        if (n_primitives <= 0) {
            return;
        }
        const int grid_size = (n_primitives + kBlockSize - 1) / kBlockSize;
        rectSelectKernel<<<grid_size, kBlockSize>>>(positions, x0, y0, x1, y1, selection, n_primitives);
    }

    void rect_select_mode(
        const float2* const positions,
        const float x0,
        const float y0,
        const float x1,
        const float y1,
        bool* const selection,
        const int n_primitives,
        const bool add_mode) {
        if (n_primitives <= 0) {
            return;
        }
        const int grid_size = (n_primitives + kBlockSize - 1) / kBlockSize;
        rectSelectModeKernel<<<grid_size, kBlockSize>>>(positions, x0, y0, x1, y1, selection, n_primitives, add_mode);
    }

    void polygon_select(
        const float2* const positions,
        const float2* const polygon,
        const int num_vertices,
        bool* const selection,
        const int n_primitives) {
        if (n_primitives <= 0 || num_vertices < 3) {
            return;
        }
        const int grid_size = (n_primitives + kBlockSize - 1) / kBlockSize;
        polygonSelectKernel<<<grid_size, kBlockSize>>>(positions, polygon, num_vertices, selection, n_primitives);
    }

    void polygon_select_mode(
        const float2* const positions,
        const float2* const polygon,
        const int num_vertices,
        bool* const selection,
        const int n_primitives,
        const bool add_mode) {
        if (n_primitives <= 0 || num_vertices < 3) {
            return;
        }
        const int grid_size = (n_primitives + kBlockSize - 1) / kBlockSize;
        polygonSelectModeKernel<<<grid_size, kBlockSize>>>(positions, polygon, num_vertices, selection, n_primitives, add_mode);
    }

    void set_selection_element(bool* const selection, const int index, const bool value) {
        cudaMemcpy(selection + index, &value, sizeof(bool), cudaMemcpyHostToDevice);
    }

    void brush_select_tensor(
        const Tensor& screen_positions,
        const float mouse_x,
        const float mouse_y,
        const float radius,
        Tensor& selection_out) {
        if (!screen_positions.is_valid() || screen_positions.size(0) == 0) {
            return;
        }
        const int n = checkedToInt(screen_positions.size(0), "n_primitives exceeds int range");
        brush_select(reinterpret_cast<const float2*>(screen_positions.ptr<float>()),
                     mouse_x,
                     mouse_y,
                     radius,
                     selection_out.ptr<uint8_t>(),
                     n);
    }

    void rect_select_tensor(
        const Tensor& screen_positions,
        const float x0,
        const float y0,
        const float x1,
        const float y1,
        Tensor& selection_out) {
        if (!screen_positions.is_valid() || screen_positions.size(0) == 0) {
            return;
        }
        const int n = checkedToInt(screen_positions.size(0), "n_primitives exceeds int range");
        rect_select(reinterpret_cast<const float2*>(screen_positions.ptr<float>()),
                    x0,
                    y0,
                    x1,
                    y1,
                    selection_out.ptr<bool>(),
                    n);
    }

    void rect_select_mode_tensor(
        const Tensor& screen_positions,
        const float x0,
        const float y0,
        const float x1,
        const float y1,
        Tensor& selection_out,
        const bool add_mode) {
        if (!screen_positions.is_valid() || screen_positions.size(0) == 0) {
            return;
        }
        const int n = checkedToInt(screen_positions.size(0), "n_primitives exceeds int range");
        rect_select_mode(reinterpret_cast<const float2*>(screen_positions.ptr<float>()),
                         x0,
                         y0,
                         x1,
                         y1,
                         selection_out.ptr<bool>(),
                         n,
                         add_mode);
    }

    void polygon_select_tensor(
        const Tensor& screen_positions,
        const Tensor& polygon_vertices,
        Tensor& selection_out) {
        if (!screen_positions.is_valid() || screen_positions.size(0) == 0) {
            return;
        }
        if (!polygon_vertices.is_valid() || polygon_vertices.size(0) < 3) {
            return;
        }
        const int num_vertices = checkedToInt(polygon_vertices.size(0), "polygon vertex count exceeds int range");
        const int n = checkedToInt(screen_positions.size(0), "n_primitives exceeds int range");
        polygon_select(reinterpret_cast<const float2*>(screen_positions.ptr<float>()),
                       reinterpret_cast<const float2*>(polygon_vertices.ptr<float>()),
                       num_vertices,
                       selection_out.ptr<bool>(),
                       n);
    }

    void polygon_select_mode_tensor(
        const Tensor& screen_positions,
        const Tensor& polygon_vertices,
        Tensor& selection_out,
        const bool add_mode) {
        if (!screen_positions.is_valid() || screen_positions.size(0) == 0) {
            return;
        }
        if (!polygon_vertices.is_valid() || polygon_vertices.size(0) < 3) {
            return;
        }
        const int num_vertices = checkedToInt(polygon_vertices.size(0), "polygon vertex count exceeds int range");
        const int n = checkedToInt(screen_positions.size(0), "n_primitives exceeds int range");
        polygon_select_mode(reinterpret_cast<const float2*>(screen_positions.ptr<float>()),
                            reinterpret_cast<const float2*>(polygon_vertices.ptr<float>()),
                            num_vertices,
                            selection_out.ptr<bool>(),
                            n,
                            add_mode);
    }

    void apply_selection_group_tensor(
        const Tensor& cumulative_selection,
        const Tensor& existing_mask,
        Tensor& output_mask,
        const uint8_t group_id,
        const uint32_t* const locked_groups,
        const bool add_mode,
        const Tensor* const transform_indices,
        const int target_node_index) {
        if (!cumulative_selection.is_valid() || cumulative_selection.size(0) == 0) {
            return;
        }
        const int n = checkedToInt(cumulative_selection.size(0), "selection size exceeds int range");
        const uint8_t* const existing_ptr =
            (existing_mask.is_valid() && existing_mask.numel() == static_cast<std::size_t>(n))
                ? existing_mask.ptr<uint8_t>()
                : nullptr;
        const int* const node_indices_ptr =
            (transform_indices && transform_indices->is_valid() &&
             transform_indices->numel() == static_cast<std::size_t>(n))
                ? transform_indices->ptr<int>()
                : nullptr;

        const int grid_size = (n + kBlockSize - 1) / kBlockSize;
        applySelectionGroupKernel<<<grid_size, kBlockSize>>>(
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

    void apply_selection_group_tensor_mask(
        const Tensor& cumulative_selection,
        const Tensor& existing_mask,
        Tensor& output_mask,
        const uint8_t group_id,
        const uint32_t* const locked_groups,
        const bool add_mode,
        const Tensor* const transform_indices,
        const std::vector<bool>& valid_nodes,
        const bool replace_mode) {
        if (!cumulative_selection.is_valid() || cumulative_selection.size(0) == 0) {
            return;
        }

        const int n = checkedToInt(cumulative_selection.size(0), "selection size exceeds int range");
        std::vector<bool> default_valid_nodes;
        const std::vector<bool>* effective_valid_nodes = &valid_nodes;
        if (effective_valid_nodes->empty()) {
            default_valid_nodes.push_back(true);
            effective_valid_nodes = &default_valid_nodes;
        }

        const int num_nodes = checkedToInt(effective_valid_nodes->size(), "node count exceeds int range");
        const uint8_t* const existing_ptr =
            (existing_mask.is_valid() && existing_mask.numel() == static_cast<std::size_t>(n))
                ? existing_mask.ptr<uint8_t>()
                : nullptr;
        const int* const node_indices_ptr =
            (transform_indices && transform_indices->is_valid() &&
             transform_indices->numel() == static_cast<std::size_t>(n))
                ? transform_indices->ptr<int>()
                : nullptr;

        const Tensor valid_nodes_gpu = uploadBoolMask(*effective_valid_nodes);
        const int grid_size = (n + kBlockSize - 1) / kBlockSize;
        applySelectionGroupMaskKernel<<<grid_size, kBlockSize>>>(
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

    void filter_selection_by_node(
        Tensor& selection,
        const Tensor& transform_indices,
        const int target_node_index) {
        if (!selection.is_valid() || !transform_indices.is_valid() || target_node_index < 0) {
            return;
        }
        const int n = checkedToInt(selection.size(0), "selection size exceeds int range");
        if (transform_indices.numel() != static_cast<std::size_t>(n)) {
            return;
        }
        const int grid_size = (n + kBlockSize - 1) / kBlockSize;
        filterSelectionByNodeKernel<<<grid_size, kBlockSize>>>(
            selection.ptr<bool>(),
            transform_indices.ptr<int>(),
            n,
            target_node_index);
    }

    void filter_selection_by_node_mask(
        Tensor& selection,
        const Tensor& transform_indices,
        const std::vector<bool>& valid_nodes) {
        if (!selection.is_valid() || !transform_indices.is_valid() || valid_nodes.empty()) {
            return;
        }
        const int n = checkedToInt(selection.size(0), "selection size exceeds int range");
        if (transform_indices.numel() != static_cast<std::size_t>(n)) {
            return;
        }
        const int num_nodes = checkedToInt(valid_nodes.size(), "node count exceeds int range");
        const Tensor valid_nodes_gpu = uploadBoolMask(valid_nodes);
        const int grid_size = (n + kBlockSize - 1) / kBlockSize;
        filterSelectionByNodeMaskKernel<<<grid_size, kBlockSize>>>(
            selection.ptr<bool>(),
            transform_indices.ptr<int>(),
            reinterpret_cast<const bool*>(valid_nodes_gpu.ptr<uint8_t>()),
            n,
            num_nodes);
    }

    void filter_selection_by_crop(
        Tensor& selection,
        const Tensor& means,
        const Tensor* const crop_box_transform,
        const Tensor* const crop_box_min,
        const Tensor* const crop_box_max,
        const bool crop_inverse,
        const Tensor* const ellipsoid_transform,
        const Tensor* const ellipsoid_radii,
        const bool ellipsoid_inverse,
        const Tensor* const model_transforms,
        const Tensor* const transform_indices) {
        if (!selection.is_valid() || !means.is_valid()) {
            return;
        }

        const int n = checkedToInt(selection.size(0), "selection size exceeds int range");
        if (means.size(0) != static_cast<std::size_t>(n)) {
            return;
        }

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

        if (!crop_t_ptr && !ellip_t_ptr) {
            return;
        }

        const auto prepared_transforms = PreparedModelTransforms::from(model_transforms);
        Tensor transform_indices_contig;
        const int* transform_indices_ptr = nullptr;
        if (transform_indices != nullptr && transform_indices->is_valid() &&
            transform_indices->numel() == static_cast<std::size_t>(n)) {
            transform_indices_contig = transform_indices->is_contiguous()
                                           ? *transform_indices
                                           : transform_indices->contiguous();
            transform_indices_ptr = transform_indices_contig.ptr<int>();
        }

        const int grid_size = (n + kBlockSize - 1) / kBlockSize;
        filterSelectionByCropKernel<<<grid_size, kBlockSize>>>(
            selection.ptr<bool>(),
            reinterpret_cast<const float3*>(means.ptr<float>()),
            crop_t_ptr,
            crop_min_ptr,
            crop_max_ptr,
            crop_inverse,
            ellip_t_ptr,
            ellip_radii_ptr,
            ellipsoid_inverse,
            prepared_transforms.ptr,
            transform_indices_ptr,
            prepared_transforms.count,
            n);
    }

    namespace config {
        void setSelectionGroupColor(int, float3) {}
        void setSelectionPreviewColor(float3) {}
    } // namespace config

} // namespace lfs::rendering
