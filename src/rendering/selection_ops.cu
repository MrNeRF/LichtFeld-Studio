/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "selection_ops.hpp"

#include "core/cuda_error.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace lfs::rendering {
    namespace {
        constexpr float kInvalidScreenPositionThreshold = -1000.0f;
        constexpr int kBlockSize = 256;
        constexpr int kCountMaxBlocks = 4096;
        constexpr int kSelectionGroupCount = 256;
        constexpr std::size_t kSelectionGroupCountBins = kSelectionGroupCount;
        constexpr std::size_t kSelectionChangedCountIndex = kSelectionGroupCountBins;
        constexpr std::size_t kSelectionGroupScratchWords = kSelectionGroupCountBins + 1;

        [[nodiscard]] int checkedToInt(const std::size_t value, const char* const message) {
            if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error(message);
            }
            return static_cast<int>(value);
        }

        [[nodiscard]] cudaStream_t currentSelectionStream(const Tensor* const tensor = nullptr) {
            if (const cudaStream_t stream = lfs::core::getCurrentCUDAStream()) {
                return stream;
            }
            return tensor != nullptr && tensor->is_valid() ? tensor->stream() : nullptr;
        }

        template <std::size_t N>
        void copySelectionCountsToHost(const Tensor& counts_scratch,
                                       std::array<int, N>& host_counts) {
            const cudaStream_t stream = currentSelectionStream(&counts_scratch);
            if (const cudaError_t status = cudaMemcpyAsync(host_counts.data(),
                                                           counts_scratch.ptr<int>(),
                                                           sizeof(host_counts),
                                                           cudaMemcpyDeviceToHost,
                                                           stream);
                status != cudaSuccess) {
                throw std::runtime_error(cudaGetErrorString(status));
            }
            if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
                throw std::runtime_error(cudaGetErrorString(status));
            }
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

        [[nodiscard]] bool nodeMaskRestrictsSelection(const std::vector<bool>& mask) {
            return std::any_of(mask.begin(), mask.end(), [](const bool enabled) { return !enabled; });
        }

        void prepareSelectionGroupCountsScratch(Tensor& counts_scratch) {
            if (!counts_scratch.is_valid() ||
                counts_scratch.device() != lfs::core::Device::CUDA ||
                counts_scratch.dtype() != lfs::core::DataType::Int32 ||
                counts_scratch.numel() != kSelectionGroupScratchWords) {
                counts_scratch = Tensor::zeros(
                    {kSelectionGroupScratchWords}, lfs::core::Device::CUDA, lfs::core::DataType::Int32);
            } else {
                counts_scratch.zero_();
            }
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

        __global__ void setSelectionElementKernel(
            bool* __restrict__ selection,
            const int index,
            const bool value) {
            if (threadIdx.x == 0 && blockIdx.x == 0) {
                selection[index] = value;
            }
        }

        __device__ __forceinline__ void writeInvalidScreenPosition(float2* const output, const int idx) {
            output[idx] = make_float2(kInvalidScreenPositionThreshold * 100000.0f,
                                      kInvalidScreenPositionThreshold * 100000.0f);
        }

        __global__ void projectScreenPositionsKernel(
            const float3* __restrict__ means,
            float2* __restrict__ output,
            const int n,
            const int width,
            const int height,
            const float3 view_row0,
            const float3 view_row1,
            const float3 view_row2,
            const float3 translation,
            const float pixel_focal_x,
            const float pixel_focal_y,
            const bool orthographic,
            const float ortho_scale,
            const float* __restrict__ model_transforms,
            const int* __restrict__ transform_indices,
            const int num_model_transforms,
            const uint8_t* __restrict__ node_visibility,
            const int node_visibility_count) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) {
                return;
            }

            int transform_idx = transform_indices != nullptr ? transform_indices[idx] : 0;
            if (node_visibility != nullptr && transform_indices != nullptr) {
                if (transform_idx < 0 ||
                    transform_idx >= node_visibility_count ||
                    node_visibility[transform_idx] == 0) {
                    writeInvalidScreenPosition(output, idx);
                    return;
                }
            }

            float3 pos = means[idx];
            if (model_transforms != nullptr && num_model_transforms > 0) {
                transform_idx = min(max(transform_idx, 0), num_model_transforms - 1);
                const float* const m = model_transforms + transform_idx * 16;
                pos = make_float3(
                    m[0] * pos.x + m[1] * pos.y + m[2] * pos.z + m[3],
                    m[4] * pos.x + m[5] * pos.y + m[6] * pos.z + m[7],
                    m[8] * pos.x + m[9] * pos.y + m[10] * pos.z + m[11]);
            }

            const float dx = pos.x - translation.x;
            const float dy = pos.y - translation.y;
            const float dz = pos.z - translation.z;
            const float view_x = view_row0.x * dx + view_row0.y * dy + view_row0.z * dz;
            const float view_y = view_row1.x * dx + view_row1.y * dy + view_row1.z * dz;
            const float view_z = view_row2.x * dx + view_row2.y * dy + view_row2.z * dz;
            if (!isfinite(view_x) || !isfinite(view_y) || !isfinite(view_z) || view_z >= -1.0e-6f) {
                writeInvalidScreenPosition(output, idx);
                return;
            }

            const float cx = static_cast<float>(width) * 0.5f;
            const float cy = static_cast<float>(height) * 0.5f;
            if (orthographic) {
                if (!isfinite(ortho_scale) || ortho_scale <= 0.0f) {
                    writeInvalidScreenPosition(output, idx);
                    return;
                }
                output[idx] = make_float2(cx + view_x * ortho_scale, cy - view_y * ortho_scale);
                return;
            }

            const float depth = -view_z;
            output[idx] = make_float2(
                cx + view_x * pixel_focal_x / depth,
                cy - view_y * pixel_focal_y / depth);
        }

        __device__ __forceinline__ bool betterPickCandidate(
            const float dist_sq,
            const int index,
            const float best_dist_sq,
            const int best_index) {
            return index >= 0 &&
                   (best_index < 0 ||
                    dist_sq < best_dist_sq ||
                    (dist_sq == best_dist_sq && index > best_index));
        }

        __global__ void pickProjectedGaussianBlocksKernel(
            const float2* __restrict__ positions,
            const float x,
            const float y,
            const float max_dist_sq,
            float* __restrict__ block_dist_sq,
            int* __restrict__ block_index,
            const int n) {
            __shared__ float shared_dist[kBlockSize];
            __shared__ int shared_index[kBlockSize];

            float best_dist_sq = max_dist_sq;
            int best_index = -1;
            for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
                 idx < n;
                 idx += blockDim.x * gridDim.x) {
                const float2 pos = positions[idx];
                if (pos.x < kInvalidScreenPositionThreshold ||
                    pos.y < kInvalidScreenPositionThreshold ||
                    !isfinite(pos.x) ||
                    !isfinite(pos.y)) {
                    continue;
                }

                const float dx = pos.x - x;
                const float dy = pos.y - y;
                const float dist_sq = dx * dx + dy * dy;
                if (dist_sq <= max_dist_sq &&
                    betterPickCandidate(dist_sq, idx, best_dist_sq, best_index)) {
                    best_dist_sq = dist_sq;
                    best_index = idx;
                }
            }

            shared_dist[threadIdx.x] = best_dist_sq;
            shared_index[threadIdx.x] = best_index;
            __syncthreads();

            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) {
                    const float other_dist = shared_dist[threadIdx.x + stride];
                    const int other_index = shared_index[threadIdx.x + stride];
                    if (betterPickCandidate(
                            other_dist, other_index, shared_dist[threadIdx.x], shared_index[threadIdx.x])) {
                        shared_dist[threadIdx.x] = other_dist;
                        shared_index[threadIdx.x] = other_index;
                    }
                }
                __syncthreads();
            }

            if (threadIdx.x == 0) {
                block_dist_sq[blockIdx.x] = shared_dist[0];
                block_index[blockIdx.x] = shared_index[0];
            }
        }

        __global__ void reduceProjectedGaussianPickKernel(
            const float* __restrict__ block_dist_sq,
            const int* __restrict__ block_index,
            int* __restrict__ result_index,
            const int block_count) {
            __shared__ float shared_dist[kBlockSize];
            __shared__ int shared_index[kBlockSize];

            float best_dist_sq = 0.0f;
            int best_index = -1;
            for (int idx = threadIdx.x; idx < block_count; idx += blockDim.x) {
                const int candidate = block_index[idx];
                const float dist_sq = block_dist_sq[idx];
                if (betterPickCandidate(dist_sq, candidate, best_dist_sq, best_index)) {
                    best_dist_sq = dist_sq;
                    best_index = candidate;
                }
            }

            shared_dist[threadIdx.x] = best_dist_sq;
            shared_index[threadIdx.x] = best_index;
            __syncthreads();

            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) {
                    const float other_dist = shared_dist[threadIdx.x + stride];
                    const int other_index = shared_index[threadIdx.x + stride];
                    if (betterPickCandidate(
                            other_dist, other_index, shared_dist[threadIdx.x], shared_index[threadIdx.x])) {
                        shared_dist[threadIdx.x] = other_dist;
                        shared_index[threadIdx.x] = other_index;
                    }
                }
                __syncthreads();
            }

            if (threadIdx.x == 0) {
                result_index[0] = shared_index[0];
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
            if (pos.x < kInvalidScreenPositionThreshold || pos.y < kInvalidScreenPositionThreshold) {
                return;
            }
            if (pos.x >= x0 && pos.x <= x1 && pos.y >= y0 && pos.y <= y1) {
                selection[idx] = true;
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
            if (pos.x < kInvalidScreenPositionThreshold || pos.y < kInvalidScreenPositionThreshold) {
                return;
            }
            if (pointInPolygon(pos.x, pos.y, polygon, num_verts)) {
                selection[idx] = true;
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
            const bool replace_mode,
            int* __restrict__ group_deltas) {
            __shared__ int local_deltas[kSelectionGroupCount];
            __shared__ int local_changed_count;
            if (group_deltas) {
                for (int bin = threadIdx.x; bin < kSelectionGroupCount; bin += blockDim.x) {
                    local_deltas[bin] = 0;
                }
            }
            if (group_deltas && threadIdx.x == 0) {
                local_changed_count = 0;
            }
            if (group_deltas) {
                __syncthreads();
            }

            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            uint8_t output_group = 0;

            if (idx < n) {
                const uint8_t existing_group = existing ? existing[idx] : 0;
                output_group = existing_group;

                bool node_valid = true;
                if (node_indices && valid_nodes) {
                    const int node_idx = node_indices[idx];
                    node_valid = node_idx >= 0 && node_idx < num_nodes && valid_nodes[node_idx];
                }

                if (node_valid) {
                    const bool selected = cumulative[idx];
                    const bool is_other_locked = existing_group != 0 &&
                                                 existing_group != group_id &&
                                                 locked_groups &&
                                                 (locked_groups[existing_group / 32] &
                                                  (1u << (existing_group % 32)));

                    if (replace_mode) {
                        if (selected) {
                            output_group = is_other_locked ? existing_group : group_id;
                        } else if (existing_group == group_id) {
                            output_group = 0;
                        }
                    } else if (add_mode) {
                        output_group = (selected && !is_other_locked) ? group_id : existing_group;
                    } else {
                        output_group = (selected && existing_group == group_id) ? 0 : existing_group;
                    }
                }

                output[idx] = output_group;
                if (group_deltas && output_group != existing_group) {
                    if (existing_group != 0) {
                        atomicAdd(&local_deltas[existing_group], -1);
                    }
                    if (output_group != 0) {
                        atomicAdd(&local_deltas[output_group], 1);
                    }
                    atomicAdd(&local_changed_count, 1);
                }
            }

            if (group_deltas) {
                __syncthreads();
                for (int bin = threadIdx.x; bin < kSelectionGroupCount; bin += blockDim.x) {
                    const int delta = local_deltas[bin];
                    if (delta != 0) {
                        atomicAdd(&group_deltas[bin], delta);
                    }
                }
                if (threadIdx.x == 0 && local_changed_count != 0) {
                    atomicAdd(&group_deltas[kSelectionChangedCountIndex], local_changed_count);
                }
            }
        }

        __global__ void applySelectionGroupIndexedMaskKernel(
            const bool* __restrict__ cumulative,
            const int* __restrict__ visible_indices,
            const uint8_t* __restrict__ existing,
            uint8_t* __restrict__ output,
            const int visible_n,
            const int output_n,
            const uint8_t group_id,
            const uint32_t* __restrict__ locked_groups,
            const bool add_mode,
            const int* __restrict__ node_indices,
            const bool* __restrict__ valid_nodes,
            const int num_nodes,
            const bool replace_mode) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= visible_n) {
                return;
            }

            const int output_idx = visible_indices[idx];
            if (output_idx < 0 || output_idx >= output_n) {
                return;
            }

            const uint8_t existing_group = existing ? existing[output_idx] : 0;
            if (node_indices && valid_nodes) {
                const int node_idx = node_indices[idx];
                if (node_idx < 0 || node_idx >= num_nodes || !valid_nodes[node_idx]) {
                    output[output_idx] = existing_group;
                    return;
                }
            }

            const bool selected = cumulative[idx];
            if (!selected) {
                return;
            }
            const bool is_other_locked = existing_group != 0 &&
                                         existing_group != group_id &&
                                         locked_groups &&
                                         (locked_groups[existing_group / 32] &
                                          (1u << (existing_group % 32)));

            if (replace_mode) {
                output[output_idx] = is_other_locked ? existing_group : group_id;
            } else if (add_mode) {
                if (!is_other_locked) {
                    output[output_idx] = group_id;
                }
            } else {
                if (existing_group == group_id) {
                    output[output_idx] = 0;
                }
            }
        }

        __global__ void clearSelectionGroupIndexedMaskKernel(
            const bool* __restrict__ cumulative,
            const int* __restrict__ visible_indices,
            const uint8_t* __restrict__ existing,
            uint8_t* __restrict__ output,
            const int visible_n,
            const int output_n,
            const uint8_t group_id,
            const int* __restrict__ node_indices,
            const bool* __restrict__ valid_nodes,
            const int num_nodes) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= visible_n || cumulative[idx]) {
                return;
            }

            const int output_idx = visible_indices[idx];
            if (output_idx < 0 || output_idx >= output_n) {
                return;
            }

            const uint8_t existing_group = existing ? existing[output_idx] : 0;
            if (existing_group != group_id) {
                return;
            }

            if (node_indices && valid_nodes) {
                const int node_idx = node_indices[idx];
                if (node_idx < 0 || node_idx >= num_nodes || !valid_nodes[node_idx]) {
                    return;
                }
            }

            output[output_idx] = 0;
        }

        __global__ void countSelectionGroupsKernel(
            const uint8_t* __restrict__ mask,
            const int n,
            int* __restrict__ counts) {
            __shared__ int local_counts[kSelectionGroupCount];
            for (int bin = threadIdx.x; bin < kSelectionGroupCount; bin += blockDim.x) {
                local_counts[bin] = 0;
            }
            __syncthreads();

            const int stride = blockDim.x * gridDim.x;
            for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
                const uint8_t group = mask[idx];
                if (group != 0) {
                    atomicAdd(&local_counts[group], 1);
                }
            }
            __syncthreads();

            for (int bin = threadIdx.x; bin < kSelectionGroupCount; bin += blockDim.x) {
                const int count = local_counts[bin];
                if (count != 0) {
                    atomicAdd(&counts[bin], count);
                }
            }
        }

        __global__ void mergeSelectionMaskOrKernel(
            bool* __restrict__ accumulated,
            const bool* __restrict__ delta,
            const int n) {
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                accumulated[idx] = accumulated[idx] || delta[idx];
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
        brushSelectKernel<<<grid_size, kBlockSize, 0, currentSelectionStream()>>>(
            screen_positions, mouse_x, mouse_y, radius * radius, selection_out, n_primitives);
        LFS_CUDA_LAUNCH_CHECK(currentSelectionStream(), "render.selection.brush");
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
        rectSelectKernel<<<grid_size, kBlockSize, 0, currentSelectionStream()>>>(
            positions, x0, y0, x1, y1, selection, n_primitives);
        LFS_CUDA_LAUNCH_CHECK(currentSelectionStream(), "render.selection.rect");
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
        polygonSelectKernel<<<grid_size, kBlockSize, 0, currentSelectionStream()>>>(
            positions, polygon, num_vertices, selection, n_primitives);
        LFS_CUDA_LAUNCH_CHECK(currentSelectionStream(), "render.selection.polygon");
    }

    void set_selection_element(bool* const selection, const int index, const bool value) {
        if (selection == nullptr || index < 0) {
            return;
        }
        setSelectionElementKernel<<<1, 1, 0, currentSelectionStream()>>>(selection, index, value);
        LFS_CUDA_LAUNCH_CHECK(currentSelectionStream(), "render.selection.set_element");
    }

    Tensor project_screen_positions_tensor(
        const Tensor& means,
        const int width,
        const int height,
        const std::array<float, 9>& view_rotation_rows,
        const std::array<float, 3>& translation,
        const float pixel_focal_x,
        const float pixel_focal_y,
        const bool orthographic,
        const float ortho_scale) {
        return project_screen_positions_tensor(
            means, width, height, view_rotation_rows, translation,
            pixel_focal_x, pixel_focal_y, orthographic, ortho_scale,
            nullptr, nullptr, {});
    }

    Tensor project_screen_positions_tensor(
        const Tensor& means,
        const int width,
        const int height,
        const std::array<float, 9>& view_rotation_rows,
        const std::array<float, 3>& translation,
        const float pixel_focal_x,
        const float pixel_focal_y,
        const bool orthographic,
        const float ortho_scale,
        const Tensor* const model_transforms) {
        return project_screen_positions_tensor(
            means, width, height, view_rotation_rows, translation,
            pixel_focal_x, pixel_focal_y, orthographic, ortho_scale,
            model_transforms, nullptr, {});
    }

    Tensor project_screen_positions_tensor(
        const Tensor& means,
        const int width,
        const int height,
        const std::array<float, 9>& view_rotation_rows,
        const std::array<float, 3>& translation,
        const float pixel_focal_x,
        const float pixel_focal_y,
        const bool orthographic,
        const float ortho_scale,
        const Tensor* const model_transforms,
        const Tensor* const transform_indices) {
        return project_screen_positions_tensor(
            means, width, height, view_rotation_rows, translation,
            pixel_focal_x, pixel_focal_y, orthographic, ortho_scale,
            model_transforms, transform_indices, {});
    }

    Tensor project_screen_positions_tensor(
        const Tensor& means,
        const int width,
        const int height,
        const std::array<float, 9>& view_rotation_rows,
        const std::array<float, 3>& translation,
        const float pixel_focal_x,
        const float pixel_focal_y,
        const bool orthographic,
        const float ortho_scale,
        const Tensor* const model_transforms,
        const Tensor* const transform_indices,
        const std::vector<bool>& node_visibility_mask) {
        if (!means.is_valid() || means.size(0) == 0) {
            return {};
        }
        if (means.device() != lfs::core::Device::CUDA ||
            means.dtype() != lfs::core::DataType::Float32 ||
            means.ndim() != 2 ||
            means.size(1) != 3) {
            throw std::runtime_error("project_screen_positions_tensor expects a CUDA Float32 [N, 3] means tensor");
        }
        if (width <= 0 || height <= 0) {
            return {};
        }

        const int n = checkedToInt(means.size(0), "screen position count exceeds int range");
        const Tensor means_contig = means.is_contiguous() ? means : means.contiguous();
        Tensor output = Tensor::empty(
            {static_cast<std::size_t>(n), std::size_t{2}},
            lfs::core::Device::CUDA,
            lfs::core::DataType::Float32);

        Tensor model_transforms_contig;
        const Tensor* prepared_model_transforms = model_transforms;
        if (model_transforms && model_transforms->is_valid() && model_transforms->numel() > 0) {
            model_transforms_contig = *model_transforms;
            if (model_transforms_contig.dtype() != lfs::core::DataType::Float32) {
                model_transforms_contig = model_transforms_contig.to(lfs::core::DataType::Float32);
            }
            if (model_transforms_contig.device() != lfs::core::Device::CUDA) {
                model_transforms_contig = model_transforms_contig.cuda();
            }
            if (!model_transforms_contig.is_contiguous()) {
                model_transforms_contig = model_transforms_contig.contiguous();
            }
            prepared_model_transforms = &model_transforms_contig;
        }
        const auto prepared_transforms = PreparedModelTransforms::from(prepared_model_transforms);

        Tensor transform_indices_contig;
        const int* transform_indices_ptr = nullptr;
        if (transform_indices && transform_indices->is_valid() &&
            transform_indices->numel() >= static_cast<std::size_t>(n)) {
            transform_indices_contig = *transform_indices;
            if (transform_indices_contig.dtype() != lfs::core::DataType::Int32) {
                transform_indices_contig = transform_indices_contig.to(lfs::core::DataType::Int32);
            }
            if (transform_indices_contig.device() != lfs::core::Device::CUDA) {
                transform_indices_contig = transform_indices_contig.cuda();
            }
            if (!transform_indices_contig.is_contiguous()) {
                transform_indices_contig = transform_indices_contig.contiguous();
            }
            transform_indices_ptr = transform_indices_contig.ptr<int>();
        }

        Tensor visibility_cuda;
        const uint8_t* visibility_ptr = nullptr;
        int visibility_count = 0;
        if (transform_indices_ptr != nullptr && !node_visibility_mask.empty()) {
            visibility_cuda = uploadBoolMask(node_visibility_mask);
            visibility_ptr = visibility_cuda.ptr<uint8_t>();
            visibility_count = checkedToInt(node_visibility_mask.size(), "node visibility count exceeds int range");
        }

        const int grid_size = std::min((n + kBlockSize - 1) / kBlockSize, kCountMaxBlocks);
        const cudaStream_t stream = currentSelectionStream(&output);
        projectScreenPositionsKernel<<<grid_size, kBlockSize, 0, stream>>>(
            reinterpret_cast<const float3*>(means_contig.ptr<float>()),
            reinterpret_cast<float2*>(output.ptr<float>()),
            n,
            width,
            height,
            make_float3(view_rotation_rows[0], view_rotation_rows[1], view_rotation_rows[2]),
            make_float3(view_rotation_rows[3], view_rotation_rows[4], view_rotation_rows[5]),
            make_float3(view_rotation_rows[6], view_rotation_rows[7], view_rotation_rows[8]),
            make_float3(translation[0], translation[1], translation[2]),
            pixel_focal_x,
            pixel_focal_y,
            orthographic,
            ortho_scale,
            prepared_transforms.ptr,
            transform_indices_ptr,
            prepared_transforms.count,
            visibility_ptr,
            visibility_count);
        LFS_CUDA_LAUNCH_CHECK(stream, "render.selection.project_screen");
        if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
            throw std::runtime_error(std::string("projectScreenPositionsKernel: ") + cudaGetErrorString(status));
        }

        return output;
    }

    int pick_projected_gaussian_tensor(
        const Tensor& screen_positions,
        const float x,
        const float y,
        const float radius) {
        if (!screen_positions.is_valid() || screen_positions.size(0) == 0) {
            return -1;
        }
        if (screen_positions.device() != lfs::core::Device::CUDA ||
            screen_positions.dtype() != lfs::core::DataType::Float32 ||
            screen_positions.ndim() != 2 ||
            screen_positions.size(1) != 2) {
            throw std::runtime_error("pick_projected_gaussian_tensor expects a CUDA Float32 [N, 2] tensor");
        }

        const int n = checkedToInt(screen_positions.size(0), "n_primitives exceeds int range");
        const int block_count = std::min((n + kBlockSize - 1) / kBlockSize, kCountMaxBlocks);
        Tensor block_dist_sq = Tensor::empty(
            {static_cast<std::size_t>(block_count)}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
        Tensor block_index = Tensor::empty(
            {static_cast<std::size_t>(block_count)}, lfs::core::Device::CUDA, lfs::core::DataType::Int32);
        Tensor result_index = Tensor::empty({1}, lfs::core::Device::CUDA, lfs::core::DataType::Int32);

        const cudaStream_t stream = currentSelectionStream(&screen_positions);
        pickProjectedGaussianBlocksKernel<<<block_count, kBlockSize, 0, stream>>>(
            reinterpret_cast<const float2*>(screen_positions.ptr<float>()),
            x,
            y,
            radius * radius,
            block_dist_sq.ptr<float>(),
            block_index.ptr<int>(),
            n);
        LFS_CUDA_LAUNCH_CHECK(stream, "render.selection.pick_blocks");
        reduceProjectedGaussianPickKernel<<<1, kBlockSize, 0, stream>>>(
            block_dist_sq.ptr<float>(),
            block_index.ptr<int>(),
            result_index.ptr<int>(),
            block_count);
        LFS_CUDA_LAUNCH_CHECK(stream, "render.selection.pick_reduce");

        const auto result_cpu = result_index.cpu().contiguous();
        return result_cpu.ptr<int>()[0];
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
                     reinterpret_cast<uint8_t*>(selection_out.ptr<bool>()),
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

    void apply_selection_group_tensor_mask(
        const Tensor& cumulative_selection,
        const Tensor& existing_mask,
        Tensor& output_mask,
        const uint8_t group_id,
        const uint32_t* const locked_groups,
        const bool add_mode,
        const Tensor* const transform_indices,
        const std::vector<bool>& valid_nodes,
        const bool replace_mode,
        Tensor* const group_counts_scratch) {
        if (!cumulative_selection.is_valid() || cumulative_selection.size(0) == 0) {
            return;
        }

        const int n = checkedToInt(cumulative_selection.size(0), "selection size exceeds int range");
        const bool restricts_nodes = nodeMaskRestrictsSelection(valid_nodes);
        const int num_nodes = restricts_nodes
                                  ? checkedToInt(valid_nodes.size(), "node count exceeds int range")
                                  : 0;
        const uint8_t* const existing_ptr =
            (existing_mask.is_valid() && existing_mask.numel() == static_cast<std::size_t>(n))
                ? existing_mask.ptr<uint8_t>()
                : nullptr;
        const int* const node_indices_ptr =
            (restricts_nodes &&
             transform_indices && transform_indices->is_valid() &&
             transform_indices->numel() == static_cast<std::size_t>(n))
                ? transform_indices->ptr<int>()
                : nullptr;
        const Tensor valid_nodes_gpu = node_indices_ptr ? uploadBoolMask(valid_nodes) : Tensor{};
        const bool* const valid_nodes_ptr = valid_nodes_gpu.is_valid()
                                                ? reinterpret_cast<const bool*>(valid_nodes_gpu.ptr<uint8_t>())
                                                : nullptr;
        if (group_counts_scratch != nullptr) {
            prepareSelectionGroupCountsScratch(*group_counts_scratch);
        }

        const int grid_size = (n + kBlockSize - 1) / kBlockSize;
        applySelectionGroupMaskKernel<<<grid_size, kBlockSize, 0, currentSelectionStream(&output_mask)>>>(
            cumulative_selection.ptr<bool>(),
            existing_ptr,
            output_mask.ptr<uint8_t>(),
            n,
            group_id,
            locked_groups,
            add_mode,
            node_indices_ptr,
            valid_nodes_ptr,
            num_nodes,
            replace_mode,
            group_counts_scratch != nullptr ? group_counts_scratch->ptr<int>() : nullptr);
        LFS_CUDA_LAUNCH_CHECK(currentSelectionStream(&output_mask), "render.selection.apply_group_mask");
    }

    void apply_selection_group_indexed_tensor_mask(
        const Tensor& visible_selection,
        const Tensor& visible_indices,
        const Tensor& existing_mask,
        Tensor& output_mask,
        const uint8_t group_id,
        const uint32_t* const locked_groups,
        const bool add_mode,
        const Tensor* const transform_indices,
        const std::vector<bool>& valid_nodes,
        const bool replace_mode) {
        if (!visible_selection.is_valid() || !visible_indices.is_valid() ||
            !output_mask.is_valid() || visible_selection.size(0) == 0) {
            return;
        }

        const int visible_n = checkedToInt(visible_selection.size(0), "selection size exceeds int range");
        const int output_n = checkedToInt(output_mask.size(0), "selection output size exceeds int range");
        if (visible_indices.numel() != static_cast<std::size_t>(visible_n)) {
            return;
        }

        if (existing_mask.is_valid() && existing_mask.numel() == static_cast<std::size_t>(output_n)) {
            output_mask.copy_from(existing_mask);
        } else {
            output_mask.zero_();
        }

        const bool restricts_nodes = nodeMaskRestrictsSelection(valid_nodes);
        const int num_nodes = restricts_nodes
                                  ? checkedToInt(valid_nodes.size(), "node count exceeds int range")
                                  : 0;
        const int* const node_indices_ptr =
            (restricts_nodes &&
             transform_indices && transform_indices->is_valid() &&
             transform_indices->numel() == static_cast<std::size_t>(visible_n))
                ? transform_indices->ptr<int>()
                : nullptr;
        const Tensor valid_nodes_gpu = node_indices_ptr ? uploadBoolMask(valid_nodes) : Tensor{};
        const bool* const valid_nodes_ptr = valid_nodes_gpu.is_valid()
                                                ? reinterpret_cast<const bool*>(valid_nodes_gpu.ptr<uint8_t>())
                                                : nullptr;
        const uint8_t* const existing_ptr =
            (existing_mask.is_valid() && existing_mask.numel() == static_cast<std::size_t>(output_n))
                ? existing_mask.ptr<uint8_t>()
                : nullptr;

        const int grid_size = (visible_n + kBlockSize - 1) / kBlockSize;
        const cudaStream_t stream = currentSelectionStream(&output_mask);
        if (replace_mode) {
            clearSelectionGroupIndexedMaskKernel<<<grid_size, kBlockSize, 0, stream>>>(
                visible_selection.ptr<bool>(),
                visible_indices.ptr<int>(),
                existing_ptr,
                output_mask.ptr<uint8_t>(),
                visible_n,
                output_n,
                group_id,
                node_indices_ptr,
                valid_nodes_ptr,
                num_nodes);
            LFS_CUDA_LAUNCH_CHECK(stream, "render.selection.clear_group_indexed_mask");
        }
        applySelectionGroupIndexedMaskKernel<<<grid_size, kBlockSize, 0, stream>>>(
            visible_selection.ptr<bool>(),
            visible_indices.ptr<int>(),
            existing_ptr,
            output_mask.ptr<uint8_t>(),
            visible_n,
            output_n,
            group_id,
            locked_groups,
            add_mode,
            node_indices_ptr,
            valid_nodes_ptr,
            num_nodes,
            replace_mode);
        LFS_CUDA_LAUNCH_CHECK(stream, "render.selection.apply_group_indexed_mask");
    }

    std::array<size_t, 256> count_selection_groups(
        const Tensor& selection_mask,
        Tensor& counts_scratch) {
        std::array<size_t, 256> result{};
        if (!selection_mask.is_valid() || selection_mask.numel() == 0) {
            return result;
        }
        if (selection_mask.device() != lfs::core::Device::CUDA) {
            const auto mask_cpu = selection_mask.cpu();
            const auto* const data = mask_cpu.ptr<uint8_t>();
            const size_t n = mask_cpu.numel();
            for (size_t i = 0; i < n; ++i) {
                const uint8_t group = data[i];
                if (group != 0) {
                    ++result[group];
                }
            }
            return result;
        }

        prepareSelectionGroupCountsScratch(counts_scratch);

        const int n = checkedToInt(selection_mask.numel(), "selection mask size exceeds int range");
        const int grid_size = std::min((n + kBlockSize - 1) / kBlockSize, kCountMaxBlocks);
        countSelectionGroupsKernel<<<grid_size, kBlockSize, 0, currentSelectionStream(&counts_scratch)>>>(
            selection_mask.ptr<uint8_t>(),
            n,
            counts_scratch.ptr<int>());
        LFS_CUDA_LAUNCH_CHECK(currentSelectionStream(&counts_scratch), "render.selection.count_groups");

        return read_selection_group_counts(counts_scratch);
    }

    SelectionGroupCountResult read_selection_group_count_result(const Tensor& counts_scratch) {
        SelectionGroupCountResult result{};
        if (!counts_scratch.is_valid() ||
            counts_scratch.device() != lfs::core::Device::CUDA ||
            counts_scratch.dtype() != lfs::core::DataType::Int32 ||
            counts_scratch.numel() != kSelectionGroupScratchWords) {
            return result;
        }

        std::array<int, kSelectionGroupScratchWords> host_counts{};
        copySelectionCountsToHost(counts_scratch, host_counts);
        for (size_t i = 0; i < result.group_counts.size(); ++i) {
            result.group_counts[i] = static_cast<size_t>(std::max(host_counts[i], 0));
        }
        result.changed_count = static_cast<size_t>(std::max(host_counts[kSelectionChangedCountIndex], 0));
        return result;
    }

    SelectionGroupDeltaResult read_selection_group_delta_result(const Tensor& counts_scratch) {
        SelectionGroupDeltaResult result{};
        if (!counts_scratch.is_valid() ||
            counts_scratch.device() != lfs::core::Device::CUDA ||
            counts_scratch.dtype() != lfs::core::DataType::Int32 ||
            counts_scratch.numel() != kSelectionGroupScratchWords) {
            return result;
        }

        std::array<int, kSelectionGroupScratchWords> host_counts{};
        copySelectionCountsToHost(counts_scratch, host_counts);
        for (size_t i = 0; i < result.group_deltas.size(); ++i) {
            result.group_deltas[i] = host_counts[i];
        }
        result.changed_count = static_cast<size_t>(std::max(host_counts[kSelectionChangedCountIndex], 0));
        return result;
    }

    std::array<size_t, 256> read_selection_group_counts(const Tensor& counts_scratch) {
        return read_selection_group_count_result(counts_scratch).group_counts;
    }

    void merge_selection_mask_or(Tensor& accumulated_mask, const Tensor& delta_mask) {
        if (!accumulated_mask.is_valid() || !delta_mask.is_valid() ||
            accumulated_mask.numel() == 0 ||
            accumulated_mask.numel() != delta_mask.numel()) {
            return;
        }
        if (accumulated_mask.device() != lfs::core::Device::CUDA ||
            delta_mask.device() != lfs::core::Device::CUDA ||
            accumulated_mask.dtype() != lfs::core::DataType::Bool ||
            delta_mask.dtype() != lfs::core::DataType::Bool) {
            accumulated_mask = accumulated_mask | delta_mask;
            return;
        }

        const int n = checkedToInt(accumulated_mask.numel(), "selection mask size exceeds int range");
        const int grid_size = (n + kBlockSize - 1) / kBlockSize;
        mergeSelectionMaskOrKernel<<<grid_size, kBlockSize, 0, currentSelectionStream(&accumulated_mask)>>>(
            accumulated_mask.ptr<bool>(),
            delta_mask.ptr<bool>(),
            n);
        LFS_CUDA_LAUNCH_CHECK(currentSelectionStream(&accumulated_mask), "render.selection.merge_mask_or");
    }

    void filter_selection_by_node_mask(
        Tensor& selection,
        const Tensor& transform_indices,
        const std::vector<bool>& valid_nodes) {
        if (!selection.is_valid() || !transform_indices.is_valid() || valid_nodes.empty()) {
            return;
        }
        if (!nodeMaskRestrictsSelection(valid_nodes)) {
            return;
        }
        const int n = checkedToInt(selection.size(0), "selection size exceeds int range");
        if (transform_indices.numel() != static_cast<std::size_t>(n)) {
            return;
        }
        const int num_nodes = checkedToInt(valid_nodes.size(), "node count exceeds int range");
        const Tensor valid_nodes_gpu = uploadBoolMask(valid_nodes);
        const int grid_size = (n + kBlockSize - 1) / kBlockSize;
        filterSelectionByNodeMaskKernel<<<grid_size, kBlockSize, 0, currentSelectionStream(&selection)>>>(
            selection.ptr<bool>(),
            transform_indices.ptr<int>(),
            reinterpret_cast<const bool*>(valid_nodes_gpu.ptr<uint8_t>()),
            n,
            num_nodes);
        LFS_CUDA_LAUNCH_CHECK(currentSelectionStream(&selection), "render.selection.filter_node_mask");
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
        filterSelectionByCropKernel<<<grid_size, kBlockSize, 0, currentSelectionStream(&selection)>>>(
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
        LFS_CUDA_LAUNCH_CHECK(currentSelectionStream(&selection), "render.selection.filter_crop");
    }

    namespace config {
        void setSelectionGroupColor(int, float3) {}
        void setSelectionPreviewColor(float3) {}
    } // namespace config

} // namespace lfs::rendering
