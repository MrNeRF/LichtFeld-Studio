/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "selection_ops.hpp"

#include "core/tensor_backend.hpp"
#include "rendering/render_constants.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace lfs::rendering {
    namespace {
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::GpuBackend;
        using lfs::core::GpuBackendScope;
        using lfs::core::Tensor;

        constexpr std::size_t kSelectionGroupCountBins = 256;
        constexpr std::size_t kSelectionChangedCountIndex = kSelectionGroupCountBins;
        constexpr std::size_t kSelectionGroupScratchWords = kSelectionGroupCountBins + 1;
        constexpr float kPi = 3.14159265358979323846f;

        [[nodiscard]] bool nodeMaskRestrictsSelection(const std::vector<bool>& mask) {
            return std::any_of(mask.begin(), mask.end(), [](const bool enabled) { return !enabled; });
        }

        [[nodiscard]] GpuBackend requireGpuBackend(const Tensor& tensor, const char* message) {
            const auto backend = lfs::core::gpu_backend_of(tensor);
            if (!backend) {
                throw std::runtime_error(message);
            }
            return *backend;
        }

        [[nodiscard]] Tensor matchBackend(const Tensor& src, const GpuBackend backend) {
            const auto src_backend = lfs::core::gpu_backend_of(src);
            if (src.device() == Device::GPU && src_backend && *src_backend == backend) {
                return src.is_contiguous() ? src : src.contiguous();
            }
            GpuBackendScope scope(backend);
            if (src.device() == Device::CPU) {
                return src.to(Device::GPU);
            }
            return src.cpu().to(Device::GPU);
        }

        [[nodiscard]] Tensor asBool(const Tensor& tensor) {
            if (tensor.dtype() == DataType::Bool) {
                return tensor;
            }
            return tensor != 0;
        }

        [[nodiscard]] Tensor col1(const Tensor& matrix, const int start) {
            return matrix.slice(1, start, start + 1).flatten();
        }

        [[nodiscard]] Tensor catColumns(const Tensor& x, const Tensor& y, const Tensor& z) {
            const auto n = static_cast<std::size_t>(x.numel());
            return Tensor::cat({x.reshape(lfs::core::TensorShape({n, std::size_t{1}})),
                                y.reshape(lfs::core::TensorShape({n, std::size_t{1}})),
                                z.reshape(lfs::core::TensorShape({n, std::size_t{1}}))},
                               1);
        }

        void prepareSelectionGroupCountsScratch(Tensor& counts_scratch) {
            if (!counts_scratch.is_valid() ||
                counts_scratch.device() != Device::GPU ||
                counts_scratch.dtype() != DataType::Int32 ||
                counts_scratch.numel() != kSelectionGroupScratchWords) {
                counts_scratch = Tensor::zeros(
                    {kSelectionGroupScratchWords}, Device::GPU, DataType::Int32);
            } else {
                counts_scratch.zero_();
            }
        }

        [[nodiscard]] Tensor lockedGroupTable(const uint32_t* const locked_groups,
                                              const GpuBackend backend) {
            std::vector<int> bits(kSelectionGroupCountBins, 0);
            if (locked_groups != nullptr) {
                for (int group = 0; group < static_cast<int>(kSelectionGroupCountBins); ++group) {
                    bits[static_cast<std::size_t>(group)] =
                        (locked_groups[static_cast<std::size_t>(group) / 32] &
                         (1u << (static_cast<std::uint32_t>(group) % 32)))
                            ? 1
                            : 0;
                }
            }
            GpuBackendScope scope(backend);
            return Tensor::from_vector(bits, {kSelectionGroupCountBins}, Device::GPU);
        }

        [[nodiscard]] Tensor existingGroups(const Tensor& existing_mask,
                                            const std::size_t n,
                                            const GpuBackend backend) {
            if (existing_mask.is_valid() && existing_mask.numel() == n) {
                auto groups = matchBackend(existing_mask, backend);
                if (groups.dtype() != DataType::UInt8) {
                    groups = groups.to(DataType::UInt8);
                }
                return groups.flatten().contiguous();
            }
            GpuBackendScope scope(backend);
            return Tensor::zeros({n}, Device::GPU, DataType::UInt8);
        }

        [[nodiscard]] Tensor nodeValidMask(const std::size_t n,
                                           const Tensor* const transform_indices,
                                           const std::vector<bool>& valid_nodes,
                                           const GpuBackend backend) {
            GpuBackendScope scope(backend);
            auto valid = Tensor::full_bool({n}, true, Device::GPU);
            if (!nodeMaskRestrictsSelection(valid_nodes)) {
                return valid;
            }
            if (transform_indices == nullptr || !transform_indices->is_valid() ||
                transform_indices->numel() != n) {
                return valid;
            }
            const int num_nodes = static_cast<int>(valid_nodes.size());
            const Tensor table = Tensor::from_vector(valid_nodes, {valid_nodes.size()}, Device::GPU);
            Tensor idx = matchBackend(*transform_indices, backend).flatten().to(DataType::Int32);
            const Tensor in_range = (idx >= 0).logical_and(idx < num_nodes);
            const Tensor safe = Tensor::where(in_range, idx, Tensor::zeros_like(idx));
            const Tensor gathered = asBool(table.index_select(0, safe));
            return in_range.logical_and(gathered);
        }

        [[nodiscard]] Tensor otherLockedMask(const Tensor& existing_u8,
                                             const uint8_t group_id,
                                             const uint32_t* const locked_groups,
                                             const GpuBackend backend) {
            const Tensor existing_i = existing_u8.to(DataType::Int32);
            const Tensor table = lockedGroupTable(locked_groups, backend);
            const Tensor locked_at = table.index_select(0, existing_i) != 0;
            return (existing_i != 0)
                .logical_and(existing_i != static_cast<int>(group_id))
                .logical_and(locked_at);
        }

        void indexFillMasked(Tensor& dest,
                             const Tensor& output_indices,
                             const Tensor& mask,
                             const float value) {
            if (!mask.is_valid() || mask.numel() == 0 || !mask.any_scalar()) {
                return;
            }
            const Tensor positions = mask.nonzero().flatten().to(DataType::Int32);
            if (positions.numel() == 0) {
                return;
            }
            dest.index_fill_(0, output_indices.index_select(0, positions), value);
        }

        void writeGroupDeltas(const Tensor& existing_u8,
                              const Tensor& output_u8,
                              Tensor& counts_scratch) {
            prepareSelectionGroupCountsScratch(counts_scratch);
            const Tensor existing_i = existing_u8.to(DataType::Int32);
            const Tensor output_i = output_u8.to(DataType::Int32);
            const Tensor changed = output_i != existing_i;
            const auto n_changed = changed.count_nonzero();
            if (n_changed == 0) {
                return;
            }
            const Tensor changed_idx = changed.nonzero().flatten().to(DataType::Int32);
            const Tensor old_g = existing_i.index_select(0, changed_idx);
            const Tensor new_g = output_i.index_select(0, changed_idx);
            const Tensor old_nz = old_g != 0;
            if (old_nz.any_scalar()) {
                const Tensor old_pos = old_nz.nonzero().flatten().to(DataType::Int32);
                const Tensor neg = Tensor::full({old_pos.numel()}, -1.0f, Device::GPU, DataType::Int32);
                counts_scratch.index_add_(0, old_g.index_select(0, old_pos), neg);
            }
            const Tensor new_nz = new_g != 0;
            if (new_nz.any_scalar()) {
                const Tensor new_pos = new_nz.nonzero().flatten().to(DataType::Int32);
                const Tensor pos = Tensor::full({new_pos.numel()}, 1.0f, Device::GPU, DataType::Int32);
                counts_scratch.index_add_(0, new_g.index_select(0, new_pos), pos);
            }
            counts_scratch.slice(0, static_cast<int>(kSelectionChangedCountIndex),
                                 static_cast<int>(kSelectionChangedCountIndex + 1))
                .fill_(static_cast<float>(n_changed));
        }

        [[nodiscard]] Tensor applyGroupLogic(const Tensor& selected,
                                             const Tensor& existing_u8,
                                             const Tensor& node_valid,
                                             const Tensor& is_other_locked,
                                             const uint8_t group_id,
                                             const bool add_mode,
                                             const bool replace_mode) {
            const auto n = static_cast<std::size_t>(existing_u8.numel());
            const Tensor gid = Tensor::full({n}, static_cast<float>(group_id), Device::GPU, DataType::UInt8);
            const Tensor zeros = Tensor::zeros({n}, Device::GPU, DataType::UInt8);
            Tensor updated;
            if (replace_mode) {
                const Tensor selected_val = Tensor::where(is_other_locked, existing_u8, gid);
                const Tensor unselected_val =
                    Tensor::where(existing_u8 == static_cast<int>(group_id), zeros, existing_u8);
                updated = Tensor::where(selected, selected_val, unselected_val);
            } else if (add_mode) {
                updated = Tensor::where(selected.logical_and(is_other_locked.logical_not()), gid, existing_u8);
            } else {
                updated = Tensor::where(
                    selected.logical_and(existing_u8 == static_cast<int>(group_id)), zeros, existing_u8);
            }
            return Tensor::where(node_valid, updated, existing_u8);
        }

        [[nodiscard]] Tensor meansNx3(const Tensor& means, const std::size_t n, const GpuBackend backend) {
            Tensor contig = matchBackend(means, backend);
            if (contig.dtype() != DataType::Float32) {
                contig = contig.to(DataType::Float32);
            }
            contig = contig.contiguous();
            if (contig.ndim() == 2 && contig.size(1) == 3 && contig.size(0) == n) {
                return contig;
            }
            if (contig.numel() == n * 3) {
                return contig.reshape(lfs::core::TensorShape({n, std::size_t{3}}));
            }
            throw std::runtime_error("selection mask ops expect means with shape [N, 3]");
        }

        [[nodiscard]] Tensor applyRowMajorModelTransforms(const Tensor& means,
                                                          const Tensor* const model_transforms,
                                                          const Tensor* const transform_indices,
                                                          const std::size_t n,
                                                          const GpuBackend backend) {
            if (model_transforms == nullptr || !model_transforms->is_valid() ||
                model_transforms->numel() == 0) {
                return means;
            }
            Tensor mats = matchBackend(*model_transforms, backend);
            if (mats.dtype() != DataType::Float32) {
                mats = mats.to(DataType::Float32);
            }
            if (mats.numel() % 16 != 0) {
                throw std::runtime_error(
                    "model_transforms tensor must contain a multiple of 16 float values (N x 4 x 4).");
            }
            const int count = static_cast<int>(mats.numel() / 16);
            if (count <= 0) {
                return means;
            }
            mats = mats.contiguous().reshape(
                lfs::core::TensorShape({static_cast<std::size_t>(count), std::size_t{16}}));
            Tensor idx;
            if (transform_indices != nullptr && transform_indices->is_valid() &&
                transform_indices->numel() == n) {
                idx = matchBackend(*transform_indices, backend).flatten().to(DataType::Int32);
                idx = idx.clamp(0.0f, static_cast<float>(count - 1));
            } else {
                GpuBackendScope scope(backend);
                idx = Tensor::zeros({n}, Device::GPU, DataType::Int32);
            }
            const Tensor m = mats.index_select(0, idx);
            const Tensor x = col1(means, 0);
            const Tensor y = col1(means, 1);
            const Tensor z = col1(means, 2);
            const Tensor ox = col1(m, 0) * x + col1(m, 1) * y + col1(m, 2) * z + col1(m, 3);
            const Tensor oy = col1(m, 4) * x + col1(m, 5) * y + col1(m, 6) * z + col1(m, 7);
            const Tensor oz = col1(m, 8) * x + col1(m, 9) * y + col1(m, 10) * z + col1(m, 11);
            return catColumns(ox, oy, oz);
        }

        [[nodiscard]] std::vector<float> hostFloats(const Tensor& tensor) {
            return tensor.cpu().contiguous().to_vector();
        }

        [[nodiscard]] Tensor atan2Tensor(const Tensor& y, const Tensor& x) {
            const Tensor x_zero = x == 0.0f;
            const Tensor safe_x = Tensor::where(x_zero, Tensor::ones_like(x), x);
            const Tensor base = (y / safe_x).atan();
            const Tensor from_neg_x = Tensor::where(y >= 0.0f, base + kPi, base - kPi);
            const Tensor from_zero_x = Tensor::where(
                y > 0.0f,
                Tensor::full_like(y, kPi * 0.5f),
                Tensor::where(y < 0.0f, Tensor::full_like(y, -kPi * 0.5f), Tensor::zeros_like(y)));
            return Tensor::where(x > 0.0f, base, Tensor::where(x < 0.0f, from_neg_x, from_zero_x));
        }

        [[nodiscard]] Tensor inRangeIndices(const Tensor& indices, const int bound) {
            const Tensor idx = indices.flatten().to(DataType::Int32);
            return (idx >= 0).logical_and(idx < bound);
        }
    } // namespace

    void apply_selection_group_tensor_mask_program(
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
        const GpuBackend backend =
            requireGpuBackend(cumulative_selection, "apply_selection_group_tensor_mask requires a GPU mask");
        GpuBackendScope scope(backend);
        const auto n = static_cast<std::size_t>(cumulative_selection.numel());
        const Tensor selected = asBool(matchBackend(cumulative_selection, backend).flatten());
        const Tensor existing_u8 = existingGroups(existing_mask, n, backend);
        const Tensor node_valid = nodeValidMask(n, transform_indices, valid_nodes, backend);
        const Tensor is_other_locked = otherLockedMask(existing_u8, group_id, locked_groups, backend);
        const Tensor result = applyGroupLogic(
            selected, existing_u8, node_valid, is_other_locked, group_id, add_mode, replace_mode);
        if (!output_mask.is_valid() || output_mask.numel() != n ||
            output_mask.dtype() != DataType::UInt8 || output_mask.device() != Device::GPU) {
            output_mask = Tensor::empty({n}, Device::GPU, DataType::UInt8);
        }
        output_mask.copy_from(result);
        if (group_counts_scratch != nullptr) {
            writeGroupDeltas(existing_u8, result, *group_counts_scratch);
        }
    }

    void clear_selection_group_indexed_mask_program(
        const Tensor& visible_selection,
        const Tensor& visible_indices,
        const Tensor& existing_mask,
        Tensor& output_mask,
        const uint8_t group_id,
        const Tensor* const transform_indices,
        const std::vector<bool>& valid_nodes) {
        if (!visible_selection.is_valid() || !visible_indices.is_valid() ||
            !output_mask.is_valid() || visible_selection.size(0) == 0) {
            return;
        }
        const GpuBackend backend =
            requireGpuBackend(output_mask, "clear_selection_group_indexed_mask requires a GPU mask");
        GpuBackendScope scope(backend);
        const auto visible_n = static_cast<std::size_t>(visible_selection.numel());
        const int output_n = static_cast<int>(output_mask.numel());
        if (visible_indices.numel() != visible_n) {
            return;
        }
        const Tensor selected = asBool(matchBackend(visible_selection, backend).flatten());
        const Tensor vis_idx = matchBackend(visible_indices, backend).flatten().to(DataType::Int32);
        const Tensor in_range = inRangeIndices(vis_idx, output_n);
        const Tensor safe_vis = Tensor::where(in_range, vis_idx, Tensor::zeros_like(vis_idx));
        const Tensor existing_u8 = existingGroups(existing_mask, static_cast<std::size_t>(output_n), backend);
        const Tensor existing_at = existing_u8.index_select(0, safe_vis);
        const Tensor node_valid = nodeValidMask(visible_n, transform_indices, valid_nodes, backend);
        const Tensor clear_mask = selected.logical_not()
                                      .logical_and(in_range)
                                      .logical_and(node_valid)
                                      .logical_and(existing_at == static_cast<int>(group_id));
        indexFillMasked(output_mask, vis_idx, clear_mask, 0.0f);
    }

    void apply_selection_group_indexed_tensor_mask_program(
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
        const GpuBackend backend =
            requireGpuBackend(output_mask, "apply_selection_group_indexed_tensor_mask requires a GPU mask");
        GpuBackendScope scope(backend);
        const auto visible_n = static_cast<std::size_t>(visible_selection.numel());
        const auto output_n = static_cast<std::size_t>(output_mask.numel());
        if (visible_indices.numel() != visible_n) {
            return;
        }
        if (existing_mask.is_valid() && existing_mask.numel() == output_n) {
            output_mask.copy_from(matchBackend(existing_mask, backend));
        } else {
            output_mask.zero_();
        }
        if (replace_mode) {
            clear_selection_group_indexed_mask_program(
                visible_selection, visible_indices, existing_mask, output_mask, group_id,
                transform_indices, valid_nodes);
        }
        const Tensor selected = asBool(matchBackend(visible_selection, backend).flatten());
        const Tensor vis_idx = matchBackend(visible_indices, backend).flatten().to(DataType::Int32);
        const Tensor in_range = inRangeIndices(vis_idx, static_cast<int>(output_n));
        const Tensor safe_vis = Tensor::where(in_range, vis_idx, Tensor::zeros_like(vis_idx));
        const Tensor existing_u8 = existingGroups(existing_mask, output_n, backend);
        const Tensor existing_at = existing_u8.index_select(0, safe_vis);
        const Tensor node_valid = nodeValidMask(visible_n, transform_indices, valid_nodes, backend);
        const Tensor is_other_locked = otherLockedMask(existing_at, group_id, locked_groups, backend);
        Tensor write_mask = selected.logical_and(in_range).logical_and(node_valid);
        if (replace_mode) {
            write_mask = write_mask.logical_and(is_other_locked.logical_not());
            indexFillMasked(output_mask, vis_idx, write_mask, static_cast<float>(group_id));
        } else if (add_mode) {
            write_mask = write_mask.logical_and(is_other_locked.logical_not());
            indexFillMasked(output_mask, vis_idx, write_mask, static_cast<float>(group_id));
        } else {
            write_mask = write_mask.logical_and(existing_at == static_cast<int>(group_id));
            indexFillMasked(output_mask, vis_idx, write_mask, 0.0f);
        }
    }

    void merge_selection_mask_or_program(Tensor& accumulated_mask, const Tensor& delta_mask) {
        if (!accumulated_mask.is_valid() || !delta_mask.is_valid() ||
            accumulated_mask.numel() == 0 ||
            accumulated_mask.numel() != delta_mask.numel()) {
            return;
        }
        const GpuBackend backend =
            requireGpuBackend(accumulated_mask, "merge_selection_mask_or requires a GPU mask");
        GpuBackendScope scope(backend);
        const Tensor delta = matchBackend(delta_mask, backend);
        if (accumulated_mask.dtype() == DataType::Bool && delta.dtype() == DataType::Bool) {
            accumulated_mask.masked_fill_(asBool(delta), 1.0f);
            return;
        }
        accumulated_mask.copy_from(accumulated_mask | delta);
    }

    void filter_selection_by_node_mask_program(
        Tensor& selection,
        const Tensor& transform_indices,
        const std::vector<bool>& valid_nodes) {
        if (!selection.is_valid() || !transform_indices.is_valid() || valid_nodes.empty()) {
            return;
        }
        if (!nodeMaskRestrictsSelection(valid_nodes)) {
            return;
        }
        const GpuBackend backend =
            requireGpuBackend(selection, "filter_selection_by_node_mask requires a GPU mask");
        GpuBackendScope scope(backend);
        const auto n = static_cast<std::size_t>(selection.size(0));
        if (n == 0 || transform_indices.numel() != n) {
            return;
        }
        const Tensor keep = nodeValidMask(n, &transform_indices, valid_nodes, backend);
        selection.masked_fill_(keep.logical_not(), 0.0f);
    }

    void filter_selection_by_crop_program(
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
        const GpuBackend backend =
            requireGpuBackend(selection, "filter_selection_by_crop requires a GPU mask");
        GpuBackendScope scope(backend);
        const auto n = static_cast<std::size_t>(selection.size(0));
        if (means.size(0) != n) {
            return;
        }
        const bool has_crop = crop_box_transform && crop_box_transform->is_valid() &&
                              crop_box_min && crop_box_min->is_valid() &&
                              crop_box_max && crop_box_max->is_valid();
        const bool has_ellipsoid = ellipsoid_transform && ellipsoid_transform->is_valid() &&
                                   ellipsoid_radii && ellipsoid_radii->is_valid();
        if (!has_crop && !has_ellipsoid) {
            return;
        }
        const Tensor world = applyRowMajorModelTransforms(
            meansNx3(means, n, backend), model_transforms, transform_indices, n, backend);
        const Tensor px = col1(world, 0);
        const Tensor py = col1(world, 1);
        const Tensor pz = col1(world, 2);
        if (has_crop) {
            const auto c = hostFloats(*crop_box_transform);
            const auto bmin = hostFloats(*crop_box_min);
            const auto bmax = hostFloats(*crop_box_max);
            if (c.size() >= 16 && bmin.size() >= 3 && bmax.size() >= 3) {
                const Tensor lx = px * c[0] + py * c[4] + pz * c[8] + c[12];
                const Tensor ly = px * c[1] + py * c[5] + pz * c[9] + c[13];
                const Tensor lz = px * c[2] + py * c[6] + pz * c[10] + c[14];
                const Tensor inside = (lx >= bmin[0])
                                          .logical_and(lx <= bmax[0])
                                          .logical_and(ly >= bmin[1])
                                          .logical_and(ly <= bmax[1])
                                          .logical_and(lz >= bmin[2])
                                          .logical_and(lz <= bmax[2]);
                const Tensor pass = crop_inverse ? inside.logical_not() : inside;
                selection.masked_fill_(pass.logical_not(), 0.0f);
            }
        }
        if (has_ellipsoid) {
            const auto e = hostFloats(*ellipsoid_transform);
            const auto r = hostFloats(*ellipsoid_radii);
            if (e.size() >= 16 && r.size() >= 3) {
                const Tensor lx = px * e[0] + py * e[4] + pz * e[8] + e[12];
                const Tensor ly = px * e[1] + py * e[5] + pz * e[9] + e[13];
                const Tensor lz = px * e[2] + py * e[6] + pz * e[10] + e[14];
                const Tensor norm = (lx * lx) / (r[0] * r[0]) +
                                    (ly * ly) / (r[1] * r[1]) +
                                    (lz * lz) / (r[2] * r[2]);
                const Tensor inside = norm <= 1.0f;
                const Tensor pass = ellipsoid_inverse ? inside.logical_not() : inside;
                selection.masked_fill_(pass.logical_not(), 0.0f);
            }
        }
    }

    void filter_selection_by_screen_window_program(
        Tensor& selection,
        const Tensor& means,
        const std::array<float, 9>& view_rotation_rows,
        const std::array<float, 3>& translation,
        const ScreenWindowCameraModel camera_model,
        const int width,
        const int height,
        const float pixel_focal_x,
        const float pixel_focal_y,
        const float center_x,
        const float center_y,
        const float ortho_scale,
        const float near_depth,
        const float far_depth,
        const float scale_x,
        const float scale_y,
        const float offset_x,
        const float offset_y,
        const Tensor* const model_transforms,
        const Tensor* const transform_indices) {
        const auto model = static_cast<std::uint32_t>(camera_model);
        if (model != static_cast<std::uint32_t>(ScreenWindowCameraModel::Pinhole) &&
            model != static_cast<std::uint32_t>(ScreenWindowCameraModel::Orthographic) &&
            model != static_cast<std::uint32_t>(ScreenWindowCameraModel::Equirectangular)) {
            return;
        }
        if (!selection.is_valid() || !means.is_valid() || width <= 0 || height <= 0) {
            return;
        }
        const GpuBackend backend =
            requireGpuBackend(selection, "filter_selection_by_screen_window requires a GPU mask");
        GpuBackendScope scope(backend);
        const auto n = static_cast<std::size_t>(selection.size(0));
        if (means.size(0) != n) {
            return;
        }
        const float sanitized_ortho_scale =
            (std::isfinite(ortho_scale) && ortho_scale > 1.0e-5f) ? ortho_scale : DEFAULT_ORTHO_SCALE;
        const Tensor world = applyRowMajorModelTransforms(
            meansNx3(means, n, backend), model_transforms, transform_indices, n, backend);
        const Tensor dx = col1(world, 0) - translation[0];
        const Tensor dy = col1(world, 1) - translation[1];
        const Tensor dz = col1(world, 2) - translation[2];
        const Tensor visualizer_x =
            dx * view_rotation_rows[0] + dy * view_rotation_rows[1] + dz * view_rotation_rows[2];
        const Tensor visualizer_y =
            dx * view_rotation_rows[3] + dy * view_rotation_rows[4] + dz * view_rotation_rows[5];
        const Tensor visualizer_z =
            dx * view_rotation_rows[6] + dy * view_rotation_rows[7] + dz * view_rotation_rows[8];
        const Tensor view_x = visualizer_x;
        const Tensor view_y = -visualizer_y;
        const Tensor view_z = -visualizer_z;
        const float image_width = static_cast<float>(width);
        const float image_height = static_cast<float>(height);
        Tensor px;
        Tensor py;
        Tensor depth = view_z;
        if (camera_model == ScreenWindowCameraModel::Pinhole) {
            px = view_x * (pixel_focal_x) / view_z + center_x;
            py = view_y * (pixel_focal_y) / view_z + center_y;
        } else if (camera_model == ScreenWindowCameraModel::Orthographic) {
            px = view_x * sanitized_ortho_scale + 0.5f * image_width;
            py = view_y * sanitized_ortho_scale + 0.5f * image_height;
        } else {
            const Tensor len = (view_x * view_x + view_y * view_y + view_z * view_z).sqrt();
            const Tensor bad = (len <= 1.0e-6f).logical_or(len.isfinite().logical_not());
            const Tensor safe_len = Tensor::where(bad, Tensor::ones_like(len), len);
            const Tensor dir_x = view_x / safe_len;
            const Tensor dir_y = view_y / safe_len;
            const Tensor dir_z = view_z / safe_len;
            px = (atan2Tensor(dir_x, dir_z) / (2.0f * kPi) + 0.5f) * image_width;
            py = (dir_y.clamp(-1.0f, 1.0f).asin() / kPi + 0.5f) * image_height;
            depth = Tensor::where(bad, Tensor::full_like(len, -1.0f), len);
        }
        const float half_w = 0.5f * scale_x * image_width;
        const float half_h = 0.5f * scale_y * image_height;
        const float cx = 0.5f * image_width + offset_x * (0.5f * image_width - half_w);
        const float cy = 0.5f * image_height + offset_y * (0.5f * image_height - half_h);
        const Tensor inside_rect = ((px - cx).abs() <= half_w).logical_and((py - cy).abs() <= half_h);
        const Tensor inside = inside_rect
                                  .logical_and(depth >= near_depth)
                                  .logical_and(depth <= far_depth)
                                  .logical_and(depth > 0.0f);
        selection.masked_fill_(inside.logical_not(), 0.0f);
    }

} // namespace lfs::rendering
