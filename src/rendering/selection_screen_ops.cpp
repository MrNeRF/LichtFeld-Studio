/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "selection_ops.hpp"

#include "core/tensor_backend.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace lfs::rendering {

    namespace {
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::GpuBackend;
        using lfs::core::GpuBackendScope;
        using lfs::core::Tensor;

        constexpr float kPi = 3.14159265358979323846f;
        constexpr float kInvalidScreenFill = kInvalidScreenPositionThreshold * 100000.0f;

        struct ScreenAxes {
            Tensor x;
            Tensor y;
            Tensor valid;
        };

        ScreenAxes screen_axes(const Tensor& screen_positions) {
            ScreenAxes axes{
                .x = screen_positions.slice(1, 0, 1),
                .y = screen_positions.slice(1, 1, 2),
            };
            axes.valid = (axes.x >= kInvalidScreenPositionThreshold)
                             .logical_and(axes.y >= kInvalidScreenPositionThreshold);
            return axes;
        }

        [[nodiscard]] bool isCudaGpu(const Tensor& tensor) {
            return tensor.is_valid() && tensor.device() == Device::GPU &&
                   lfs::core::gpu_backend_of(tensor) == GpuBackend::CUDA;
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

        [[nodiscard]] Tensor stackXY(const Tensor& x, const Tensor& y) {
            const auto n = static_cast<std::size_t>(x.numel());
            return Tensor::cat({x.reshape(lfs::core::TensorShape({n, std::size_t{1}})),
                                y.reshape(lfs::core::TensorShape({n, std::size_t{1}}))},
                               1);
        }

        [[nodiscard]] Tensor stackXYZ(const Tensor& x, const Tensor& y, const Tensor& z) {
            const auto n = static_cast<std::size_t>(x.numel());
            return Tensor::cat({x.reshape(lfs::core::TensorShape({n, std::size_t{1}})),
                                y.reshape(lfs::core::TensorShape({n, std::size_t{1}})),
                                z.reshape(lfs::core::TensorShape({n, std::size_t{1}}))},
                               1);
        }

        [[nodiscard]] Tensor applyOneRowMajorMat4(const Tensor& x,
                                                  const Tensor& y,
                                                  const Tensor& z,
                                                  const float* const m) {
            return stackXYZ(x * m[0] + y * m[1] + z * m[2] + m[3],
                            x * m[4] + y * m[5] + z * m[6] + m[7],
                            x * m[8] + y * m[9] + z * m[10] + m[11]);
        }

        // Match atan2f, including atan2(y, -0) == ±π. Equirect hits that seam when
        // vis view_z is +0 (eq_z = -view_z is -0). Treating ±0 as +0 would write
        // width/2 instead of width.
        [[nodiscard]] Tensor atan2Tensor(const Tensor& y, const Tensor& x) {
            const Tensor x_zero = x == 0.0f;
            const Tensor safe_x = Tensor::where(x_zero, Tensor::ones_like(x), x);
            const Tensor base = (y / safe_x).atan();
            const Tensor negative_y = (y < 0.0f).logical_or(
                (y == 0.0f).logical_and(y.reciprocal() < 0.0f));
            const Tensor from_neg_x = Tensor::where(negative_y, base - kPi, base + kPi);
            const Tensor from_pos_zero = Tensor::where(
                y > 0.0f,
                Tensor::full_like(y, kPi * 0.5f),
                Tensor::where(y < 0.0f, Tensor::full_like(y, -kPi * 0.5f), Tensor::zeros_like(y)));
            const Tensor x_neg_zero = x_zero.logical_and(x.reciprocal() < 0.0f);
            const Tensor from_neg_zero = Tensor::where(
                negative_y, Tensor::full_like(y, -kPi), Tensor::full_like(y, kPi));
            return Tensor::where(
                x > 0.0f,
                base,
                Tensor::where(x < 0.0f, from_neg_x,
                              Tensor::where(x_neg_zero.logical_and(y == 0.0f),
                                            from_neg_zero, from_pos_zero)));
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
            throw std::runtime_error("project_screen_positions_tensor expects a GPU Float32 [N, 3] means tensor");
        }

        // KEEP IN SYNC with projectScreenPositionsKernel model-transform multiply.
        [[nodiscard]] Tensor applyRowMajorModelTransforms(const Tensor& means,
                                                          const Tensor* const model_transforms,
                                                          const Tensor& idx,
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
            const Tensor x = col1(means, 0);
            const Tensor y = col1(means, 1);
            const Tensor z = col1(means, 2);
            // A single scene transform needs only 16 scalars, avoiding an
            // [N,16] expansion (about 1 GiB for 16 million points).
            if (count == 1) {
                const auto host = mats.cpu().contiguous().to_vector();
                return applyOneRowMajorMat4(x, y, z, host.data());
            }
            const Tensor clamped =
                idx.to(DataType::Int32).clamp(0.0f, static_cast<float>(count - 1)).to(DataType::Int32);
            const Tensor m = mats.index_select(0, clamped);
            const Tensor ox = col1(m, 0) * x + col1(m, 1) * y + col1(m, 2) * z + col1(m, 3);
            const Tensor oy = col1(m, 4) * x + col1(m, 5) * y + col1(m, 6) * z + col1(m, 7);
            const Tensor oz = col1(m, 8) * x + col1(m, 9) * y + col1(m, 10) * z + col1(m, 11);
            return stackXYZ(ox, oy, oz);
        }

        void orDiskIntoSelection(const Tensor& x,
                                 const Tensor& y,
                                 const Tensor& valid,
                                 const float mouse_x,
                                 const float mouse_y,
                                 const float radius,
                                 Tensor& selection_out) {
            const Tensor dx = x - mouse_x;
            const Tensor dy = y - mouse_y;
            const Tensor inside = valid.logical_and((dx * dx + dy * dy) <= radius * radius).flatten();
            selection_out.masked_fill_(inside, 1.0f);
        }
    } // namespace

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
        const ScreenAxes axes = screen_axes(screen_positions);
        const Tensor inside = axes.valid
                                  .logical_and(axes.x >= x0)
                                  .logical_and(axes.x <= x1)
                                  .logical_and(axes.y >= y0)
                                  .logical_and(axes.y <= y1)
                                  .flatten();
        selection_out.masked_fill_(inside, 1.0f);
    }

    int pick_projected_gaussian_tensor(
        const Tensor& screen_positions,
        const float x,
        const float y,
        const float radius) {
        if (!screen_positions.is_valid() || screen_positions.size(0) == 0) {
            return -1;
        }
        if (screen_positions.device() != Device::GPU ||
            screen_positions.dtype() != DataType::Float32 ||
            screen_positions.ndim() != 2 ||
            screen_positions.size(1) != 2) {
            throw std::runtime_error("pick_projected_gaussian_tensor expects a GPU Float32 [N, 2] tensor");
        }
        const ScreenAxes axes = screen_axes(screen_positions);
        const Tensor finite = axes.valid.logical_and(axes.x.isfinite()).logical_and(axes.y.isfinite());
        const Tensor dx = axes.x - x;
        const Tensor dy = axes.y - y;
        const Tensor dist_sq = (dx * dx + dy * dy)
                                   .masked_fill(finite.logical_not(), std::numeric_limits<float>::infinity())
                                   .flatten();
        const float best = dist_sq.min_scalar();
        if (!(best <= radius * radius)) {
            return -1;
        }
        // Equal distances pick the largest index, as the kernel did.
        const auto candidates = (dist_sq == best).nonzero().flatten().to_vector_int();
        return *std::max_element(candidates.begin(), candidates.end());
    }

    void set_selection_element(Tensor& selection, const int index, const bool value) {
        if (!selection.is_valid() || index < 0 ||
            static_cast<size_t>(index) >= selection.numel()) {
            return;
        }
        if (lfs::core::gpu_backend_of(selection) == lfs::core::GpuBackend::Vulkan) {
            lfs::core::GpuBackendScope scope(lfs::core::GpuBackend::Vulkan);
            Tensor idx = Tensor::from_vector(std::vector<int>{index}, {1}, selection.device());
            selection.index_fill_(0, idx, value ? 1.0f : 0.0f);
            return;
        }
        set_selection_element(selection.ptr<bool>(), index, value);
    }

    void brush_select_tensor_program(
        const Tensor& screen_positions,
        const float mouse_x,
        const float mouse_y,
        const float radius,
        Tensor& selection_out) {
        if (!screen_positions.is_valid() || screen_positions.size(0) == 0 ||
            !selection_out.is_valid() || selection_out.numel() == 0) {
            return;
        }
        if (screen_positions.device() != Device::GPU || selection_out.device() != Device::GPU) {
            return;
        }
        const GpuBackend backend =
            requireGpuBackend(selection_out, "brush_select_tensor requires a GPU selection tensor");
        GpuBackendScope scope(backend);
        const Tensor positions = matchBackend(screen_positions, backend);
        const ScreenAxes axes = screen_axes(positions);
        orDiskIntoSelection(axes.x, axes.y, axes.valid, mouse_x, mouse_y, radius, selection_out);
    }

    void brush_select_disks_tensor(
        const Tensor& screen_positions,
        const std::vector<float>& disk_xy,
        const float radius,
        Tensor& selection_out) {
        if (disk_xy.size() < 2 || !screen_positions.is_valid() || screen_positions.size(0) == 0 ||
            !selection_out.is_valid() || selection_out.numel() == 0) {
            return;
        }
        const std::size_t disk_count = disk_xy.size() / 2;
        if (isCudaGpu(screen_positions) && isCudaGpu(selection_out)) {
            for (std::size_t i = 0; i < disk_count; ++i) {
                brush_select_tensor(
                    screen_positions, disk_xy[i * 2], disk_xy[i * 2 + 1], radius, selection_out);
            }
            return;
        }
        if (screen_positions.device() != Device::GPU || selection_out.device() != Device::GPU) {
            return;
        }
        const GpuBackend backend =
            requireGpuBackend(selection_out, "brush_select_disks_tensor requires a GPU selection tensor");
        GpuBackendScope scope(backend);
        const Tensor positions = matchBackend(screen_positions, backend);
        const ScreenAxes axes = screen_axes(positions);
        for (std::size_t i = 0; i < disk_count; ++i) {
            orDiskIntoSelection(
                axes.x, axes.y, axes.valid, disk_xy[i * 2], disk_xy[i * 2 + 1], radius, selection_out);
        }
    }

    void polygon_select_tensor_program(
        const Tensor& screen_positions,
        const Tensor& polygon_vertices,
        Tensor& selection_out) {
        if (!screen_positions.is_valid() || screen_positions.size(0) == 0 ||
            !polygon_vertices.is_valid() || polygon_vertices.size(0) < 3 ||
            !selection_out.is_valid() || selection_out.numel() == 0) {
            return;
        }
        if (screen_positions.device() != Device::GPU || selection_out.device() != Device::GPU) {
            return;
        }
        const GpuBackend backend =
            requireGpuBackend(selection_out, "polygon_select_tensor requires a GPU selection tensor");
        GpuBackendScope scope(backend);
        const Tensor positions = matchBackend(screen_positions, backend);
        const ScreenAxes axes = screen_axes(positions);
        const Tensor px = axes.x.flatten();
        const Tensor py = axes.y.flatten();
        const Tensor valid = axes.valid.flatten();
        const auto verts = matchBackend(polygon_vertices, backend).cpu().contiguous().to_vector();
        const int num_verts = static_cast<int>(polygon_vertices.size(0));
        if (static_cast<int>(verts.size()) < num_verts * 2 || num_verts < 3) {
            return;
        }
        Tensor inside = Tensor::full_bool({static_cast<std::size_t>(px.numel())}, false, Device::GPU);
        // Even-odd inclusion matching polygonSelectKernel. Iterate edges so we
        // never allocate an N x vertex matrix.
        for (int i = 0, j = num_verts - 1; i < num_verts; j = i++) {
            const float xi = verts[static_cast<std::size_t>(i) * 2];
            const float yi = verts[static_cast<std::size_t>(i) * 2 + 1];
            const float xj = verts[static_cast<std::size_t>(j) * 2];
            const float yj = verts[static_cast<std::size_t>(j) * 2 + 1];
            // polygonSelectKernel: (yi > py) != (yj > py) is false when yi == yj.
            if (yi == yj) {
                continue;
            }
            const Tensor crosses = (py < yi).logical_xor(py < yj);
            const Tensor hit_x = (py - yi) * (xj - xi) / (yj - yi) + xi;
            const Tensor hit = crosses.logical_and(px < hit_x);
            inside = inside.logical_xor(hit);
        }
        selection_out.masked_fill_(valid.logical_and(inside), 1.0f);
    }

    Tensor project_screen_positions_tensor_program(
        const Tensor& means,
        const int width,
        const int height,
        const std::array<float, 9>& view_rotation_rows,
        const std::array<float, 3>& translation,
        const float pixel_focal_x,
        const float pixel_focal_y,
        const float center_x,
        const float center_y,
        const ScreenWindowCameraModel camera_model,
        const float ortho_scale,
        const Tensor* const model_transforms,
        const Tensor* const transform_indices,
        const std::vector<bool>& node_visibility_mask) {
        if (!means.is_valid() || means.size(0) == 0) {
            return {};
        }
        if (means.device() != Device::GPU ||
            means.dtype() != DataType::Float32 ||
            means.ndim() != 2 ||
            means.size(1) != 3) {
            throw std::runtime_error("project_screen_positions_tensor expects a GPU Float32 [N, 3] means tensor");
        }
        if (width <= 0 || height <= 0) {
            return {};
        }

        const GpuBackend backend =
            requireGpuBackend(means, "project_screen_positions_tensor requires GPU means");
        GpuBackendScope scope(backend);
        const auto n = static_cast<std::size_t>(means.size(0));
        const Tensor world_means = meansNx3(means, n, backend);

        Tensor idx;
        const bool has_indices = transform_indices != nullptr && transform_indices->is_valid() &&
                                 transform_indices->numel() >= n;
        if (has_indices) {
            idx = matchBackend(*transform_indices, backend).flatten().to(DataType::Int32);
            if (idx.numel() > n) {
                idx = idx.slice(0, 0, static_cast<int>(n));
            }
        } else {
            idx = Tensor::zeros({n}, Device::GPU, DataType::Int32);
        }

        Tensor valid = Tensor::full_bool({n}, true, Device::GPU);
        if (has_indices && !node_visibility_mask.empty()) {
            const int vis_count = static_cast<int>(node_visibility_mask.size());
            const Tensor table = asBool(
                Tensor::from_vector(node_visibility_mask, {node_visibility_mask.size()}, Device::GPU));
            const Tensor in_range = (idx >= 0).logical_and(idx < vis_count);
            const Tensor safe = Tensor::where(in_range, idx, Tensor::zeros_like(idx));
            const Tensor visible = asBool(table.index_select(0, safe));
            valid = in_range.logical_and(visible);
        }

        // KEEP IN SYNC with projectScreenPositionsKernel: vis view is +X right,
        // +Y up, -Z forward. Invalid fill is kInvalidScreenPositionThreshold * 1e5.
        const Tensor world = applyRowMajorModelTransforms(world_means, model_transforms, idx, backend);
        const Tensor dx = col1(world, 0) - translation[0];
        const Tensor dy = col1(world, 1) - translation[1];
        const Tensor dz = col1(world, 2) - translation[2];
        // view_rotation_rows is R^T rows (GLM columns of viewport.rotation).
        const Tensor view_x =
            dx * view_rotation_rows[0] + dy * view_rotation_rows[1] + dz * view_rotation_rows[2];
        const Tensor view_y =
            dx * view_rotation_rows[3] + dy * view_rotation_rows[4] + dz * view_rotation_rows[5];
        const Tensor view_z =
            dx * view_rotation_rows[6] + dy * view_rotation_rows[7] + dz * view_rotation_rows[8];
        const Tensor finite = view_x.isfinite().logical_and(view_y.isfinite()).logical_and(view_z.isfinite());
        valid = valid.logical_and(finite);

        const bool equirectangular = camera_model == ScreenWindowCameraModel::Equirectangular;
        const bool orthographic = camera_model == ScreenWindowCameraModel::Orthographic;
        const float image_width = static_cast<float>(width);
        const float image_height = static_cast<float>(height);

        Tensor px;
        Tensor py;
        if (equirectangular) {
            // Same vis→vk negation as projectScreenPositionsKernel.
            const Tensor eq_x = view_x;
            const Tensor eq_y = -view_y;
            const Tensor eq_z = -view_z;
            const Tensor len = (eq_x * eq_x + eq_y * eq_y + eq_z * eq_z).sqrt();
            const Tensor bad = (len <= 1.0e-6f).logical_or(len.isfinite().logical_not());
            valid = valid.logical_and(bad.logical_not());
            const Tensor safe_len = Tensor::where(bad, Tensor::ones_like(len), len);
            const Tensor dir_x = eq_x / safe_len;
            const Tensor dir_y = eq_y / safe_len;
            const Tensor dir_z = eq_z / safe_len;
            px = (atan2Tensor(dir_x, dir_z) / (2.0f * kPi) + 0.5f) * image_width;
            py = (dir_y.clamp(-1.0f, 1.0f).asin() / kPi + 0.5f) * image_height;
        } else {
            // Pinhole/ortho reject at or behind the camera (visualizer -Z forward).
            valid = valid.logical_and(view_z < -1.0e-6f);
            if (orthographic) {
                if (!std::isfinite(ortho_scale) || ortho_scale <= 0.0f) {
                    valid = Tensor::full_bool({n}, false, Device::GPU);
                    px = Tensor::full({n}, kInvalidScreenFill, Device::GPU, DataType::Float32);
                    py = Tensor::full({n}, kInvalidScreenFill, Device::GPU, DataType::Float32);
                } else {
                    px = view_x * ortho_scale + center_x;
                    py = -view_y * ortho_scale + center_y;
                }
            } else {
                const Tensor depth = -view_z;
                const Tensor safe_depth = Tensor::where(valid, depth, Tensor::ones_like(depth));
                px = view_x * pixel_focal_x / safe_depth + center_x;
                py = -view_y * pixel_focal_y / safe_depth + center_y;
            }
        }

        const Tensor invalid = Tensor::full_like(px, kInvalidScreenFill);
        px = Tensor::where(valid, px, invalid);
        py = Tensor::where(valid, py, invalid);
        return stackXY(px, py);
    }

} // namespace lfs::rendering
