/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor_splat.hpp"
#include "internal/tensor_impl.hpp"
#include <tbb/parallel_for.h>
namespace lfs::core {
    static void affine_geometry(const splat_transform::LinearTransform& linear, const Tensor* matrices,
                                const Tensor& scales, const Tensor& rotations,
                                Tensor& output_scales, Tensor& output_rotations) {
        const auto check = [](const Tensor& t, size_t columns) {
            return t.is_valid() && t.dtype() == DataType::Float32 && t.ndim() == 2 && t.size(1) == columns;
        };
        LFS_ASSERT_MSG(check(scales, 3) && check(rotations, 4) && scales.size(0) == rotations.size(0),
                       "affine_splat_geometry expects Float32 [N,3] and [N,4]");
        LFS_ASSERT_MSG(check(output_scales, 3) && check(output_rotations, 4) &&
                           output_scales.shape() == scales.shape() && output_rotations.shape() == rotations.shape() &&
                           output_scales.is_contiguous() && output_rotations.is_contiguous() &&
                           !output_scales.has_zero_stride() && !output_rotations.has_zero_stride() &&
                           !internal::shares_storage(output_scales, output_rotations),
                       "affine_splat_geometry requires distinct contiguous outputs");
        for (const auto* t : {&rotations, static_cast<const Tensor*>(&output_scales), static_cast<const Tensor*>(&output_rotations)}) {
            LFS_ASSERT_MSG(t->device() == scales.device(), "affine_splat_geometry requires the same device");
            internal::require_same_gpu_backend(scales, *t, "affine_splat_geometry");
        }
        internal::preserve_lazy_snapshots_before_write(output_scales);
        internal::preserve_lazy_snapshots_before_write(output_rotations);
        const auto input = [&](const Tensor& t) {
            return internal::shares_storage(t, output_scales) || internal::shares_storage(t, output_rotations) ? t.clone() : t.contiguous();
        };
        const Tensor s = input(scales), r = input(rotations);
        const Tensor transforms = matrices ? matrices->contiguous() : Tensor{};
        const size_t n = scales.size(0);
        if (!n)
            return;
        if (scales.device() == Device::GPU) {
            pin_operands({&s, &r, &output_scales, &output_rotations});
            if (matrices)
                pin_operands({&transforms});
            const auto stream = matrices ? prepare_inputs_for_stream({&s, &r, &transforms, &output_scales, &output_rotations})
                                         : prepare_inputs_for_stream({&s, &r, &output_scales, &output_rotations});
            output_scales.set_stream(stream);
            output_rotations.set_stream(stream);
            internal::backend_ops_for(s).affine_splat_geometry(internal::storage_ref(s), internal::storage_ref(r),
                                                               internal::storage_ref(output_scales), internal::storage_ref(output_rotations), linear, n,
                                                               matrices ? std::optional(internal::storage_ref(transforms)) : std::nullopt, {stream});
        } else {
            const auto* sp = s.ptr<float>();
            const auto* rp = r.ptr<float>();
            auto* os = output_scales.ptr<float>();
            auto* oq = output_rotations.ptr<float>();
            const auto* m = matrices ? transforms.ptr<float>() : nullptr;
            tbb::parallel_for(size_t{0}, n, [&](size_t i) {
                auto local = linear;
                if (m)
                    std::copy_n(m + 9 * i, 9, local.rows);
                splat_transform::affine_geometry(local, sp + 3 * i, rp + 4 * i, os + 3 * i, oq + 4 * i);
            });
        }
    }
    void affine_splat_geometry(const splat_transform::LinearTransform& linear, const Tensor& scales, const Tensor& rotations,
                               Tensor& output_scales, Tensor& output_rotations) {
        affine_geometry(linear, nullptr, scales, rotations, output_scales, output_rotations);
    }
    void affine_splat_geometry(const Tensor& linear, const Tensor& scales, const Tensor& rotations,
                               Tensor& output_scales, Tensor& output_rotations) {
        LFS_ASSERT_MSG(linear.is_valid() && linear.ndim() == 2 && linear.size(1) == 9 && linear.size(0) == scales.size(0) &&
                           linear.dtype() == DataType::Float32 && linear.device() == scales.device(),
                       std::format("Batched affine matrices require Float32 [N,9] on scales device (rank={}, rows={}, columns={}, dtype={}, device={}, scales_rows={}, scales_device={})",
                                   linear.ndim(), linear.ndim() ? linear.size(0) : 0, linear.ndim() == 2 ? linear.size(1) : 0, int(linear.dtype()), int(linear.device()), scales.size(0), int(scales.device())));
        internal::require_same_gpu_backend(linear, scales, "affine_splat_geometry");
        affine_geometry({}, &linear, scales, rotations, output_scales, output_rotations);
    }
} // namespace lfs::core
