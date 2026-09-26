/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor_splat.hpp"
#include "internal/tensor_impl.hpp"
#include <tbb/parallel_for.h>
namespace lfs::core {
    void affine_splat_geometry(const splat_transform::LinearTransform& linear,
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
        const size_t n = scales.size(0);
        if (!n)
            return;
        if (scales.device() == Device::GPU) {
            pin_operands({&s, &r, &output_scales, &output_rotations});
            const auto stream = prepare_inputs_for_stream({&s, &r, &output_scales, &output_rotations});
            output_scales.set_stream(stream);
            output_rotations.set_stream(stream);
            internal::backend_ops_for(s).affine_splat_geometry(internal::storage_ref(s), internal::storage_ref(r),
                                                               internal::storage_ref(output_scales), internal::storage_ref(output_rotations), linear, n, {stream});
        } else {
            const auto* sp = s.ptr<float>();
            const auto* rp = r.ptr<float>();
            auto* os = output_scales.ptr<float>();
            auto* oq = output_rotations.ptr<float>();
            tbb::parallel_for(size_t{0}, n, [&](size_t i) {
                splat_transform::affine_geometry(linear, sp + 3 * i, rp + 4 * i, os + 3 * i, oq + 4 * i);
            });
        }
    }
} // namespace lfs::core
