/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor_ppisp.hpp"
#include "core/tensor_backend.hpp"
#include "internal/ppisp.hpp"
#include "internal/tensor_impl.hpp"
namespace lfs::core {
    Tensor ppisp_apply(const Tensor& rgb, const PpispParams& p) {
        LFS_ASSERT_MSG(rgb.is_valid() && rgb.dtype() == DataType::Float32 && rgb.ndim() == 3 && rgb.size(0) == 3 &&
                           rgb.size(1) > 0 && rgb.size(2) > 0 && rgb.numel() <= INT32_MAX,
                       "PPISP requires float RGB CHW");
        const int height = int(rgb.size(1)), width = int(rgb.size(2)), full_height = p.full_height > 0 ? p.full_height : height;
        LFS_ASSERT_MSG(p.y_offset >= 0 && height <= full_height && p.y_offset <= full_height - height, "PPISP region is out of bounds");
        const GpuBackendScope scope(gpu_backend_of(rgb).value_or(default_gpu_backend()));
        const auto input = rgb.contiguous();
        auto output = Tensor::empty(rgb.shape(), rgb.device());
        if (rgb.device() == Device::GPU) {
            pin_operands({&input, &output});
            const auto stream = prepare_inputs_for_stream({&input}, output.stream());
            output.set_stream(stream);
            internal::backend_ops_for(output).ppisp_apply(internal::storage_ref(input), internal::storage_ref(output),
                                                          width, height, p, internal::ExecContext{stream});
        } else
            for (int i = 0; i < width * height; ++i)
                internal::ppisp_pixel(input.ptr<float>(), output.ptr<float>(), width, height, p, i);
        return output;
    }
} // namespace lfs::core
