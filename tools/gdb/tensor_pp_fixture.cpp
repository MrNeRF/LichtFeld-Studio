/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Debug fixture for tools/gdb/lfs_pretty_printers.py. Built with -g -O0 by
// tools/gdb/verify_pretty_printers.sh; not part of the application build.

#include <core/tensor.hpp>

using namespace lfs::core;

__attribute__((noinline)) void inspect_here() {
    asm volatile("" ::
                     : "memory");
}

int main() {
    Tensor empty_tensor;
    Tensor cpu_f32 = Tensor::ones(TensorShape{2, 3}, Device::CPU);
    Tensor cpu_i32 = Tensor::zeros(TensorShape{4}, Device::CPU, DataType::Int32);
    Tensor cpu_rank3 = Tensor::full(TensorShape{2, 3, 4}, 1.5f, Device::CPU);
    Tensor cuda_f32 = Tensor::ones(TensorShape{3, 256, 256}, Device::CUDA);
    TensorShape shape{7, 11};

    inspect_here();

    return static_cast<int>(cpu_f32.shape().elements() + cuda_f32.shape().elements() +
                            cpu_i32.shape().elements() + cpu_rank3.shape().elements() +
                            shape.elements() + (empty_tensor.is_empty() ? 0 : 1));
}
