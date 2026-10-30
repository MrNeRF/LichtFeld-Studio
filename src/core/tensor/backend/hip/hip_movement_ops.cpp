/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "hip_backend_ops.hpp"
#include <algorithm>
#include <core/tensor.hpp>
#include <limits>
#include <stdexcept>

namespace lfs::core::internal {
    namespace {
        void movement_require(bool value, const char* message) {
            if (!value)
                throw std::invalid_argument(message);
        }
        size_t movement_product(size_t a, size_t b) {
            movement_require(!a || b <= SIZE_MAX / a, "HIP movement size overflow");
            return a * b;
        }
        size_t movement_span(const StridedLayout& layout, size_t width) {
            movement_require(layout.rank <= MAX_TENSOR_RANK && (width == 1 || width == 2 || width == 4 || width == 8), "Invalid HIP movement layout or dtype");
            for (size_t axis = 0; axis < layout.rank; ++axis) {
                if (!layout.dims[axis]) {
                    movement_require(!layout.element_count, "HIP empty movement layout has nonzero elements");
                    return 0;
                }
            }
            size_t count = 1, last = 0;
            for (size_t axis = 0; axis < layout.rank; ++axis) {
                count = movement_product(count, layout.dims[axis]);
                const auto part = movement_product(layout.dims[axis] - 1, layout.strides[axis]);
                movement_require(part <= SIZE_MAX - last, "HIP movement stride overflow");
                last += part;
            }
            movement_require(count == layout.element_count && last != SIZE_MAX, "HIP movement element count mismatch");
            return movement_product(last + 1, width);
        }
        class MovementTemporary {
            HipBackendOps& owner_;
            ExecContext execution_;

        public:
            StorageRef storage;
            MovementTemporary(HipBackendOps& owner, size_t bytes, DataType dtype, ExecContext execution)
                : owner_(owner), execution_(execution), storage(owner.allocate(bytes, 16, execution)) {
                storage.dtype = dtype;
            }
            ~MovementTemporary() { owner_.deallocate(storage, execution_); }
            MovementTemporary(const MovementTemporary&) = delete;
            MovementTemporary& operator=(const MovementTemporary&) = delete;
        };
    } // namespace

    void HipBackendOps::strided_upload(StorageRef host_input, StorageRef output,
                                       const StridedLayout& layout, ExecContext execution) {
        movement_require(!execution.cuda_stream, "HIP strided upload received a foreign queue");
        movement_require(host_input.dtype == output.dtype, "HIP strided upload dtype mismatch");
        const auto bytes = movement_span(layout, dtype_size(output.dtype));
        if (!bytes)
            return;
        movement_require(host_input.data && host_input.byte_offset <= SIZE_MAX - bytes, "Invalid HIP strided host input");
        // Snapshot the complete addressed host span before returning. The data
        // rearrangement is performed by the native GPU strided-copy kernel.
        MovementTemporary staging(*this, bytes, output.dtype, execution);
        copy_host_to_device(CopyRequest{.src = host_input, .dst = staging.storage, .bytes = bytes, .synchronous = true, .context = execution});
        strided_copy(staging.storage, output, layout, execution);
    }

    void HipBackendOps::pad(StorageRef input, StorageRef output,
                            const StridedLayout& input_layout, const StridedLayout& output_layout,
                            const std::array<size_t, MAX_TENSOR_RANK>& before, ExecContext execution) {
        movement_require(!execution.cuda_stream, "HIP padding received a foreign queue");
        movement_require(input.dtype == output.dtype && input_layout.rank == output_layout.rank, "HIP padding rank or dtype mismatch");
        const size_t width = dtype_size(input.dtype);
        const auto source_bytes = movement_span(input_layout, width);
        (void)movement_span(output_layout, width);
        size_t offset = 0;
        auto interior = input_layout;
        for (size_t axis = 0; axis < input_layout.rank; ++axis) {
            movement_require(before[axis] <= output_layout.dims[axis] &&
                                 input_layout.dims[axis] <= output_layout.dims[axis] - before[axis],
                             "HIP padding interior exceeds output shape");
            const auto part = movement_product(before[axis], output_layout.strides[axis]);
            movement_require(part <= SIZE_MAX - offset, "HIP padding offset overflow");
            offset += part;
            interior.strides[axis] = output_layout.strides[axis];
        }
        if (!source_bytes)
            return;
        offset = movement_product(offset, width);
        movement_require(offset <= SIZE_MAX - output.byte_offset, "HIP padding byte offset overflow");
        output.byte_offset += offset;
        bool contiguous = true;
        size_t stride = 1;
        for (size_t axis = input_layout.rank; axis; --axis) {
            if (input_layout.dims[axis - 1] > 1 && input_layout.strides[axis - 1] != stride)
                contiguous = false;
            stride = movement_product(stride, input_layout.dims[axis - 1]);
        }
        // Preserve input when source and destination share a backing allocation.
        // Padding values are initialized by the shared facade; copy only the interior.
        if (contiguous && input.data != output.data) {
            strided_scatter(input, output, interior, execution);
        } else {
            MovementTemporary packed(*this, movement_product(input_layout.element_count, width), input.dtype, execution);
            strided_copy(input, packed.storage, input_layout, execution);
            strided_scatter(packed.storage, output, interior, execution);
        }
    }
} // namespace lfs::core::internal
