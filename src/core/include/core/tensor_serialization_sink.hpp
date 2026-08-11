/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "core/export.hpp"
#include "core/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>

namespace lfs::core {

    enum class TensorPayloadEncoding : std::uint8_t {
        NativeContiguous = 0,
        SwizzledShToCanonical = 1,
        QuantizedShToCanonical = 2,
    };

    struct TensorSerializationDescriptor {
        TensorShape serialized_shape;
        DataType dtype = DataType::Float32;
        Device serialized_device = Device::CPU;
        TensorPayloadEncoding encoding =
            TensorPayloadEncoding::NativeContiguous;
        std::size_t sh_primitives = 0;
        std::uint32_t sh_coefficients_rest = 0;
        std::uint32_t sh_layout_coefficients_rest = 0;

        [[nodiscard]] LFS_CORE_API std::uint64_t payload_bytes() const;
    };

    class LFS_CORE_API TensorSerializationSink {
    public:
        virtual ~TensorSerializationSink() = default;

        virtual void write_tensor_payload(
            std::ostream& destination,
            const Tensor& source,
            const Tensor* auxiliary_source,
            const TensorSerializationDescriptor& descriptor) = 0;
    };

    // Thread-local and deliberately scoped: checkpoint serialization remains
    // byte-compatible, while the training thread may redirect only tensor
    // payloads into a banded snapshot engine.
    class LFS_CORE_API TensorSerializationSinkScope {
    public:
        explicit TensorSerializationSinkScope(
            TensorSerializationSink& sink) noexcept;
        TensorSerializationSinkScope(
            const TensorSerializationSinkScope&) = delete;
        TensorSerializationSinkScope&
        operator=(const TensorSerializationSinkScope&) = delete;
        ~TensorSerializationSinkScope();

    private:
        TensorSerializationSink* previous_ = nullptr;
    };

    [[nodiscard]] LFS_CORE_API TensorSerializationSink*
    current_tensor_serialization_sink() noexcept;

    // Writes the normal tensor header/dimensions, then delegates only the
    // payload bytes to the active sink. Callers use this for an alternate
    // on-disk layout such as canonical SH while retaining the live swizzle.
    LFS_CORE_API void serialize_tensor_with_descriptor(
        std::ostream& destination,
        const Tensor& source,
        const TensorSerializationDescriptor& descriptor,
        const Tensor* auxiliary_source = nullptr);

} // namespace lfs::core
