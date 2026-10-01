/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#if defined(__APPLE__) && defined(__OBJC__)
#include "core/export.hpp"
#include "core/tensor_fwd.hpp"
#import <Metal/Metal.h>
#include <functional>
#include <memory>
#include <span>

namespace lfs::core {
    struct MetalTensorView {
        id<MTLBuffer> buffer = nil;
        NSUInteger offset = 0;
        size_t bytes = 0;
    };

    // Zero-copy native access to resident Metal tensors. submit is read-only;
    // submitWrites names mutable outputs explicitly. Tensor work and native
    // access are ordered by GPU events; no CPU completion wait.
    // The caller must serialize model mutations while taking/submitting its snapshot,
    // exactly as for TensorVulkanInterop. This object does not acquire a scene lock.
    class LFS_CORE_API MetalTensorReader {
    public:
        MetalTensorReader();
        ~MetalTensorReader();
        MetalTensorReader(const MetalTensorReader&) = delete;
        MetalTensorReader& operator=(const MetalTensorReader&) = delete;
        [[nodiscard]] id<MTLDevice> device() const;
        using Encode = std::function<void(id<MTLCommandBuffer>, std::span<const MetalTensorView>)>;
        // Commits the command after encode returns. Invalid/empty tensors yield empty
        // views. Nonempty tensors must be contiguous and resident on this device.
        // Tensor owners are retained until the native command completes.
        // On encoding failure the command is discarded; discard its frame reservation.
        [[nodiscard]] id<MTLCommandBuffer> submit(std::span<const Tensor* const> tensors, const Encode& encode);

        using EncodeWrite = std::function<void(id<MTLCommandBuffer>, std::span<const MetalTensorView>, std::span<const MetalTensorView>)>;
        // Explicit mutable outputs use the same GPU producer/consumer ordering.
        // Later tensor reads, writes, host access and pool reuse wait for them.
        // Encoding failure discards the command; submitted write failure is sticky
        // in the tensor context, so partial output can never become valid data.
        // Callers must serialize inputs/outputs while taking this snapshot.
        [[nodiscard]] id<MTLCommandBuffer> submitWrites(std::span<const Tensor* const> inputs,
                                                        std::span<Tensor* const> outputs, const EncodeWrite& encode);

    private:
        id<MTLCommandBuffer> submitAccess(std::span<const Tensor* const>, const Encode&, bool writes);
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::core
#endif
