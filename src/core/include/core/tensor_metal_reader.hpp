// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#if defined(__APPLE__) && defined(__OBJC__)
#import <Metal/Metal.h>
#include "core/export.hpp"
#include "core/tensor_fwd.hpp"
#include <functional>
#include <memory>
#include <span>

namespace lfs::core {
struct MetalTensorView {
    id<MTLBuffer> buffer = nil;
    NSUInteger offset = 0;
    size_t bytes = 0;
};

// Read-only, zero-copy consumer of resident Metal tensors. Producer work and
// later tensor mutations are ordered by GPU events; no CPU completion wait.
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
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace lfs::core
#endif
