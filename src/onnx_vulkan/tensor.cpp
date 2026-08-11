/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs_onnx_vulkan/onnx_vulkan.hpp"

namespace lfs::onnx_vulkan {

    Tensor::Tensor(const ElementType type,
                   std::vector<std::int64_t> shape,
                   std::vector<std::byte> bytes)
        : type_(type),
          shape_(std::move(shape)),
          bytes_(std::move(bytes)) {}

} // namespace lfs::onnx_vulkan
