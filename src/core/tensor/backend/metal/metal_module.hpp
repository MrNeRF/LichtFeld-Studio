/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/internal/private_access.hpp"

#include "core/detail/descriptors.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>

namespace lfs::core::internal {

    // MSL compiled at first launch into its own library, dispatched through the
    // tensor context so launches order with tensor work on the same storage.
    class MetalModule {
    public:
        virtual ~MetalModule() = default;
        [[nodiscard]] virtual uint64_t address(const StorageRef& storage) = 0;
        virtual void launch(std::string_view function, std::span<const std::pair<uint32_t, uint32_t>> constants,
                            std::span<const StorageRef> uses, std::span<const std::byte> params,
                            std::array<uint32_t, 3> groups, std::array<uint32_t, 3> group) = 0;
    };

    // Builds without Metal throw.
    std::unique_ptr<MetalModule> make_metal_module(std::string source, bool fast_math);

} // namespace lfs::core::internal
