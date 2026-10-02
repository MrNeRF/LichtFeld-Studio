/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/tensor/backend/vulkan/vk_context.hpp"

#include <cstdint>
#include <map>
#include <mutex>
#include <span>
#include <string_view>
#include <tuple>
#include <type_traits>

namespace lfs::training::vulkan {
    struct EmbeddedShader {
        std::string_view name;
        std::span<const uint32_t> words;
    };

    std::span<const EmbeddedShader> embedded_training_shaders();

    // Static pipeline caches keyed by (context id, ...) outlive the device. Each
    // insert asks the context to erase its entries at shutdown, so the pipelines
    // and layouts are destroyed before vkDestroyDevice.
    template <class Key, class Value>
    void release_at_shutdown(core::internal::VulkanContext& context, std::mutex& mutex,
                             std::map<Key, Value>& cache) {
        context.on_shutdown([&mutex, &cache, id = context.context_id()] {
            std::lock_guard lock(mutex);
            std::erase_if(cache, [id](const auto& entry) {
                if constexpr (std::is_integral_v<Key>)
                    return entry.first == id;
                else
                    return std::get<0>(entry.first) == id;
            });
        });
    }
} // namespace lfs::training::vulkan
