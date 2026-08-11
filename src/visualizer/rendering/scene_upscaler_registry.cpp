/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "rendering/scene_upscaler_registry.hpp"

#include <array>
#include <mutex>
#include <ranges>
#include <shared_mutex>
#include <utility>

namespace lfs::vis {
    namespace {
        constexpr std::array DESCRIPTORS{
            SceneUpscalerDescriptor{
                .backend = SceneUpscalerBackend::Native,
                .id = "native",
                .requirements = {},
                .requires_adapter = false,
                .available = true,
            },
            SceneUpscalerDescriptor{
                .backend = SceneUpscalerBackend::Spatial,
                .id = "spatial",
                .requirements = {},
                .requires_adapter = true,
                .available = true,
            },
            SceneUpscalerDescriptor{
                .backend = SceneUpscalerBackend::Temporal,
                .id = "temporal",
                .requirements = {
                    .depth = true,
                    .motion_vectors = true,
                    .jitter = true,
                    .history = true,
                },
                .requires_adapter = true,
                .available = true,
            },
        };
    } // namespace

    std::span<const SceneUpscalerDescriptor> sceneUpscalerDescriptors() {
        return DESCRIPTORS;
    }

    const SceneUpscalerDescriptor& nativeSceneUpscalerDescriptor() {
        return DESCRIPTORS.front();
    }

    const SceneUpscalerDescriptor& spatialSceneUpscalerDescriptor() {
        return DESCRIPTORS[1];
    }

    const SceneUpscalerDescriptor& temporalSceneUpscalerDescriptor() {
        return DESCRIPTORS[2];
    }

    const SceneUpscalerDescriptor& sceneUpscalerDescriptor(const SceneUpscalerBackend backend) {
        const auto descriptor = std::ranges::find(DESCRIPTORS, backend,
                                                  &SceneUpscalerDescriptor::backend);
        return descriptor != DESCRIPTORS.end() ? *descriptor : nativeSceneUpscalerDescriptor();
    }

    std::optional<SceneUpscalerBackend> sceneUpscalerBackendFromId(const std::string_view id) {
        const auto descriptor = std::ranges::find(DESCRIPTORS, id, &SceneUpscalerDescriptor::id);
        if (descriptor == DESCRIPTORS.end())
            return std::nullopt;
        return descriptor->backend;
    }

    SceneUpscalerSelection resolveSceneUpscalerSelection(
        const SceneUpscalerBackend requested, const bool adapter_available) {
        if (requested == SceneUpscalerBackend::Native) {
            return {};
        }
        if ((requested == SceneUpscalerBackend::Spatial ||
             requested == SceneUpscalerBackend::Temporal) &&
            adapter_available) {
            return {.requested = requested, .effective = requested, .fallback = false};
        }
        return {
            .requested = requested,
            .effective = SceneUpscalerBackend::Native,
            .fallback = true,
        };
    }

    bool OptionalSceneUpscalerRegistry::registerAdapter(
        OptionalSceneUpscalerDescriptor descriptor,
        const SceneUpscalerAdapterFactory factory) {
        if (descriptor.id.empty() || descriptor.label_key.empty() || factory == nullptr ||
            sceneUpscalerBackendFromId(descriptor.id).has_value()) {
            return false;
        }
        const std::unique_lock lock(mutex_);
        const auto duplicate = std::ranges::find(
            registrations_, descriptor.id, [](const Registration& registration) {
                return std::string_view(registration.descriptor.id);
            });
        if (duplicate != registrations_.end())
            return false;
        registrations_.push_back({std::move(descriptor), factory});
        return true;
    }

    std::vector<OptionalSceneUpscalerDescriptor> OptionalSceneUpscalerRegistry::descriptors() const {
        const std::shared_lock lock(mutex_);
        std::vector<OptionalSceneUpscalerDescriptor> result;
        result.reserve(registrations_.size());
        for (const auto& registration : registrations_)
            result.push_back(registration.descriptor);
        return result;
    }

    std::optional<OptionalSceneUpscalerDescriptor> OptionalSceneUpscalerRegistry::descriptor(
        const std::string_view id) const {
        const std::shared_lock lock(mutex_);
        const auto registration = std::ranges::find(
            registrations_, id, [](const Registration& candidate) {
                return std::string_view(candidate.descriptor.id);
            });
        if (registration == registrations_.end())
            return std::nullopt;
        return registration->descriptor;
    }

    SceneUpscalerAvailability OptionalSceneUpscalerRegistry::probe(
        const std::string_view id,
        const SceneUpscalerProbeContext& context) const {
        SceneUpscalerAdapterFactory factory = nullptr;
        {
            const std::shared_lock lock(mutex_);
            const auto registration = std::ranges::find(
                registrations_, id, [](const Registration& candidate) {
                    return std::string_view(candidate.descriptor.id);
                });
            if (registration == registrations_.end())
                return {};
            factory = registration->factory;
        }
        if (context.safe_mode)
            return {.reason = SceneUpscalerAvailabilityReason::SafeMode};
        auto result = factory();
        if (!result.has_value())
            return {.reason = result.error()};
        auto& adapter = result.value();
        return adapter ? adapter->probe(context)
                       : SceneUpscalerAvailability{
                             .reason = SceneUpscalerAvailabilityReason::ProbeFailed};
    }

    std::unique_ptr<SceneUpscalerAdapter> OptionalSceneUpscalerRegistry::createAvailable(
        const std::string_view id,
        const SceneUpscalerProbeContext& context) const {
        SceneUpscalerAdapterFactory factory = nullptr;
        {
            const std::shared_lock lock(mutex_);
            const auto registration = std::ranges::find(
                registrations_, id, [](const Registration& candidate) {
                    return std::string_view(candidate.descriptor.id);
                });
            if (registration == registrations_.end())
                return nullptr;
            factory = registration->factory;
        }
        if (context.safe_mode)
            return nullptr;
        auto result = factory();
        if (!result.has_value())
            return nullptr;
        auto adapter = std::move(result.value());
        if (!adapter || !adapter->probe(context).available())
            return nullptr;
        return adapter;
    }

    OptionalSceneUpscalerRegistry& optionalSceneUpscalerRegistry() {
        static OptionalSceneUpscalerRegistry registry;
        return registry;
    }
} // namespace lfs::vis
