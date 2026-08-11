/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace lfs::vis {

    enum class SceneUpscalerBackend : std::uint8_t {
        Native = 0,
        Spatial,
        Temporal,
    };

    struct SceneUpscalerRequirements {
        bool depth = false;
        bool motion_vectors = false;
        bool jitter = false;
        bool history = false;
        bool reactive_mask = false;
        bool exposure = false;

        [[nodiscard]] constexpr bool any() const {
            return depth || motion_vectors || jitter || history || reactive_mask || exposure;
        }

        [[nodiscard]] constexpr bool temporal() const {
            return motion_vectors || jitter || history;
        }

        constexpr bool operator==(const SceneUpscalerRequirements&) const = default;
    };

    struct SceneUpscalerDescriptor {
        SceneUpscalerBackend backend = SceneUpscalerBackend::Native;
        std::string_view id;
        SceneUpscalerRequirements requirements;
        bool requires_adapter = false;
        bool available = false;
    };

    struct SceneUpscalerSelection {
        SceneUpscalerBackend requested = SceneUpscalerBackend::Native;
        SceneUpscalerBackend effective = SceneUpscalerBackend::Native;
        bool fallback = false;

        constexpr bool operator==(const SceneUpscalerSelection&) const = default;
    };

    enum class SceneUpscalerGraphicsApi : std::uint8_t {
        Vulkan = 0,
    };

    enum class SceneUpscalerAvailabilityReason : std::uint8_t {
        Ready = 0,
        SafeMode,
        NotCompiled,
        RuntimeMissing,
        DeviceUnsupported,
        GraphicsApiUnsupported,
        ProbeFailed,
    };

    struct SceneUpscalerProbeContext {
        SceneUpscalerGraphicsApi graphics_api = SceneUpscalerGraphicsApi::Vulkan;
        std::uint32_t graphics_api_version = 0;
        std::uint32_t vendor_id = 0;
        std::uint32_t device_id = 0;
        bool safe_mode = false;
    };

    struct SceneUpscalerAvailability {
        SceneUpscalerAvailabilityReason reason = SceneUpscalerAvailabilityReason::NotCompiled;

        [[nodiscard]] constexpr bool available() const {
            return reason == SceneUpscalerAvailabilityReason::Ready;
        }

        constexpr bool operator==(const SceneUpscalerAvailability&) const = default;
    };

    struct OptionalSceneUpscalerDescriptor {
        std::string id;
        std::string label_key;
        SceneUpscalerRequirements requirements;
    };

    struct SceneUpscalerCatalogEntry {
        std::string id;
        std::string label_key;
        SceneUpscalerRequirements requirements;
        SceneUpscalerAvailability availability;
        bool internal = false;

        [[nodiscard]] constexpr bool selectable() const {
            return availability.available();
        }
    };

    class SceneUpscalerAdapter {
    public:
        virtual ~SceneUpscalerAdapter() = default;
        [[nodiscard]] virtual SceneUpscalerAvailability probe(
            const SceneUpscalerProbeContext& context) const noexcept = 0;
    };

    using SceneUpscalerAdapterFactoryResult =
        std::expected<std::unique_ptr<SceneUpscalerAdapter>,
                      SceneUpscalerAvailabilityReason>;
    using SceneUpscalerAdapterFactory = SceneUpscalerAdapterFactoryResult (*)() noexcept;

    class LFS_VIS_API OptionalSceneUpscalerRegistry {
    public:
        [[nodiscard]] bool registerAdapter(OptionalSceneUpscalerDescriptor descriptor,
                                           SceneUpscalerAdapterFactory factory);
        [[nodiscard]] std::vector<OptionalSceneUpscalerDescriptor> descriptors() const;
        [[nodiscard]] std::optional<OptionalSceneUpscalerDescriptor> descriptor(
            std::string_view id) const;
        [[nodiscard]] SceneUpscalerAvailability probe(
            std::string_view id, const SceneUpscalerProbeContext& context) const;
        [[nodiscard]] std::unique_ptr<SceneUpscalerAdapter> createAvailable(
            std::string_view id, const SceneUpscalerProbeContext& context) const;

    private:
        struct Registration {
            OptionalSceneUpscalerDescriptor descriptor;
            SceneUpscalerAdapterFactory factory = nullptr;
        };
        mutable std::shared_mutex mutex_;
        std::vector<Registration> registrations_;
    };

    [[nodiscard]] LFS_VIS_API OptionalSceneUpscalerRegistry& optionalSceneUpscalerRegistry();
    [[nodiscard]] LFS_VIS_API std::vector<SceneUpscalerCatalogEntry> buildSceneUpscalerCatalog(
        const OptionalSceneUpscalerRegistry& optional_registry,
        const SceneUpscalerProbeContext& context);
    [[nodiscard]] LFS_VIS_API std::vector<SceneUpscalerCatalogEntry>
    availableSceneUpscalerCatalog(const OptionalSceneUpscalerRegistry& optional_registry,
                                  const SceneUpscalerProbeContext& context);

    [[nodiscard]] LFS_VIS_API std::span<const SceneUpscalerDescriptor> sceneUpscalerDescriptors();
    [[nodiscard]] LFS_VIS_API const SceneUpscalerDescriptor& nativeSceneUpscalerDescriptor();
    [[nodiscard]] LFS_VIS_API const SceneUpscalerDescriptor& spatialSceneUpscalerDescriptor();
    [[nodiscard]] LFS_VIS_API const SceneUpscalerDescriptor& temporalSceneUpscalerDescriptor();
    [[nodiscard]] LFS_VIS_API const SceneUpscalerDescriptor& sceneUpscalerDescriptor(
        SceneUpscalerBackend backend);
    [[nodiscard]] LFS_VIS_API std::optional<SceneUpscalerBackend> sceneUpscalerBackendFromId(
        std::string_view id);
    [[nodiscard]] LFS_VIS_API SceneUpscalerSelection resolveSceneUpscalerSelection(
        SceneUpscalerBackend requested, bool adapter_available);

} // namespace lfs::vis
