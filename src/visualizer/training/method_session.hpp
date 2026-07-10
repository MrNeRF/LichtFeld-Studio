/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <core/export.hpp>

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <expected>
#include <filesystem>
#include <functional>
#include <glm/glm.hpp>
#include <initializer_list>
#include <map>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace lfs::core {
    class Scene;
    class Tensor;

    namespace param {
        struct TrainingParameters;
    }
} // namespace lfs::core

namespace lfs::vis {

    enum class MethodCapability : std::uint32_t {
        FramePreview = 1U << 0U,
        HostCheckpointIO = 1U << 1U,
        HostEvaluation = 1U << 2U,
        GaussianSceneModel = 1U << 3U,
        GaussianParameterUI = 1U << 4U,
        GaussianEditing = 1U << 5U,
        LiveModelVkSplat = 1U << 6U,
        AppearanceCorrection = 1U << 7U,
    };

    struct LFS_VIS_API MethodCapabilities {
        std::uint32_t mask = 0;

        constexpr MethodCapabilities() = default;
        constexpr explicit MethodCapabilities(const std::uint32_t value) noexcept
            : mask(value) {}
        constexpr MethodCapabilities(const std::initializer_list<MethodCapability> values) noexcept {
            for (const auto value : values) {
                mask |= static_cast<std::uint32_t>(value);
            }
        }

        [[nodiscard]] constexpr bool has(const MethodCapability capability) const noexcept {
            return (mask & static_cast<std::uint32_t>(capability)) != 0;
        }
    };

    using MethodOptionValue = std::variant<bool, std::int64_t, double, std::string>;
    using MethodOptions = std::map<std::string, MethodOptionValue>;

    struct LFS_VIS_API MethodOptionSpec {
        enum class Type {
            Bool,
            Int,
            Float,
            String,
            Path,
            Enum,
        };

        std::string key;
        std::string display_name;
        std::string doc;
        Type type = Type::String;
        MethodOptionValue default_value = std::string{};
        std::optional<double> min_value;
        std::optional<double> max_value;
        std::vector<std::string> choices;
        bool restart_required = false;
    };

    struct LFS_VIS_API MethodDescriptor {
        std::string id;
        std::string display_name;
        std::string primitive_noun;
        MethodCapabilities capabilities;
        std::vector<MethodOptionSpec> options;
        std::string version;
    };

    struct LFS_VIS_API MethodStatus {
        int iteration = 0;
        int total_iterations = 0;
        float loss = 0.0F;
        std::optional<std::size_t> primitive_count;
        std::string phase;
    };

    // Camera pose follows rendering::FrameView: visualizer-space camera-to-world,
    // with the rotation columns representing camera right, up, and backward and
    // translation representing the camera position. The camera looks along local -Z.
    struct LFS_VIS_API CameraRenderRequest {
        glm::mat3 camera_to_world_rotation{1.0F};
        glm::vec3 camera_to_world_translation{0.0F};
        int width = 0;
        int height = 0;
        float focal_x = 0.0F;
        float focal_y = 0.0F;
        float principal_x = 0.0F;
        float principal_y = 0.0F;
        float near_plane = 0.01F;
        float far_plane = 100000.0F;
        bool orthographic = false;
        // Full vertical world-space extent. Ignored for perspective cameras.
        float ortho_size = 0.0F;
        glm::vec3 background_color{0.0F};
        // The method must enqueue all render work on this host-owned stream and
        // return tensors homed to it without synchronizing the stream.
        cudaStream_t stream = nullptr;
        // Monotonic for this consumer; changes whenever camera or output geometry changes.
        std::uint64_t view_generation = 0;
        bool allow_cached = false;
    };

    struct LFS_VIS_API CameraOutputs {
        // Required CUDA image. HWC or CHW, UInt8 or Float32; dimensions must
        // agree with width/height. Optional outputs use the same pixel size.
        std::shared_ptr<const lfs::core::Tensor> rgb;
        std::shared_ptr<const lfs::core::Tensor> depth;
        std::shared_ptr<const lfs::core::Tensor> alpha;
        int width = 0;
        int height = 0;
        bool flip_y = false;
        // Change only when fresh image content is returned. Cached responses
        // retain their previous generation so downstream uploads can be skipped.
        std::uint64_t content_generation = 0;
    };

    struct LFS_VIS_API MethodCreateContext {
        lfs::core::Scene* scene = nullptr;
        const lfs::core::param::TrainingParameters* params = nullptr;
        MethodOptions options;
        std::filesystem::path output_path;
        std::function<void()> request_redraw;
    };

    class LFS_VIS_API IMethodSession {
    public:
        virtual ~IMethodSession() = default;

        virtual std::expected<void, std::string> initialize(const MethodCreateContext& context) = 0;
        virtual void run(std::stop_token stop_token) = 0;
        virtual void request_pause() = 0;
        virtual void request_resume() = 0;
        [[nodiscard]] virtual bool is_paused() const = 0;
        // The GUI calls status concurrently with run(); implementations must return a thread-safe snapshot.
        [[nodiscard]] virtual MethodStatus status() const = 0;
        // Called from the render thread concurrently with run(). Sessions that
        // advertise FramePreview must implement this method and make it thread-safe.
        [[nodiscard]] virtual std::optional<CameraOutputs> render_camera(
            const CameraRenderRequest&) {
            return std::nullopt;
        }
        // May be called more than once across explicit clearing and host teardown;
        // implementations must make this idempotent.
        virtual void shutdown() = 0;
    };

    [[nodiscard]] LFS_VIS_API std::expected<MethodOptions, std::string> resolve_method_options(
        const MethodDescriptor& descriptor,
        const std::vector<std::pair<std::string, std::string>>& raw_options);

} // namespace lfs::vis
