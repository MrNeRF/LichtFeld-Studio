/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <core/export.hpp>

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <functional>
#include <initializer_list>
#include <map>
#include <optional>
#include <stop_token>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace lfs::core {
    class Scene;

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
        // May be called more than once across explicit clearing and host teardown;
        // implementations must make this idempotent.
        virtual void shutdown() = 0;
    };

    [[nodiscard]] LFS_VIS_API std::expected<MethodOptions, std::string> resolve_method_options(
        const MethodDescriptor& descriptor,
        const std::vector<std::pair<std::string, std::string>>& raw_options);

} // namespace lfs::vis
