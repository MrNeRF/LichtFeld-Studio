/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/utils/native_file_dialog.hpp"
#include "mcp/mcp_protocol.hpp"
#include "visualizer/post_work_utils.hpp"
#include "visualizer/visualizer.hpp"

#include "core/error.hpp"
#include "core/path_utils.hpp"
#include "core/scene.hpp"

#include <chrono>
#include <expected>
#include <filesystem>
#include <format>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lfs::app {

    class SceneNodeCounts {
    public:
        explicit SceneNodeCounts(const core::Scene& scene) : scene_(scene) {}

        // One live-count snapshot per MCP response, shared across its nodes.
        // Physical row counts remain unchanged for selection offsets and topology.
        size_t get(const core::SceneNode& node) {
            if (node.type == core::NodeType::SPLAT) {
                if (!active_counts_) {
                    active_counts_ = scene_.getActiveGaussianCountsByNode();
                }
                if (const auto it = active_counts_->find(node.id); it != active_counts_->end()) {
                    return it->second;
                }
            }
            return node.gaussian_count.load(std::memory_order_acquire);
        }

    private:
        const core::Scene& scene_;
        std::optional<std::unordered_map<core::NodeId, size_t>> active_counts_;
    };

    namespace detail {

        template <typename T>
        struct dependent_false : std::false_type {};

        template <typename T>
        struct is_string_expected : std::false_type {};

        template <typename T>
        struct is_string_expected<std::expected<T, std::string>> : std::true_type {};

        template <typename T>
        struct is_lfs_result : std::false_type {};

        template <typename T>
        struct is_lfs_result<lfs::Result<T>> : std::true_type {};

    } // namespace detail

    template <typename R>
    R make_post_failure(const std::string& error) {
        if constexpr (std::is_same_v<R, nlohmann::json>) {
            return nlohmann::json{{"error", error}};
        } else if constexpr (detail::is_string_expected<R>::value) {
            return std::unexpected(error);
        } else if constexpr (detail::is_lfs_result<R>::value) {
            auto typed_error = lfs::make_error(
                lfs::ErrorInit{
                    .code = lfs::ErrorCode::Unavailable,
                    .domain = lfs::ErrorDomain::MCP,
                    .severity = lfs::Severity::Error,
                    .retryability =
                        lfs::Retryability::NotRetryable,
                    .operation_id = {},
                    .user_message = error,
                    .detail =
                        "The GUI work queue rejected the MCP request",
                    .detection =
                        LFS_SOURCE_SITE_CURRENT(),
                    .fields = {},
                    .native = std::nullopt,
                });
            if constexpr (
                std::same_as<
                    typename R::value_type, void>) {
                return R::failure(
                    std::move(typed_error));
            } else {
                return R(std::move(typed_error));
            }
        } else {
            static_assert(detail::dependent_false<R>::value, "Unsupported post_and_wait return type");
        }
    }

    namespace detail {

        template <typename R>
        R make_native_dialog_failure() {
            constexpr std::string_view message =
                "This MCP operation requested a native file dialog. Use project_save_as, project_open, "
                "scene.load_ply, or scene.load_dataset with an explicit path instead.";
            if constexpr (std::is_same_v<R, nlohmann::json>) {
                return nlohmann::json{{"error", std::string(message)}};
            } else if constexpr (is_string_expected<R>::value) {
                return std::unexpected(std::string(message));
            } else if constexpr (is_lfs_result<R>::value) {
                auto error = lfs::make_error(lfs::ErrorInit{
                    .code = lfs::ErrorCode::FailedPrecondition,
                    .domain = lfs::ErrorDomain::MCP,
                    .severity = lfs::Severity::Error,
                    .retryability = lfs::Retryability::NotRetryable,
                    .operation_id = {},
                    .user_message = std::string(message),
                    .detail = "Native file dialogs cannot be answered by an MCP client",
                    .detection = LFS_SOURCE_SITE_CURRENT(),
                    .fields = {},
                    .native = std::nullopt,
                });
                if constexpr (std::same_as<typename R::value_type, void>)
                    return R::failure(std::move(error));
                else
                    return R(std::move(error));
            } else {
                return R{};
            }
        }

        template <typename F, typename PostFn>
        auto post_and_wait_impl(PostFn&& post_fn, F&& fn) {
            using R = std::invoke_result_t<F>;
            constexpr const char* shutdown_error = "Viewer is shutting down";
            return vis::post_work_and_wait(
                std::forward<PostFn>(post_fn),
                std::forward<F>(fn),
                [] { return make_post_failure<R>(shutdown_error); });
        }

    } // namespace detail

    template <typename F>
    auto post_and_wait(vis::Visualizer* viewer, F&& fn) {
        using R = std::invoke_result_t<F>;
        auto guarded = [work = std::forward<F>(fn)]() mutable -> R {
            vis::gui::ScopedNativeFileDialogBlock block;
            if constexpr (std::is_void_v<R>) {
                std::invoke(std::move(work));
            } else {
                R result = std::invoke(std::move(work));
                if (vis::gui::nativeFileDialogAttempted())
                    return detail::make_native_dialog_failure<R>();
                return result;
            }
        };

        if (viewer->isOnViewerThread()) {
            if (!viewer->acceptsPostedWork())
                return make_post_failure<R>("Viewer is shutting down");
            return std::invoke(std::move(guarded));
        }

        return detail::post_and_wait_impl(
            [viewer](vis::Visualizer::WorkItem work) { return viewer->postWork(std::move(work)); },
            std::move(guarded));
    }

    inline lfs::Result<vis::ProjectInfo>
    wait_for_project_generation(
        vis::Visualizer* viewer,
        const std::uint64_t previous_generation,
        const std::filesystem::path& expected_path,
        const bool require_newer_generation) {
        constexpr auto TIMEOUT =
            std::chrono::minutes(2);
        constexpr auto POLL_INTERVAL =
            std::chrono::milliseconds(25);
        const auto deadline =
            std::chrono::steady_clock::now() +
            TIMEOUT;
        std::uint64_t last_generation =
            previous_generation;
        do {
            auto poll = post_and_wait(
                viewer, [viewer] {
                    return viewer->projectPollWrite();
                });
            if (!poll) {
                return std::move(poll).error();
            }
            last_generation = poll->generation;
            if (!poll->running) {
                break;
            }
            std::this_thread::sleep_for(
                POLL_INTERVAL);
        } while (
            std::chrono::steady_clock::now() <
            deadline);
        auto latest = post_and_wait(
            viewer, [viewer] {
                return viewer->projectGetInfo();
            });
        if (!latest) {
            return std::move(latest).error();
        }
        last_generation = latest->generation;
        if (!latest->project_write_running &&
            !latest->project_write_error.empty()) {
            return lfs::make_error(
                lfs::ErrorInit{
                    .code = latest->project_write_error_code.value_or(
                        lfs::ErrorCode::Unavailable),
                    .domain = lfs::ErrorDomain::MCP,
                    .severity = lfs::Severity::Error,
                    .retryability = lfs::Retryability::NotRetryable,
                    .operation_id = {},
                    .user_message = "The project save failed.",
                    .detail = latest->project_write_error,
                    .detection = LFS_SOURCE_SITE_CURRENT(),
                    .fields = {},
                    .native = std::nullopt});
        }
        if (latest->path &&
            latest->path->lexically_normal() ==
                expected_path.lexically_normal() &&
            (!require_newer_generation ||
             latest->generation >
                 previous_generation)) {
            return latest;
        }
        return lfs::make_error(
            lfs::ErrorInit{
                .code =
                    lfs::ErrorCode::
                        DeadlineExceeded,
                .domain =
                    lfs::ErrorDomain::MCP,
                .severity =
                    lfs::Severity::Error,
                .retryability =
                    lfs::Retryability::
                        RetryableWithBackoff,
                .operation_id = {},
                .user_message =
                    "The project save did not publish before the MCP deadline.",
                .detail = std::format(
                    "Timed out waiting for the explicit .licht generation at '{}' (last generation {})",
                    core::path_to_utf8(
                        expected_path),
                    last_generation),
                .detection =
                    LFS_SOURCE_SITE_CURRENT(),
                .fields =
                    lfs::SmallFields{}
                        .add(
                            "path",
                            core::path_to_utf8(
                                expected_path))
                        .add(
                            "last_generation",
                            last_generation),
                .native = std::nullopt,
            });
    }

    inline lfs::Result<vis::ProjectInfo>
    wait_for_project_write(
        vis::Visualizer* viewer,
        const std::string_view operation) {
        constexpr auto TIMEOUT =
            std::chrono::minutes(2);
        constexpr auto POLL_INTERVAL =
            std::chrono::milliseconds(25);
        const auto deadline =
            std::chrono::steady_clock::now() +
            TIMEOUT;
        do {
            auto poll = post_and_wait(
                viewer, [viewer] {
                    return viewer->projectPollWrite();
                });
            if (!poll) {
                return std::move(poll).error();
            }
            if (!poll->running) {
                break;
            }
            std::this_thread::sleep_for(
                POLL_INTERVAL);
        } while (
            std::chrono::steady_clock::now() <
            deadline);
        auto latest = post_and_wait(
            viewer, [viewer] {
                return viewer->projectGetInfo();
            });
        if (!latest) {
            return std::move(latest).error();
        }
        if (!latest->project_write_running) {
            if (!latest->project_write_error.empty()) {
                return lfs::make_error(
                    lfs::ErrorInit{
                        .code = latest
                                    ->project_write_error_code
                                    .value_or(
                                        lfs::ErrorCode::Unavailable),
                        .domain =
                            lfs::ErrorDomain::
                                MCP,
                        .severity =
                            lfs::Severity::
                                Error,
                        .retryability =
                            lfs::Retryability::
                                NotRetryable,
                        .operation_id = {},
                        .user_message =
                            std::format(
                                "{} failed.",
                                operation),
                        .detail =
                            latest
                                ->project_write_error,
                        .detection =
                            LFS_SOURCE_SITE_CURRENT(),
                        .fields = {},
                        .native =
                            std::nullopt,
                    });
            }
            return latest;
        }
        return lfs::make_error(
            lfs::ErrorInit{
                .code =
                    lfs::ErrorCode::
                        DeadlineExceeded,
                .domain =
                    lfs::ErrorDomain::MCP,
                .severity =
                    lfs::Severity::Error,
                .retryability =
                    lfs::Retryability::
                        RetryableWithBackoff,
                .operation_id = {},
                .user_message =
                    std::format(
                        "{} did not finish before the MCP deadline.",
                        operation),
                .detail =
                    "Timed out waiting for the shared project-write job",
                .detection =
                    LFS_SOURCE_SITE_CURRENT(),
                .fields = {},
                .native = std::nullopt,
            });
    }

    inline std::expected<std::vector<mcp::McpResourceContent>, std::string> single_json_resource(
        const std::string& uri,
        nlohmann::json payload) {
        return std::vector<mcp::McpResourceContent>{
            mcp::McpResourceContent{
                .uri = uri,
                .mime_type = "application/json",
                .content = payload.dump(2)}};
    }

    inline std::expected<std::vector<mcp::McpResourceContent>, std::string> single_blob_resource(
        const std::string& uri,
        const std::string& mime_type,
        std::string base64_payload) {
        return std::vector<mcp::McpResourceContent>{
            mcp::McpResourceContent{
                .uri = uri,
                .mime_type = mime_type,
                .content = std::move(base64_payload)}};
    }

} // namespace lfs::app
