/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "mcp/mcp_protocol.hpp"
#include "visualizer/post_work_utils.hpp"
#include "visualizer/visualizer.hpp"

#include <expected>
#include <functional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace lfs::app {

    namespace detail {

        template <typename T>
        struct dependent_false : std::false_type {};

        template <typename T>
        struct is_string_expected : std::false_type {};

        template <typename T>
        struct is_string_expected<std::expected<T, std::string>> : std::true_type {};

    } // namespace detail

    template <typename R>
    R make_post_failure(const std::string& error) {
        if constexpr (std::is_same_v<R, nlohmann::json>) {
            return nlohmann::json{{"error", error}};
        } else if constexpr (detail::is_string_expected<R>::value) {
            return std::unexpected(error);
        } else {
            static_assert(detail::dependent_false<R>::value, "Unsupported post_and_wait return type");
        }
    }

    namespace detail {

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

        if (viewer->isOnViewerThread()) {
            if (!viewer->acceptsPostedWork())
                return make_post_failure<R>("Viewer is shutting down");
            return std::invoke(std::forward<F>(fn));
        }

        return detail::post_and_wait_impl(
            [viewer](vis::Visualizer::WorkItem work) { return viewer->postWork(std::move(work)); },
            std::forward<F>(fn));
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
