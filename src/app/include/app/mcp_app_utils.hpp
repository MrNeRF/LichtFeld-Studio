/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "mcp/mcp_protocol.hpp"
#include "visualizer/visualizer.hpp"

#include <atomic>
#include <expected>
#include <functional>
#include <future>
#include <memory>
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

    template <typename F>
    auto post_and_wait(vis::Visualizer* viewer, F&& fn) {
        using R = std::invoke_result_t<F>;
        constexpr const char* shutdown_error = "Viewer is shutting down";

        if (viewer->isOnViewerThread()) {
            if (!viewer->acceptsPostedWork()) {
                return make_post_failure<R>(shutdown_error);
            }
            return std::invoke(std::forward<F>(fn));
        }

        auto task = std::make_shared<std::decay_t<F>>(std::forward<F>(fn));
        auto promise = std::make_shared<std::promise<R>>();
        auto completed = std::make_shared<std::atomic_bool>(false);
        auto future = promise->get_future();

        auto finish_with_value = [promise, completed](auto&& value) mutable {
            if (!completed->exchange(true)) {
                promise->set_value(std::forward<decltype(value)>(value));
            }
        };
        auto finish_with_exception = [promise, completed](std::exception_ptr error) {
            if (!completed->exchange(true)) {
                promise->set_exception(std::move(error));
            }
        };

        const bool posted = viewer->postWork(vis::Visualizer::WorkItem{
            .run =
                [task, finish_with_value, finish_with_exception]() mutable {
                    try {
                        finish_with_value(std::invoke(*task));
                    } catch (...) {
                        finish_with_exception(std::current_exception());
                    }
                },
            .cancel =
                [finish_with_value]() mutable {
                    finish_with_value(make_post_failure<R>(shutdown_error));
                }});

        if (!posted) {
            return make_post_failure<R>(shutdown_error);
        }

        return future.get();
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
