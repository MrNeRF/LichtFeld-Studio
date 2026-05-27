/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_store.hpp"

#include "core/logger.hpp"
#include "python/gil.hpp"
#include "visualizer/app_store.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace nb = nanobind;

namespace lfs::python {

    namespace {
        struct PyStoreSubscription {
            nb::object callback;
            lfs::core::reactive::SubscriptionToken token;
        };

        std::mutex g_subscriptions_mutex;
        std::unordered_map<std::uint64_t, std::shared_ptr<PyStoreSubscription>> g_subscriptions;
        std::atomic_uint64_t g_next_subscription_id{1};

        template <typename Observable>
        std::uint64_t subscribe_observable(Observable& observable, nb::object callback) {
            const auto id = g_next_subscription_id.fetch_add(1, std::memory_order_relaxed);
            auto subscription = std::make_shared<PyStoreSubscription>();
            subscription->callback = std::move(callback);
            const std::weak_ptr<PyStoreSubscription> weak_subscription = subscription;
            subscription->token = observable.subscribe([weak_subscription](const auto& value) {
                const auto subscription = weak_subscription.lock();
                if (!subscription)
                    return;
                if (!can_acquire_gil())
                    return;
                const GilAcquire gil;
                try {
                    subscription->callback(value);
                } catch (const nb::python_error& e) {
                    LOG_ERROR("Python app store subscriber failed: {}", e.what());
                } catch (const std::exception& e) {
                    LOG_ERROR("Python app store subscriber failed: {}", e.what());
                }
            });

            {
                std::lock_guard lock(g_subscriptions_mutex);
                g_subscriptions.emplace(id, std::move(subscription));
            }
            return id;
        }

        [[noreturn]] void throw_unknown_field(const std::string_view field) {
            throw std::invalid_argument(std::string("Unknown app store field: ") + std::string(field));
        }

        void set_field(const std::string& field, const nb::object& value) {
            auto& store = lfs::vis::app_store();
            if (field == "iteration")
                store.iteration.set(nb::cast<int>(value));
            else if (field == "total_iterations")
                store.total_iterations.set(nb::cast<int>(value));
            else if (field == "loss")
                store.loss.set(nb::cast<float>(value));
            else if (field == "num_gaussians")
                store.num_gaussians.set(nb::cast<std::int64_t>(value));
            else if (field == "max_gaussians")
                store.max_gaussians.set(nb::cast<std::int64_t>(value));
            else if (field == "training_running")
                store.training_running.set(nb::cast<bool>(value));
            else if (field == "training_state")
                store.training_state.set(nb::cast<std::string>(value));
            else if (field == "trainer_loaded")
                store.trainer_loaded.set(nb::cast<bool>(value));
            else if (field == "eval_psnr")
                store.eval_psnr.set(value.is_none() ? std::optional<float>{}
                                                    : std::optional<float>{nb::cast<float>(value)});
            else if (field == "eval_ssim")
                store.eval_ssim.set(value.is_none() ? std::optional<float>{}
                                                    : std::optional<float>{nb::cast<float>(value)});
            else if (field == "scene_generation")
                store.scene_generation.set(nb::cast<std::uint64_t>(value));
            else if (field == "selection_generation")
                store.selection_generation.set(nb::cast<std::uint64_t>(value));
            else if (field == "fps")
                store.fps.set(nb::cast<float>(value));
            else if (field == "mode_text")
                store.mode_text.set(nb::cast<std::string>(value));
            else if (field == "active_tool")
                store.active_tool.set(nb::cast<std::string>(value));
            else if (field == "active_submode")
                store.active_submode.set(nb::cast<std::string>(value));
            else if (field == "transform_space")
                store.transform_space.set(nb::cast<int>(value));
            else if (field == "pivot_mode")
                store.pivot_mode.set(nb::cast<int>(value));
            else
                throw_unknown_field(field);
        }

        nb::object get_field(const std::string& field) {
            auto& store = lfs::vis::app_store();
            if (field == "iteration")
                return nb::cast(store.iteration.get());
            if (field == "total_iterations")
                return nb::cast(store.total_iterations.get());
            if (field == "loss")
                return nb::cast(store.loss.get());
            if (field == "num_gaussians")
                return nb::cast(store.num_gaussians.get());
            if (field == "max_gaussians")
                return nb::cast(store.max_gaussians.get());
            if (field == "training_running")
                return nb::cast(store.training_running.get());
            if (field == "training_state")
                return nb::cast(store.training_state.get());
            if (field == "trainer_loaded")
                return nb::cast(store.trainer_loaded.get());
            if (field == "eval_psnr")
                return nb::cast(store.eval_psnr.get());
            if (field == "eval_ssim")
                return nb::cast(store.eval_ssim.get());
            if (field == "scene_generation")
                return nb::cast(store.scene_generation.get());
            if (field == "selection_generation")
                return nb::cast(store.selection_generation.get());
            if (field == "fps")
                return nb::cast(store.fps.get());
            if (field == "mode_text")
                return nb::cast(store.mode_text.get());
            if (field == "active_tool")
                return nb::cast(store.active_tool.get());
            if (field == "active_submode")
                return nb::cast(store.active_submode.get());
            if (field == "transform_space")
                return nb::cast(store.transform_space.get());
            if (field == "pivot_mode")
                return nb::cast(store.pivot_mode.get());
            throw_unknown_field(field);
        }

        std::uint64_t subscribe_field(const std::string& field, nb::object callback) {
            auto& store = lfs::vis::app_store();
            if (field == "iteration")
                return subscribe_observable(store.iteration, std::move(callback));
            if (field == "total_iterations")
                return subscribe_observable(store.total_iterations, std::move(callback));
            if (field == "loss")
                return subscribe_observable(store.loss, std::move(callback));
            if (field == "num_gaussians")
                return subscribe_observable(store.num_gaussians, std::move(callback));
            if (field == "max_gaussians")
                return subscribe_observable(store.max_gaussians, std::move(callback));
            if (field == "training_running")
                return subscribe_observable(store.training_running, std::move(callback));
            if (field == "training_state")
                return subscribe_observable(store.training_state, std::move(callback));
            if (field == "trainer_loaded")
                return subscribe_observable(store.trainer_loaded, std::move(callback));
            if (field == "eval_psnr")
                return subscribe_observable(store.eval_psnr, std::move(callback));
            if (field == "eval_ssim")
                return subscribe_observable(store.eval_ssim, std::move(callback));
            if (field == "scene_generation")
                return subscribe_observable(store.scene_generation, std::move(callback));
            if (field == "selection_generation")
                return subscribe_observable(store.selection_generation, std::move(callback));
            if (field == "fps")
                return subscribe_observable(store.fps, std::move(callback));
            if (field == "mode_text")
                return subscribe_observable(store.mode_text, std::move(callback));
            if (field == "active_tool")
                return subscribe_observable(store.active_tool, std::move(callback));
            if (field == "active_submode")
                return subscribe_observable(store.active_submode, std::move(callback));
            if (field == "transform_space")
                return subscribe_observable(store.transform_space, std::move(callback));
            if (field == "pivot_mode")
                return subscribe_observable(store.pivot_mode, std::move(callback));
            throw_unknown_field(field);
        }

        void unsubscribe_field(const std::uint64_t token) {
            std::lock_guard lock(g_subscriptions_mutex);
            g_subscriptions.erase(token);
        }

        void begin_batch() {
            lfs::vis::app_store().store().begin_batch();
        }

        void end_batch() {
            lfs::vis::app_store().store().end_batch();
        }
    } // namespace

    void shutdown_store_bridge() {
        if (!can_acquire_gil())
            return;

        const GilAcquire gil;
        std::unordered_map<std::uint64_t, std::shared_ptr<PyStoreSubscription>> subscriptions;
        {
            std::lock_guard lock(g_subscriptions_mutex);
            subscriptions.swap(g_subscriptions);
        }
    }

    void register_store(nb::module_& ui_module) {
        auto store = ui_module.def_submodule("store", "Reactive C++ app store bridge");
        store.def("set", &set_field, "Set an app store field");
        store.def("get", &get_field, "Get an app store field");
        store.def("subscribe", &subscribe_field, "Subscribe to an app store field");
        store.def("unsubscribe", &unsubscribe_field, "Unsubscribe from an app store field");
        store.def("begin_batch", &begin_batch, "Begin a batched app store update");
        store.def("end_batch", &end_batch, "End a batched app store update");
    }

} // namespace lfs::python
