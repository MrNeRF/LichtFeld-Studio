/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_store.hpp"

#include "core/logger.hpp"
#include "py_error.hpp"
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
        thread_local bool g_test_drain_with_current_gil = false;

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
                const bool can_acquire = can_acquire_gil();
                const bool use_current_gil = !can_acquire && g_test_drain_with_current_gil && PyGILState_Check();
                if (!can_acquire && !use_current_gil)
                    return;
                try {
                    if (can_acquire) {
                        const GilAcquire gil;
                        subscription->callback(value);
                    } else {
                        subscription->callback(value);
                    }
                } catch (nb::python_error& e) {
                    // Containment needs the GIL to extract and report; if it is
                    // gone the interpreter is dead or finalizing, and swallowing
                    // here is deliberate — the pre-Phase 9 e.what() log ran
                    // GIL-unsafe and was itself a shutdown crash class.
                    if (can_acquire_gil()) {
                        const GilAcquire gil;
                        (void)contain_python_callback(e, PyCallbackPolicy::WarnAndContinue);
                    }
                } catch (const std::exception& e) {
                    (void)contain_cxx_callback(e.what(), PyCallbackPolicy::WarnAndContinue);
                }
            });

            {
                std::lock_guard lock(g_subscriptions_mutex);
                g_subscriptions.emplace(id, std::move(subscription));
            }
            return id;
        }

        template <typename Observable, typename Convert>
        std::uint64_t subscribe_observable_as(Observable& observable, nb::object callback, Convert convert) {
            const auto id = g_next_subscription_id.fetch_add(1, std::memory_order_relaxed);
            auto subscription = std::make_shared<PyStoreSubscription>();
            subscription->callback = std::move(callback);
            const std::weak_ptr<PyStoreSubscription> weak_subscription = subscription;
            subscription->token = observable.subscribe([weak_subscription, convert = std::move(convert)](const auto& value) {
                const auto subscription = weak_subscription.lock();
                if (!subscription)
                    return;
                const bool can_acquire = can_acquire_gil();
                const bool use_current_gil = !can_acquire && g_test_drain_with_current_gil && PyGILState_Check();
                if (!can_acquire && !use_current_gil)
                    return;
                try {
                    if (can_acquire) {
                        const GilAcquire gil;
                        subscription->callback(convert(value));
                    } else {
                        subscription->callback(convert(value));
                    }
                } catch (nb::python_error& e) {
                    // Containment needs the GIL to extract and report; if it is
                    // gone the interpreter is dead or finalizing, and swallowing
                    // here is deliberate — the pre-Phase 9 e.what() log ran
                    // GIL-unsafe and was itself a shutdown crash class.
                    if (can_acquire_gil()) {
                        const GilAcquire gil;
                        (void)contain_python_callback(e, PyCallbackPolicy::WarnAndContinue);
                    }
                } catch (const std::exception& e) {
                    (void)contain_cxx_callback(e.what(), PyCallbackPolicy::WarnAndContinue);
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

        template <typename T>
        T dict_value(const nb::dict& dict, const char* key, T fallback) {
            const auto py_key = nb::str(key);
            if (!dict.contains(py_key))
                return fallback;
            return nb::cast<T>(dict[py_key]);
        }

        nb::dict import_overlay_state_to_dict(const lfs::vis::AppStore::ImportOverlayState& value) {
            nb::dict state;
            state["active"] = value.active;
            state["show_completion"] = value.show_completion;
            state["progress"] = value.progress;
            state["stage"] = value.stage;
            state["dataset_type"] = value.dataset_type;
            state["path"] = value.path;
            state["success"] = value.success;
            state["error"] = value.error;
            state["num_images"] = value.num_images;
            state["num_points"] = value.num_points;
            state["seconds_since_completion"] = value.seconds_since_completion;
            return state;
        }

        lfs::vis::AppStore::ImportOverlayState import_overlay_state_from_object(const nb::object& value) {
            if (value.is_none())
                return {};
            if (!nb::isinstance<nb::dict>(value))
                throw nb::type_error("import_overlay_state must be a dict");

            const nb::dict dict = nb::cast<nb::dict>(value);
            lfs::vis::AppStore::ImportOverlayState state;
            state.active = dict_value(dict, "active", false);
            state.show_completion = dict_value(dict, "show_completion", false);
            state.progress = dict_value(dict, "progress", 0.0f);
            state.stage = dict_value(dict, "stage", std::string{});
            state.dataset_type = dict_value(dict, "dataset_type", std::string{});
            state.path = dict_value(dict, "path", std::string{});
            state.success = dict_value(dict, "success", false);
            state.error = dict_value(dict, "error", std::string{});
            state.num_images = dict_value(dict, "num_images", std::uint64_t{0});
            state.num_points = dict_value(dict, "num_points", std::uint64_t{0});
            state.seconds_since_completion = dict_value(dict, "seconds_since_completion", 0.0f);
            return state;
        }

        nb::dict video_export_overlay_state_to_dict(const lfs::vis::AppStore::VideoExportOverlayState& value) {
            nb::dict state;
            state["active"] = value.active;
            state["progress"] = value.progress;
            state["current_frame"] = value.current_frame;
            state["total_frames"] = value.total_frames;
            state["stage"] = value.stage;
            return state;
        }

        lfs::vis::AppStore::VideoExportOverlayState video_export_overlay_state_from_object(const nb::object& value) {
            if (value.is_none())
                return {};
            if (!nb::isinstance<nb::dict>(value))
                throw nb::type_error("video_export_overlay_state must be a dict");

            const nb::dict dict = nb::cast<nb::dict>(value);
            lfs::vis::AppStore::VideoExportOverlayState state;
            state.active = dict_value(dict, "active", false);
            state.progress = dict_value(dict, "progress", 0.0f);
            state.current_frame = dict_value(dict, "current_frame", 0);
            state.total_frames = dict_value(dict, "total_frames", 0);
            state.stage = dict_value(dict, "stage", std::string{});
            return state;
        }

        nb::dict export_progress_state_to_dict(const lfs::vis::AppStore::ExportProgressState& value) {
            nb::dict state;
            state["active"] = value.active;
            state["progress"] = value.progress;
            state["stage"] = value.stage;
            state["format"] = value.format;
            state["error"] = value.error;
            state["path"] = value.path;
            return state;
        }

        lfs::vis::AppStore::ExportProgressState export_progress_state_from_object(const nb::object& value) {
            if (value.is_none())
                return {};
            if (!nb::isinstance<nb::dict>(value))
                throw nb::type_error("export_progress_state must be a dict");

            const nb::dict dict = nb::cast<nb::dict>(value);
            lfs::vis::AppStore::ExportProgressState state;
            state.active = dict_value(dict, "active", false);
            state.progress = dict_value(dict, "progress", 0.0f);
            state.stage = dict_value(dict, "stage", std::string{});
            state.format = dict_value(dict, "format", std::string{});
            state.error = dict_value(dict, "error", std::string{});
            state.path = dict_value(dict, "path", std::string{});
            return state;
        }

        nb::dict task_progress_state_to_dict(const lfs::vis::AppStore::TaskProgressState& value) {
            nb::dict state;
            state["active"] = value.active;
            state["progress"] = value.progress;
            state["stage"] = value.stage;
            state["error"] = value.error;
            state["source_name"] = value.source_name;
            state["output_name"] = value.output_name;
            return state;
        }

        lfs::vis::AppStore::TaskProgressState task_progress_state_from_object(const nb::object& value) {
            if (value.is_none())
                return {};
            if (!nb::isinstance<nb::dict>(value))
                throw nb::type_error("task progress state must be a dict");

            const nb::dict dict = nb::cast<nb::dict>(value);
            lfs::vis::AppStore::TaskProgressState state;
            state.active = dict_value(dict, "active", false);
            state.progress = dict_value(dict, "progress", 0.0f);
            state.stage = dict_value(dict, "stage", std::string{});
            state.error = dict_value(dict, "error", std::string{});
            state.source_name = dict_value(dict, "source_name", std::string{});
            state.output_name = dict_value(dict, "output_name", std::string{});
            return state;
        }

        void set_field(const std::string& field, nb::object value) {
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
            else if (field == "multi_transform_mode")
                store.multi_transform_mode.set(nb::cast<int>(value));
            else if (field == "import_overlay_state")
                store.import_overlay_state.set(import_overlay_state_from_object(value));
            else if (field == "video_export_overlay_state")
                store.video_export_overlay_state.set(video_export_overlay_state_from_object(value));
            else if (field == "export_progress_state")
                store.export_progress_state.set(export_progress_state_from_object(value));
            else if (field == "mesh2splat_state")
                store.mesh2splat_state.set(task_progress_state_from_object(value));
            else if (field == "splat_simplify_state")
                store.splat_simplify_state.set(task_progress_state_from_object(value));
            else if (field == "scripts_generation")
                store.scripts_generation.set(nb::cast<std::uint64_t>(value));
            else if (field == "language_generation")
                store.language_generation.set(nb::cast<std::uint64_t>(value));
            else if (field == "render_settings_generation")
                store.render_settings_generation.set(nb::cast<std::uint64_t>(value));
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
            if (field == "multi_transform_mode")
                return nb::cast(store.multi_transform_mode.get());
            if (field == "import_overlay_state")
                return import_overlay_state_to_dict(store.import_overlay_state.get());
            if (field == "video_export_overlay_state")
                return video_export_overlay_state_to_dict(store.video_export_overlay_state.get());
            if (field == "export_progress_state")
                return export_progress_state_to_dict(store.export_progress_state.get());
            if (field == "mesh2splat_state")
                return task_progress_state_to_dict(store.mesh2splat_state.get());
            if (field == "splat_simplify_state")
                return task_progress_state_to_dict(store.splat_simplify_state.get());
            if (field == "scripts_generation")
                return nb::cast(store.scripts_generation.get());
            if (field == "language_generation")
                return nb::cast(store.language_generation.get());
            if (field == "render_settings_generation")
                return nb::cast(store.render_settings_generation.get());
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
            if (field == "multi_transform_mode")
                return subscribe_observable(store.multi_transform_mode, std::move(callback));
            if (field == "import_overlay_state")
                return subscribe_observable_as(
                    store.import_overlay_state, std::move(callback), import_overlay_state_to_dict);
            if (field == "video_export_overlay_state")
                return subscribe_observable_as(
                    store.video_export_overlay_state, std::move(callback), video_export_overlay_state_to_dict);
            if (field == "export_progress_state")
                return subscribe_observable_as(
                    store.export_progress_state, std::move(callback), export_progress_state_to_dict);
            if (field == "mesh2splat_state")
                return subscribe_observable_as(
                    store.mesh2splat_state, std::move(callback), task_progress_state_to_dict);
            if (field == "splat_simplify_state")
                return subscribe_observable_as(
                    store.splat_simplify_state, std::move(callback), task_progress_state_to_dict);
            if (field == "scripts_generation")
                return subscribe_observable(store.scripts_generation, std::move(callback));
            if (field == "language_generation")
                return subscribe_observable(store.language_generation, std::move(callback));
            if (field == "render_settings_generation")
                return subscribe_observable(store.render_settings_generation, std::move(callback));
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

        bool drain_for_tests() {
            struct ScopedTestDrain {
                bool previous;
                explicit ScopedTestDrain(const bool enabled)
                    : previous(g_test_drain_with_current_gil) {
                    g_test_drain_with_current_gil = enabled;
                }
                ~ScopedTestDrain() {
                    g_test_drain_with_current_gil = previous;
                }
            };

            const ScopedTestDrain allow_current_gil(true);
            return lfs::vis::app_store().store().drain_dirty_into_frame();
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
        store.def("set", &set_field, nb::arg("field"), nb::arg("value").none(), "Set an app store field");
        store.def("get", &get_field, "Get an app store field");
        store.def("subscribe", &subscribe_field, "Subscribe to an app store field");
        store.def("unsubscribe", &unsubscribe_field, "Unsubscribe from an app store field");
        store.def("begin_batch", &begin_batch, "Begin a batched app store update");
        store.def("end_batch", &end_batch, "End a batched app store update");
        store.def("_drain_for_tests", &drain_for_tests, "Drain pending app store notifications in tests");
    }

} // namespace lfs::python
