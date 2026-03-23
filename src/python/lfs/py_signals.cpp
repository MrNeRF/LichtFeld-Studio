/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "py_signals.hpp"

#include <chrono>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "core/logger.hpp"
#include "python/python_runtime.hpp"

namespace nb = nanobind;

namespace lfs::python {

    namespace {

        constexpr int THROTTLE_MS = 100;

        struct TrainingBuffer {
            std::chrono::steady_clock::time_point last_update{};
            bool dirty = false;

            int iteration = 0;
            float loss = 0.0f;
            std::size_t num_gaussians = 0;
        };

        TrainingBuffer g_training;
        nb::object g_app_state;
        bool g_initialized = false;

        template <typename T>
        void set_signal_value(const char* signal_name, T value) {
            if (!g_initialized || !g_app_state) {
                return;
            }

            try {
                nb::object signal = g_app_state.attr(signal_name);
                signal.attr("value") = value;
                request_redraw();
            } catch (const std::exception& e) {
                LOG_ERROR("Failed to set signal '{}': {}", signal_name, e.what());
            }
        }

        void do_training_progress(int iteration, float loss, std::size_t num_gaussians) {
            g_training.iteration = iteration;
            g_training.loss = loss;
            g_training.num_gaussians = num_gaussians;
            g_training.dirty = true;
        }

        void do_training_state(bool is_training, const char* state) {
            if (!g_initialized) {
                return;
            }

            nb::gil_scoped_acquire gil;

            set_signal_value("is_training", is_training);
            set_signal_value("trainer_state", state);
        }

        void do_trainer_loaded(const bool has_trainer, const int max_iterations) {
            if (!g_initialized) {
                return;
            }

            g_training = TrainingBuffer{};

            nb::gil_scoped_acquire gil;

            set_signal_value("has_trainer", has_trainer);
            set_signal_value("max_iterations", max_iterations);
            set_signal_value("iteration", 0);
            set_signal_value("loss", 0.0f);
            set_signal_value("num_gaussians", 0);
        }

        void do_psnr(float psnr) {
            if (!g_initialized) {
                return;
            }

            nb::gil_scoped_acquire gil;

            set_signal_value("psnr", psnr);
        }

        void do_scene(bool has_scene, const char* path) {
            if (!g_initialized) {
                return;
            }

            nb::gil_scoped_acquire gil;

            set_signal_value("has_scene", has_scene);
            set_signal_value("scene_path", path);

            try {
                nb::object gen_signal = g_app_state.attr("scene_generation");
                int current = nb::cast<int>(gen_signal.attr("value"));
                gen_signal.attr("value") = current + 1;
                request_redraw();
            } catch (const std::exception& e) {
                LOG_ERROR("Failed to increment scene_generation: {}", e.what());
            }
        }

        void do_selection(bool has_selection, int count) {
            if (!g_initialized) {
                return;
            }

            nb::gil_scoped_acquire gil;

            set_signal_value("has_selection", has_selection);
            set_signal_value("selection_count", count);

            try {
                nb::object gen_signal = g_app_state.attr("selection_generation");
                int current = nb::cast<int>(gen_signal.attr("value"));
                gen_signal.attr("value") = current + 1;
                request_redraw();
            } catch (const std::exception& e) {
                LOG_ERROR("Failed to increment selection_generation: {}", e.what());
            }
        }

        void do_flush() {
            if (!g_initialized || !g_training.dirty) {
                return;
            }

            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                               now - g_training.last_update)
                               .count();

            if (elapsed < THROTTLE_MS) {
                return;
            }

            nb::gil_scoped_acquire gil;

            set_signal_value("iteration", g_training.iteration);
            set_signal_value("loss", g_training.loss);
            set_signal_value("num_gaussians", static_cast<int>(g_training.num_gaussians));

            g_training.last_update = now;
            g_training.dirty = false;
        }

        void init_bridge() {
            if (g_initialized) {
                return;
            }

            nb::gil_scoped_acquire gil;

            try {
                nb::module_ ui_module = nb::module_::import_("lfs_plugins.ui.state");
                g_app_state = ui_module.attr("AppState");
                g_initialized = true;

                SignalBridgeCallbacks callbacks{};
                callbacks.flush = do_flush;
                callbacks.training_progress = do_training_progress;
                callbacks.training_state = do_training_state;
                callbacks.trainer_loaded = do_trainer_loaded;
                callbacks.psnr = do_psnr;
                callbacks.scene = do_scene;
                callbacks.selection = do_selection;
                set_signal_bridge_callbacks(callbacks);

                LOG_INFO("Signal bridge initialized");
            } catch (const std::exception& e) {
                LOG_ERROR("Failed to initialize signal bridge: {}", e.what());
                g_initialized = false;
            }
        }

        void shutdown_bridge() {
            if (!g_initialized) {
                return;
            }

            SignalBridgeCallbacks empty{};
            set_signal_bridge_callbacks(empty);

            nb::gil_scoped_acquire gil;

            try {
                g_app_state.attr("reset")();
            } catch (const std::exception& e) {
                LOG_ERROR("Failed to reset AppState during shutdown: {}", e.what());
            }

            g_app_state = nb::object();
            g_initialized = false;
            g_training = TrainingBuffer{};
            LOG_INFO("Signal bridge shutdown");
        }

    } // namespace

    void shutdown_signal_bridge() {
        shutdown_bridge();
    }

    void register_signals(nb::module_& m) {
        auto signals = m.def_submodule("signals", "Signal bridge for reactive UI updates");

        signals.def("init", &init_bridge, "Initialize the signal bridge");
        signals.def("shutdown", &shutdown_bridge, "Shutdown the signal bridge");
    }

} // namespace lfs::python
