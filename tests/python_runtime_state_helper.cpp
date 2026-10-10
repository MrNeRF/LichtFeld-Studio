/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "python_runtime_state_helper.hpp"
#include "python/package_manager.hpp"
#include "python/runner.hpp"

void runtime_state_set_output(void (*callback)(const char*, bool)) {
    lfs::python::set_output_callback([callback](const std::string& text, bool error) {
        callback(text.c_str(), error);
    });
}
void runtime_state_write_output() { lfs::python::write_output("from helper", true); }
void runtime_state_set_scene_callback() {
    lfs::python::set_scene_time_callback([](float) {});
}
bool runtime_state_has_scene_callback() { return lfs::python::has_scene_time_callback(); }
void runtime_state_set_frame_callback() {
    lfs::python::set_frame_callback([](float) {}, 60.0);
}
bool runtime_state_has_frame_callback() { return lfs::python::has_frame_callback(); }
void runtime_state_force_init_failure() {
    lfs::python::force_python_init_failure_for_testing(true);
    (void)lfs::python::ensure_initialized();
}
int runtime_state_init_state() { return static_cast<int>(lfs::python::init_state().state); }
const void* runtime_state_package_manager() { return &lfs::python::PackageManager::instance(); }
