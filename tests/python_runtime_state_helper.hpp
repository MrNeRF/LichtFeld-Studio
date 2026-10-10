/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#ifdef LFS_TEST_PYTHON_STATE_EXPORTS
#define LFS_TEST_PYTHON_STATE_API __declspec(dllexport)
#else
#define LFS_TEST_PYTHON_STATE_API __declspec(dllimport)
#endif

extern "C" {
LFS_TEST_PYTHON_STATE_API void runtime_state_set_output(void (*callback)(const char*, bool));
LFS_TEST_PYTHON_STATE_API void runtime_state_write_output();
LFS_TEST_PYTHON_STATE_API void runtime_state_set_scene_callback();
LFS_TEST_PYTHON_STATE_API bool runtime_state_has_scene_callback();
LFS_TEST_PYTHON_STATE_API void runtime_state_set_frame_callback();
LFS_TEST_PYTHON_STATE_API bool runtime_state_has_frame_callback();
LFS_TEST_PYTHON_STATE_API void runtime_state_force_init_failure();
LFS_TEST_PYTHON_STATE_API int runtime_state_init_state();
LFS_TEST_PYTHON_STATE_API const void* runtime_state_package_manager();
}
