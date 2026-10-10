/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#ifdef _WIN32
#ifdef LFS_PYTHON_RUNTIME_EXPORTS
#define LFS_PYTHON_RUNTIME_API __declspec(dllexport)
#else
#define LFS_PYTHON_RUNTIME_API __declspec(dllimport)
#endif
#else
#define LFS_PYTHON_RUNTIME_API
#endif
