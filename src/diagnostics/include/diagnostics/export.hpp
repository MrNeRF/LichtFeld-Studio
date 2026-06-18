/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#ifdef _WIN32
#ifdef LFS_DIAGNOSTICS_EXPORTS
#define LFS_DIAGNOSTICS_API __declspec(dllexport)
#else
#define LFS_DIAGNOSTICS_API __declspec(dllimport)
#endif
#else
#define LFS_DIAGNOSTICS_API __attribute__((visibility("default")))
#endif
