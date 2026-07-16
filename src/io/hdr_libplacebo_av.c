/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

/* libplacebo's FFmpeg adapter is header-only by design. It must have exactly
 * one C translation unit defining PL_LIBAV_IMPLEMENTATION=1; C++ users then
 * include the declarations with PL_LIBAV_IMPLEMENTATION=0. */
#define PL_LIBAV_IMPLEMENTATION 1
#include <vulkan/vulkan.h>
#include <libplacebo/utils/libav.h>
