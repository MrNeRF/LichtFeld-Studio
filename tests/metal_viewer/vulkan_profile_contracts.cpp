/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "config.h"
// Compiled both ways on the same Mac: __APPLE__ must not change production.
#ifdef LFS_VULKAN_MACOS_REFERENCE
static_assert(RASTER_BATCH_SIZE == 256);
static_assert(RADIX_WORKGROUP_SIZE == 256);
static_assert(RADIX_PARTITION_SIZE == 2048);
#else
static_assert(RASTER_BATCH_SIZE == 1024);
static_assert(RADIX_WORKGROUP_SIZE == 512);
static_assert(RADIX_PARTITION_SIZE == 4096);
#endif
int main() { return 0; }
