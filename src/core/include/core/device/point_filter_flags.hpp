/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#define LFS_FILTER_NODES             1u
#define LFS_FILTER_BOX               2u
#define LFS_FILTER_ELLIPSOID         4u
#define LFS_FILTER_WINDOW            8u
#define LFS_FILTER_INVERSE_BOX       16u
#define LFS_FILTER_INVERSE_ELLIPSOID 32u
#define LFS_FILTER_GEOMETRY          (LFS_FILTER_BOX | LFS_FILTER_ELLIPSOID | LFS_FILTER_WINDOW)
