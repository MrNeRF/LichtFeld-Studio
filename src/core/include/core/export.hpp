/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#ifdef _WIN32
#ifdef LFS_LOGGER_EXPORTS
#define LFS_LOGGER_API __declspec(dllexport)
#else
#define LFS_LOGGER_API __declspec(dllimport)
#endif
#ifdef LFS_CORE_EXPORTS
#define LFS_CORE_API __declspec(dllexport)
#else
#define LFS_CORE_API __declspec(dllimport)
#endif
#define LFS_IO_API
#ifdef LFS_VIS_EXPORTS
#define LFS_VIS_API __declspec(dllexport)
#else
#define LFS_VIS_API __declspec(dllimport)
#endif
#ifdef LFS_MCP_EXPORTS
#define LFS_MCP_API __declspec(dllexport)
#else
#define LFS_MCP_API __declspec(dllimport)
#endif
#else
#define LFS_LOGGER_API __attribute__((visibility("default")))
#define LFS_CORE_API   __attribute__((visibility("default")))
#define LFS_IO_API     __attribute__((visibility("default")))
#define LFS_VIS_API    __attribute__((visibility("default")))
#define LFS_MCP_API    __attribute__((visibility("default")))
#endif
