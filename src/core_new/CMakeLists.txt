# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

# ============================================================================
# core_new module - New LibTorch-free core module with lfs::core namespace
# ============================================================================

# Add tensor library
add_subdirectory(tensor)

message(STATUS "╔════════════════════════════════════════════════════════════════╗")
message(STATUS "║  core_new - LibTorch-free Core Module                         ║")
message(STATUS "╚════════════════════════════════════════════════════════════════╝")
message(STATUS "  • Namespace: lfs::core")
message(STATUS "  • Tensor library: lfs_tensor")
