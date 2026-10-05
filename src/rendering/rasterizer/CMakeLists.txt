# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Production Vulkan unless the Metal viewer is built; then it is an explicit
# comparison target for Mac tests.
if(LFS_USE_VULKAN AND (NOT LFS_TENSOR_METAL OR BUILD_TESTS))
    add_subdirectory(vulkan)
endif()

# Single-source Slang rasterizer through the tensor library (Metal and Vulkan).
if(LFS_TENSOR_METAL OR LFS_TENSOR_VULKAN)
    add_subdirectory(tensor)
endif()
