# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Production Vulkan unless the native Metal viewer is built; then it is an
# explicit comparison target for Mac tests.
if(LFS_USE_VULKAN AND (NOT LFS_TENSOR_METAL OR BUILD_TESTS))
    add_subdirectory(vulkan)
endif()

# Native Metal scene rendering, for either compositor. It reads model
# tensors through the tensor library; its hand-written MSL kernels are slated
# for the single-source Slang migration shared with the Vulkan rasterizer.
if(LFS_TENSOR_METAL)
    add_subdirectory(metal)
endif()
