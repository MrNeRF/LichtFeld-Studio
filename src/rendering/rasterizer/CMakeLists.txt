# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

# Production Vulkan outside Apple; explicit comparison target for Mac tests.
if(NOT APPLE OR BUILD_TESTS)
    add_subdirectory(vulkan)
endif()

# Native Metal scene rendering; the desktop compositor remains Vulkan.
if(APPLE)
    add_subdirectory(metal)
endif()
