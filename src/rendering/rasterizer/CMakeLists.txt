# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

if(NOT APPLE OR BUILD_TESTS)
    add_subdirectory(vulkan)
endif()

# Native Metal viewer stages; desktop presentation remains Vulkan until parity.
if(APPLE)
    add_subdirectory(metal)
endif()
