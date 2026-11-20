/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

// CUDA builds take Common.h as before. Host builds of Cameras.cuh need only the camera types.
#ifdef __CUDACC__
#include "Common.h"
#endif
#include "CameraTypes.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <limits>
#include <variant>
