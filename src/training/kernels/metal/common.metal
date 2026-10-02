// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// Trainer kernels for the Metal backend. Each kernel takes one parameter block
// at buffer(0) whose pointers are tensor device addresses. The files that
// follow are concatenated into one library in the order CMake lists them.

#include <metal_stdlib>
using namespace metal;

constant constexpr uint kThreadgroupWidth = 256;
