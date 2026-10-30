/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

template <typename T>
struct PixelTraits;

template <>
struct PixelTraits<uint8_t> {
    __device__ __forceinline__ static float load_and_normalize(const uint8_t* ptr, int idx) {
        return static_cast<float>(ptr[idx]) / 255.0f;
    }
};

template <>
struct PixelTraits<float> {
    __device__ __forceinline__ static float load_and_normalize(const float* ptr, int idx) {
        return ptr[idx];
    }
};
