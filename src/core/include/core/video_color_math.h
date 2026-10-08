// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "color_transfer.h"
#ifdef __cplusplus
#include <algorithm>
#include <cmath>
#include <cstdint>
namespace lfs::core::color {
    using uint = uint32_t;
    using lfs::core::color::bt709ToLinear;
    using lfs::core::color::srgbToLinear;
    using std::ceil;
    using std::floor;
    using std::max;
    using std::min;
#endif
    struct VideoLinearRgb {
        float channel[3];
    };
    struct VideoColorComponent {
        uint offset, pitch, step, shift, depth, width, height;
    };
    struct VideoColorParameters {
        VideoColorComponent component[3];
        uint width, height, rgb, big_endian, transfer;
        float chroma_x_scale, chroma_x_offset, chroma_y_scale, chroma_y_offset;
        float decode[9];
        float bias[3];
        float scale[3];
    };
// The reader supplies byte loads; layout/range/chroma/curve/resize math is the
// same source on the portable CPU and CUDA/Vulkan/Metal tensor backends.
#ifdef __cplusplus
    template <class Reader>
#endif
    struct VideoColorSampler {
        VideoColorParameters parameters;
        Reader reader;
        float code(uint channel, int x, int y) {
            VideoColorComponent c = parameters.component[channel];
            x = max(0, min(x, int(c.width) - 1));
            y = max(0, min(y, int(c.height) - 1));
            uint address = c.offset + uint(y) * c.pitch + uint(x) * c.step;
            uint first = reader.load(address);
            uint value = first;
            uint bytes = (c.depth + c.shift + 7u) / 8u;
            for (uint i = 1u; i < bytes; ++i)
                value = parameters.big_endian != 0u ? (value << 8u) | reader.load(address + i) : value | (reader.load(address + i) << (8u * i));
            return float((value >> c.shift) & ((1u << c.depth) - 1u));
        }
        float component(uint channel, int x, int y) {
            if (channel == 0u || parameters.rgb != 0u)
                return code(channel, x, y);
            float sx = parameters.component[channel].width == parameters.width ? float(x) : float(x) * parameters.chroma_x_scale + parameters.chroma_x_offset;
            float sy = parameters.component[channel].height == parameters.height ? float(y) : float(y) * parameters.chroma_y_scale + parameters.chroma_y_offset;
            int x0 = int(floor(sx)), y0 = int(floor(sy));
            float fx = sx - float(x0), fy = sy - float(y0);
            float a = code(channel, x0, y0), b = code(channel, x0 + 1, y0);
            float c = code(channel, x0, y0 + 1), d = code(channel, x0 + 1, y0 + 1);
            return (a + (b - a) * fx) + ((c + (d - c) * fx) - (a + (b - a) * fx)) * fy;
        }
        VideoLinearRgb linearRgb(int x, int y) {
            float a = component(0u, x, y) * parameters.scale[0] - parameters.bias[0];
            float b = component(1u, x, y) * parameters.scale[1] - parameters.bias[1];
            float c = component(2u, x, y) * parameters.scale[2] - parameters.bias[2];
            VideoLinearRgb result;
            for (uint channel = 0u; channel < 3u; ++channel) {
                float value = parameters.decode[channel * 3u] * a + parameters.decode[channel * 3u + 1u] * b + parameters.decode[channel * 3u + 2u] * c;
                result.channel[channel] = parameters.transfer == 1u ? srgbToLinear(value) : parameters.transfer == 2u ? bt709ToLinear(value)
                                                                                                                      : value;
            }
            return result;
        }
        // Area prefilter for reduction, pixel-center bilinear for enlargement.
        // Sampling operates in linear light, without clipping or RGB8 intermediates.
        VideoLinearRgb resizedRgb(uint x, uint y, uint width, uint height) {
            if (width == parameters.width && height == parameters.height)
                return linearRgb(int(x), int(y));
            float rx = float(parameters.width) / float(width), ry = float(parameters.height) / float(height);
            bool down_x = rx > 1.0f, down_y = ry > 1.0f;
            float left = down_x ? float(x) * rx : (float(x) + 0.5f) * rx - 0.5f;
            float top = down_y ? float(y) * ry : (float(y) + 0.5f) * ry - 0.5f;
            float right = down_x ? float(x + 1u) * rx : left + 1.0f;
            float bottom = down_y ? float(y + 1u) * ry : top + 1.0f;
            int first_x = int(floor(left)), first_y = int(floor(top));
            int last_x = down_x ? int(ceil(right)) - 1 : first_x + 1;
            int last_y = down_y ? int(ceil(bottom)) - 1 : first_y + 1;
            VideoLinearRgb result;
            for (uint c = 0u; c < 3u; ++c)
                result.channel[c] = 0.0f;
            float weight = 0.0f;
            for (int sy = first_y; sy <= last_y; ++sy) {
                float wy = down_y ? min(bottom, float(sy + 1)) - max(top, float(sy)) : (sy == first_y ? 1.0f - (top - float(first_y)) : top - float(first_y));
                for (int sx = first_x; sx <= last_x; ++sx) {
                    float wx = down_x ? min(right, float(sx + 1)) - max(left, float(sx)) : (sx == first_x ? 1.0f - (left - float(first_x)) : left - float(first_x));
                    VideoLinearRgb pixel = linearRgb(sx, sy);
                    for (uint c = 0u; c < 3u; ++c)
                        result.channel[c] += wx * wy * pixel.channel[c];
                    weight += wx * wy;
                }
            }
            for (uint c = 0u; c < 3u; ++c)
                result.channel[c] /= weight;
            return result;
        }
    };
#ifdef __cplusplus
} // namespace lfs::core::color
#endif
