// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "rgb_to_yuv_math.h"
struct Parameters {
    Ptr<float, Access.Read> rgb;
    Ptr<uint8_t, Access.ReadWrite> y;
    Ptr<uint8_t, Access.ReadWrite> u;
    Ptr<uint8_t, Access.ReadWrite> v;
    uint width;
    uint height;
};
[[vk::push_constant]] ConstantBuffer<Parameters> p;
[shader("compute")]
[numthreads(16, 16, 1)]
void rgbToYuv420p(uint3 tid : SV_DispatchThreadID) {
    uint x = 2 * tid.x, y = 2 * tid.y;
    if (x >= p.width || y >= p.height)
        return;
    int rs = 0, gs = 0, bs = 0;
    bool invalidChroma = false;
    for (uint row = 0; row < 2; ++row) {
        for (uint col = 0; col < 2; ++col) {
            uint pixel = (y + row) * p.width + x + col;
            float rf = p.rgb[3 * pixel], gf = p.rgb[3 * pixel + 1], bf = p.rgb[3 * pixel + 2];
            bool invalid = isnan(rf) || isnan(gf) || isnan(bf);
            invalidChroma = invalidChroma || invalid;
            int r = colorRgbByte(rf), g = colorRgbByte(gf), b = colorRgbByte(bf);
            p.y[pixel] = uint8_t(invalid ? 0 : colorLuma(r, g, b));
            rs += r;
            gs += g;
            bs += b;
        }
    }
    uint chroma = tid.y * (p.width / 2) + tid.x;
    p.u[chroma] = uint8_t(invalidChroma ? 0 : colorChromaU(rs / 4, gs / 4, bs / 4));
    p.v[chroma] = uint8_t(invalidChroma ? 0 : colorChromaV(rs / 4, gs / 4, bs / 4));
}
