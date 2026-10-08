// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
struct Reader {
    Ptr<uint8_t, Access.Read> bytes;
    uint load(uint address) { return bytes[address]; }
};
#include "../media/video_color_math.h"
struct Parameters {
    Ptr<uint8_t, Access.Read> planes;
    Ptr<VideoColorParameters, Access.Read> color;
    Ptr<float, Access.ReadWrite> output;
    uint width, height;
};
[[vk::push_constant]] ConstantBuffer<Parameters> p;
[shader("compute")]
[numthreads(8, 8, 1)]
void linearVideo(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= p.width || tid.y >= p.height) return;
    VideoColorSampler sampler;
    sampler.parameters = p.color[0]; sampler.reader.bytes = p.planes;
    for (uint c = 0; c < 3; ++c)
        p.output[(tid.y * p.width + tid.x) * 3 + c] = sampler.resized(c, tid.x, tid.y, p.width, p.height);
}
