// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
struct MaskParameters {
    Ptr<uint8_t, Access.Read> coverage;
    Ptr<uint8_t, Access.ReadWrite> mask;
    uint width;
    uint height;
};
[[vk::push_constant]] ConstantBuffer<MaskParameters> m;

[shader("compute")]
[numthreads(64, 1, 1)]
void intersectMask(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= m.width || tid.y >= m.height) return;
    uint i = (tid.y * m.width + tid.x) * 4;
    uint value = (uint(m.mask[i + 3]) * uint(m.coverage[i + 3]) + 127) / 255;
    m.mask[i] = m.mask[i+1] = m.mask[i+2] = m.mask[i+3] = uint8_t(value);
}
