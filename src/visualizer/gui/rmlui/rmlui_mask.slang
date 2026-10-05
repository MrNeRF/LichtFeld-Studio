// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
struct MaskParameters {
    Ptr<uint8_t, Access.Read> coverage;
    Ptr<uint8_t, Access.ReadWrite> mask;
    uint target_width;
    uint x;
    uint y;
    uint width;
    uint height;
    uint clear_value;
};
[[vk::push_constant]] ConstantBuffer<MaskParameters> m;

[shader("compute")]
[numthreads(64, 1, 1)]
void intersectMask(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= m.width || tid.y >= m.height) return;
    uint i = ((m.y + tid.y) * m.target_width + m.x + tid.x) * 4;
    uint value = (uint(m.mask[i + 3]) * uint(m.coverage[i + 3]) + 127) / 255;
    m.mask[i] = m.mask[i+1] = m.mask[i+2] = m.mask[i+3] = uint8_t(value);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void clearRegion(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= m.width || tid.y >= m.height) return;
    uint i = ((m.y + tid.y) * m.target_width + m.x + tid.x) * 4;
    m.mask[i] = m.mask[i+1] = m.mask[i+2] = m.mask[i+3] = uint8_t(m.clear_value);
}
