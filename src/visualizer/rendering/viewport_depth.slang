// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// Seeds the mesh depth attachment with the splats' depth so meshes are hidden
// behind nearer splats, like resources/viewport/depth_blit.frag. The splat
// depth is linear view depth; empty pixels carry a value >= 1e9.
struct Parameters {
    Ptr<float, Access.Read> splat;
    Ptr<float, Access.ReadWrite> depth;
    uint width;
    uint height;
    int viewportX;
    int viewportY;
    uint viewportWidth;
    uint viewportHeight;
    uint splatWidth;
    uint splatHeight;
    float nearPlane;
    float farPlane;
    uint hasSplat;
    uint orthographic;
};
[[vk::push_constant]] ConstantBuffer<Parameters> p;

[shader("compute")]
[numthreads(64, 1, 1)]
void seedDepth(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= p.width || tid.y >= p.height)
        return;
    int2 pixel = int2(tid.xy) - int2(p.viewportX, p.viewportY);
    float ndc = 1.0;
    if (p.hasSplat != 0 && pixel.x >= 0 && pixel.y >= 0 &&
        pixel.x < int(p.viewportWidth) && pixel.y < int(p.viewportHeight)) {
        uint2 q = uint2((float2(pixel) + 0.5) * float2(p.splatWidth, p.splatHeight) /
                        float2(p.viewportWidth, p.viewportHeight));
        q = min(q, uint2(p.splatWidth - 1u, p.splatHeight - 1u));
        float z = p.splat[q.y * p.splatWidth + q.x];
        // Zero-to-one depth of glm::perspective / glm::ortho (GLM_FORCE_DEPTH_ZERO_TO_ONE).
        if (z > 0.0 && z < 1.0e9)
            ndc = saturate(p.orthographic != 0
                               ? (z - p.nearPlane) / (p.farPlane - p.nearPlane)
                               : p.farPlane / (p.farPlane - p.nearPlane) * (1.0 - p.nearPlane / max(z, p.nearPlane)));
    }
    p.depth[tid.y * p.width + tid.x] = ndc;
}
