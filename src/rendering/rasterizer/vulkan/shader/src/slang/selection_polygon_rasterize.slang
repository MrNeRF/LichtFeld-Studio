#include "config.slang"

struct PolygonRasterizeUniforms
{
    uint vertex_count;
    uint aabb_x0;
    uint aabb_y0;
    uint aabb_w;
    uint aabb_h;
    uint pad0;
    uint pad1;
    uint pad2;
}

layout(binding=0) StructuredBuffer<float2> polygon_vertices;
layout(binding=1) RWByteAddressBuffer coverage_mask;

static const uint kBlockX = 8u;
static const uint kBlockY = 8u;
static const uint kMaxSharedVertices = 4096u;
groupshared float2 shared_vertices[kMaxSharedVertices];

[shader("compute")]
[numthreads(kBlockX, kBlockY, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID,
          uint group_thread_index : SV_GroupIndex,
          uniform PolygonRasterizeUniforms uniforms)
{
    const uint V = uniforms.vertex_count;
    const uint shared_count = min(V, kMaxSharedVertices);
    for (uint i = group_thread_index; i < shared_count; i += kBlockX * kBlockY) {
        shared_vertices[i] = polygon_vertices[i];
    }
    GroupMemoryBarrierWithGroupSync();

    const uint local_x = dispatchThreadID.x;
    const uint local_y = dispatchThreadID.y;
    if (local_x >= uniforms.aabb_w || local_y >= uniforms.aabb_h)
        return;

    const float px = float(uniforms.aabb_x0 + local_x) + 0.5f;
    const float py = float(uniforms.aabb_y0 + local_y) + 0.5f;

    bool inside = false;
    uint j = V - 1u;
    for (uint i = 0u; i < V; ++i) {
        const float2 vi = (i < shared_count) ? shared_vertices[i] : polygon_vertices[i];
        const float2 vj = (j < shared_count) ? shared_vertices[j] : polygon_vertices[j];
        if ((vi.y > py) != (vj.y > py)) {
            const float x_intersect = (vj.x - vi.x) * (py - vi.y) / (vj.y - vi.y) + vi.x;
            if (px < x_intersect)
                inside = !inside;
        }
        j = i;
    }

    if (inside) {
        const uint idx = local_y * uniforms.aabb_w + local_x;
        const uint aligned = idx & ~3u;
        const uint shift = (idx & 3u) * 8u;
        coverage_mask.InterlockedOr(aligned, 0xFFu << shift);
    }
}
