// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
#include "gsplat_common.slangh"
struct GsplatBinParams {
    Ptr<int64_t, Access.ReadWrite> counts, ends;
    Ptr<int, Access.ReadWrite> radii;
    Ptr<float, Access.ReadWrite> means2d;
    Ptr<uint, Access.ReadWrite> depths;
    Ptr<uint64_t, Access.ReadWrite> keys;
    Ptr<uint, Access.ReadWrite> ids, offsets;
    uint count, tiles_x, tiles_y;
    Ptr<uint, Access.ReadWrite> control;
};
struct GsplatRasterParams {
    Ptr<float, Access.ReadWrite> camera;
    Ptr<float, Access.ReadWrite> means, scales, quats, opacities, colors;
    Ptr<float, Access.ReadWrite> bg_color, bg_image;
    Ptr<uint, Access.ReadWrite> offsets, ids;
    Ptr<float, Access.ReadWrite> image, alpha;
    Ptr<int, Access.ReadWrite> last;
    Ptr<float, Access.ReadWrite> depth;
    Ptr<uint, Access.ReadWrite> depths;
    uint count, width, height, tiles_x, tiles_y, mode;
    Ptr<uint, Access.ReadWrite> control;
};
struct ControlParams {
    Ptr<int64_t, Access.Read> ends;
    Ptr<uint, Access.ReadWrite> control;
    uint count, capacity, tiles, max_groups;
};
struct ScanParams {
    Ptr<int64_t, Access.ReadWrite> input, output, totals;
    uint count;
};
struct Push {
    uint64_t parameters;
};
[[vk::push_constant]] Push push;
[SpecializationConstant] const uint stage = 0;
groupshared float4 m0[64], m1[64], m2[64], colors[64];
groupshared float opacity[64];
groupshared uint finished[256];
groupshared int64_t prefix[256];
void raster(GsplatRasterParams p, uint tile, uint lane) {
    uint x = (tile % p.tiles_x) * 16 + lane % 16;
    uint y = (tile / p.tiles_x) * 16 + lane / 16;
    bool inside = x < p.width && y < p.height;
    // Match the synchronous empty-frame contract, including background.
    if (p.control != 0 && p.control[0] == 0) {
        if (inside) {
            uint pixel = y * p.width + x, plane = p.width * p.height;
            p.alpha[pixel] = 0;
            p.last[pixel] = 0;
            if (p.image != 0)
                for (uint c = 0; c < 3; ++c) p.image[c * plane + pixel] = 0;
            if (p.depth != 0) p.depth[pixel] = 0;
        }
        return;
    }
    const GsplatCamera camera = gsplat_camera_load(p.camera);
    GsplatRay ray = gsplat_pixel_ray(camera, float2(x, y) + 0.5f);
    bool done = !inside || !ray.valid;
    uint begin = p.offsets[tile], end = p.offsets[tile + 1];
    float T = 1, z = 0;
    float3 rgb = 0;
    int last = 0;
    for (uint batch = begin; batch < end; batch += 64) {
        // Publish one completion flag per subgroup instead of reducing 256
        // flags through eight workgroup barriers for every batch.
        const bool wave_done = WaveActiveAllTrue(done);
        if (WaveIsFirstLane())
            finished[lane / WaveGetLaneCount()] = wave_done ? 1u : 0u;
        GroupMemoryBarrierWithGroupSync();
        uint all_done = 1;
        for (uint wave = 0; wave < (256 + WaveGetLaneCount() - 1) / WaveGetLaneCount(); ++wave)
            all_done &= finished[wave];
        if (all_done != 0)
            break;
        if (lane < 64 && batch + lane < end) {
            uint g = p.ids[batch + lane];
            float3x3 M = gsplat_world_to_unit(gsplat_load4(p.quats, g), exp(gsplat_load3(p.scales, g)));
            // Global-shutter rays share their origin across the whole tile.
            const float3 gro = mul(camera.origin - gsplat_load3(p.means, g), M);
            opacity[lane] = gsplat_sigmoid(p.opacities[g]);
            m0[lane] = float4(M[0], gro.x);
            m1[lane] = float4(M[1], gro.y);
            m2[lane] = float4(M[2], gro.z);
            colors[lane] = float4(gsplat_load3(p.colors, g), asfloat(p.depths[g]));
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint t = 0; t < min(64u, end - batch) && !done; ++t) {
            float3x3 M = float3x3(m0[t].xyz, m1[t].xyz, m2[t].xyz);
            const float3 gro = float3(m0[t].w, m1[t].w, m2[t].w);
            float3 grd = gsplat_safe_normalize(mul(ray.dir, M));
            float3 crossed = cross(grd, gro);
            float a = min(0.999f, opacity[t] * exp(-0.5f * dot(crossed, crossed)));
            if (a < kGsplatAlphaMin)
                continue;
            float next = T * (1 - a);
            if (next <= 1e-4f) {
                done = true;
                break;
            }
            rgb += colors[t].xyz * (a * T);
            z += colors[t].w * (a * T);
            last = int(batch + t);
            T = next;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (!inside)
        return;
    uint pixel = y * p.width + x, plane = p.width * p.height;
    p.alpha[pixel] = 1 - T;
    p.last[pixel] = last;
    if (p.image != 0) {
        for (uint c = 0; c < 3; ++c) {
            float bg = p.bg_image != 0 ? p.bg_image[c * plane + pixel] : p.bg_color != 0 ? p.bg_color[c]
                                                                                         : 0;
            p.image[c * plane + pixel] = rgb[c] + T * bg;
        }
    }
    if (p.depth != 0)
        p.depth[pixel] = (p.mode == 2 || p.mode == 4) ? z / max(1 - T, 1e-10f) : z;
}
[numthreads(256, 1, 1)] void main(uint3 gid : SV_GroupID, uint lane : SV_GroupIndex) {
    uint3 grid = spirv_asm {
result:
        $$uint3 = OpLoad builtin(NumWorkgroups : uint3);
    };
    if (stage == 0) {
        if (lane == 0 && gid.x == 0)
            gsplat_camera_setup(((Ptr<GsplatSetupParams, Access.Read>)push.parameters)[0]);
    } else if (stage == 1) {
        GsplatProjectParams p = ((Ptr<GsplatProjectParams, Access.Read>)push.parameters)[0];
        for (uint i = gid.x * 256 + lane; i < p.count; i += grid.x * 256)
            gsplat_project(p, i);
    } else if (stage == 2 || stage == 3) {
        GsplatBinParams p = ((Ptr<GsplatBinParams, Access.Read>)push.parameters)[0];
        if (stage == 3 && p.control != 0) p.count = p.control[0];
        for (uint i = gid.x * 256 + lane; i < p.count; i += grid.x * 256) {
            if (stage == 2) {
                if (p.counts[i] == 0)
                    continue;
                GsplatTileRect rect = gsplat_tile_rect(float2(p.means2d[i * 2], p.means2d[i * 2 + 1]), int2(p.radii[i * 2], p.radii[i * 2 + 1]), p.tiles_x, p.tiles_y);
                uint cursor = uint(p.ends[i] - p.counts[i]);
                for (uint y = rect.y0; y < rect.y1; ++y)
                    for (uint x = rect.x0; x < rect.x1; ++x) {
                        p.keys[cursor] = (uint64_t(y * p.tiles_x + x) << 32) | p.depths[i];
                        p.ids[cursor++] = i;
                    }
            } else {
                uint current = uint(p.keys[i] >> 32);
                uint previous = i == 0 ? 0 : uint(p.keys[i - 1] >> 32) + 1;
                for (uint t = previous; t <= current; ++t)
                    p.offsets[t] = i;
                if (i + 1 == p.count)
                    for (uint t = current + 1; t <= p.tiles_x * p.tiles_y; ++t)
                        p.offsets[t] = p.count;
            }
        }
    } else if (stage == 4 || stage == 5) {
        ScanParams p = ((Ptr<ScanParams, Access.Read>)push.parameters)[0];
        for (uint block = gid.x; block * 256 < p.count; block += grid.x) {
            uint i = block * 256 + lane;
            if (stage == 5) {
                if (i < p.count && block != 0)
                    p.output[i] += p.totals[block - 1];
                continue;
            }
            prefix[lane] = i < p.count ? p.input[i] : 0;
            GroupMemoryBarrierWithGroupSync();
            for (uint step = 1; step < 256; step <<= 1) {
                int64_t add = lane >= step ? prefix[lane - step] : 0;
                GroupMemoryBarrierWithGroupSync();
                prefix[lane] += add;
                GroupMemoryBarrierWithGroupSync();
            }
            if (i < p.count)
                p.output[i] = prefix[lane];
            if (lane == 255)
                p.totals[block] = prefix[lane];
            GroupMemoryBarrierWithGroupSync();
        }
    } else if (stage == 7) {
        if (lane != 0 || gid.x != 0) return;
        ControlParams p = ((Ptr<ControlParams, Access.Read>)push.parameters)[0];
        int64_t total = p.count != 0 ? p.ends[p.count - 1] : 0;
        bool invalid = total < 0 || total > 0x7fffffff;
        bool overflow = invalid || total > int64_t(p.capacity);
        uint n = overflow ? 0 : uint(total);
        p.control[0] = n;
        p.control[1] = invalid ? 2u : overflow ? 1u : 0u;
        p.control[2] = uint(total);
        uint partitions = (n + 2047u) / 2048u;
        p.control[4] = n != 0 ? min((p.count + 255u) / 256u, p.max_groups) : 0u;
        p.control[8] = partitions;
        p.control[12] = min((n + 255u) / 256u, p.max_groups);
        p.control[16] = n != 0 ? 256u : 0u;
        p.control[20] = overflow ? 0u : min(p.tiles, p.max_groups);
        p.control[28] = n != 0 ? 1u : 0u;
        for (uint word = 4; word <= 28; word += 4) {
            p.control[word + 1] = 1;
            p.control[word + 2] = 1;
        }
    } else {
        GsplatRasterParams p = ((Ptr<GsplatRasterParams, Access.Read>)push.parameters)[0];
        for (uint tile = gid.x; tile < p.tiles_x * p.tiles_y; tile += grid.x)
            raster(p, tile, lane);
    }
}
