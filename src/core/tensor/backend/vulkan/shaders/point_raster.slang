// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

// Point-cloud rasterization, ported from the renderer's point_cloud_raster.cu.
// Each point splats a disk of pixels; CUDA packs (depth, color) into one
// 64-bit atomic min, which here takes two 32-bit passes with the same winner:
// kPhase 1 keeps each pixel's nearest depth, kPhase 2 the smallest color among
// the points at that depth. kPhase 0 clears and kPhase 3 writes the image.
static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 128;
[SpecializationConstant] const uint kPhase = 0;

static const uint kCropBox = 1u, kCropEllipsoid = 2u, kCropInverse = 4u, kCropDesaturate = 8u;
static const uint kEquirectangular = 16u, kOrthographic = 32u, kTransparent = 64u;
static const uint kEmpty = 0xffffffffu;

#include "byte_read.slangh"

struct Parameters {
    uint64_t positions, colors, parameters, scratch, image, depth, transforms, indices, visibility, deleted;
    uint count, width, height, channels, transformCount, visibilityCount, flags;
    float orthoScale, focalY, voxelSize, farPlane;
    uint padding;
};

// Row `row` of a column-major 4x4 matrix at `m` times (x, y, z, 1).
float rasterRow(Ptr<float, Access.Read> m, int row, float3 v) {
    return m[row] * v.x + m[row + 4] * v.y + m[row + 8] * v.z + m[row + 12];
}

struct Splat {
    int x, y, radius;
    uint depth, color;
};

// Projects point i; false when it is hidden, cropped out or off screen.
bool splatOf(Parameters p, uint i, out Splat splat) {
    splat = {};
    if (p.deleted != 0 && loadByte(p.deleted, i) != 0) return false;
    const int transform = p.indices != 0 ? clamp(((Ptr<int, Access.Read>)p.indices)[i], 0, max(int(p.transformCount) - 1, 0)) : 0;
    if (p.visibility != 0 && uint(transform) < p.visibilityCount && loadByte(p.visibility, uint(transform)) == 0) return false;
    Ptr<float, Access.Read> positions = (Ptr<float, Access.Read>)p.positions;
    float3 v = float3(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
    if (p.transforms != 0 && p.transformCount > 0) {
        Ptr<float, Access.Read> m = (Ptr<float, Access.Read>)p.transforms + transform * 16;
        const float nw = rasterRow(m, 3, v);
        v = float3(rasterRow(m, 0, v), rasterRow(m, 1, v), rasterRow(m, 2, v)) * (abs(nw) > 1e-6f ? 1.0f / nw : 1.0f);
    }
    Ptr<float, Access.Read> parameters = (Ptr<float, Access.Read>)p.parameters;
    bool desaturate = false;
    if ((p.flags & (kCropBox | kCropEllipsoid)) != 0) {
        Ptr<float, Access.Read> toLocal = parameters + 32;
        const float3 local = float3(rasterRow(toLocal, 0, v), rasterRow(toLocal, 1, v), rasterRow(toLocal, 2, v));
        const float3 low = float3(parameters[48], parameters[49], parameters[50]);
        const float3 high = float3(parameters[51], parameters[52], parameters[53]);
        bool inside;
        if ((p.flags & kCropBox) != 0) {
            inside = all(local >= low) && all(local <= high);
        } else {
            const float3 radii = max(abs(low), float3(1e-8f));
            inside = dot(local * local, 1.0f / (radii * radii)) <= 1.0f;
        }
        if (inside == ((p.flags & kCropInverse) != 0)) {
            if ((p.flags & kCropDesaturate) == 0) return false;
            desaturate = true;
        }
    }
    const float3 view = float3(rasterRow(parameters, 0, v), rasterRow(parameters, 1, v), rasterRow(parameters, 2, v));
    const bool orthographic = (p.flags & kOrthographic) != 0;
    float2 pixel;
    float depth;
    if ((p.flags & kEquirectangular) != 0) {
        const float len = length(view);
        if (len <= 1e-6f) return false;
        const float pi = 3.14159265358979323846f;
        const float u = 0.5f + atan2(view.x / len, -view.z / len) / (2.0f * pi);
        const float w = 0.5f - asin(clamp(view.y / len, -1.0f, 1.0f)) / pi;
        pixel = float2(u * float(p.width - 1), w * float(p.height - 1));
        if (!isfinite(pixel.x) || !isfinite(pixel.y) || pixel.x < 0.0f || pixel.x >= float(p.width) ||
            pixel.y < 0.0f || pixel.y >= float(p.height)) return false;
        depth = len;
    } else {
        Ptr<float, Access.Read> viewProjection = parameters + 16;
        const float cw = rasterRow(viewProjection, 3, v);
        if (abs(cw) <= 1e-6f) return false;
        const float3 ndc = float3(rasterRow(viewProjection, 0, v), rasterRow(viewProjection, 1, v), rasterRow(viewProjection, 2, v)) / cw;
        if (!all(isfinite(ndc)) || ndc.x < -1.0f || ndc.x > 1.0f || ndc.y < -1.0f || ndc.y > 1.0f || ndc.z < 0.0f || ndc.z > 1.0f) return false;
        pixel = float2((ndc.x * 0.5f + 0.5f) * float(p.width - 1), (0.5f - ndc.y * 0.5f) * float(p.height - 1));
        depth = orthographic ? -view.z : max(-view.z, 0.0f);
        if (depth <= 0.0f && !orthographic) return false;
    }
    const float voxel = max(p.voxelSize, 1e-5f);
    splat.radius = orthographic ? max(1, int(ceil(voxel * float(p.height) / max(p.orthoScale, 1e-5f) * 0.5f)))
                                : max(1, int(ceil(voxel * p.focalY / max(depth, 1e-4f))));
    Ptr<float, Access.Read> colors = (Ptr<float, Access.Read>)p.colors;
    float3 color = clamp(float3(colors[i * 3], colors[i * 3 + 1], colors[i * 3 + 2]), 0.0f, 1.0f);
    if (desaturate) color += (dot(color, float3(0.299f, 0.587f, 0.114f)) - color) * 0.75f;
    const uint3 bytes = uint3(clamp(color, 0.0f, 1.0f) * 255.0f + 0.5f);
    splat.color = bytes.x | (bytes.y << 8) | (bytes.z << 16);
    splat.depth = asuint(max(depth, 0.0f));
    splat.x = int(round(pixel.x));
    splat.y = int(round(pixel.y));
    return true;
}

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID, uniform Parameters p) {
    const uint i = tid.x;
    const uint pixels = p.width * p.height;
    Ptr<uint, Access.ReadWrite> scratch = (Ptr<uint, Access.ReadWrite>)p.scratch;
    if (kPhase == 0 || kPhase == 3) {
        if (i >= pixels) return;
        if (kPhase == 0) {
            scratch[i] = kEmpty;
            scratch[pixels + i] = kEmpty;
            return;
        }
        Ptr<float, Access.Read> background = (Ptr<float, Access.Read>)p.parameters + 54;
        Ptr<float, Access.ReadWrite> image = (Ptr<float, Access.ReadWrite>)p.image;
        const uint depthBits = scratch[i];
        const bool empty = depthBits == kEmpty;
        const uint color = scratch[pixels + i];
        for (uint c = 0; c < 3; ++c)
            image[c * pixels + i] = empty ? background[c] : float((color >> (8 * c)) & 0xffu) / 255.0f;
        if (p.channels == 4)
            image[3 * pixels + i] = empty && (p.flags & kTransparent) != 0 ? 0.0f : 1.0f;
        ((Ptr<float, Access.ReadWrite>)p.depth)[i] = empty ? p.farPlane : asfloat(depthBits);
        return;
    }
    Splat splat;
    if (i >= p.count || !splatOf(p, i, splat)) return;
    const int radiusSquared = splat.radius * splat.radius;
    for (int dy = -splat.radius; dy <= splat.radius; ++dy) {
        const int y = splat.y + dy;
        if (y < 0 || y >= int(p.height)) continue;
        for (int dx = -splat.radius; dx <= splat.radius; ++dx) {
            const int x = splat.x + dx;
            if (x < 0 || x >= int(p.width) || dx * dx + dy * dy > radiusSquared) continue;
            const uint pixel = uint(y) * p.width + uint(x);
            if (kPhase == 1)
                InterlockedMin(scratch[pixel], splat.depth);
            else if (scratch[pixel] == splat.depth)
                InterlockedMin(scratch[pixels + pixel], splat.color);
        }
    }
}
