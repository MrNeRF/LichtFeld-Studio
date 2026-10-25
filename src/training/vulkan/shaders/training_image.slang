[SpecializationConstant] const uint operation = 0;
[SpecializationConstant] const uint imageIsBytes = 0;

struct Parameters {
    uint64_t input, output, secondary, tertiary;
    uint64_t camera, rays, depth, alpha;
    uint count, width, height, sourceWidth;
    uint sourceHeight, channels, slot, slotCount;
    float fx, fy, cx, cy;
    float outsideWeight, emaAlpha, skipBelow, scalar0;
    float4 cropRows[3];
    float4 cropMin;
    uint inverse;
    float4 cropMax;
    uint seed;
};

static float3 transformPoint(Parameters p, float3 v) {
    return float3(dot(p.cropRows[0].xyz, v) + p.cropRows[0].w,
                  dot(p.cropRows[1].xyz, v) + p.cropRows[1].w,
                  dot(p.cropRows[2].xyz, v) + p.cropRows[2].w);
}
static float3 transformDirection(Parameters p, float3 v) {
    return float3(dot(p.cropRows[0].xyz, v), dot(p.cropRows[1].xyz, v), dot(p.cropRows[2].xyz, v));
}
static bool rayHitsBox(float3 origin, float3 direction, float3 lo, float3 hi) {
    float enter = -3.402823466e+38f, leave = 3.402823466e+38f;
    for (uint axis = 0; axis < 3; ++axis) {
        if (abs(direction[axis]) < 1.0e-8f) {
            if (origin[axis] < lo[axis] || origin[axis] > hi[axis]) return false;
        } else {
            float inverseDirection = 1.0f / direction[axis];
            float a = (lo[axis] - origin[axis]) * inverseDirection;
            float b = (hi[axis] - origin[axis]) * inverseDirection;
            if (a > b) { float t = a; a = b; b = t; }
            enter = max(enter, a); leave = min(leave, b);
            if (leave < enter) return false;
        }
    }
    return leave >= 0.0f;
}
// Crop polygon ROI (operation 1 with slot = vertex count): the outline, x and y pairs at
// secondary, in the crop frame over |z| <= cropMax.z; cropMin and cropMax bound it. Mirrors
// the CUDA polygon kernel in kernels/roi_weight_map.cu.
static float2 polygonVertex(Parameters p, uint i) {
    Ptr<float, Access.Read> v = (Ptr<float, Access.Read>)p.secondary;
    return float2(v[2u * i], v[2u * i + 1u]);
}
static bool outlineContains(Parameters p, float2 q) {
    bool inside = false;
    float2 b = polygonVertex(p, p.slot - 1u);
    for (uint i = 0; i < p.slot; ++i) {
        float2 a = polygonVertex(p, i);
        if ((a.y > q.y) != (b.y > q.y) && q.x < (b.x - a.x) * (q.y - a.y) / (b.y - a.y) + a.x) inside = !inside;
        b = a;
    }
    return inside;
}
static float cross2(float2 a, float2 b) { return a.x * b.y - a.y * b.x; }
// The ray (t >= 0) clipped to the slab and the outline bounds must leave a segment that
// touches the outline: an endpoint inside it, or a crossing with one of its edges.
static bool rayHitsPrism(Parameters p, float3 origin, float3 direction) {
    float enter = 0.0f, leave = 3.402823466e+38f;
    for (uint axis = 0; axis < 3; ++axis) {
        if (abs(direction[axis]) < 1.0e-8f) {
            if (origin[axis] < p.cropMin[axis] || origin[axis] > p.cropMax[axis]) return false;
        } else {
            float inverseDirection = 1.0f / direction[axis];
            float a = (p.cropMin[axis] - origin[axis]) * inverseDirection;
            float b = (p.cropMax[axis] - origin[axis]) * inverseDirection;
            if (a > b) { float t = a; a = b; b = t; }
            enter = max(enter, a); leave = min(leave, b);
            if (leave < enter) return false;
        }
    }
    if (leave == 3.402823466e+38f) leave = enter;
    float2 p0 = origin.xy + enter * direction.xy, p1 = origin.xy + leave * direction.xy;
    if (outlineContains(p, p0) || outlineContains(p, p1)) return true;
    float2 d = p1 - p0;
    float2 a = polygonVertex(p, p.slot - 1u);
    for (uint i = 0; i < p.slot; ++i) {
        float2 b = polygonVertex(p, i), e = b - a;
        if (((cross2(d, a - p0) > 0.0f) != (cross2(d, b - p0) > 0.0f)) &&
            ((cross2(e, p0 - a) > 0.0f) != (cross2(e, p1 - a) > 0.0f))) return true;
        a = b;
    }
    return false;
}
static uint pcgHash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
static float grayAt(Parameters p, int x, int y) {
    x = clamp(x, 0, int(p.width) - 1); y = clamp(y, 0, int(p.height) - 1);
    uint at = uint(y) * p.width + uint(x), plane = p.width * p.height;
    Ptr<float, Access.Read> f = (Ptr<float, Access.Read>)p.input;
    Ptr<uint8_t, Access.Read> b = (Ptr<uint8_t, Access.Read>)p.input;
    float r = imageIsBytes != 0 ? float(b[at]) * (1.0f / 255.0f) : f[at];
    float g = imageIsBytes != 0 ? float(b[plane + at]) * (1.0f / 255.0f) : f[plane + at];
    float bl = imageIsBytes != 0 ? float(b[2u * plane + at]) * (1.0f / 255.0f) : f[2u * plane + at];
    return 0.299f * r + 0.587f * g + 0.114f * bl;
}
static bool maskKeeps(Parameters p, int x, int y) {
    if (p.channels == 0) return true;
    x = clamp(x, 0, int(p.width) - 1); y = clamp(y, 0, int(p.height) - 1);
    uint at = uint(y) * p.width + uint(x);
    return ((Ptr<float, Access.Read>)p.tertiary)[at] > 0.0f;
}
groupshared float grayTile[576];
groupshared float blurTile[400];
groupshared float2 gradientTile[324];

[numthreads(256, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint lane : SV_GroupIndex, uniform Parameters p) {
    uint3 groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    if (operation == 0) {
        if (lane == 0 && p.slot < p.slotCount) {
            float value = ((Ptr<float, Access.Read>)p.input)[0];
            ((Ptr<float, Access.ReadWrite>)p.output)[p.slot] = value;
            Ptr<float, Access.ReadWrite> ema = (Ptr<float, Access.ReadWrite>)p.secondary;
            float old = ema[p.slot]; ema[p.slot] = old >= 0.0f ? old + p.emaAlpha * (value - old) : value;
        }
        return;
    }
    uint index = groupId.x * 256u + lane;
    Ptr<float, Access.ReadWrite> output = (Ptr<float, Access.ReadWrite>)p.output;
    if (operation == 1) {
        if (index >= p.count) return;
        uint x = index % p.width, y = index / p.width;
        float3 camera = float3((float(x) + 0.5f - p.cx) / p.fx, (float(y) + 0.5f - p.cy) / p.fy, 1.0f);
        if (p.rays != 0) {
            Ptr<float, Access.Read> rays = (Ptr<float, Access.Read>)p.rays;
            camera = float3(rays[3 * index], rays[3 * index + 1], rays[3 * index + 2]);
            // The lens has no ray for this pixel.
            if (!all(isfinite(camera))) {
                output[index] = p.inverse != 0 ? 1.0f : p.outsideWeight;
                return;
            }
        }
        Ptr<float, Access.Read> view = (Ptr<float, Access.Read>)p.input;
        float3 world = float3(view[0]*camera.x + view[4]*camera.y + view[8]*camera.z,
                              view[1]*camera.x + view[5]*camera.y + view[9]*camera.z,
                              view[2]*camera.x + view[6]*camera.y + view[10]*camera.z);
        Ptr<float, Access.Read> cam = (Ptr<float, Access.Read>)p.camera;
        float3 originInBox = transformPoint(p, float3(cam[0],cam[1],cam[2]));
        float3 directionInBox = transformDirection(p, world);
        bool hit;
        if (p.depth != 0 && p.alpha != 0) {
            // Below half coverage a pixel shows mostly background, so no surface in the box.
            float alpha = ((Ptr<float, Access.Read>)p.alpha)[index];
            float3 point = originInBox + (((Ptr<float, Access.Read>)p.depth)[index] / alpha) * directionInBox;
            hit = alpha >= 0.5f && all(point >= p.cropMin.xyz) && all(point <= p.cropMax.xyz) &&
                  (p.slot == 0 || outlineContains(p, point.xy));
        } else {
            hit = p.slot != 0 ? rayHitsPrism(p, originInBox, directionInBox)
                              : rayHitsBox(originInBox, directionInBox, p.cropMin.xyz, p.cropMax.xyz);
        }
        output[index] = ((p.inverse != 0) != hit) ? 1.0f : p.outsideWeight;
        return;
    }
    if (operation == 2) {
        if (index >= p.count) return;
        uint dy = index / p.width, dx = index % p.width;
        float sy = (float(dy) + 0.5f) * (float(p.sourceHeight) / float(p.height)) - 0.5f;
        float sx = (float(dx) + 0.5f) * (float(p.sourceWidth) / float(p.width)) - 0.5f;
        sy = clamp(sy, 0.0f, float(p.sourceHeight - 1)); sx = clamp(sx, 0.0f, float(p.sourceWidth - 1));
        int y0 = int(sy), x0 = int(sx), y1 = min(y0 + 1, int(p.sourceHeight - 1)), x1 = min(x0 + 1, int(p.sourceWidth - 1));
        float wy = sy - float(y0), wx = sx - float(x0);
        float w00 = (1.0f-wy)*(1.0f-wx), w01 = (1.0f-wy)*wx, w10 = wy*(1.0f-wx), w11 = wy*wx;
        uint sourcePlane = p.sourceHeight * p.sourceWidth, destinationPlane = p.height * p.width;
        Ptr<float, Access.Read> source = (Ptr<float, Access.Read>)p.input;
        for (uint c=0; c<p.channels; ++c) {
            Ptr<float, Access.ReadWrite> dst = (Ptr<float, Access.ReadWrite>)p.output;
            uint base = c * sourcePlane;
            dst[c * destinationPlane + index] = w00*source[base + uint(y0)*p.sourceWidth+uint(x0)] +
                w01*source[base + uint(y0)*p.sourceWidth+uint(x1)] + w10*source[base + uint(y1)*p.sourceWidth+uint(x0)] +
                w11*source[base + uint(y1)*p.sourceWidth+uint(x1)];
        }
        return;
    }
    if (operation == 3) {
        if (index >= p.count) return;
        uint base = p.seed ^ index;
        output[index] = float(pcgHash(base)) * (1.0f / 4294967296.0f);
        output[p.count + index] = float(pcgHash(base + 0x9E3779B9u)) * (1.0f / 4294967296.0f);
        output[2u*p.count + index] = float(pcgHash(base + 0x6C8E9CF9u)) * (1.0f / 4294967296.0f);
        return;
    }
    if (operation == 4) {
        int2 origin = int2(groupId.xy) * 16;
        for (uint t = lane; t < 576u; t += 256u) {
            int x = origin.x + int(t % 24u) - 4;
            int y = origin.y + int(t / 24u) - 4;
            grayTile[t] = maskKeeps(p, x, y) ? grayAt(p, x, y) : 0.0f;
        }
        GroupMemoryBarrierWithGroupSync();
        const float blurWeights[25] = {2,4,5,4,2, 4,9,12,9,4, 5,12,15,12,5,
                                       4,9,12,9,4, 2,4,5,4,2};
        for (uint t = lane; t < 400u; t += 256u) {
            uint x = t % 20u, y = t / 20u;
            float sum = 0.0f;
            for (uint cy = 0; cy < 5u; ++cy)
                for (uint cx = 0; cx < 5u; ++cx)
                    sum += (blurWeights[cy * 5u + cx] / 159.0f) * grayTile[(y + cy) * 24u + x + cx];
            blurTile[t] = sum;
        }
        GroupMemoryBarrierWithGroupSync();
        const float sobel[9] = {-1,0,1,-2,0,2,-1,0,1};
        for (uint t = lane; t < 324u; t += 256u) {
            uint x = t % 18u, y = t / 18u;
            float gx = 0.0f, gy = 0.0f;
            for (uint cy = 0; cy < 3u; ++cy)
                for (uint cx = 0; cx < 3u; ++cx) {
                    float value = blurTile[(y + cy) * 20u + x + cx];
                    gx += sobel[cy * 3u + cx] * value;
                    gy += sobel[cx * 3u + cy] * value;
                }
            gradientTile[t] = float2(gx, gy);
        }
        GroupMemoryBarrierWithGroupSync();
        uint x = lane % 16u, y = lane / 16u;
        uint outputX = uint(origin.x) + x, outputY = uint(origin.y) + y;
        if (outputX >= p.width || outputY >= p.height) return;
        int at = int((y + 1u) * 18u + x + 1u);
        float2 g = gradientTile[at];
        float magnitude = length(g);
        if (magnitude > 0.0f) {
            int dx = clamp(int(round(g.x / magnitude)), -1, 1), dy = clamp(int(round(g.y / magnitude)), -1, 1);
            float2 forward = gradientTile[at + dy * 18 + dx], backward = gradientTile[at - dy * 18 - dx];
            if (magnitude < length(forward) || magnitude < length(backward)) magnitude = 0.0f;
        }
        if (!maskKeeps(p, int(outputX), int(outputY))) {
            output[outputY * p.width + outputX] = 0.0f;
            return;
        }
        output[outputY * p.width + outputX] = magnitude;
        return;
    }
    if (operation == 5) {
        if (index >= p.count) return;
        float value = ((Ptr<float, Access.Read>)p.secondary)[0];
        if (!(value <= p.skipBelow)) output[index] /= max(value, 1.0e-9f);
    }
}
