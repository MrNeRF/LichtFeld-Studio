[SpecializationConstant] const uint operation = 0;
[SpecializationConstant] const uint imageIsBytes = 0;

struct Parameters {
    uint64_t input, output, secondary, tertiary;
    uint64_t camera, unused0, unused1, unused2;
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
        Ptr<float, Access.Read> view = (Ptr<float, Access.Read>)p.input;
        float3 world = float3(view[0]*camera.x + view[4]*camera.y + view[8],
                              view[1]*camera.x + view[5]*camera.y + view[9],
                              view[2]*camera.x + view[6]*camera.y + view[10]);
        Ptr<float, Access.Read> cam = (Ptr<float, Access.Read>)p.camera;
        bool hit = rayHitsBox(transformPoint(p, float3(cam[0],cam[1],cam[2])), transformDirection(p, world), p.cropMin.xyz, p.cropMax.xyz);
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
        for (uint t = lane; t < 576u; t += 256u)
            grayTile[t] = grayAt(p, origin.x + int(t % 24u) - 4, origin.y + int(t / 24u) - 4);
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
        output[outputY * p.width + outputX] = magnitude;
        return;
    }
    if (operation == 5) {
        if (index >= p.count) return;
        float value = ((Ptr<float, Access.Read>)p.secondary)[0];
        if (!(value <= p.skipBelow)) output[index] /= max(value, 1.0e-9f);
    }
}
