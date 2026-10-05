// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
struct Parameters {
    Ptr<uint8_t, Access.ReadWrite> destination;
    Ptr<uint8_t, Access.Read> left;
    Ptr<uint8_t, Access.Read> right;
    uint destinationWidth;
    uint destinationHeight;
    int viewportX;
    int viewportY;
    uint viewportWidth;
    uint viewportHeight;
    int contentX;
    int contentY;
    uint contentWidth;
    uint contentHeight;
    uint leftWidth;
    uint leftHeight;
    uint leftChannels;
    uint leftFloat;
    uint leftChw;
    uint rightWidth;
    uint rightHeight;
    uint rightChannels;
    uint rightFloat;
    uint rightChw;
    float splitPosition;
    float leftStart;
    float leftEnd;
    float rightStart;
    float rightEnd;
    uint leftNormalize;
    uint rightNormalize;
    uint leftFlipY;
    uint rightFlipY;
    uint leftFilter;
    uint rightFilter;
    uint lossVisualization;
    uint padding;
    float2 leftUvScale;
    float2 leftUvClamp;
    float2 rightUvScale;
    float2 rightUvClamp;
    float2 leftTexcoordScale;
    float2 leftTexcoordOffset;
    float2 rightTexcoordScale;
    float2 rightTexcoordOffset;
    float4 background;
};
[[vk::push_constant]] ConstantBuffer<Parameters> p;

float readChannel(Ptr<uint8_t, Access.Read> image, uint width, uint height,
                  uint channels, uint floatingPoint, uint chw,
                  uint x, uint y, uint channel) {
    x = min(x, max(width, 1u) - 1u);
    y = min(y, max(height, 1u) - 1u);
    channel = min(channel, max(channels, 1u) - 1u);
    uint index = chw != 0 ? channel * width * height + y * width + x
                          : (y * width + x) * channels + channel;
    return floatingPoint != 0 ? ((Ptr<float, Access.Read>)image)[index]
                              : float(image[index]) * (1.0 / 255.0);
}

float3 sampleLinear(Ptr<uint8_t, Access.Read> image, uint width, uint height,
                    uint channels, uint floatingPoint, uint chw, float2 uv) {
    float2 extent = float2(max(width, 1u), max(height, 1u));
    float2 pixel = clamp(uv * extent - 0.5, 0.0, extent - 1.0);
    uint2 a = uint2(floor(pixel));
    uint2 b = min(a + 1u, uint2(max(width, 1u) - 1u, max(height, 1u) - 1u));
    float2 f = frac(pixel);
    uint c1 = min(1u, channels - 1u), c2 = min(2u, channels - 1u);
    float3 c00 = float3(readChannel(image, width, height, channels, floatingPoint, chw, a.x, a.y, 0),
                        readChannel(image, width, height, channels, floatingPoint, chw, a.x, a.y, c1),
                        readChannel(image, width, height, channels, floatingPoint, chw, a.x, a.y, c2));
    float3 c10 = float3(readChannel(image, width, height, channels, floatingPoint, chw, b.x, a.y, 0),
                        readChannel(image, width, height, channels, floatingPoint, chw, b.x, a.y, c1),
                        readChannel(image, width, height, channels, floatingPoint, chw, b.x, a.y, c2));
    float3 c01 = float3(readChannel(image, width, height, channels, floatingPoint, chw, a.x, b.y, 0),
                        readChannel(image, width, height, channels, floatingPoint, chw, a.x, b.y, c1),
                        readChannel(image, width, height, channels, floatingPoint, chw, a.x, b.y, c2));
    float3 c11 = float3(readChannel(image, width, height, channels, floatingPoint, chw, b.x, b.y, 0),
                        readChannel(image, width, height, channels, floatingPoint, chw, b.x, b.y, c1),
                        readChannel(image, width, height, channels, floatingPoint, chw, b.x, b.y, c2));
    return lerp(lerp(c00, c10, f.x), lerp(c01, c11, f.x), f.y);
}

float3 samplePanel(Ptr<uint8_t, Access.Read> image, uint width, uint height,
                   uint channels, uint floatingPoint, uint chw, float2 uv,
                   float start, float end, uint normalize, uint flipY, uint spatialFilter,
                   float2 uvScale, float2 uvClamp, float2 texcoordScale, float2 texcoordOffset) {
    float u = normalize != 0 ? (uv.x - start) / max(end - start, 1e-6) : uv.x;
    float v = flipY != 0 ? 1.0 - uv.y : uv.y;
    float2 st = min((float2(u, v) * texcoordScale + texcoordOffset) * uvScale, uvClamp);
    float3 center = sampleLinear(image, width, height, channels, floatingPoint, chw, st);
    if (spatialFilter == 0) return center;
    const float strength = 0.18;
    float2 texel = 1.0 / float2(max(width, 1u), max(height, 1u));
    float3 left = sampleLinear(image, width, height, channels, floatingPoint, chw,
                               clamp(st - float2(texel.x, 0.0), 0.0, uvClamp));
    float3 right = sampleLinear(image, width, height, channels, floatingPoint, chw,
                                clamp(st + float2(texel.x, 0.0), 0.0, uvClamp));
    float3 up = sampleLinear(image, width, height, channels, floatingPoint, chw,
                             clamp(st - float2(0.0, texel.y), 0.0, uvClamp));
    float3 down = sampleLinear(image, width, height, channels, floatingPoint, chw,
                               clamp(st + float2(0.0, texel.y), 0.0, uvClamp));
    float3 sharpened = center * (1.0 + 4.0 * strength) - (left + right + up + down) * strength;
    return clamp(sharpened, min(center, min(min(left, right), min(up, down))),
                 max(center, max(max(left, right), max(up, down))));
}

float3 lossHeatmap(float3 groundTruth, float3 rendered) {
    float3 difference = abs(groundTruth - rendered);
    float intensity = saturate(1.0 - exp(-8.0 * max((difference.r + difference.g + difference.b) / 3.0, 0.0)));
    const float3 black = float3(0.0, 0.0, 0.0), purple = float3(0.22, 0.02, 0.47);
    const float3 red = float3(0.72, 0.12, 0.29), yellow = float3(0.99, 0.65, 0.04);
    const float3 white = float3(1.0, 1.0, 0.75);
    if (intensity < 0.25) return lerp(black, purple, intensity * 4.0);
    if (intensity < 0.5) return lerp(purple, red, (intensity - 0.25) * 4.0);
    if (intensity < 0.75) return lerp(red, yellow, (intensity - 0.5) * 4.0);
    return lerp(yellow, white, (intensity - 0.75) * 4.0);
}

void writeDestination(uint2 pixel, float3 color) {
    uint index = (pixel.y * p.destinationWidth + pixel.x) * 4u;
    p.destination[index] = uint8_t(saturate(color.r) * 255.0 + 0.5);
    p.destination[index + 1] = uint8_t(saturate(color.g) * 255.0 + 0.5);
    p.destination[index + 2] = uint8_t(saturate(color.b) * 255.0 + 0.5);
    p.destination[index + 3] = 255;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void composeSplitView(uint3 tid : SV_DispatchThreadID) {
    if (tid.x >= p.destinationWidth || tid.y >= p.destinationHeight) return;
    int2 pixel = int2(tid.xy);
    if (pixel.x < p.viewportX || pixel.y < p.viewportY ||
        pixel.x >= p.viewportX + int(p.viewportWidth) ||
        pixel.y >= p.viewportY + int(p.viewportHeight)) return;

    float2 px = float2(pixel) + 0.5;
    if (pixel.x < p.contentX || pixel.y < p.contentY ||
        pixel.x >= p.contentX + int(p.contentWidth) ||
        pixel.y >= p.contentY + int(p.contentHeight)) {
        writeDestination(tid.xy, p.background.rgb);
        return;
    }
    float2 uv = (px - float2(p.contentX, p.contentY)) /
                float2(max(p.contentWidth, 1u), max(p.contentHeight, 1u));
    float3 left = samplePanel(p.left, p.leftWidth, p.leftHeight, p.leftChannels,
                              p.leftFloat, p.leftChw, uv, p.leftStart, p.leftEnd,
                              p.leftNormalize, p.leftFlipY, p.leftFilter,
                              p.leftUvScale, p.leftUvClamp,
                              p.leftTexcoordScale, p.leftTexcoordOffset);
    float3 right = samplePanel(p.right, p.rightWidth, p.rightHeight, p.rightChannels,
                               p.rightFloat, p.rightChw, uv, p.rightStart, p.rightEnd,
                               p.rightNormalize, p.rightFlipY, p.rightFilter,
                               p.rightUvScale, p.rightUvClamp,
                               p.rightTexcoordScale, p.rightTexcoordOffset);
    if (p.lossVisualization != 0) {
        writeDestination(tid.xy, lossHeatmap(left, right));
        return;
    }

    float splitX = float(p.contentX) + saturate(p.splitPosition) * float(p.contentWidth);
    float dividerPixel = float(p.contentX) + floor(p.splitPosition * float(p.contentWidth) + 0.5);
    float3 color = px.x < dividerPixel ? left : right;
    float distFromSplit = abs(px.x - splitX);
    if (distFromSplit < 2.0) {
        color = float3(0.29, 0.33, 0.42);
        float centerY = float(p.contentY) + float(p.contentHeight) * 0.5;
        float distFromCenter = abs(px.y - centerY);
        float handleHalfWidth = min(12.0, float(p.contentWidth) * 0.5);
        float handleHalfHeight = min(40.0, float(p.contentHeight) * 0.5);
        if (distFromCenter < handleHalfHeight && distFromSplit < handleHalfWidth) {
            float radius = min(6.0, min(handleHalfWidth, handleHalfHeight));
            float2 corner = float2(distFromSplit, distFromCenter) -
                            (float2(handleHalfWidth, handleHalfHeight) - radius);
            if (corner.x <= 0.0 || corner.y <= 0.0 || length(corner) <= radius) {
                color *= 0.8;
                float localY = px.y - centerY;
                for (int i = -2; i <= 2; ++i) {
                    if (abs(localY - float(i) * 10.0) < 2.0 && distFromSplit < 6.0) {
                        color = 0.9;
                        break;
                    }
                }
            }
        }
    }
    writeDestination(tid.xy, color);
}
