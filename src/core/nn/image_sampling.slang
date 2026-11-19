// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
struct SamplingParams {
    Ptr<float, Access.Read> source, locations, weights;
    Ptr<float, Access.ReadWrite> output;
    Ptr<int, Access.Read> bounds, coefficients;
    uint count, spatial, queries, heads, channels, points, levels, taps;
    uint inH, inW, outH, outW, horizontal;
    uint groups;
};
[[vk::push_constant]] ConstantBuffer<SamplingParams> p;

[shader("compute")]
[numthreads(256,1,1)]
void sampleAttention(uint3 tid : SV_DispatchThreadID) {
    for (uint64_t i = tid.x; i < p.count; i += p.groups*256) {
        uint channel = i % p.channels;
        uint head = (i / p.channels) % p.heads;
        uint query = (i / p.channels / p.heads) % p.queries;
        uint batch = i / p.channels / p.heads / p.queries;
        float sum = 0;
        for (uint level = 0; level < p.levels; ++level) {
            int height = p.bounds[level*3], width = p.bounds[level*3+1], offset = p.bounds[level*3+2];
            for (uint point = 0; point < p.points; ++point) {
                uint64_t sample = ((((uint64_t(batch)*p.queries+query)*p.heads+head)*p.levels+level)*p.points+point);
                float x = p.locations[sample*2]*width-0.5f, y = p.locations[sample*2+1]*height-0.5f;
                if (!isfinite(x) || !isfinite(y)) { sum = asfloat(0x7fc00000u); continue; }
                if (x <= -1 || x >= width || y <= -1 || y >= height) continue;
                int x0 = int(floor(x)), y0 = int(floor(y));
                float dx = x-x0, dy = y-y0, v = 0;
                [unroll] for (int iy = 0; iy < 2; ++iy) {
                    [unroll] for (int ix = 0; ix < 2; ++ix) {
                        int sx = x0+ix, sy = y0+iy;
                        if (sx >= 0 && sx < width && sy >= 0 && sy < height) {
                            uint64_t row = uint64_t(batch)*p.spatial + offset + uint64_t(sy)*width + sx;
                            v += p.source[(row*p.heads+head)*p.channels+channel] * (ix != 0 ? dx : 1-dx) * (iy != 0 ? dy : 1-dy);
                        }
                    }
                }
                sum += v*p.weights[sample];
            }
        }
        p.output[i] = sum;
    }
}

[shader("compute")]
[numthreads(256,1,1)]
void resampleAxis(uint3 tid : SV_DispatchThreadID) {
    for (uint64_t i = tid.x; i < p.count; i += p.groups*256) {
        int x = i%p.outW, y = (i/p.outW)%p.outH;
        uint64_t plane = i/p.outW/p.outH;
        int target = p.horizontal != 0 ? x : y;
        int first = p.bounds[2*target], taps = p.bounds[2*target+1];
        int sum = 1 << 21;
        for (int j=0;j<taps;++j) {
            uint64_t pos = plane*p.inH*p.inW + (p.horizontal != 0 ? y*p.inW+first+j : (first+j)*p.inW+x);
            sum += int(round(p.source[pos]*255.0f)) * p.coefficients[target*p.taps+j];
        }
        p.output[i] = float(clamp(sum >> 22, 0, 255))/255.0f;
    }
}
