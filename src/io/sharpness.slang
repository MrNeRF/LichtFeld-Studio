// SPDX-License-Identifier: GPL-3.0-or-later
struct Parameters {
    Ptr<uint8_t, Access.Read> rgb;
    Ptr<uint, Access.ReadWrite> counts;
    uint width, height, algorithm, groups;
};
[[vk::push_constant]] ConstantBuffer<Parameters> p;
groupshared uint counts[256];

int green(uint64_t pixel) { return int(p.rgb[3 * pixel + 1]); }

[numthreads(256, 1, 1)]
void countEdges(uint3 group : SV_GroupID, uint lane : SV_GroupIndex) {
    uint count = 0;
    uint64_t n = uint64_t(p.width - 2) * (p.height - 2);
    for (uint64_t i = group.x * 256 + lane; i < n; i += p.groups * 256) {
        uint64_t pixel = (i / (p.width - 2) + 1) * p.width + i % (p.width - 2) + 1;
        int left = green(pixel - 1), right = green(pixel + 1);
        int up = green(pixel - p.width), down = green(pixel + p.width);
        bool edge = p.algorithm != 1 && abs(4 * green(pixel) - left - right - up - down) > 10;
        if (!edge && p.algorithm != 0) {
            int ul = green(pixel - p.width - 1), ur = green(pixel - p.width + 1);
            int dl = green(pixel + p.width - 1), dr = green(pixel + p.width + 1);
            int gx = ur - ul + 2 * (right - left) + dr - dl;
            int gy = dl - ul + 2 * (down - up) + dr - ur;
            edge = abs(gx) + abs(gy) > 40;
        }
        count += uint(edge);
    }
    counts[lane] = count;
    for (uint step = 128; step > 0; step /= 2) {
        GroupMemoryBarrierWithGroupSync();
        if (lane < step)
            counts[lane] += counts[lane + step];
    }
    if (lane == 0)
        p.counts[group.x] = counts[0];
}
