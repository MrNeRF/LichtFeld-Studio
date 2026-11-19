[SpecializationConstant] const uint kPhase = 0;

struct Parameters {
    uint64_t means;
    uint64_t codes;
    uint64_t indices;
    float centre_x;
    float centre_y;
    float centre_z;
    float inv_radius;
    uint count;
};

// The contracted code space of lfs/training/morton_contraction.hpp: the grid spans [-2, 2]^3.
static const float kContractedHalfExtent = 2.0;

uint part1by2(uint value) {
    value &= 0x3ffu;
    value = (value ^ (value << 16)) & 0xff0000ffu;
    value = (value ^ (value << 8)) & 0x0300f00fu;
    value = (value ^ (value << 4)) & 0x030c30c3u;
    value = (value ^ (value << 2)) & 0x09249249u;
    return value;
}

uint quantize(float value, float low, float multiplier) {
    const float normalized = (value - low) * multiplier;
    return normalized > 0.0 ? min(1023u, uint(normalized)) : 0u;
}

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uniform Parameters p) {
    const uint i = id.x;
    if (i >= p.count) return;
    if (kPhase == 1u) {
        Ptr<uint, Access.Read> source = (Ptr<uint, Access.Read>)p.means;
        Ptr<int64_t, Access.ReadWrite> destination = (Ptr<int64_t, Access.ReadWrite>)p.indices;
        destination[i] = int64_t(source[i]);
        return;
    }
    Ptr<float, Access.Read> means = (Ptr<float, Access.Read>)p.means;
    Ptr<uint, Access.ReadWrite> codes = (Ptr<uint, Access.ReadWrite>)p.codes;
    Ptr<uint, Access.ReadWrite> indices = (Ptr<uint, Access.ReadWrite>)p.indices;
    float3 q = (float3(means[i * 3], means[i * 3 + 1], means[i * 3 + 2]) - float3(p.centre_x, p.centre_y, p.centre_z)) *
               p.inv_radius;
    const float r = sqrt(dot(q, q));
    if (r > 1.0)
        q *= (2.0 - 1.0 / r) / r;
    const float origin = -kContractedHalfExtent;
    const float mul = 1024.0 / (2.0 * kContractedHalfExtent);
    codes[i] = (part1by2(quantize(q.z, origin, mul)) << 2) + (part1by2(quantize(q.y, origin, mul)) << 1) +
               part1by2(quantize(q.x, origin, mul));
    indices[i] = i;
}
