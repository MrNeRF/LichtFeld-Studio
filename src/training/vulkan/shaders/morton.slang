[SpecializationConstant] const uint kPhase = 0;

struct Parameters {
    uint64_t means;
    uint64_t minimum;
    uint64_t maximum;
    uint64_t codes;
    uint64_t indices;
    uint count;
};

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
    Ptr<float, Access.Read> minimum = (Ptr<float, Access.Read>)p.minimum;
    Ptr<float, Access.Read> maximum = (Ptr<float, Access.Read>)p.maximum;
    Ptr<uint, Access.ReadWrite> codes = (Ptr<uint, Access.ReadWrite>)p.codes;
    Ptr<uint, Access.ReadWrite> indices = (Ptr<uint, Access.ReadWrite>)p.indices;
    const float3 lo = float3(minimum[0], minimum[1], minimum[2]);
    const float3 hi = float3(maximum[0], maximum[1], maximum[2]);
    const float3 extent = hi - lo;
    const float3 mul = float3(extent.x == 0.0 ? 0.0 : 1024.0 / extent.x,
                              extent.y == 0.0 ? 0.0 : 1024.0 / extent.y,
                              extent.z == 0.0 ? 0.0 : 1024.0 / extent.z);
    const uint x = quantize(means[i * 3], lo.x, mul.x);
    const uint y = quantize(means[i * 3 + 1], lo.y, mul.y);
    const uint z = quantize(means[i * 3 + 2], lo.z, mul.z);
    codes[i] = (part1by2(z) << 2) + (part1by2(y) << 1) + part1by2(x);
    indices[i] = i;
}
