#include "../../../../include/core/rounded_math.slangh"

static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 56;

[SpecializationConstant]
const uint kMode = 0;

struct Parameters
{
    uint64_t points;
    uint64_t references;
    uint64_t heads;
    uint64_t next;
    uint64_t output;
    uint count;
    uint bucketMask;
    float radius;
    uint padding;
};

#include "byte_read.slangh"

float3 position(Parameters p, uint i)
{
    Ptr<float, Access.Read> data = (Ptr<float, Access.Read>)p.points;
    const uint64_t base = uint64_t(i) * 3;
    return float3(data[base], data[base + 1], data[base + 2]);
}

int3 cell(float3 point, float radius)
{
    return int3(clamp(floor((point / radius) * 0.5f), -268435456.0f, 268435456.0f));
}

uint hashCell(int3 c, uint mask)
{
    return ((uint(c.x) * 73856093u) ^ (uint(c.y) * 19349663u) ^ (uint(c.z) * 83492791u)) & mask;
}

bool within(float3 a, float3 b, float radius)
{
    float3 d = float3(roundedSub(a.x, b.x), roundedSub(a.y, b.y), roundedSub(a.z, b.z));
    float r2 = roundedMul(radius, radius);
    if (r2 < 1.17549435e-38f || !isfinite(r2))
    {
        d = float3(roundedDiv(d.x, radius), roundedDiv(d.y, radius), roundedDiv(d.z, radius));
        r2 = 1.0f;
    }
    const float squared = dotRounded(d, d);
    return squared <= r2;
}

bool query(Parameters p, uint i)
{
    const float3 point = position(p, i);
    if (!all(isfinite(point))) return false;
    if (loadByte(p.references, i) != 0u) return true;
    const int3 center = cell(point, p.radius);
    Ptr<int, Access.Read> heads = (Ptr<int, Access.Read>)p.heads;
    Ptr<int, Access.Read> next = (Ptr<int, Access.Read>)p.next;
    for (int z = -1; z <= 1; ++z)
    {
        for (int y = -1; y <= 1; ++y)
        {
            for (int x = -1; x <= 1; ++x)
            {
                const uint bucket = hashCell(center + int3(x, y, z), p.bucketMask);
                for (int j = heads[bucket]; j >= 0; j = next[j])
                {
                    if (within(point, position(p, uint(j)), p.radius)) return true;
                }
            }
        }
    }
    return false;
}

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID, uniform Parameters p)
{
    const uint3 numWorkgroups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    const uint stride = numWorkgroups.x * LFS_LOCAL_SIZE_X;
    if (kMode == 0)
    {
        Ptr<int, Access.ReadWrite> heads = (Ptr<int, Access.ReadWrite>)p.heads;
        Ptr<int, Access.ReadWrite> next = (Ptr<int, Access.ReadWrite>)p.next;
        for (uint i = dispatchThreadId.x; i < p.count; i += stride)
        {
            if (loadByte(p.references, i) == 0u) continue;
            const float3 point = position(p, i);
            if (!all(isfinite(point))) continue;
            const uint bucket = hashCell(cell(point, p.radius), p.bucketMask);
            int previous;
            InterlockedExchange(heads[bucket], int(i), previous);
            next[i] = previous;
        }
    }
    else
    {
        Ptr<uint, Access.ReadWrite> output = (Ptr<uint, Access.ReadWrite>)p.output;
        const uint words = (p.count + 3u) / 4u;
        for (uint word = dispatchThreadId.x; word < words; word += stride)
        {
            uint packed = 0u;
            for (uint lane = 0; lane < 4; ++lane)
            {
                const uint i = word * 4u + lane;
                if (i < p.count && query(p, i)) packed |= 1u << (lane * 8u);
            }
            output[word] = packed;
        }
    }
}
