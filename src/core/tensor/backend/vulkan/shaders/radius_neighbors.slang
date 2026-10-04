#include "../../../../include/core/rounded_math.slangh"

static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 88;

[SpecializationConstant] const uint kMode = 0;

struct Parameters {
    uint64_t points;
    uint64_t references;
    uint64_t heads;
    uint64_t next;
    uint64_t output;
    uint64_t queries;
    uint64_t values;
    uint64_t radii;
    uint count;
    uint bucketMask;
    float radius;
    uint excludeSelf; // Mode 1: boolean; mode 2: positive count saturation limit.
    uint queryBegin;
    uint queryEnd;
};

#include "byte_read.slangh"

float3 position(Parameters p, uint i) {
    Ptr<float, Access.Read> data = (Ptr<float, Access.Read>)p.points;
    const uint64_t base = uint64_t(i) * 3;
    return float3(data[base], data[base + 1], data[base + 2]);
}

int3 cell(float3 point, float radius) {
    return int3(clamp(floor(point / radius), -268435456.0f, 268435456.0f));
}

uint hashCell(int3 c, uint mask) {
    return ((uint(c.x) * 73856093u) ^ (uint(c.y) * 19349663u) ^ (uint(c.z) * 83492791u)) & mask;
}

bool within(float3 a, float3 b, float radius) {
    float3 d = float3(roundedSub(a.x, b.x), roundedSub(a.y, b.y), roundedSub(a.z, b.z));
    float r2 = roundedMul(radius, radius);
    if (r2 < 1.17549435e-38f || !isfinite(r2)) {
        d = float3(roundedDiv(d.x, radius), roundedDiv(d.y, radius), roundedDiv(d.z, radius));
        r2 = 1.0f;
    }
    const float squared = dotRounded(d, d);
    return squared <= r2;
}

bool query(Parameters p, uint i) {
    if (p.queries != 0 && loadByte(p.queries, i) == 0u)
        return false;
    const float3 point = position(p, i);
    if (!all(isfinite(point)))
        return false;
    if (p.excludeSelf == 0u && loadByte(p.references, i) != 0u)
        return true;
    const int3 center = cell(point, p.radius);
    Ptr<int, Access.Read> heads = (Ptr<int, Access.Read>)p.heads;
    Ptr<int, Access.Read> next = (Ptr<int, Access.Read>)p.next;
    for (int j = heads[hashCell(center, p.bucketMask)]; j >= 0; j = next[j])
        if ((p.excludeSelf == 0u || uint(j) != i) && within(point, position(p, uint(j)), p.radius))
            return true;
    for (int z = -1; z <= 1; ++z) {
        for (int y = -1; y <= 1; ++y) {
            for (int x = -1; x <= 1; ++x) {
                if (x == 0 && y == 0 && z == 0)
                    continue;
                const uint bucket = hashCell(center + int3(x, y, z), p.bucketMask);
                for (int j = heads[bucket]; j >= 0; j = next[j]) {
                    if ((p.excludeSelf == 0u || uint(j) != i) && within(point, position(p, uint(j)), p.radius))
                        return true;
                }
            }
        }
    }
    return false;
}

uint queryCount(Parameters p, uint i) {
    if (p.queries != 0 && loadByte(p.queries, i) == 0u)
        return 0;
    const float3 point = position(p, i);
    if (!all(isfinite(point)))
        return 0;
    const int3 center = cell(point, p.radius);
    Ptr<int, Access.Read> heads = (Ptr<int, Access.Read>)p.heads;
    Ptr<int, Access.Read> next = (Ptr<int, Access.Read>)p.next;
    uint count = 0;
    for (int slot = 0; slot < 27; ++slot) {
        const int neighbor = slot == 0 ? 13 : (slot <= 13 ? slot - 1 : slot);
        const int x = neighbor % 3 - 1;
        const int y = neighbor / 3 % 3 - 1;
        const int z = neighbor / 9 - 1;
        const int3 target = center + int3(x, y, z);
        for (int j = heads[hashCell(target, p.bucketMask)]; j >= 0; j = next[j]) {
            const float3 other = position(p, uint(j));
            // Checking the actual cell prevents double counts on hash collisions.
            if (uint(j) != i && all(cell(other, p.radius) == target) && within(point, other, p.radius)) {
                if (++count == p.excludeSelf)
                    return count;
            }
        }
    }
    return count;
}

int queryMinInt(Parameters p, uint i) {
    Ptr<int, Access.Read> values = (Ptr<int, Access.Read>)p.values;
    int result = values[i];
    Ptr<float, Access.Read> radii = (Ptr<float, Access.Read>)p.radii;
    if (p.radii != 0 && !(radii[i] > p.radius * 0.5f))
        return result;
    const float3 point = position(p, i);
    if (!all(isfinite(point)))
        return result;
    const int3 center = cell(point, p.radius);
    Ptr<int, Access.Read> heads = (Ptr<int, Access.Read>)p.heads;
    Ptr<int, Access.Read> next = (Ptr<int, Access.Read>)p.next;
    for (int z = -1; z <= 1; ++z) for (int y = -1; y <= 1; ++y) for (int x = -1; x <= 1; ++x) {
        const int3 target = center + int3(x, y, z);
        for (int j = heads[hashCell(target, p.bucketMask)]; j >= 0; j = next[j]) {
            const float3 other = position(p, uint(j));
            if (all(cell(other, p.radius) == target) && within(point, other, p.radii != 0 ? min(p.radius, min(radii[i], radii[j])) : p.radius))
                result = min(result, values[j]);
        }
    }
    return result;
}

float queryMinFloat(Parameters p, uint i) {
    Ptr<float, Access.Read> values = (Ptr<float, Access.Read>)p.values;
    float result = values[i];
    Ptr<float, Access.Read> radii = (Ptr<float, Access.Read>)p.radii;
    if (p.radii != 0 && !(radii[i] > p.radius * 0.5f))
        return result;
    const float3 point = position(p, i);
    if (!all(isfinite(point)))
        return result;
    const int3 center = cell(point, p.radius);
    Ptr<int, Access.Read> heads = (Ptr<int, Access.Read>)p.heads;
    Ptr<int, Access.Read> next = (Ptr<int, Access.Read>)p.next;
    for (int z = -1; z <= 1; ++z) for (int y = -1; y <= 1; ++y) for (int x = -1; x <= 1; ++x) {
        const int3 target = center + int3(x, y, z);
        for (int j = heads[hashCell(target, p.bucketMask)]; j >= 0; j = next[j]) {
            const float3 other = position(p, uint(j));
            if (all(cell(other, p.radius) == target) && within(point, other, p.radii != 0 ? min(p.radius, min(radii[i], radii[j])) : p.radius))
                result = min(result, values[j]);
        }
    }
    return result;
}

float querySpacing(Parameters p, uint i) {
    const float3 point = position(p, i);
    if (!all(isfinite(point)))
        return 0;
    const int3 center = cell(point, p.radius);
    float3 best = float3(asfloat(0x7f800000u));
    int found = 0;
    Ptr<int, Access.Read> heads = (Ptr<int, Access.Read>)p.heads;
    Ptr<int, Access.Read> next = (Ptr<int, Access.Read>)p.next;
    for (int extent = 1; extent <= 2; ++extent) {
        for (int z = -extent; z <= extent; ++z) {
            for (int y = -extent; y <= extent; ++y) {
                for (int x = -extent; x <= extent; ++x) {
                    if (extent == 2 && abs(x) <= 1 && abs(y) <= 1 && abs(z) <= 1)
                        continue;
                    const int3 target = center + int3(x, y, z);
                    // Every candidate participates, independent of atomic insertion order.
                    for (int j = heads[hashCell(target, p.bucketMask)]; j >= 0; j = next[j]) {
                        const float3 other = position(p, uint(j));
                        if (uint(j) == i || !all(cell(other, p.radius) == target))
                            continue;
                        const float3 delta = point - other;
                        const float distance = dotRounded(delta, delta);
                        if (distance < best.z) {
                            best.z = max(best.y, distance);
                            best.y = max(best.x, min(best.y, distance));
                            best.x = min(best.x, distance);
                        }
                        found = min(found + 1, 3);
                    }
                }
            }
        }
        if (found == 3)
            break;
    }
    float sum = 0;
    for (int k = 0; k < found; ++k)
        sum += sqrt(best[k]);
    return found ? sum / found : p.radius * 4;
}
[numthreads(LFS_LOCAL_SIZE_X, 1, 1)] void main(uint3 dispatchThreadId : SV_DispatchThreadID, uniform Parameters p) {
    const uint3 numWorkgroups = spirv_asm {
result:
        $$uint3 = OpLoad builtin(NumWorkgroups : uint3);
    };
    const uint stride = numWorkgroups.x * LFS_LOCAL_SIZE_X;
    if (kMode == 0) {
        Ptr<int, Access.ReadWrite> heads = (Ptr<int, Access.ReadWrite>)p.heads;
        Ptr<int, Access.ReadWrite> next = (Ptr<int, Access.ReadWrite>)p.next;
        for (uint i = p.queryBegin + dispatchThreadId.x; i < p.queryEnd; i += stride) {
            if (loadByte(p.references, i) == 0u)
                continue;
            const float3 point = position(p, i);
            if (!all(isfinite(point)))
                continue;
            const uint bucket = hashCell(cell(point, p.radius), p.bucketMask);
            int previous;
            InterlockedExchange(heads[bucket], int(i), previous);
            next[i] = previous;
        }
    } else if (kMode == 5) {
        Ptr<float, Access.ReadWrite> output = (Ptr<float, Access.ReadWrite>)p.output;
        for (uint i = p.queryBegin + dispatchThreadId.x; i < p.queryEnd; i += stride)
            output[i] = queryMinFloat(p, i);
    } else if (kMode == 4) {
        Ptr<int, Access.ReadWrite> output = (Ptr<int, Access.ReadWrite>)p.output;
        for (uint i = p.queryBegin + dispatchThreadId.x; i < p.queryEnd; i += stride)
            output[i] = queryMinInt(p, i);
    } else if (kMode == 3) {
        Ptr<float, Access.ReadWrite> output = (Ptr<float, Access.ReadWrite>)p.output;
        for (uint i = p.queryBegin + dispatchThreadId.x; i < p.queryEnd; i += stride)
            output[i] = querySpacing(p, i);
    } else if (kMode == 2) {
        Ptr<int, Access.ReadWrite> output = (Ptr<int, Access.ReadWrite>)p.output;
        for (uint i = p.queryBegin + dispatchThreadId.x; i < p.queryEnd; i += stride)
            output[i] = int(queryCount(p, i));
    } else {
        Ptr<uint, Access.ReadWrite> output = (Ptr<uint, Access.ReadWrite>)p.output;
        const uint words = (p.queryEnd + 3u) / 4u;
        for (uint word = p.queryBegin / 4u + dispatchThreadId.x; word < words; word += stride) {
            uint packed = 0u;
            for (uint lane = 0; lane < 4; ++lane) {
                const uint i = word * 4u + lane;
                if (i < p.queryEnd && query(p, i))
                    packed |= 1u << (lane * 8u);
            }
            output[word] = packed;
        }
    }
}
