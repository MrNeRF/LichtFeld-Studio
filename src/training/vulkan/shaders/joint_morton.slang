#include "byte_read.slangh"
#include "byte_word.slangh"

[SpecializationConstant]
const uint kPhase = 0;

struct Parameters {
    uint64_t source;
    uint64_t sourceBounds;
    uint64_t destination;
    uint64_t destinationBounds;
    uint64_t permutation;
    uint primitives;
    uint width;
    uint swizzled;
    uint bits;
    uint slotBegin;
    uint slotEnd;
};

groupshared float reductions[256 * 4];

uint64_t cellIndex(uint primitive, uint attribute, uint component, uniform Parameters p) {
    if (p.swizzled == 0u) return uint64_t(primitive) * p.width + attribute;
    const uint slot = (primitive / 32u) * (p.width * 32u) + attribute * 32u + (primitive % 32u);
    return uint64_t(slot) * 4u + component;
}

uint64_t destinationCellIndex(uint primitive, uint attribute, uint component, uniform Parameters p) {
    if (p.swizzled == 0u) return uint64_t(primitive) * p.width + attribute;
    const uint groupWidth = p.slotEnd - p.slotBegin;
    const uint slot = (primitive / 32u) * (groupWidth * 32u) + attribute * 32u + (primitive % 32u);
    return uint64_t(slot) * 4u + component;
}

float2 decode(uint64_t cell, uint sourceRow, uniform Parameters p) {
    Ptr<float, Access.Read> bounds = (Ptr<float, Access.Read>)p.sourceBounds;
    const uint block = sourceRow / 256u;
    const float uMin = bounds[block * 4u];
    const float uMax = bounds[block * 4u + 1u];
    const float sMin = bounds[block * 4u + 2u];
    const float sMax = bounds[block * 4u + 3u];
    float uq, sq;
    if (p.bits == 16u) {
        Ptr<uint16_t, Access.Read> codes = (Ptr<uint16_t, Access.Read>)p.source;
        uq = float(codes[cell * 2u]);
        sq = float(codes[cell * 2u + 1u]);
    } else {
        uq = float(loadByte(p.source, cell * 2u));
        sq = float(loadByte(p.source, cell * 2u + 1u));
    }
    const float maxCode = p.bits == 16u ? 65535.0 : 255.0;
    return float2(uMin + (uMax - uMin) * (uq / maxCode),
                  sMin + (sMax - sMin) * (sq / maxCode));
}

void encode(uint64_t cell, float2 value, uint destinationRow, uniform Parameters p) {
    Ptr<float, Access.Read> bounds = (Ptr<float, Access.Read>)p.destinationBounds;
    const uint block = destinationRow / 256u;
    const float uMin = bounds[block * 4u];
    const float uMax = bounds[block * 4u + 1u];
    const float sMin = bounds[block * 4u + 2u];
    const float sMax = bounds[block * 4u + 3u];
    const float maxCode = p.bits == 16u ? 65535.0 : 255.0;
    const uint uq = uint(clamp(floor(maxCode * (value.x - uMin) / max(uMax - uMin, 1e-15) + 0.5), 0.0, maxCode));
    const uint sq = uint(clamp(floor(maxCode * (value.y - sMin) / max(sMax - sMin, 1e-15) + 0.5), 0.0, maxCode));
    if (p.bits == 16u) {
        Ptr<uint16_t, Access.ReadWrite> codes = (Ptr<uint16_t, Access.ReadWrite>)p.destination;
        codes[cell * 2u] = uint16_t(uq);
        codes[cell * 2u + 1u] = uint16_t(sq);
    } else {
        storeByteWord(p.destination + cell * 2u, uq | (sq << 8u), 0xffffu);
    }
}

[numthreads(256, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint3 laneId : SV_GroupThreadID, uniform Parameters p) {
    const uint lane = laneId.x;
    Ptr<int64_t, Access.Read> permutation = (Ptr<int64_t, Access.Read>)p.permutation;
    if (kPhase == 0u) {
        const uint row = groupId.x * 256u + lane;
        float uMin = 1e30, uMax = -1e30, sMin = 1e30, sMax = -1e30;
        if (row < p.primitives) {
            const int64_t sourceRow = permutation[row];
            if (sourceRow >= 0 && sourceRow < int64_t(p.primitives)) {
                const uint components = p.swizzled != 0u ? p.width * 4u : p.width;
                for (uint a = 0u; a < components; ++a) {
                    const uint attribute = p.swizzled != 0u ? a / 4u : a;
                    const uint component = p.swizzled != 0u ? a % 4u : 0u;
                    const float2 value = decode(cellIndex(uint(sourceRow), attribute, component, p), uint(sourceRow), p);
                    uMin = min(uMin, value.x); uMax = max(uMax, value.x);
                    sMin = min(sMin, value.y); sMax = max(sMax, value.y);
                }
            }
        }
        reductions[lane] = uMin; reductions[256u + lane] = uMax;
        reductions[512u + lane] = sMin; reductions[768u + lane] = sMax;
        GroupMemoryBarrierWithGroupSync();
        for (uint step = 128u; step > 0u; step >>= 1u) {
            if (lane < step) {
                reductions[lane] = min(reductions[lane], reductions[lane + step]);
                reductions[256u + lane] = max(reductions[256u + lane], reductions[256u + lane + step]);
                reductions[512u + lane] = min(reductions[512u + lane], reductions[512u + lane + step]);
                reductions[768u + lane] = max(reductions[768u + lane], reductions[768u + lane + step]);
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if (lane == 0u) {
            Ptr<float, Access.ReadWrite> output = (Ptr<float, Access.ReadWrite>)p.destinationBounds;
            output[groupId.x * 4u] = reductions[0]; output[groupId.x * 4u + 1u] = reductions[256u];
            output[groupId.x * 4u + 2u] = reductions[512u]; output[groupId.x * 4u + 3u] = reductions[768u];
        }
        return;
    }
    const uint row = groupId.x * 256u + lane;
    if (row >= p.primitives) return;
    const int64_t sourceRow = permutation[row];
    if (sourceRow < 0 || sourceRow >= int64_t(p.primitives)) return;
    const uint first = p.swizzled != 0u ? p.slotBegin : 0u;
    const uint last = p.swizzled != 0u ? p.slotEnd : p.width;
    for (uint attribute = first; attribute < last; ++attribute) {
        const uint componentCount = p.swizzled != 0u ? 4u : 1u;
        for (uint component = 0u; component < componentCount; ++component) {
            const float2 value = decode(cellIndex(uint(sourceRow), attribute, component, p), uint(sourceRow), p);
            const uint destinationAttribute = p.swizzled != 0u ? attribute - first : attribute;
            encode(destinationCellIndex(row, destinationAttribute, component, p), value, row, p);
        }
    }
}
