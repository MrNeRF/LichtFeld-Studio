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

// Swizzled slot indices are 64-bit: with 12 SH slots (degree 3) a 32-bit slot index wraps past 357.9M rows.
uint64_t cellIndex(uint primitive, uint attribute, uint component, uniform Parameters p) {
    if (p.swizzled == 0u) return uint64_t(primitive) * p.width + attribute;
    const uint64_t slot = uint64_t(primitive / 32u) * (p.width * 32u) + attribute * 32u + (primitive % 32u);
    return slot * 4u + component;
}

uint64_t destinationCellIndex(uint primitive, uint attribute, uint component, uniform Parameters p) {
    if (p.swizzled == 0u) return uint64_t(primitive) * p.width + attribute;
    const uint groupWidth = p.slotEnd - p.slotBegin;
    const uint64_t slot = uint64_t(primitive / 32u) * (groupWidth * 32u) + attribute * 32u + (primitive % 32u);
    return slot * 4u + component;
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
        // A cell's codes at their 64-bit byte address (the 8-bit path's byte helpers already address in integer
        // math): NVIDIA's Vulkan computes an access chain's byte offset in 32 bits without
        // VK_EXT_shader_64bit_indexing.
        Ptr<uint16_t, Access.Read> codes = (Ptr<uint16_t, Access.Read>)(p.source + cell * 4u);
        uq = float(codes[0]);
        sq = float(codes[1]);
    } else {
        uq = float(loadByte(p.source, cell * 2u));
        sq = float(loadByte(p.source, cell * 2u + 1u));
    }
    // joint_adam_codec.cuh decode_us: multiply by the reciprocal code range.
    const float invMaxCode = p.bits == 16u ? 1.0f / 65535.0f : 1.0f / 255.0f;
    return float2(uMin + (uMax - uMin) * (uq * invMaxCode),
                  sMin + (sMax - sMin) * (sq * invMaxCode));
}

// roundf (half away from zero), then clamp to the code range.
uint codeOf(float scaled, float maxCode) {
    const float magnitude = floor(abs(scaled));
    const float rounded = sign(scaled) * (magnitude + (abs(scaled) - magnitude >= 0.5f ? 1.0f : 0.0f));
    return uint(clamp(rounded, 0.0f, maxCode));
}

void encode(uint64_t cell, float2 value, uint destinationRow, uniform Parameters p) {
    Ptr<float, Access.Read> bounds = (Ptr<float, Access.Read>)p.destinationBounds;
    const uint block = destinationRow / 256u;
    const float uMin = bounds[block * 4u];
    const float uMax = bounds[block * 4u + 1u];
    const float sMin = bounds[block * 4u + 2u];
    const float sMax = bounds[block * 4u + 3u];
    const float maxCode = p.bits == 16u ? 65535.0 : 255.0;
    // joint_adam_codec.cuh encode_us: hoisted reciprocal ranges and roundf.
    const uint uq = codeOf(maxCode * (value.x - uMin) * (1.0f / max(uMax - uMin, 1e-15f)), maxCode);
    const uint sq = codeOf(maxCode * (value.y - sMin) * (1.0f / max(sMax - sMin, 1e-15f)), maxCode);
    if (p.bits == 16u) {
        Ptr<uint16_t, Access.ReadWrite> codes = (Ptr<uint16_t, Access.ReadWrite>)(p.destination + cell * 4u);
        codes[0] = uint16_t(uq);
        codes[1] = uint16_t(sq);
    } else {
        // Odd cells sit in the upper half of their aligned word.
        const uint64_t address = p.destination + cell * 2u;
        const uint shift = uint(address & 3u) * 8u;
        storeByteWord(address & ~uint64_t(3u), (uq | (sq << 8u)) << shift, 0xffffu << shift);
    }
}

[numthreads(256, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint3 laneId : SV_GroupThreadID, uniform Parameters p) {
    const uint lane = laneId.x;
    if (kPhase == 2u) {
        uint3 grid = spirv_asm {
        result:
            $$uint3 = OpLoad builtin(NumWorkgroups : uint3);
        };
        const uint wordsPerSlot = 32u * (p.bits == 16u ? 4u : 2u);
        const uint groupWords = (p.slotEnd - p.slotBegin) * wordsPerSlot;
        const uint64_t words = uint64_t((p.primitives + 31u) / 32u) * groupWords;
        // Words at their 64-bit byte address: the moments pass 4 GiB at 44.7M rows (8-bit, degree 3), past which
        // NVIDIA's Vulkan wraps an access chain's 32-bit byte offset without VK_EXT_shader_64bit_indexing.
        for (uint64_t word = uint64_t(groupId.x) * 256u + lane;
             word < words; word += uint64_t(grid.x) * 256u) {
            const uint64_t tile = word / groupWords;
            const uint local = uint(word % groupWords);
            const uint64_t to = tile * p.width * wordsPerSlot + p.slotBegin * wordsPerSlot + local;
            ((Ptr<uint, Access.ReadWrite>)(p.destination + to * 4u))[0] = ((Ptr<uint, Access.Read>)(p.source + word * 4u))[0];
        }
        return;
    }
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
