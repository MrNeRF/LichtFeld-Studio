#ifndef PAIR_SORT_SUBGROUP_MATCH
#define PAIR_SORT_SUBGROUP_MATCH 1
#endif

static const uint kWorkers = PAIR_SORT_SUBGROUP_MATCH != 0 ? 128 : 32;
static const uint kItemsPerWorker = PAIR_SORT_SUBGROUP_MATCH != 0 ? 16 : 64;
static const uint kPartitionWidth = kWorkers * kItemsPerWorker;
static const uint kDigitCount = 256;
static const uint kMaxSubgroups = PAIR_SORT_SUBGROUP_MATCH != 0 ? 4 : 16;

[SpecializationConstant] const uint kStage = 0;

struct SortParameters {
    uint64_t sourceKeys;
    uint64_t destinationKeys;
    uint64_t sourceValues;
    uint64_t destinationValues;
    uint64_t partitionCounts;
    uint64_t digitOffsets;
    uint elementCount;
    uint partitionCount;
    uint bitOffset;
    uint keyWidth;
    uint significantBits;
    uint passCount;
    uint passIndex;
};

groupshared uint localCounts[kDigitCount];
groupshared uint roundPrefix[kDigitCount];
groupshared uint subgroupCounts[kMaxSubgroups * kDigitCount];
groupshared uint subgroupTotals[kMaxSubgroups];
groupshared uint chunkCarry;
#if PAIR_SORT_SUBGROUP_MATCH
groupshared uint sortedIndex[kPartitionWidth];
void subgroupBarrier() {
    const uint scope = 3, semantics = 264;
    spirv_asm { OpControlBarrier $scope $scope $semantics; };
}
#endif
#if PAIR_SORT_SUBGROUP_MATCH == 0
groupshared uint laneDigits[kWorkers];
#endif

uint loadDigitAt(uint2 key, SortParameters parameters, uint bitOffset) {
    const uint shift = bitOffset & 31u;
    const uint lowerWord = bitOffset < 32u ? key.x : key.y;
    uint word = lowerWord >> shift;
    if (bitOffset < 32u && shift != 0u)
        word |= key.y << (32u - shift);
    const uint remaining = parameters.significantBits - bitOffset;
    const uint mask = remaining < 8u ? (1u << remaining) - 1u : 255u;
    return word & mask;
}

uint loadDigit(uint2 key, SortParameters parameters) {
    return loadDigitAt(key, parameters, parameters.bitOffset);
}

uint2 loadKey(SortParameters parameters, uint index) {
    if (parameters.keyWidth == 32u) {
        Ptr<uint, Access.Read> keys = (Ptr<uint, Access.Read>)parameters.sourceKeys;
        return uint2(keys[index], 0u);
    }
    Ptr<uint2, Access.Read> keys = (Ptr<uint2, Access.Read>)parameters.sourceKeys;
    return keys[index];
}

void storeKey(SortParameters parameters, uint index, uint2 key) {
    if (parameters.keyWidth == 32u) {
        Ptr<uint, Access.ReadWrite> keys = (Ptr<uint, Access.ReadWrite>)parameters.destinationKeys;
        keys[index] = key.x;
        return;
    }
    Ptr<uint2, Access.ReadWrite> keys = (Ptr<uint2, Access.ReadWrite>)parameters.destinationKeys;
    keys[index] = key;
}

uint matchCount(uint4 matchMask) {
    return countbits(matchMask.x) + countbits(matchMask.y) +
           countbits(matchMask.z) + countbits(matchMask.w);
}

uint matchPrefix(uint4 matchMask, uint lane) {
    const uint word = lane >> 5u;
    const uint bit = lane & 31u;
    uint result = 0u;
    if (word > 0u) result += countbits(matchMask.x);
    if (word > 1u) result += countbits(matchMask.y);
    if (word > 2u) result += countbits(matchMask.z);
    const uint bits = bit == 0u ? 0u : (1u << bit) - 1u;
    if (word == 0u) result += countbits(matchMask.x & bits);
    else if (word == 1u) result += countbits(matchMask.y & bits);
    else if (word == 2u) result += countbits(matchMask.z & bits);
    else result += countbits(matchMask.w & bits);
    return result;
}

bool firstMatchingLane(uint4 matchMask, uint lane) {
    uint first = 0u;
    if (matchMask.x != 0u) first = firstbitlow(matchMask.x);
    else if (matchMask.y != 0u) first = 32u + firstbitlow(matchMask.y);
    else if (matchMask.z != 0u) first = 64u + firstbitlow(matchMask.z);
    else first = 96u + firstbitlow(matchMask.w);
    return lane == first;
}

[numthreads(kWorkers, 1, 1)] void main(uint3 dispatchThreadId : SV_DispatchThreadID,
                                       uint3 groupId : SV_GroupID,
                                       uint3 groupThreadId : SV_GroupThreadID,
                                       uniform SortParameters parameters) {
    Ptr<uint, Access.Read> valuesIn = (Ptr<uint, Access.Read>)parameters.sourceValues;
    Ptr<uint, Access.ReadWrite> valuesOut = (Ptr<uint, Access.ReadWrite>)parameters.destinationValues;
    Ptr<uint, Access.ReadWrite> counts = (Ptr<uint, Access.ReadWrite>)parameters.partitionCounts;
    Ptr<uint, Access.ReadWrite> offsets = (Ptr<uint, Access.ReadWrite>)parameters.digitOffsets;
    const uint lane = groupThreadId.x;

    if (kStage == 0u) {
        const uint first = groupId.x * kPartitionWidth;

            for (uint digit = lane; digit < kDigitCount; digit += kWorkers)
                localCounts[digit] = 0u;
            GroupMemoryBarrierWithGroupSync();
            for (uint item = 0u; item < kItemsPerWorker; ++item) {
                const uint index = first + item * kWorkers + lane;
                const bool valid = index < parameters.elementCount;
                const uint digit = valid ? loadDigit(loadKey(parameters, index), parameters)
                                         : kDigitCount;
#if PAIR_SORT_SUBGROUP_MATCH
                const uint4 matches = WaveMatch(digit);
                const uint subgroupLane = WaveGetLaneIndex();
                if (valid && firstMatchingLane(matches, subgroupLane))
                    InterlockedAdd(localCounts[digit], matchCount(matches));
#else
                if (valid)
                    InterlockedAdd(localCounts[digit], 1u);
#endif
            }
            GroupMemoryBarrierWithGroupSync();
            for (uint digit = lane; digit < kDigitCount; digit += kWorkers)
                counts[(parameters.passIndex * kDigitCount + digit) * parameters.partitionCount +
                       groupId.x] = localCounts[digit];
        return;
    }

    if (kStage == 1u) {
        const uint digit = groupId.x;
        const uint row = (parameters.passIndex * kDigitCount + digit) * parameters.partitionCount;
        const uint subgroup = lane / WaveGetLaneCount();
        const uint subgroupLane = WaveGetLaneIndex();
        uint carry = 0u;
        for (uint first = 0u; first < parameters.partitionCount; first += kWorkers) {
            const uint partition = first + lane;
            const uint amount = partition < parameters.partitionCount ? counts[row + partition] : 0u;
            const uint subgroupSum = WaveActiveSum(amount);
            if (subgroupLane == 0u)
                subgroupTotals[subgroup] = subgroupSum;
            GroupMemoryBarrierWithGroupSync();
            uint subgroupPrefix = 0u;
            for (uint previous = 0u; previous < subgroup; ++previous)
                subgroupPrefix += subgroupTotals[previous];
            if (partition < parameters.partitionCount)
                counts[row + partition] = carry + subgroupPrefix + WavePrefixSum(amount);
            GroupMemoryBarrierWithGroupSync();
            if (lane == 0u) {
                uint chunkTotal = 0u;
                for (uint previous = 0u; previous < kMaxSubgroups; ++previous) {
                    if (previous < kWorkers / WaveGetLaneCount())
                        chunkTotal += subgroupTotals[previous];
                }
                chunkCarry = carry + chunkTotal;
            }
            GroupMemoryBarrierWithGroupSync();
            carry = chunkCarry;
        }
        return;
    }

    if (kStage == 2u) {
        const uint pass = parameters.passIndex;
        const uint digit = groupId.x % kDigitCount;
        const uint row = (pass * kDigitCount + digit) * parameters.partitionCount;
        uint localTotal = 0u;
        for (uint partition = lane; partition < parameters.partitionCount; partition += kWorkers)
            localTotal += counts[row + partition];
        const uint subgroup = lane / WaveGetLaneCount();
        const uint subgroupLane = WaveGetLaneIndex();
        const uint subgroupTotal = WaveActiveSum(localTotal);
        if (subgroupLane == 0u)
            subgroupTotals[subgroup] = subgroupTotal;
        GroupMemoryBarrierWithGroupSync();
        if (lane == 0u) {
            uint total = 0u;
            for (uint previous = 0u; previous < kWorkers / WaveGetLaneCount(); ++previous)
                total += subgroupTotals[previous];
            offsets[pass * kDigitCount + digit] = total;
        }
        return;
    }

    if (kStage == 3u) {
        const uint pass = parameters.passIndex;
        if (lane == 0u) {
            uint base = 0u;
            for (uint digit = 0u; digit < kDigitCount; ++digit) {
                const uint index = pass * kDigitCount + digit;
                const uint amount = offsets[index];
                offsets[index] = base;
                base += amount;
            }
        }
        return;
    }

    if (kStage == 4u) {
#if PAIR_SORT_SUBGROUP_MATCH
        // Each subgroup owns a contiguous input segment, preserving producer order.
        // Rank locally before exchanging subgroup prefixes and coalescing the scatter.
        const uint partition = groupId.x;
        const uint subgroup = lane / 32, subgroupLane = lane % 32;
        for (uint slot = lane; slot < 4 * kDigitCount; slot += kWorkers)
            subgroupCounts[slot] = 0;
        GroupMemoryBarrierWithGroupSync();
        uint ranks[kItemsPerWorker];
        for (uint item = 0; item < kItemsPerWorker; ++item) {
            uint localIndex = subgroup * 32 * kItemsPerWorker + item * 32 + subgroupLane;
            uint index = partition * kPartitionWidth + localIndex;
            bool valid = index < parameters.elementCount;
            uint digit = valid ? loadDigit(loadKey(parameters,index),parameters) : 0;
            uint4 matches = WaveMatch(valid ? digit : 256);
            uint rank = subgroupCounts[subgroup*kDigitCount+digit] + matchPrefix(matches,subgroupLane);
            subgroupBarrier();
            if (valid && firstMatchingLane(matches,subgroupLane))
                subgroupCounts[subgroup*kDigitCount+digit] += matchCount(matches);
            subgroupBarrier();
            ranks[item] = rank;
        }
        GroupMemoryBarrierWithGroupSync();
        uint carry = 0;
        for (uint chunk = 0; chunk < kDigitCount; chunk += kWorkers) {
            uint digit = chunk + lane, amount = 0;
            for (uint wave = 0; wave < 4; ++wave) {
                uint count = subgroupCounts[wave*kDigitCount+digit];
                subgroupCounts[wave*kDigitCount+digit] = amount;
                amount += count;
            }
            uint prefix = WavePrefixSum(amount), sum = WaveActiveSum(amount);
            if (subgroupLane == 0) subgroupTotals[subgroup] = sum;
            GroupMemoryBarrierWithGroupSync();
            uint earlier = 0;
            for (uint previous = 0; previous < subgroup; ++previous) earlier += subgroupTotals[previous];
            roundPrefix[digit] = carry + earlier + prefix;
            for (uint previous = 0; previous < 4; ++previous) carry += subgroupTotals[previous];
            GroupMemoryBarrierWithGroupSync();
        }
        for (uint item = 0; item < kItemsPerWorker; ++item) {
            uint localIndex = subgroup * 32 * kItemsPerWorker + item * 32 + subgroupLane;
            uint index = partition * kPartitionWidth + localIndex;
            if (index < parameters.elementCount) {
                uint digit = loadDigit(loadKey(parameters,index),parameters);
                uint rank = roundPrefix[digit] + subgroupCounts[subgroup*kDigitCount+digit] + ranks[item];
                sortedIndex[rank] = localIndex;
            }
        }
        GroupMemoryBarrierWithGroupSync();
        uint validCount = min(kPartitionWidth, parameters.elementCount-partition*kPartitionWidth);
        for (uint localIndex = lane; localIndex < validCount; localIndex += kWorkers) {
            uint sourceIndex = partition*kPartitionWidth + sortedIndex[localIndex];
            uint2 key = loadKey(parameters,sourceIndex);
            uint digit = loadDigit(key,parameters);
            uint row = (parameters.passIndex*kDigitCount+digit)*parameters.partitionCount;
            uint outIndex = offsets[parameters.passIndex*kDigitCount+digit]+counts[row+partition]+localIndex-roundPrefix[digit];
            storeKey(parameters,outIndex,key);
            valuesOut[outIndex] = valuesIn[sourceIndex];
        }
#else

        const uint partition = groupId.x;
        const uint subgroup = lane / WaveGetLaneCount();
        const uint subgroupLane = WaveGetLaneIndex();
        for (uint digit = lane; digit < kDigitCount; digit += kWorkers)
            roundPrefix[digit] = 0u;
        GroupMemoryBarrierWithGroupSync();
        for (uint item = 0u; item < kItemsPerWorker; ++item) {
            const uint index = partition * kPartitionWidth + item * kWorkers + lane;
            const bool valid = index < parameters.elementCount;
            const uint2 key = valid ? loadKey(parameters, index) : uint2(0u, 0u);
            const uint digit = valid ? loadDigit(key, parameters) : kDigitCount;
            laneDigits[lane] = digit;
            GroupMemoryBarrierWithGroupSync();
            uint localRank = 0u;
            for (uint previous = 0u; previous < lane; ++previous)
                localRank += laneDigits[previous] == digit ? 1u : 0u;
            if (valid) {
                const uint row = (parameters.passIndex * kDigitCount + digit) * parameters.partitionCount;
                const uint outIndex = offsets[parameters.passIndex * kDigitCount + digit] +
                                      counts[row + partition] + roundPrefix[digit] + localRank;
                storeKey(parameters, outIndex, key);
                valuesOut[outIndex] = valuesIn[index];
            }
            GroupMemoryBarrierWithGroupSync();
            for (uint digitIndex = lane; digitIndex < kDigitCount; digitIndex += kWorkers) {
                uint amount = 0u;
                for (uint current = 0u; current < kWorkers; ++current)
                    amount += laneDigits[current] == digitIndex ? 1u : 0u;
                roundPrefix[digitIndex] += amount;
            }
            GroupMemoryBarrierWithGroupSync();
        }
#endif
    }
}
