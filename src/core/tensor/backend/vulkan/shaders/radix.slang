static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 88;

// Least-significant-digit radix sort over 4-bit digits for lines longer than
// the shared-memory bitonic capacity. Every phase is deterministic: block
// histograms, an inclusive digit-major scan per line (the tensor cumsum), and a
// stable scatter whose in-block ranks come from a prefix over packed per-thread
// digit counts.
// 0 extract keys and positions, 1 block histograms, 3 scatter (source A to B or
// B to A by pass parity), 4 gather values by position into the key buffer,
// 5 write values and int64 indices.
[SpecializationConstant]
const uint kPhase = 0;
// 0 float values mapped to order-preserving keys, 1 raw int32 indices (used to
// group duplicate scatter targets deterministically).
[SpecializationConstant]
const uint kKeyKind = 0;

static const uint kDigits = 16;
static const uint kElementsPerThread = 8;
static const uint kBlockElements = LFS_LOCAL_SIZE_X * kElementsPerThread;

struct Parameters
{
    uint64_t valuesAddress;
    uint64_t indicesAddress;
    uint64_t keysAAddress;
    uint64_t keysBAddress;
    uint64_t positionsAAddress;
    uint64_t positionsBAddress;
    uint64_t histogramAddress;
    uint lines;
    uint dimSize;
    uint inner;
    uint blocksPerLine;
    uint shift;
    uint parity;
    uint descending;
    uint total;
};

groupshared uint sharedHistogram[kDigits];
groupshared uint sharedScan[LFS_LOCAL_SIZE_X];
groupshared uint4 sharedLow[LFS_LOCAL_SIZE_X];
groupshared uint4 sharedHigh[LFS_LOCAL_SIZE_X];

uint sortableKey(float value, bool descending)
{
    // Negative zero compares equal to positive zero, so both take the same key.
    uint bits = asuint(value);
    if ((bits & 0x7fffffffu) == 0u)
        bits = 0u;
    uint key;
    if ((bits & 0x7fffffffu) > 0x7f800000u)
        key = 0xffffffffu;
    else
        key = (bits & 0x80000000u) != 0u ? ~bits : (bits | 0x80000000u);
    return descending ? ~key : key;
}

uint64_t lineBase(uint line, Parameters parameters)
{
    const uint outerIndex = line / parameters.inner;
    const uint innerIndex = line - outerIndex * parameters.inner;
    return uint64_t(outerIndex) * parameters.dimSize * parameters.inner + innerIndex;
}

// Packed 16-bit counters: digit d lives in component d / 2, half d % 2.
void addPacked(inout uint4 low, inout uint4 high, uint digit, uint amount)
{
    const uint lane = digit >> 1;
    const uint value = amount << ((digit & 1u) * 16u);
    if (lane < 4)
        low[lane] += value;
    else
        high[lane - 4] += value;
}

uint readPacked(uint4 low, uint4 high, uint digit)
{
    const uint lane = digit >> 1;
    const uint word = lane < 4 ? low[lane] : high[lane - 4];
    return (word >> ((digit & 1u) * 16u)) & 0xffffu;
}

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID, uint3 groupId : SV_GroupID,
          uint3 groupThreadId : SV_GroupThreadID, uniform Parameters parameters)
{
    const uint3 numWorkgroups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    const uint stride = numWorkgroups.x * LFS_LOCAL_SIZE_X;
    const uint thread = groupThreadId.x;
    Ptr<float, Access.ReadWrite> values = (Ptr<float, Access.ReadWrite>)parameters.valuesAddress;
    Ptr<uint, Access.ReadWrite> keysA = (Ptr<uint, Access.ReadWrite>)parameters.keysAAddress;
    Ptr<uint, Access.ReadWrite> keysB = (Ptr<uint, Access.ReadWrite>)parameters.keysBAddress;
    Ptr<uint, Access.ReadWrite> positionsA = (Ptr<uint, Access.ReadWrite>)parameters.positionsAAddress;
    Ptr<uint, Access.ReadWrite> positionsB = (Ptr<uint, Access.ReadWrite>)parameters.positionsBAddress;
    Ptr<uint, Access.ReadWrite> histogram = (Ptr<uint, Access.ReadWrite>)parameters.histogramAddress;

    if (kPhase == 0)
    {
        for (uint64_t index = dispatchThreadId.x; index < parameters.total; index += stride)
        {
            const uint line = uint(index / parameters.dimSize);
            const uint element = uint(index - uint64_t(line) * parameters.dimSize);
            const uint64_t source = lineBase(line, parameters) + uint64_t(element) * parameters.inner;
            keysA[index] = kKeyKind == 0
                               ? sortableKey(values[source], parameters.descending != 0u)
                               : ((Ptr<uint, Access.Read>)parameters.valuesAddress)[source];
            positionsA[index] = element;
        }
        return;
    }
    if (kPhase == 4)
    {
        for (uint64_t index = dispatchThreadId.x; index < parameters.total; index += stride)
        {
            const uint line = uint(index / parameters.dimSize);
            const uint64_t source = lineBase(line, parameters) + uint64_t(positionsA[index]) * parameters.inner;
            keysB[index] = asuint(values[source]);
        }
        return;
    }
    if (kPhase == 5)
    {
        Ptr<int64_t, Access.ReadWrite> indices = (Ptr<int64_t, Access.ReadWrite>)parameters.indicesAddress;
        for (uint64_t index = dispatchThreadId.x; index < parameters.total; index += stride)
        {
            const uint line = uint(index / parameters.dimSize);
            const uint element = uint(index - uint64_t(line) * parameters.dimSize);
            const uint64_t target = lineBase(line, parameters) + uint64_t(element) * parameters.inner;
            values[target] = asfloat(keysB[index]);
            indices[target] = int64_t(positionsA[index]);
        }
        return;
    }
    const bool sourceIsA = parameters.parity == 0u;
    for (uint block = groupId.x; block < parameters.lines * parameters.blocksPerLine; block += numWorkgroups.x)
    {
        const uint line = block / parameters.blocksPerLine;
        const uint localBlock = block - line * parameters.blocksPerLine;
        const uint64_t lineOffset = uint64_t(line) * parameters.dimSize;
        const uint first = localBlock * kBlockElements + thread * kElementsPerThread;
        uint digits[kElementsPerThread];
        uint keys[kElementsPerThread];
        uint positions[kElementsPerThread];
        uint4 low = uint4(0, 0, 0, 0);
        uint4 high = uint4(0, 0, 0, 0);
        for (uint slot = 0; slot < kElementsPerThread; ++slot)
        {
            const uint element = first + slot;
            digits[slot] = kDigits;
            if (element < parameters.dimSize)
            {
                keys[slot] = sourceIsA ? keysA[lineOffset + element] : keysB[lineOffset + element];
                positions[slot] = sourceIsA ? positionsA[lineOffset + element] : positionsB[lineOffset + element];
                digits[slot] = (keys[slot] >> parameters.shift) & (kDigits - 1u);
                addPacked(low, high, digits[slot], 1u);
            }
        }
        if (kPhase == 1)
        {
            if (thread < kDigits)
                sharedHistogram[thread] = 0u;
            GroupMemoryBarrierWithGroupSync();
            for (uint digit = 0; digit < kDigits; ++digit)
            {
                const uint count = readPacked(low, high, digit);
                if (count != 0u)
                {
                    uint previous;
                    InterlockedAdd(sharedHistogram[digit], count, previous);
                }
            }
            GroupMemoryBarrierWithGroupSync();
            if (thread < kDigits)
                histogram[(uint64_t(line) * kDigits + thread) * parameters.blocksPerLine + localBlock] =
                    sharedHistogram[thread];
            GroupMemoryBarrierWithGroupSync();
            continue;
        }
        sharedLow[thread] = low;
        sharedHigh[thread] = high;
        GroupMemoryBarrierWithGroupSync();
        for (uint offset = 1; offset < LFS_LOCAL_SIZE_X; offset <<= 1)
        {
            uint4 partialLow = uint4(0, 0, 0, 0);
            uint4 partialHigh = uint4(0, 0, 0, 0);
            if (thread >= offset)
            {
                partialLow = sharedLow[thread - offset];
                partialHigh = sharedHigh[thread - offset];
            }
            GroupMemoryBarrierWithGroupSync();
            sharedLow[thread] += partialLow;
            sharedHigh[thread] += partialHigh;
            GroupMemoryBarrierWithGroupSync();
        }
        const uint4 exclusiveLow = sharedLow[thread] - low;
        const uint4 exclusiveHigh = sharedHigh[thread] - high;
        // The histogram scan is inclusive: this block's own digit counts come off it.
        const uint4 totalLow = sharedLow[LFS_LOCAL_SIZE_X - 1];
        const uint4 totalHigh = sharedHigh[LFS_LOCAL_SIZE_X - 1];
        uint4 seenLow = uint4(0, 0, 0, 0);
        uint4 seenHigh = uint4(0, 0, 0, 0);
        for (uint slot = 0; slot < kElementsPerThread; ++slot)
        {
            if (digits[slot] == kDigits)
                continue;
            const uint rank = readPacked(exclusiveLow, exclusiveHigh, digits[slot]) +
                              readPacked(seenLow, seenHigh, digits[slot]);
            addPacked(seenLow, seenHigh, digits[slot], 1u);
            const uint64_t destination =
                lineOffset +
                histogram[(uint64_t(line) * kDigits + digits[slot]) * parameters.blocksPerLine + localBlock] -
                readPacked(totalLow, totalHigh, digits[slot]) + rank;
            if (sourceIsA)
            {
                keysB[destination] = keys[slot];
                positionsB[destination] = positions[slot];
            }
            else
            {
                keysA[destination] = keys[slot];
                positionsA[destination] = positions[slot];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
