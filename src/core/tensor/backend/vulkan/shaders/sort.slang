static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 40;

// One workgroup sorts kCapacity / kWidth lines of at most kWidth elements
// in shared memory with a bitonic network over (key, position) pairs; the
// position tie-break makes the result equal to a stable sort. Keys are float
// bits mapped to an order-preserving unsigned value, inverted for descending
// order. kWidth is the line length rounded up to a power of two, so short
// lines run a short network side by side.
static const uint kCapacity = 2048;
static const uint kPerThread = kCapacity / LFS_LOCAL_SIZE_X;
[SpecializationConstant]
const uint kWidth = kCapacity;

struct Parameters
{
    uint64_t valuesAddress;
    uint64_t indicesAddress;
    uint lines;
    uint dimSize;
    uint inner;
    uint descending;
    uint pad0;
    uint pad1;
};

groupshared uint sharedKeys[kCapacity];
groupshared uint sharedPositions[kCapacity];

// NaN sorts after every number (ascending) and before every number
// (descending), matching the CUDA sort_less_op and sort_greater_op.
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

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint3 groupThreadId : SV_GroupThreadID,
          uniform Parameters parameters)
{
    const uint3 numWorkgroups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    const uint thread = groupThreadId.x;
    Ptr<float, Access.ReadWrite> values = (Ptr<float, Access.ReadWrite>)parameters.valuesAddress;
    Ptr<int64_t, Access.ReadWrite> indices = (Ptr<int64_t, Access.ReadWrite>)parameters.indicesAddress;
    const bool descending = parameters.descending != 0u;
    const uint linesPerGroup = kCapacity / kWidth;

    for (uint firstLine = groupId.x * linesPerGroup; firstLine < parameters.lines;
         firstLine += numWorkgroups.x * linesPerGroup)
    {
        uint64_t bases[kPerThread];
        bool inLine[kPerThread];
        for (uint slot = 0; slot < kPerThread; ++slot)
        {
            const uint element = thread + slot * LFS_LOCAL_SIZE_X;
            const uint line = firstLine + element / kWidth;
            const uint position = element % kWidth;
            const uint outerIndex = line / parameters.inner;
            const uint innerIndex = line - outerIndex * parameters.inner;
            bases[slot] = uint64_t(outerIndex) * parameters.dimSize * parameters.inner + innerIndex;
            inLine[slot] = line < parameters.lines && position < parameters.dimSize;
            if (inLine[slot])
            {
                sharedKeys[element] = sortableKey(values[bases[slot] + uint64_t(position) * parameters.inner], descending);
                sharedPositions[element] = position;
            }
            else
            {
                sharedKeys[element] = 0xffffffffu;
                sharedPositions[element] = 0xffffffffu;
            }
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint size = 2; size <= kWidth; size <<= 1)
        {
            for (uint span = size >> 1; span > 0; span >>= 1)
            {
                for (uint slot = 0; slot < kPerThread; ++slot)
                {
                    const uint element = thread + slot * LFS_LOCAL_SIZE_X;
                    const uint partner = element ^ span;
                    if (partner > element)
                    {
                        const bool ascending = ((element % kWidth) & size) == 0u;
                        const uint keyA = sharedKeys[element];
                        const uint keyB = sharedKeys[partner];
                        const uint positionA = sharedPositions[element];
                        const uint positionB = sharedPositions[partner];
                        const bool greater = keyA > keyB || (keyA == keyB && positionA > positionB);
                        if (greater == ascending)
                        {
                            sharedKeys[element] = keyB;
                            sharedKeys[partner] = keyA;
                            sharedPositions[element] = positionB;
                            sharedPositions[partner] = positionA;
                        }
                    }
                }
                GroupMemoryBarrierWithGroupSync();
            }
        }
        for (uint slot = 0; slot < kPerThread; ++slot)
        {
            const uint element = thread + slot * LFS_LOCAL_SIZE_X;
            if (inLine[slot])
            {
                const uint position = sharedPositions[element];
                indices[bases[slot] + uint64_t(element % kWidth) * parameters.inner] = int64_t(position);
                sharedKeys[element] = position;
            }
        }
        GroupMemoryBarrierWithGroupSync();
        // Values are permuted through a second shared pass so the strided line
        // is never read after it has been partially overwritten.
        for (uint slot = 0; slot < kPerThread; ++slot)
        {
            const uint element = thread + slot * LFS_LOCAL_SIZE_X;
            if (inLine[slot])
                sharedPositions[element] = asuint(values[bases[slot] + uint64_t(sharedKeys[element]) * parameters.inner]);
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint slot = 0; slot < kPerThread; ++slot)
        {
            const uint element = thread + slot * LFS_LOCAL_SIZE_X;
            if (inLine[slot])
                values[bases[slot] + uint64_t(element % kWidth) * parameters.inner] = asfloat(sharedPositions[element]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
