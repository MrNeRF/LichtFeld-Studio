static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 32;

// 0 nonzero bytes, 1 nonzero floats, 2 NaN present, 3 infinity present.
[SpecializationConstant]
const uint kKind = 0;

struct Parameters
{
    uint64_t inputAddress;
    uint64_t resultAddress;
    uint count;
    uint pad0;
    uint pad1;
    uint pad2;
};

groupshared uint sharedCount[LFS_LOCAL_SIZE_X];

uint loadByte(uint64_t address, uint64_t index)
{
    const uint64_t byteAddress = address + index;
    const uint shift = uint(byteAddress & uint64_t(3)) * 8u;
    const uint64_t wordAddress = byteAddress & ~uint64_t(3);
    return (((Ptr<uint, Access.Read>)wordAddress)[0] >> shift) & 255u;
}

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID, uint3 groupThreadId : SV_GroupThreadID,
          uniform Parameters parameters)
{
    const uint3 numWorkgroups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    const uint stride = numWorkgroups.x * LFS_LOCAL_SIZE_X;
    const uint thread = groupThreadId.x;
    uint matches = 0;
    for (uint64_t index = dispatchThreadId.x; index < parameters.count; index += stride)
    {
        if (kKind == 0)
        {
            matches += loadByte(parameters.inputAddress, index) != 0u ? 1u : 0u;
        }
        else
        {
            const uint bits = asuint(((Ptr<float, Access.Read>)parameters.inputAddress)[index]);
            const uint magnitude = bits & 0x7fffffffu;
            if (kKind == 1)
                matches += magnitude != 0u ? 1u : 0u;
            else if (kKind == 2)
                matches += magnitude > 0x7f800000u ? 1u : 0u;
            else
                matches += magnitude == 0x7f800000u ? 1u : 0u;
        }
    }
    sharedCount[thread] = matches;
    GroupMemoryBarrierWithGroupSync();
    for (uint span = LFS_LOCAL_SIZE_X / 2; span > 0; span >>= 1)
    {
        if (thread < span)
            sharedCount[thread] += sharedCount[thread + span];
        GroupMemoryBarrierWithGroupSync();
    }
    if (thread == 0 && sharedCount[0] != 0u)
    {
        Ptr<uint, Access.ReadWrite> result = (Ptr<uint, Access.ReadWrite>)parameters.resultAddress;
        if (kKind <= 1)
            InterlockedAdd(result[0], sharedCount[0]);
        else
            InterlockedOr(result[0], 1u);
    }
}
