static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 32;

// Element codes follow DataType: 0 Float32, 2 Int32.
[SpecializationConstant]
const uint kDType = 0;
// 0 one thread per scan line, 1 one workgroup per scan line in chunks of
// LFS_LOCAL_SIZE_X elements with a carried running total.
[SpecializationConstant]
const uint kMode = 0;

struct Parameters
{
    uint64_t dataAddress;
    uint outer;
    uint dim;
    uint inner;
    uint lines;
    uint pad0;
    uint pad1;
};

groupshared float sharedFloat[LFS_LOCAL_SIZE_X];
groupshared int sharedInt[LFS_LOCAL_SIZE_X];

uint64_t lineBase(uint line, Parameters parameters)
{
    const uint outerIndex = line / parameters.inner;
    const uint innerIndex = line - outerIndex * parameters.inner;
    return uint64_t(outerIndex) * parameters.dim * parameters.inner + innerIndex;
}

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID, uint3 groupId : SV_GroupID,
          uint3 groupThreadId : SV_GroupThreadID, uniform Parameters parameters)
{
    const uint3 numWorkgroups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    const uint thread = groupThreadId.x;
    Ptr<float, Access.ReadWrite> floats = (Ptr<float, Access.ReadWrite>)parameters.dataAddress;
    Ptr<int, Access.ReadWrite> ints = (Ptr<int, Access.ReadWrite>)parameters.dataAddress;

    if (kMode == 0)
    {
        const uint stride = numWorkgroups.x * LFS_LOCAL_SIZE_X;
        for (uint line = dispatchThreadId.x; line < parameters.lines; line += stride)
        {
            const uint64_t base = lineBase(line, parameters);
            float runningFloat = 0.0f;
            int runningInt = 0;
            for (uint element = 0; element < parameters.dim; ++element)
            {
                const uint64_t index = base + uint64_t(element) * parameters.inner;
                if (kDType == 0)
                {
                    runningFloat += floats[index];
                    floats[index] = runningFloat;
                }
                else
                {
                    runningInt += ints[index];
                    ints[index] = runningInt;
                }
            }
        }
        return;
    }

    for (uint line = groupId.x; line < parameters.lines; line += numWorkgroups.x)
    {
        const uint64_t base = lineBase(line, parameters);
        float carryFloat = 0.0f;
        int carryInt = 0;
        for (uint chunk = 0; chunk < parameters.dim; chunk += LFS_LOCAL_SIZE_X)
        {
            const uint element = chunk + thread;
            const bool live = element < parameters.dim;
            const uint64_t index = base + uint64_t(element) * parameters.inner;
            if (kDType == 0)
                sharedFloat[thread] = live ? floats[index] : 0.0f;
            else
                sharedInt[thread] = live ? ints[index] : 0;
            GroupMemoryBarrierWithGroupSync();
            for (uint offset = 1; offset < LFS_LOCAL_SIZE_X; offset <<= 1)
            {
                float partialFloat = 0.0f;
                int partialInt = 0;
                if (thread >= offset)
                {
                    if (kDType == 0)
                        partialFloat = sharedFloat[thread - offset];
                    else
                        partialInt = sharedInt[thread - offset];
                }
                GroupMemoryBarrierWithGroupSync();
                if (kDType == 0)
                    sharedFloat[thread] += partialFloat;
                else
                    sharedInt[thread] += partialInt;
                GroupMemoryBarrierWithGroupSync();
            }
            if (live)
            {
                if (kDType == 0)
                    floats[index] = sharedFloat[thread] + carryFloat;
                else
                    ints[index] = sharedInt[thread] + carryInt;
            }
            if (kDType == 0)
                carryFloat += sharedFloat[LFS_LOCAL_SIZE_X - 1];
            else
                carryInt += sharedInt[LFS_LOCAL_SIZE_X - 1];
            GroupMemoryBarrierWithGroupSync();
        }
    }
}
