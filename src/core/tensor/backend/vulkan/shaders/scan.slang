static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 32;

// Element codes follow DataType: 0 Float32, 2 Int32.
[SpecializationConstant]
const uint kDType = 0;
// 0 short lines, 1 single-block lines, 2 independent blocks, 3 scanned block offsets.
[SpecializationConstant]
const uint kMode = 0;

struct Parameters
{
    uint64_t dataAddress;
    uint outer;
    uint dim;
    uint inner;
    uint lines;
    uint64_t totalsAddress;
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

    const uint blocks = (parameters.dim + LFS_LOCAL_SIZE_X - 1) / LFS_LOCAL_SIZE_X;
    if (kMode == 3)
    {
        const uint64_t count = uint64_t(parameters.lines) * parameters.dim;
        const uint stride = numWorkgroups.x * LFS_LOCAL_SIZE_X;
        for (uint64_t index = dispatchThreadId.x; index < count; index += stride)
        {
            const uint element = uint(index / parameters.inner) % parameters.dim;
            if (element < LFS_LOCAL_SIZE_X) continue;
            const uint64_t line = index / (uint64_t(parameters.dim) * parameters.inner) * parameters.inner
                                  + index % parameters.inner;
            const uint64_t preceding = line * blocks + element / LFS_LOCAL_SIZE_X - 1;
            if (kDType == 0)
                floats[index] += ((Ptr<float, Access.Read>)parameters.totalsAddress)[preceding];
            else
                ints[index] += ((Ptr<int, Access.Read>)parameters.totalsAddress)[preceding];
        }
        return;
    }
    const uint64_t tasks = uint64_t(parameters.lines) * (kMode == 2 ? blocks : 1u);
    for (uint64_t task = groupId.x; task < tasks; task += numWorkgroups.x)
    {
        const uint line = kMode == 2 ? uint(task / blocks) : uint(task);
        const uint element = (kMode == 2 ? uint(task % blocks) * LFS_LOCAL_SIZE_X : 0u) + thread;
        const bool live = element < parameters.dim;
        const uint64_t index = lineBase(line, parameters) + uint64_t(element) * parameters.inner;
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
                if (kDType == 0) partialFloat = sharedFloat[thread - offset];
                else partialInt = sharedInt[thread - offset];
            }
            GroupMemoryBarrierWithGroupSync();
            if (kDType == 0) sharedFloat[thread] += partialFloat;
            else sharedInt[thread] += partialInt;
            GroupMemoryBarrierWithGroupSync();
        }
        if (live)
        {
            if (kDType == 0) floats[index] = sharedFloat[thread];
            else ints[index] = sharedInt[thread];
        }
        if (kMode == 2 && thread == LFS_LOCAL_SIZE_X - 1)
        {
            if (kDType == 0)
                ((Ptr<float, Access.ReadWrite>)parameters.totalsAddress)[task] = sharedFloat[thread];
            else
                ((Ptr<int, Access.ReadWrite>)parameters.totalsAddress)[task] = sharedInt[thread];
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
