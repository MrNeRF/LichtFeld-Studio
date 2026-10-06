static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 32;
// Each invocation owns this many consecutive elements of a tile.
static const uint kItems = 16;
static const uint kTile = LFS_LOCAL_SIZE_X * kItems;

// Element codes follow DataType: 0 Float32, 2 Int32.
[SpecializationConstant]
const uint kDType = 0;
// 0 short lines, 1 single-tile lines, 2 tile totals, 3 tiles scanned after the scanned totals of earlier tiles.
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
// A tile staged between coalesced global access (element i * LFS_LOCAL_SIZE_X + thread) and each invocation's
// own run (element thread * kItems + i); one pad word per 32 keeps both patterns free of bank conflicts.
groupshared uint sharedTile[kTile + kTile / 32];

uint tileSlot(uint element)
{
    return element + element / 32;
}

uint64_t lineBase(uint line, Parameters parameters)
{
    const uint outerIndex = line / parameters.inner;
    const uint innerIndex = line - outerIndex * parameters.inner;
    return uint64_t(outerIndex) * parameters.dim * parameters.inner + innerIndex;
}

// Inclusive scan of one value per invocation across the workgroup; returns the exclusive prefix and leaves the
// workgroup total in the last slot.
float workgroupExclusiveFloat(uint thread, float value)
{
    sharedFloat[thread] = value;
    GroupMemoryBarrierWithGroupSync();
    for (uint offset = 1; offset < LFS_LOCAL_SIZE_X; offset <<= 1)
    {
        const float partial = thread >= offset ? sharedFloat[thread - offset] : 0.0f;
        GroupMemoryBarrierWithGroupSync();
        sharedFloat[thread] += partial;
        GroupMemoryBarrierWithGroupSync();
    }
    return thread > 0 ? sharedFloat[thread - 1] : 0.0f;
}

int workgroupExclusiveInt(uint thread, int value)
{
    sharedInt[thread] = value;
    GroupMemoryBarrierWithGroupSync();
    for (uint offset = 1; offset < LFS_LOCAL_SIZE_X; offset <<= 1)
    {
        const int partial = thread >= offset ? sharedInt[thread - offset] : 0;
        GroupMemoryBarrierWithGroupSync();
        sharedInt[thread] += partial;
        GroupMemoryBarrierWithGroupSync();
    }
    return thread > 0 ? sharedInt[thread - 1] : 0;
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

    // One workgroup per tile of a line: each invocation scans its own run, the workgroup scans the run totals,
    // and every element adds the totals before it (and, in mode 3, those of the earlier tiles).
    const uint tiles = (parameters.dim + kTile - 1) / kTile;
    const uint64_t tasks = uint64_t(parameters.lines) * (kMode == 1 ? 1u : tiles);
    for (uint64_t task = groupId.x; task < tasks; task += numWorkgroups.x)
    {
        const uint line = kMode == 1 ? uint(task) : uint(task / tiles);
        const uint tile = kMode == 1 ? 0u : uint(task % tiles);
        const uint64_t base = lineBase(line, parameters);
        const uint tileStart = tile * kTile;
        const uint tileCount = min(kTile, parameters.dim - tileStart);
        for (uint item = 0; item < kItems; ++item)
        {
            const uint element = item * LFS_LOCAL_SIZE_X + thread;
            uint bits = 0u;
            if (element < tileCount)
            {
                const uint64_t index = base + uint64_t(tileStart + element) * parameters.inner;
                bits = kDType == 0 ? asuint(floats[index]) : asuint(ints[index]);
            }
            sharedTile[tileSlot(element)] = bits;
        }
        GroupMemoryBarrierWithGroupSync();
        if (kDType == 0)
        {
            float run[kItems];
            float running = 0.0f;
            for (uint item = 0; item < kItems; ++item)
            {
                running += asfloat(sharedTile[tileSlot(thread * kItems + item)]);
                run[item] = running;
            }
            const float before = workgroupExclusiveFloat(thread, running);
            if (kMode == 2)
            {
                if (thread == LFS_LOCAL_SIZE_X - 1)
                    ((Ptr<float, Access.ReadWrite>)parameters.totalsAddress)[task] = sharedFloat[thread];
            }
            else
            {
                const float earlier = kMode == 3 && tile > 0
                                          ? ((Ptr<float, Access.Read>)parameters.totalsAddress)[task - 1]
                                          : 0.0f;
                const float offset = earlier + before;
                for (uint item = 0; item < kItems; ++item)
                    sharedTile[tileSlot(thread * kItems + item)] = asuint(run[item] + offset);
            }
        }
        else
        {
            int run[kItems];
            int running = 0;
            for (uint item = 0; item < kItems; ++item)
            {
                running += asint(sharedTile[tileSlot(thread * kItems + item)]);
                run[item] = running;
            }
            const int before = workgroupExclusiveInt(thread, running);
            if (kMode == 2)
            {
                if (thread == LFS_LOCAL_SIZE_X - 1)
                    ((Ptr<int, Access.ReadWrite>)parameters.totalsAddress)[task] = sharedInt[thread];
            }
            else
            {
                const int earlier = kMode == 3 && tile > 0
                                        ? ((Ptr<int, Access.Read>)parameters.totalsAddress)[task - 1]
                                        : 0;
                const int offset = earlier + before;
                for (uint item = 0; item < kItems; ++item)
                    sharedTile[tileSlot(thread * kItems + item)] = asuint(run[item] + offset);
            }
        }
        if (kMode != 2)
        {
            GroupMemoryBarrierWithGroupSync();
            for (uint item = 0; item < kItems; ++item)
            {
                const uint element = item * LFS_LOCAL_SIZE_X + thread;
                if (element >= tileCount)
                    continue;
                const uint64_t index = base + uint64_t(tileStart + element) * parameters.inner;
                const uint bits = sharedTile[tileSlot(element)];
                if (kDType == 0)
                    floats[index] = asfloat(bits);
                else
                    ints[index] = asint(bits);
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
