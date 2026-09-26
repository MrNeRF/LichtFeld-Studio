static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 24;


struct Parameters {
    uint64_t input;
    uint64_t output;
    uint size;
    uint padding;
};

groupshared uint localCounts[256];

#include "byte_read.slangh"

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID, uint3 threadId : SV_GroupThreadID, uniform Parameters p) {
    const uint lane = threadId.x;
    localCounts[lane] = 0u;
    GroupMemoryBarrierWithGroupSync();
    const uint3 groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    const uint stride = groups.x * LFS_LOCAL_SIZE_X;
    for (uint index = dispatchThreadId.x; index < p.size; index += stride) {
        const uint value = loadByte(p.input, index);
        if (value != 0u) InterlockedAdd(localCounts[value], 1u);
    }
    GroupMemoryBarrierWithGroupSync();
    const uint count = localCounts[lane];
    if (count != 0u) {
        Ptr<uint, Access.ReadWrite> output = (Ptr<uint, Access.ReadWrite>)p.output;
        InterlockedAdd(output[lane], count);
    }
}
