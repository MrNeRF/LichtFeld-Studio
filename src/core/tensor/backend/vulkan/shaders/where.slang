static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 184;
#include "byte_word.slangh"

// Output element type as a DataType value; byte types store whole words.
[SpecializationConstant]
const uint kOutputDType = 0;
// Bits 0, 1 and 2 mark the condition, x and y as matching the output shape,
// so they read at the output index. The host merges axes first, so all four
// dims arrays share the output's rank.
[SpecializationConstant]
const uint kDirect = 0;

struct Parameters
{
    uint64_t conditionAddress;
    uint64_t xAddress;
    uint64_t yAddress;
    uint64_t outputAddress;
    uint conditionDims[8];
    uint xDims[8];
    uint yDims[8];
    uint outputDims[8];
    uint conditionRank;
    uint xRank;
    uint yRank;
    uint outputRank;
    uint count;
    uint pad0;
};

uint loadByte(uint64_t address, uint64_t index)
{
    const uint64_t byteAddress = address + index;
    const uint shift = uint(byteAddress & uint64_t(3)) * 8u;
    const uint64_t wordAddress = byteAddress & ~uint64_t(3);
    return (((Ptr<uint, Access.Read>)wordAddress)[0] >> shift) & 255u;
}

// The broadcast source index of an output element, in 32 bits: the count
// fits, and every source is no larger than the output.
uint broadcastIndex(uint index, uint sourceDims[8], uint outputDims[8], uint rank, uint direct)
{
    if (direct != 0u)
        return index;
    uint sourceIndex = 0;
    uint sourceStride = 1;
    for (int axis = int(rank) - 1; axis >= 0; --axis)
    {
        const uint coordinate = axis == 0 ? index : index % outputDims[axis];
        if (axis != 0)
            index /= outputDims[axis];
        const uint extent = sourceDims[axis];
        sourceIndex += (extent == 1u ? 0u : coordinate) * sourceStride;
        sourceStride *= extent;
    }
    return sourceIndex;
}

uint evaluateByte(uint index, uniform Parameters parameters)
{
    const uint conditionIndex = broadcastIndex(index, parameters.conditionDims, parameters.outputDims,
                                               parameters.outputRank, kDirect & 1u);
    const bool selected = loadByte(parameters.conditionAddress, conditionIndex) != 0u;
    if (selected)
        return loadByte(parameters.xAddress, broadcastIndex(index, parameters.xDims, parameters.outputDims,
                                                            parameters.outputRank, kDirect & 2u));
    return loadByte(parameters.yAddress, broadcastIndex(index, parameters.yDims, parameters.outputDims,
                                                        parameters.outputRank, kDirect & 4u));
}

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID, uniform Parameters parameters)
{
    if (kOutputDType == 4 || kOutputDType == 5)
    {
        const uint64_t start = parameters.outputAddress;
        const uint64_t end = start + uint64_t(parameters.count);
        const uint64_t firstWord = start & ~uint64_t(3);
        const uint wordCount = uint((end - firstWord + uint64_t(3)) / uint64_t(4));
        const uint3 groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
        for (uint wordIndex = dispatchThreadId.x; wordIndex < wordCount;
             wordIndex += groups.x * LFS_LOCAL_SIZE_X)
        {
            const uint64_t wordAddress = firstWord + uint64_t(wordIndex) * uint64_t(4);
            uint packed = 0;
            uint mask = 0;
            [unroll]
            for (uint lane = 0; lane < 4u; ++lane)
            {
                const uint64_t byteAddress = wordAddress + uint64_t(lane);
                if (byteAddress >= start && byteAddress < end)
                {
                    packed |= evaluateByte(uint(byteAddress - start), parameters) << (lane * 8u);
                    mask |= 255u << (lane * 8u);
                }
            }
            storeByteWord(wordAddress, packed, mask);
        }
        return;
    }
    const uint3 numWorkgroups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    const uint stride = numWorkgroups.x * LFS_LOCAL_SIZE_X;
    for (uint index = dispatchThreadId.x; index < parameters.count; index += stride)
    {
        const uint conditionIndex = broadcastIndex(index, parameters.conditionDims, parameters.outputDims,
                                                   parameters.outputRank, kDirect & 1u);
        const uint xIndex = broadcastIndex(index, parameters.xDims, parameters.outputDims, parameters.outputRank,
                                           kDirect & 2u);
        const uint yIndex = broadcastIndex(index, parameters.yDims, parameters.outputDims, parameters.outputRank,
                                           kDirect & 4u);
        const bool selected = loadByte(parameters.conditionAddress, conditionIndex) != 0u;
        if (kOutputDType == 1)
            ((Ptr<uint16_t, Access.ReadWrite>)parameters.outputAddress)[index] =
                selected ? ((Ptr<uint16_t, Access.Read>)parameters.xAddress)[xIndex]
                         : ((Ptr<uint16_t, Access.Read>)parameters.yAddress)[yIndex];
        else if (kOutputDType == 3)
            ((Ptr<uint64_t, Access.ReadWrite>)parameters.outputAddress)[index] =
                selected ? ((Ptr<uint64_t, Access.Read>)parameters.xAddress)[xIndex]
                         : ((Ptr<uint64_t, Access.Read>)parameters.yAddress)[yIndex];
        else
            ((Ptr<uint, Access.ReadWrite>)parameters.outputAddress)[index] =
                selected ? ((Ptr<uint, Access.Read>)parameters.xAddress)[xIndex]
                         : ((Ptr<uint, Access.Read>)parameters.yAddress)[yIndex];
    }
}
