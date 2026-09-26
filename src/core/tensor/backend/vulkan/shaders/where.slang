static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 184;
#include "byte_word.slangh"

// Output element type as a DataType value; byte types store whole words.
[SpecializationConstant]
const uint kOutputDType = 0;

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

// Right-aligned broadcast index, mirroring the CUDA compute_broadcast_index.
uint64_t broadcastIndex(uint64_t index, uint sourceDims[8], uint sourceRank, uint outputDims[8], uint outputRank)
{
    uint64_t sourceIndex = 0;
    uint64_t sourceStride = 1;
    for (int axis = int(outputRank) - 1; axis >= 0; --axis)
    {
        const uint64_t coordinate = axis == 0 ? index : index % outputDims[axis];
        if (axis != 0)
            index /= outputDims[axis];
        const int sourceAxis = axis - int(outputRank - sourceRank);
        if (sourceAxis >= 0)
        {
            const uint extent = sourceDims[sourceAxis];
            sourceIndex += (extent == 1u ? uint64_t(0) : coordinate) * sourceStride;
            sourceStride *= extent;
        }
    }
    return sourceIndex;
}

uint evaluateByte(uint index, uniform Parameters parameters)
{
    const uint64_t conditionIndex = broadcastIndex(index, parameters.conditionDims, parameters.conditionRank,
                                                   parameters.outputDims, parameters.outputRank);
    const bool selected = loadByte(parameters.conditionAddress, conditionIndex) != 0u;
    if (selected)
        return loadByte(parameters.xAddress, broadcastIndex(index, parameters.xDims, parameters.xRank,
                                                             parameters.outputDims, parameters.outputRank));
    return loadByte(parameters.yAddress, broadcastIndex(index, parameters.yDims, parameters.yRank,
                                                         parameters.outputDims, parameters.outputRank));
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
    for (uint64_t index = dispatchThreadId.x; index < parameters.count; index += stride)
    {
        const uint64_t conditionIndex = broadcastIndex(index, parameters.conditionDims, parameters.conditionRank,
                                                       parameters.outputDims, parameters.outputRank);
        const uint64_t xIndex = broadcastIndex(index, parameters.xDims, parameters.xRank,
                                               parameters.outputDims, parameters.outputRank);
        const uint64_t yIndex = broadcastIndex(index, parameters.yDims, parameters.yRank,
                                               parameters.outputDims, parameters.outputRank);
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
