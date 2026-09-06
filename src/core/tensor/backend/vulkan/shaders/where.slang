static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 184;

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
    uint64_t outputStride = 1;
    for (int axis = int(outputRank) - 1; axis >= 0; --axis)
    {
        const uint64_t coordinate = (index / outputStride) % outputDims[axis];
        const int sourceAxis = axis - int(outputRank - sourceRank);
        if (sourceAxis >= 0)
        {
            const uint extent = sourceDims[sourceAxis];
            sourceIndex += (extent == 1u ? uint64_t(0) : coordinate) * sourceStride;
            sourceStride *= extent;
        }
        outputStride *= outputDims[axis];
    }
    return sourceIndex;
}

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID, uniform Parameters parameters)
{
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
        ((Ptr<float, Access.ReadWrite>)parameters.outputAddress)[index] =
            selected ? ((Ptr<float, Access.Read>)parameters.xAddress)[xIndex]
                     : ((Ptr<float, Access.Read>)parameters.yAddress)[yIndex];
    }
}
