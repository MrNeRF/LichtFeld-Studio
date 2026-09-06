static const uint kWorkgroupSize = 256;
groupshared float subgroupSums[kWorkgroupSize];

struct Parameters
{
    uint64_t inputAddress;
    uint64_t outputAddress;
    uint totalColumns;
    uint rowCount;
    uint rowStride;
    uint segmentSize;
    uint segmentCount;
};

[numthreads(kWorkgroupSize, 1, 1)]
void main(
    uint3 groupId : SV_GroupID,
    uint3 localId : SV_GroupThreadID,
    uniform Parameters parameters)
{
    const uint row = groupId.x / parameters.segmentCount;
    const uint segment = groupId.x % parameters.segmentCount;
    if (row >= parameters.rowCount)
        return;

    Ptr<float, Access.Read> input = (Ptr<float, Access.Read>)parameters.inputAddress;
    Ptr<float, Access.ReadWrite> output = (Ptr<float, Access.ReadWrite>)parameters.outputAddress;
    const uint firstColumn = segment * parameters.segmentSize;
    const uint endColumn = min(firstColumn + parameters.segmentSize, parameters.totalColumns);
    float threadSum = 0.0f;
    for (uint column = firstColumn + localId.x; column < endColumn; column += kWorkgroupSize)
        threadSum += input[row * parameters.rowStride + column];

    const uint laneCount = WaveGetLaneCount();
    const uint subgroupIndex = localId.x / laneCount;
    const float subgroupSum = WaveActiveSum(threadSum);
    if (WaveIsFirstLane())
        subgroupSums[subgroupIndex] = subgroupSum;
    GroupMemoryBarrierWithGroupSync();

    if (localId.x == 0)
    {
        const uint subgroupCount = (kWorkgroupSize + laneCount - 1) / laneCount;
        float total = 0.0f;
        for (uint subgroup = 0; subgroup < subgroupCount; ++subgroup)
            total += subgroupSums[subgroup];
        output[row * parameters.segmentCount + segment] = total;
    }
}
