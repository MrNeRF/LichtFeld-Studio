static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 32;

// A transposed 2D view (rows contiguous in memory, columns strided) copied
// to a row-major output through 32x32 shared tiles, so both the reads and
// the writes of a warp are contiguous. kElementSize is 2, 4 or 8 bytes.
[SpecializationConstant]
const uint kElementSize = 4;

struct Parameters
{
    uint64_t inputAddress;
    uint64_t outputAddress;
    uint rows;
    uint columns;
    uint inputStride;
    uint padding;
};

groupshared uint64_t sharedWide[32 * 33];
groupshared uint sharedWords[32 * 33];

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint3 groupThreadId : SV_GroupThreadID, uniform Parameters parameters)
{
    const uint localX = groupThreadId.x % 32;
    const uint localY = groupThreadId.x / 32;
    const uint row0 = groupId.x * 32;
    const uint column0 = groupId.y * 32;
    for (uint step = 0; step < 32; step += 8)
    {
        const uint row = row0 + localX;
        const uint column = column0 + localY + step;
        if (row < parameters.rows && column < parameters.columns)
        {
            const uint64_t source = uint64_t(column) * parameters.inputStride + row;
            const uint slot = (localY + step) * 33 + localX;
            if (kElementSize == 8)
                sharedWide[slot] = ((Ptr<uint64_t, Access.Read>)parameters.inputAddress)[source];
            else if (kElementSize == 4)
                sharedWords[slot] = ((Ptr<uint, Access.Read>)parameters.inputAddress)[source];
            else
                sharedWords[slot] = uint(((Ptr<uint16_t, Access.Read>)parameters.inputAddress)[source]);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint step = 0; step < 32; step += 8)
    {
        const uint row = row0 + localY + step;
        const uint column = column0 + localX;
        if (row < parameters.rows && column < parameters.columns)
        {
            const uint64_t target = uint64_t(row) * parameters.columns + column;
            const uint slot = localX * 33 + localY + step;
            if (kElementSize == 8)
                ((Ptr<uint64_t, Access.ReadWrite>)parameters.outputAddress)[target] = sharedWide[slot];
            else if (kElementSize == 4)
                ((Ptr<uint, Access.ReadWrite>)parameters.outputAddress)[target] = sharedWords[slot];
            else
                ((Ptr<uint16_t, Access.ReadWrite>)parameters.outputAddress)[target] = uint16_t(sharedWords[slot]);
        }
    }
}
