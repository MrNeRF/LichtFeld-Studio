static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 24;

struct FillParameters
{
    uint64_t outputAddress;
    uint64_t pattern;
    uint byteCount;
    uint elementSize;
};

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID, uniform FillParameters parameters)
{
    const uint wordIndex = dispatchThreadId.x;
    const uint byteOffset = wordIndex * 4;
    if (byteOffset >= parameters.byteCount)
        return;
    Ptr<uint, Access.ReadWrite> output =
        (Ptr<uint, Access.ReadWrite>)parameters.outputAddress;
    const uint shift = (byteOffset % parameters.elementSize) * 8;
    output[wordIndex] = uint(parameters.pattern >> shift);
}
