static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 160;

[SpecializationConstant]
const uint kDType = 0;

struct FillParameters
{
    uint64_t outputAddress;
    uint64_t pattern;
    uint64_t dims[8];
    uint64_t strides[8];
    uint count;
    uint rank;
    uint strided;
    uint padding;
};

void storeByte(uint64_t address, uint64_t index, uint value)
{
    const uint64_t byteAddress = address + index;
    const uint shift = uint(byteAddress & uint64_t(3)) * 8u;
    const uint mask = 255u << shift;
    Ptr<uint, Access.ReadWrite> word =
        (Ptr<uint, Access.ReadWrite>)(byteAddress & ~uint64_t(3));
    uint assumed = word[0];
    uint observed;
    do
    {
        const uint replacement = (assumed & ~mask) | ((value & 255u) << shift);
        InterlockedCompareExchange(word[0], assumed, replacement, observed);
        if (observed == assumed) return;
        assumed = observed;
    } while (true);
}

uint64_t logicalOffset(uint linear, uniform FillParameters parameters)
{
    if (parameters.strided == 0u)
        return uint64_t(linear);
    uint64_t offset = 0;
    for (uint reverse = 0; reverse < parameters.rank; ++reverse)
    {
        const uint dimension = parameters.rank - reverse - 1;
        const uint64_t coordinate = uint64_t(linear) % parameters.dims[dimension];
        linear = uint(uint64_t(linear) / parameters.dims[dimension]);
        offset += coordinate * parameters.strides[dimension];
    }
    return offset;
}

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID, uniform FillParameters parameters)
{
    const uint index = dispatchThreadId.x;
    if (index >= parameters.count)
        return;
    const uint64_t offset = logicalOffset(index, parameters);
    if (kDType == 0 || kDType == 2 || kDType == 6)
        ((Ptr<uint, Access.ReadWrite>)parameters.outputAddress)[offset] = uint(parameters.pattern);
    else if (kDType == 1)
        ((Ptr<uint16_t, Access.ReadWrite>)parameters.outputAddress)[offset] = uint16_t(parameters.pattern);
    else if (kDType == 3)
        ((Ptr<uint64_t, Access.ReadWrite>)parameters.outputAddress)[offset] = parameters.pattern;
    else
        storeByte(parameters.outputAddress, offset, uint(parameters.pattern));
}
