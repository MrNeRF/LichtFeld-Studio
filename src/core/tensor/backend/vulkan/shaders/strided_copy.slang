static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 152;

[SpecializationConstant]
const uint kInputDType = 0;
[SpecializationConstant]
const uint kOutputDType = 0;
[SpecializationConstant]
const uint kScatter = 0;

struct Parameters
{
    uint64_t inputAddress;
    uint64_t outputAddress;
    uint64_t dims[8];
    uint64_t strides[8];
    uint rank;
    uint count;
};

uint loadByte(uint64_t address, uint64_t index)
{
    const uint64_t byteAddress = address + index;
    const uint shift = uint(byteAddress & uint64_t(3)) * 8u;
    return (((Ptr<uint, Access.Read>)(byteAddress & ~uint64_t(3)))[0] >> shift) & 255u;
}

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

uint64_t logicalOffset(uint linear, uniform Parameters parameters)
{
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
void main(uint3 dispatchThreadId : SV_DispatchThreadID, uniform Parameters parameters)
{
    const uint3 numWorkgroups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    const uint stride = numWorkgroups.x * LFS_LOCAL_SIZE_X;
    for (uint index = dispatchThreadId.x; index < parameters.count; index += stride)
    {
        const uint64_t strided = logicalOffset(index, parameters);
        const uint64_t inputIndex = kScatter != 0 ? uint64_t(index) : strided;
        const uint64_t outputIndex = kScatter != 0 ? strided : uint64_t(index);
        if (kInputDType == 2 && kOutputDType == 0)
        {
            ((Ptr<float, Access.ReadWrite>)parameters.outputAddress)[outputIndex] =
                float(((Ptr<int, Access.Read>)parameters.inputAddress)[inputIndex]);
            continue;
        }
        if (kInputDType == 0)
            ((Ptr<uint, Access.ReadWrite>)parameters.outputAddress)[outputIndex] =
                ((Ptr<uint, Access.Read>)parameters.inputAddress)[inputIndex];
        else if (kInputDType == 1)
            ((Ptr<uint16_t, Access.ReadWrite>)parameters.outputAddress)[outputIndex] =
                ((Ptr<uint16_t, Access.Read>)parameters.inputAddress)[inputIndex];
        else if (kInputDType == 2)
            ((Ptr<int, Access.ReadWrite>)parameters.outputAddress)[outputIndex] =
                ((Ptr<int, Access.Read>)parameters.inputAddress)[inputIndex];
        else if (kInputDType == 3)
            ((Ptr<int64_t, Access.ReadWrite>)parameters.outputAddress)[outputIndex] =
                ((Ptr<int64_t, Access.Read>)parameters.inputAddress)[inputIndex];
        else if (kInputDType == 6)
            ((Ptr<uint, Access.ReadWrite>)parameters.outputAddress)[outputIndex] =
                ((Ptr<uint, Access.Read>)parameters.inputAddress)[inputIndex];
        else
            storeByte(parameters.outputAddress, outputIndex,
                      loadByte(parameters.inputAddress, inputIndex));
    }
}
