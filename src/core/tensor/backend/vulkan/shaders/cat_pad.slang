static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 168;

[SpecializationConstant]
const uint kDType = 0;
[SpecializationConstant]
const uint kMode = 0;

struct Parameters
{
    uint64_t inputAddress;
    uint64_t outputAddress;
    uint inputDims[8];
    uint inputStrides[8];
    uint outputStrides[8];
    uint padBefore[8];
    uint count;
    uint rank;
    uint inputBlock;
    uint outputBlock;
    uint outputOffset;
    uint padding;
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

void copyElement(uint64_t inputAddress, uint64_t inputIndex,
                 uint64_t outputAddress, uint64_t outputIndex)
{
    if (kDType == 0 || kDType == 2 || kDType == 6)
        ((Ptr<uint, Access.ReadWrite>)outputAddress)[outputIndex] =
            ((Ptr<uint, Access.Read>)inputAddress)[inputIndex];
    else if (kDType == 1)
        ((Ptr<uint16_t, Access.ReadWrite>)outputAddress)[outputIndex] =
            ((Ptr<uint16_t, Access.Read>)inputAddress)[inputIndex];
    else if (kDType == 3)
        ((Ptr<uint64_t, Access.ReadWrite>)outputAddress)[outputIndex] =
            ((Ptr<uint64_t, Access.Read>)inputAddress)[inputIndex];
    else
        storeByte(outputAddress, outputIndex, loadByte(inputAddress, inputIndex));
}

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID, uniform Parameters parameters)
{
    const uint index = dispatchThreadId.x;
    if (index >= parameters.count)
        return;
    if (kMode == 0)
    {
        const uint outer = index / parameters.inputBlock;
        const uint within = index % parameters.inputBlock;
        copyElement(parameters.inputAddress, index, parameters.outputAddress,
                    uint64_t(outer * parameters.outputBlock + parameters.outputOffset + within));
        return;
    }
    uint remaining = index;
    uint64_t inputOffset = 0;
    uint64_t outputOffset = 0;
    for (uint dimension = 0; dimension < parameters.rank; ++dimension)
    {
        uint64_t contiguousStride = 1;
        for (uint next = dimension + 1; next < parameters.rank; ++next)
            contiguousStride *= parameters.inputDims[next];
        const uint64_t coordinate = uint64_t(remaining) / contiguousStride;
        remaining = uint(uint64_t(remaining) % contiguousStride);
        inputOffset += coordinate * parameters.inputStrides[dimension];
        outputOffset += (coordinate + parameters.padBefore[dimension]) *
                        parameters.outputStrides[dimension];
    }
    copyElement(parameters.inputAddress, inputOffset,
                parameters.outputAddress, outputOffset);
}
