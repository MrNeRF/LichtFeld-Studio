static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 24;
#include "half_bits.slangh"

[SpecializationConstant]
const uint kInputDType = 0;
[SpecializationConstant]
const uint kOutputDType = 0;

struct Parameters
{
    uint64_t inputAddress;
    uint64_t outputAddress;
    uint count;
    uint padding;
};

uint loadByte(uint64_t address, uint64_t index)
{
    const uint64_t byteAddress = address + index;
    const uint shift = uint(byteAddress & uint64_t(3)) * 8u;
    return (((Ptr<uint, Access.Read>)(byteAddress & ~uint64_t(3)))[0] >> shift) & 255u;
}

float loadAsFloat(uint64_t address, uint index)
{
    if (kInputDType == 0) return ((Ptr<float, Access.Read>)address)[index];
    if (kInputDType == 1) return halfBitsToFloat(uint(((Ptr<uint16_t, Access.Read>)address)[index]));
    if (kInputDType == 2) return float(((Ptr<int, Access.Read>)address)[index]);
    if (kInputDType == 3) return float(((Ptr<int64_t, Access.Read>)address)[index]);
    if (kInputDType == 6) return float(((Ptr<uint, Access.Read>)address)[index]);
    return float(loadByte(address, index));
}

int64_t loadAsInt64(uint64_t address, uint index)
{
    if (kInputDType == 2) return int64_t(((Ptr<int, Access.Read>)address)[index]);
    if (kInputDType == 3) return ((Ptr<int64_t, Access.Read>)address)[index];
    if (kInputDType == 6) return int64_t(((Ptr<uint, Access.Read>)address)[index]);
    if (kInputDType == 4 || kInputDType == 5)
        return int64_t(loadByte(address, index));
    return int64_t(loadAsFloat(address, index));
}

uint torchUInt8Cast(float value)
{
    if (!isfinite(value))
        return 0u;
    float wrapped = fmod(trunc(value), 256.0f);
    if (wrapped < 0.0f)
        wrapped += 256.0f;
    return uint(wrapped);
}

uint convertToByte(uint64_t inputAddress, uint index)
{
    if (kInputDType == kOutputDType)
    {
        const uint value = loadByte(inputAddress, index);
        if (kInputDType == 5)
            return value != 0u ? 1u : 0u;
        return value;
    }
    if (kOutputDType == 4)
    {
        if (kInputDType == 0 || kInputDType == 1)
            return torchUInt8Cast(loadAsFloat(inputAddress, index));
        return uint(loadAsInt64(inputAddress, index)) & 255u;
    }
    return loadAsFloat(inputAddress, index) != 0.0f ? 1u : 0u;
}

void storeMaskedWord(uint64_t wordAddress, uint packed, uint mask)
{
    Ptr<uint, Access.ReadWrite> word =
        (Ptr<uint, Access.ReadWrite>)wordAddress;
    uint assumed = word[0];
    uint observed;
    do
    {
        const uint replacement = (assumed & ~mask) | packed;
        InterlockedCompareExchange(word[0], assumed, replacement, observed);
        if (observed == assumed) return;
        assumed = observed;
    } while (true);
}

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID, uniform Parameters parameters)
{
    const uint3 numWorkgroups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    const uint stride = numWorkgroups.x * LFS_LOCAL_SIZE_X;
    // UInt8/Bool: one invocation owns one aligned output word. A word fully
    // inside the output range is a direct store; a leading/trailing partial
    // word uses CAS so neighboring bytes that are not this conversion stay
    // intact. Host dispatch is the number of covering words, not elements.
    if (kOutputDType == 4 || kOutputDType == 5)
    {
        const uint64_t start = parameters.outputAddress;
        const uint64_t end = start + uint64_t(parameters.count);
        const uint64_t firstWord = start & ~uint64_t(3);
        const uint wordCount =
            uint(((end - uint64_t(1)) >> 2) - (start >> 2) + uint64_t(1));
        for (uint wordIndex = dispatchThreadId.x; wordIndex < wordCount;
             wordIndex += stride)
        {
            const uint64_t wordAddress =
                firstWord + uint64_t(wordIndex) * uint64_t(4);
            uint packed = 0;
            uint mask = 0;
            for (uint lane = 0; lane < 4u; ++lane)
            {
                const uint64_t byteAddress = wordAddress + uint64_t(lane);
                if (byteAddress >= start && byteAddress < end)
                {
                    const uint elementIndex = uint(byteAddress - start);
                    packed |= (convertToByte(parameters.inputAddress, elementIndex) &
                               255u)
                              << (lane * 8u);
                    mask |= 255u << (lane * 8u);
                }
            }
            if (mask == 0u)
                continue;
            if (mask == 0xffffffffu)
                ((Ptr<uint, Access.ReadWrite>)wordAddress)[0] = packed;
            else
                storeMaskedWord(wordAddress, packed, mask);
        }
        return;
    }
    for (uint index = dispatchThreadId.x; index < parameters.count; index += stride)
    {
        if (kInputDType == kOutputDType)
        {
            if (kInputDType == 0)
                ((Ptr<float, Access.ReadWrite>)parameters.outputAddress)[index] =
                    ((Ptr<float, Access.Read>)parameters.inputAddress)[index];
            else if (kInputDType == 1)
                ((Ptr<uint16_t, Access.ReadWrite>)parameters.outputAddress)[index] =
                    ((Ptr<uint16_t, Access.Read>)parameters.inputAddress)[index];
            else if (kInputDType == 2)
                ((Ptr<int, Access.ReadWrite>)parameters.outputAddress)[index] =
                    ((Ptr<int, Access.Read>)parameters.inputAddress)[index];
            else if (kInputDType == 3)
                ((Ptr<int64_t, Access.ReadWrite>)parameters.outputAddress)[index] =
                    ((Ptr<int64_t, Access.Read>)parameters.inputAddress)[index];
            else
                ((Ptr<uint, Access.ReadWrite>)parameters.outputAddress)[index] =
                    ((Ptr<uint, Access.Read>)parameters.inputAddress)[index];
            continue;
        }
        if (kOutputDType == 0)
            ((Ptr<float, Access.ReadWrite>)parameters.outputAddress)[index] =
                loadAsFloat(parameters.inputAddress, index);
        else if (kOutputDType == 1)
            ((Ptr<uint16_t, Access.ReadWrite>)parameters.outputAddress)[index] =
                uint16_t(floatToHalfBits(loadAsFloat(parameters.inputAddress, index)));
        else if (kOutputDType == 2)
            ((Ptr<int, Access.ReadWrite>)parameters.outputAddress)[index] =
                int(loadAsInt64(parameters.inputAddress, index));
        else if (kOutputDType == 3)
            ((Ptr<int64_t, Access.ReadWrite>)parameters.outputAddress)[index] =
                loadAsInt64(parameters.inputAddress, index);
        else
            ((Ptr<uint, Access.ReadWrite>)parameters.outputAddress)[index] =
                uint(loadAsInt64(parameters.inputAddress, index));
    }
}
