static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 24;

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

float loadAsFloat(uint64_t address, uint index)
{
    if (kInputDType == 0) return ((Ptr<float, Access.Read>)address)[index];
    if (kInputDType == 1) return f16tof32(uint(((Ptr<uint16_t, Access.Read>)address)[index]));
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

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID, uniform Parameters parameters)
{
    const uint3 numWorkgroups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    const uint stride = numWorkgroups.x * LFS_LOCAL_SIZE_X;
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
            else if (kInputDType == 6)
                ((Ptr<uint, Access.ReadWrite>)parameters.outputAddress)[index] =
                    ((Ptr<uint, Access.Read>)parameters.inputAddress)[index];
            else if (kInputDType == 5)
                storeByte(parameters.outputAddress, index,
                          loadByte(parameters.inputAddress, index) != 0u ? 1u : 0u);
            else
                storeByte(parameters.outputAddress, index,
                          loadByte(parameters.inputAddress, index));
            continue;
        }
        if (kOutputDType == 0)
            ((Ptr<float, Access.ReadWrite>)parameters.outputAddress)[index] =
                loadAsFloat(parameters.inputAddress, index);
        else if (kOutputDType == 1)
            ((Ptr<uint16_t, Access.ReadWrite>)parameters.outputAddress)[index] =
                uint16_t(f32tof16(loadAsFloat(parameters.inputAddress, index)));
        else if (kOutputDType == 2)
            ((Ptr<int, Access.ReadWrite>)parameters.outputAddress)[index] =
                int(loadAsInt64(parameters.inputAddress, index));
        else if (kOutputDType == 3)
            ((Ptr<int64_t, Access.ReadWrite>)parameters.outputAddress)[index] =
                loadAsInt64(parameters.inputAddress, index);
        else if (kOutputDType == 4)
        {
            const uint value = kInputDType == 0 || kInputDType == 1
                                   ? torchUInt8Cast(loadAsFloat(parameters.inputAddress, index))
                                   : uint(loadAsInt64(parameters.inputAddress, index)) & 255u;
            storeByte(parameters.outputAddress, index, value);
        }
        else if (kOutputDType == 5)
            storeByte(parameters.outputAddress, index,
                      loadAsFloat(parameters.inputAddress, index) != 0.0f ? 1u : 0u);
        else
            ((Ptr<uint, Access.ReadWrite>)parameters.outputAddress)[index] =
                uint(loadAsInt64(parameters.inputAddress, index));
    }
}
