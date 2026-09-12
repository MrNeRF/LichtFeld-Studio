[SpecializationConstant]
const uint kOp = 0;

[SpecializationConstant]
const uint kDType = 0;

struct Parameters
{
    uint64_t inputAddress;
    uint64_t outputAddress;
    uint count;
};

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID, uniform Parameters parameters)
{
    const uint index = dispatchThreadId.x;
    if (index >= parameters.count)
        return;

    if (kDType == 0)
    {
        Ptr<float, Access.Read> input = (Ptr<float, Access.Read>)parameters.inputAddress;
        Ptr<float, Access.ReadWrite> output = (Ptr<float, Access.ReadWrite>)parameters.outputAddress;
        output[index] = kOp == 0 ? input[index] + 1.0f : input[index] * 2.0f;
    }
    else if (kDType == 1)
    {
        Ptr<int, Access.Read> input = (Ptr<int, Access.Read>)parameters.inputAddress;
        Ptr<int, Access.ReadWrite> output = (Ptr<int, Access.ReadWrite>)parameters.outputAddress;
        output[index] = kOp == 0 ? input[index] + 1 : input[index] ^ int(0x5a5a5a5a);
    }
    else
    {
        Ptr<uint16_t, Access.Read> input = (Ptr<uint16_t, Access.Read>)parameters.inputAddress;
        Ptr<uint16_t, Access.ReadWrite> output = (Ptr<uint16_t, Access.ReadWrite>)parameters.outputAddress;
        output[index] = uint16_t(kOp == 0 ? uint(input[index]) + 1 : uint(input[index]) * 2);
    }
}
