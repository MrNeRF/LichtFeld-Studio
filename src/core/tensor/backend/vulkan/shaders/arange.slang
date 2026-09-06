static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 24;

[SpecializationConstant]
const uint kDType = 0;

struct ArangeParameters
{
    uint64_t outputAddress;
    float start;
    float step;
    uint count;
    uint padding;
};

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID, uniform ArangeParameters parameters)
{
    const uint3 numWorkgroups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    const uint stride = numWorkgroups.x * LFS_LOCAL_SIZE_X;
    for (uint index = dispatchThreadId.x; index < parameters.count; index += stride)
    {
        const float index_f = float(index);
        const float start = parameters.start;
        const float step = parameters.step;
        const float scaled = spirv_asm {
            result:$$float = OpFMul $index_f $step;
        };
        spirv_asm { OpDecorate $scaled NoContraction; };
        const float value = spirv_asm {
            result:$$float = OpFAdd $start $scaled;
        };
        spirv_asm { OpDecorate $value NoContraction; };
        if (kDType == 2)
            ((Ptr<int, Access.ReadWrite>)parameters.outputAddress)[index] = int(value);
        else
            ((Ptr<float, Access.ReadWrite>)parameters.outputAddress)[index] = value;
    }
}
