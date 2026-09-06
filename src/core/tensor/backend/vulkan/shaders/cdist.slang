static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 48;

// out[i][j] = p-norm distance between row i of a and row j of b, with the
// p == 0 (count of differing features) and p == infinity (largest absolute
// difference) conventions of the CUDA kernel.
struct Parameters
{
    uint64_t aAddress;
    uint64_t bAddress;
    uint64_t outputAddress;
    uint rows;
    uint columns;
    uint features;
    uint pad0;
    float p;
    uint pad1;
};

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID, uniform Parameters parameters)
{
    const uint3 numWorkgroups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    const uint stride = numWorkgroups.x * LFS_LOCAL_SIZE_X;
    const uint total = parameters.rows * parameters.columns;
    Ptr<float, Access.Read> a = (Ptr<float, Access.Read>)parameters.aAddress;
    Ptr<float, Access.Read> b = (Ptr<float, Access.Read>)parameters.bAddress;
    Ptr<float, Access.ReadWrite> output = (Ptr<float, Access.ReadWrite>)parameters.outputAddress;
    const float p = parameters.p;
    for (uint index = dispatchThreadId.x; index < total; index += stride)
    {
        const uint i = index / parameters.columns;
        const uint j = index - i * parameters.columns;
        const uint64_t aBase = uint64_t(i) * parameters.features;
        const uint64_t bBase = uint64_t(j) * parameters.features;
        float distance = 0.0f;
        for (uint d = 0; d < parameters.features; ++d)
        {
            const float difference = a[aBase + d] - b[bBase + d];
            if (p == 2.0f)
                distance += difference * difference;
            else if (p == 1.0f)
                distance += abs(difference);
            else if (p == 0.0f)
                distance += difference != 0.0f ? 1.0f : 0.0f;
            else if (isinf(p))
                distance = max(distance, abs(difference));
            else
                distance += pow(abs(difference), p);
        }
        if (p == 2.0f)
            distance = sqrt(distance);
        else if (p != 1.0f && p != 0.0f && !isinf(p))
            distance = pow(distance, 1.0f / p);
        output[index] = distance;
    }
}
