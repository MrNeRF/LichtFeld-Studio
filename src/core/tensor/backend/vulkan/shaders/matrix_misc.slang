static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 40;

// 0 eye: output[rows][columns] with ones on the diagonal.
// 1 diag: output[n][n] with the diagonal vector on the diagonal.
// 2 dot partial: output[workgroup] = sum over its grid-stride slice of a * b.
// 3 dot finalize: output[0] = sum of count partials in a.
[SpecializationConstant]
const uint kKind = 0;

struct Parameters
{
    uint64_t outputAddress;
    uint64_t aAddress;
    uint64_t bAddress;
    uint rows;
    uint columns;
    uint count;
    uint pad0;
};

groupshared float sharedSum[LFS_LOCAL_SIZE_X];

// Neumaier compensated addition, as in reduce.slang.
void accumulate(inout float sum, inout float compensation, float value)
{
    const float total = sum + value;
    compensation += abs(sum) >= abs(value) ? (sum - total) + value : (value - total) + sum;
    sum = total;
}

float workgroupSum(float value, uint thread)
{
    sharedSum[thread] = value;
    GroupMemoryBarrierWithGroupSync();
    for (uint span = LFS_LOCAL_SIZE_X / 2; span > 0; span >>= 1)
    {
        if (thread < span)
            sharedSum[thread] += sharedSum[thread + span];
        GroupMemoryBarrierWithGroupSync();
    }
    return sharedSum[0];
}

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID, uint3 groupId : SV_GroupID,
          uint3 groupThreadId : SV_GroupThreadID, uniform Parameters parameters)
{
    const uint3 numWorkgroups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    const uint stride = numWorkgroups.x * LFS_LOCAL_SIZE_X;
    Ptr<float, Access.ReadWrite> output = (Ptr<float, Access.ReadWrite>)parameters.outputAddress;
    Ptr<float, Access.Read> a = (Ptr<float, Access.Read>)parameters.aAddress;
    Ptr<float, Access.Read> b = (Ptr<float, Access.Read>)parameters.bAddress;
    if (kKind <= 1)
    {
        for (uint index = dispatchThreadId.x; index < parameters.count; index += stride)
        {
            const uint row = index / parameters.columns;
            const uint column = index - row * parameters.columns;
            float value = 0.0f;
            if (row == column)
                value = kKind == 0 ? 1.0f : a[row];
            output[index] = value;
        }
        return;
    }
    float sum = 0.0f;
    float compensation = 0.0f;
    if (kKind == 2)
    {
        for (uint64_t index = dispatchThreadId.x; index < parameters.count; index += stride)
            accumulate(sum, compensation, a[index] * b[index]);
    }
    else
    {
        for (uint index = groupThreadId.x; index < parameters.count; index += LFS_LOCAL_SIZE_X)
            accumulate(sum, compensation, a[index]);
    }
    const float total = workgroupSum(isfinite(compensation) ? sum + compensation : sum, groupThreadId.x);
    if (groupThreadId.x == 0)
        output[groupId.x] = total;
}
