static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 48;
static const uint LFS_REQUIRED_FLOAT16 = 1;

[SpecializationConstant]
const uint kOp = 4;
[SpecializationConstant]
const uint kInputDType = 1;
[SpecializationConstant]
const uint kOutputDType = 1;
[SpecializationConstant]
const uint kArity = 2;
[SpecializationConstant]
const uint kVectorized = 0;
[SpecializationConstant]
const uint kNativeHalf = 1;

struct Parameters
{
    uint64_t lhsAddress;
    uint64_t rhsAddress;
    uint64_t outputAddress;
    uint64_t scalarInt64;
    float scalarFloat;
    uint count;
    uint scalarKind;
    uint flags;
};

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID, uniform Parameters parameters)
{
    const uint index = dispatchThreadId.x;
    if (index >= parameters.count)
        return;
    const half lhs = ((Ptr<half, Access.Read>)parameters.lhsAddress)[index];
    const half rhs = ((Ptr<half, Access.Read>)parameters.rhsAddress)[index];
    half result = lhs;
    if (kOp == 4) result = lhs + rhs;
    if (kOp == 5) result = lhs - rhs;
    if (kOp == 6) result = lhs * rhs;
    if (kOp == 7) result = lhs / rhs;
    ((Ptr<half, Access.ReadWrite>)parameters.outputAddress)[index] = result;
}
