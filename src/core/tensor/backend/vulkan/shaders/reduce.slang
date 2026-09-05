static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 96;

// kOp follows ReduceOp: 0 Sum, 1 Mean, 2 Max, 3 Min, 4 Prod, 5 Any, 6 All.
[SpecializationConstant]
const uint kOp = 0;
// Element codes follow DataType: 0 Float32, 2 Int32, 3 Int64, 5 Bool.
[SpecializationConstant]
const uint kInputDType = 0;
[SpecializationConstant]
const uint kOutputDType = 0;
// 0 partial: every workgroup reduces a grid-stride slice of count elements
//   into output[workgroup].
// 2 segmented: one workgroup per contiguous segment of reduce elements.
// 3 strided: one thread per (outer, inner) output, looping over the reduce
//   range of its split (workgroup y), writing output[split][outer][inner].
// 4 general: one thread per output element over the axes of reducedMask.
[SpecializationConstant]
const uint kMode = 0;
[SpecializationConstant]
const uint kChain = 0;

static const uint kFloat32 = 0;
static const uint kInt32 = 2;
static const uint kInt64 = 3;
static const uint kBool = 5;

static const uint kSum = 0;
static const uint kMean = 1;
static const uint kMax = 2;
static const uint kMin = 3;
static const uint kProd = 4;
static const uint kAny = 5;
static const uint kAll = 6;

static const bool kFloatInput = kInputDType == kFloat32;
static const bool kLogical = kOp == kAny || kOp == kAll;
static const bool kFloatAccumulator = kFloatInput && !kLogical;

struct Parameters
{
    uint64_t inputAddress;
    uint64_t outputAddress;
    uint64_t chainAddress;
    uint outer;
    uint reduce;
    uint inner;
    uint count;
    uint chainLength;
    uint splitChunk;
    uint reducedMask;
    uint rank;
    float meanScale;
    uint pad0;
    uint dims[8];
};

struct ChainOp
{
    uint kind;
    float scalar;
    uint64_t rhsAddress;
};

groupshared float sharedFloat[LFS_LOCAL_SIZE_X];
groupshared int64_t sharedInt[LFS_LOCAL_SIZE_X];

uint loadByte(uint64_t address, uint64_t index)
{
    const uint64_t byteAddress = address + index;
    const uint shift = uint(byteAddress & uint64_t(3)) * 8u;
    const uint64_t wordAddress = byteAddress & ~uint64_t(3);
    return (((Ptr<uint, Access.Read>)wordAddress)[0] >> shift) & 255u;
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
        if (observed == assumed)
            return;
        assumed = observed;
    } while (true);
}

float ieeeRound(float value)
{
    if (!isfinite(value))
        return value;
    const float lower = floor(value);
    const float fraction = value - lower;
    float rounded = fraction < 0.5f ? lower
                  : fraction > 0.5f ? lower + 1.0f
                                    : (fmod(lower, 2.0f) == 0.0f ? lower : lower + 1.0f);
    if (rounded == 0.0f && (asuint(value) & 0x80000000u) != 0u)
        return asfloat(0x80000000u);
    return rounded;
}

// First NaN wins and equal-magnitude zeros follow the sign rule, matching the
// CUDA max_reduce_op and min_reduce_op functors.
float ieeeMaximum(float lhs, float rhs)
{
    const uint lhsBits = asuint(lhs);
    const uint rhsBits = asuint(rhs);
    if ((lhsBits & 0x7fffffffu) > 0x7f800000u)
        return lhs;
    if ((rhsBits & 0x7fffffffu) > 0x7f800000u)
        return rhs;
    if ((lhsBits & 0x7fffffffu) == 0u && (rhsBits & 0x7fffffffu) == 0u)
        return asfloat((lhsBits & rhsBits) & 0x80000000u);
    return lhs < rhs ? rhs : lhs;
}

float ieeeMinimum(float lhs, float rhs)
{
    const uint lhsBits = asuint(lhs);
    const uint rhsBits = asuint(rhs);
    if ((lhsBits & 0x7fffffffu) > 0x7f800000u)
        return lhs;
    if ((rhsBits & 0x7fffffffu) > 0x7f800000u)
        return rhs;
    if ((lhsBits & 0x7fffffffu) == 0u && (rhsBits & 0x7fffffffu) == 0u)
        return asfloat((lhsBits | rhsBits) & 0x80000000u);
    return rhs < lhs ? rhs : lhs;
}

float applyChainOp(float value, ChainOp op, uint index)
{
    if (op.kind == 0) return value + op.scalar;
    if (op.kind == 1) return value - op.scalar;
    if (op.kind == 2) return value * op.scalar;
    if (op.kind == 3) return value / op.scalar;
    if (op.kind >= 4 && op.kind <= 7)
    {
        const float rhs = ((Ptr<float, Access.Read>)op.rhsAddress)[index];
        if (op.kind == 4) return value + rhs;
        if (op.kind == 5) return value - rhs;
        if (op.kind == 6) return value * rhs;
        return value / rhs;
    }
    if (op.kind == 10) return abs(value);
    if (op.kind == 11) return -value;
    if (op.kind == 12) return exp(value);
    if (op.kind == 13) return log(value);
    if (op.kind == 14) return sqrt(value);
    if (op.kind == 15) return 1.0f / (1.0f + exp(-value));
    if (op.kind == 16) return isnan(value) ? value : ieeeMaximum(value, 0.0f);
    if (op.kind == 17) return value * value;
    if (op.kind == 18) return tanh(value);
    if (op.kind == 19) return rsqrt(value);
    if (op.kind == 20) return float((value > 0.0f) - (value < 0.0f));
    if (op.kind == 21) return 1.0f / value;
    if (op.kind == 22) return floor(value);
    if (op.kind == 23) return ceil(value);
    if (op.kind == 24) return ieeeRound(value);
    return value;
}

float loadFloat(uint64_t index, Parameters parameters)
{
    float value = ((Ptr<float, Access.Read>)parameters.inputAddress)[index];
    if (kChain != 0)
    {
        const Ptr<ChainOp, Access.Read> chain = (Ptr<ChainOp, Access.Read>)parameters.chainAddress;
        for (uint operation = 0; operation < parameters.chainLength; ++operation)
            value = applyChainOp(value, chain[operation], uint(index));
    }
    return value;
}

int64_t loadInt(uint64_t index, uint64_t address)
{
    if (kInputDType == kInt32)
        return int64_t(((Ptr<int, Access.Read>)address)[index]);
    if (kInputDType == kInt64)
        return ((Ptr<int64_t, Access.Read>)address)[index];
    return int64_t(loadByte(address, index));
}

float floatIdentity()
{
    if (kOp == kMax) return -1.0f / 0.0f;
    if (kOp == kMin) return 1.0f / 0.0f;
    if (kOp == kProd) return 1.0f;
    return 0.0f;
}

int64_t intIdentity()
{
    if (kOp == kMax) return int64_t(-0x7fffffffffffffffLL - 1);
    if (kOp == kMin) return int64_t(0x7fffffffffffffffLL);
    if (kOp == kProd || kOp == kAll) return int64_t(1);
    return int64_t(0);
}

// Sum and Mean accumulate with Neumaier compensation so long sequential loops
// stay within a few ulps of the CUDA double-accumulated kernels.
void combineFloat(inout float accumulator, inout float compensation, float value)
{
    if (kOp == kSum || kOp == kMean)
    {
        const float total = accumulator + value;
        compensation += abs(accumulator) >= abs(value) ? (accumulator - total) + value
                                                       : (value - total) + accumulator;
        accumulator = total;
    }
    else if (kOp == kMax)
        accumulator = ieeeMaximum(accumulator, value);
    else if (kOp == kMin)
        accumulator = ieeeMinimum(accumulator, value);
    else
        accumulator *= value;
}

float compensated(float accumulator, float compensation)
{
    return isfinite(compensation) ? accumulator + compensation : accumulator;
}

void combineInt(inout int64_t accumulator, int64_t value)
{
    if (kOp == kSum || kOp == kMean)
        accumulator += value;
    else if (kOp == kMax)
        accumulator = value > accumulator ? value : accumulator;
    else if (kOp == kMin)
        accumulator = value < accumulator ? value : accumulator;
    else if (kOp == kProd)
        accumulator *= value;
    else if (kOp == kAny)
        accumulator = (accumulator != 0 || value != 0) ? int64_t(1) : int64_t(0);
    else
        accumulator = (accumulator != 0 && value != 0) ? int64_t(1) : int64_t(0);
}

int64_t logicalValue(uint64_t index, Parameters parameters)
{
    if (kFloatInput)
        return loadFloat(index, parameters) != 0.0f ? int64_t(1) : int64_t(0);
    return loadInt(index, parameters.inputAddress) != 0 ? int64_t(1) : int64_t(0);
}

float workgroupReduceFloat(float value, uint thread)
{
    sharedFloat[thread] = value;
    GroupMemoryBarrierWithGroupSync();
    for (uint span = LFS_LOCAL_SIZE_X / 2; span > 0; span >>= 1)
    {
        if (thread < span)
        {
            float lhs = sharedFloat[thread];
            float unused = 0.0f;
            combineFloat(lhs, unused, sharedFloat[thread + span]);
            sharedFloat[thread] = lhs;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    const float result = sharedFloat[0];
    GroupMemoryBarrierWithGroupSync();
    return result;
}

int64_t workgroupReduceInt(int64_t value, uint thread)
{
    sharedInt[thread] = value;
    GroupMemoryBarrierWithGroupSync();
    for (uint span = LFS_LOCAL_SIZE_X / 2; span > 0; span >>= 1)
    {
        if (thread < span)
        {
            int64_t lhs = sharedInt[thread];
            combineInt(lhs, sharedInt[thread + span]);
            sharedInt[thread] = lhs;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    const int64_t result = sharedInt[0];
    GroupMemoryBarrierWithGroupSync();
    return result;
}

void storeResult(uint64_t index, float floatValue, int64_t intValue, Parameters parameters)
{
    if (kOutputDType == kFloat32)
    {
        float value = kFloatAccumulator ? floatValue : float(intValue);
        if (kOp == kMean)
            value *= parameters.meanScale;
        ((Ptr<float, Access.ReadWrite>)parameters.outputAddress)[index] = value;
    }
    else if (kOutputDType == kInt32)
        ((Ptr<int, Access.ReadWrite>)parameters.outputAddress)[index] = int(intValue);
    else if (kOutputDType == kInt64)
        ((Ptr<int64_t, Access.ReadWrite>)parameters.outputAddress)[index] = intValue;
    else
        storeByte(parameters.outputAddress, index, intValue != 0 ? 1u : 0u);
}

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID, uint3 groupId : SV_GroupID,
          uint3 groupThreadId : SV_GroupThreadID, uniform Parameters parameters)
{
    const uint3 numWorkgroups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    const uint thread = groupThreadId.x;
    const uint gridStride = numWorkgroups.x * LFS_LOCAL_SIZE_X;

    if (kMode == 0)
    {
        float accumulator = floatIdentity();
        float compensation = 0.0f;
        int64_t intAccumulator = intIdentity();
        for (uint64_t index = dispatchThreadId.x; index < parameters.count; index += gridStride)
        {
            if (kFloatAccumulator)
                combineFloat(accumulator, compensation, loadFloat(index, parameters));
            else if (kLogical)
                combineInt(intAccumulator, logicalValue(index, parameters));
            else
                combineInt(intAccumulator, loadInt(index, parameters.inputAddress));
        }
        float total = 0.0f;
        int64_t intTotal = 0;
        if (kFloatAccumulator)
            total = workgroupReduceFloat(compensated(accumulator, compensation), thread);
        else
            intTotal = workgroupReduceInt(intAccumulator, thread);
        if (thread == 0)
            storeResult(groupId.x, total, intTotal, parameters);
    }
    else if (kMode == 2)
    {
        for (uint segment = groupId.x; segment < parameters.outer; segment += numWorkgroups.x)
        {
            const uint64_t base = uint64_t(segment) * parameters.reduce;
            float accumulator = floatIdentity();
            float compensation = 0.0f;
            int64_t intAccumulator = intIdentity();
            for (uint element = thread; element < parameters.reduce; element += LFS_LOCAL_SIZE_X)
            {
                const uint64_t index = base + element;
                if (kFloatAccumulator)
                    combineFloat(accumulator, compensation, loadFloat(index, parameters));
                else if (kLogical)
                    combineInt(intAccumulator, logicalValue(index, parameters));
                else
                    combineInt(intAccumulator, loadInt(index, parameters.inputAddress));
            }
            float total = 0.0f;
            int64_t intTotal = 0;
            if (kFloatAccumulator)
                total = workgroupReduceFloat(compensated(accumulator, compensation), thread);
            else
                intTotal = workgroupReduceInt(intAccumulator, thread);
            if (thread == 0)
                storeResult(segment, total, intTotal, parameters);
        }
    }
    else if (kMode == 3)
    {
        const uint outputs = parameters.outer * parameters.inner;
        const uint reduceBegin = groupId.y * parameters.splitChunk;
        const uint reduceEnd = min(parameters.reduce, reduceBegin + parameters.splitChunk);
        for (uint output = dispatchThreadId.x; output < outputs; output += gridStride)
        {
            const uint outerIndex = output / parameters.inner;
            const uint innerIndex = output - outerIndex * parameters.inner;
            const uint64_t base = uint64_t(outerIndex) * parameters.reduce * parameters.inner + innerIndex;
            float accumulator = floatIdentity();
            float compensation = 0.0f;
            int64_t intAccumulator = intIdentity();
            for (uint element = reduceBegin; element < reduceEnd; ++element)
            {
                const uint64_t index = base + uint64_t(element) * parameters.inner;
                if (kFloatAccumulator)
                    combineFloat(accumulator, compensation, loadFloat(index, parameters));
                else if (kLogical)
                    combineInt(intAccumulator, logicalValue(index, parameters));
                else
                    combineInt(intAccumulator, loadInt(index, parameters.inputAddress));
            }
            storeResult(uint64_t(groupId.y) * outputs + output,
                        compensated(accumulator, compensation), intAccumulator, parameters);
        }
    }
    else
    {
        uint strides[8];
        uint stride = 1;
        for (int axis = int(parameters.rank) - 1; axis >= 0; --axis)
        {
            strides[axis] = stride;
            stride *= parameters.dims[axis];
        }
        const uint outputs = parameters.outer;
        for (uint output = dispatchThreadId.x; output < outputs; output += gridStride)
        {
            uint remaining = output;
            uint64_t base = 0;
            for (int axis = int(parameters.rank) - 1; axis >= 0; --axis)
            {
                if ((parameters.reducedMask & (1u << uint(axis))) == 0u)
                {
                    const uint coordinate = remaining % parameters.dims[axis];
                    remaining /= parameters.dims[axis];
                    base += uint64_t(coordinate) * strides[axis];
                }
            }
            float accumulator = floatIdentity();
            float compensation = 0.0f;
            int64_t intAccumulator = intIdentity();
            for (uint element = 0; element < parameters.reduce; ++element)
            {
                uint rest = element;
                uint64_t index = base;
                for (int axis = int(parameters.rank) - 1; axis >= 0; --axis)
                {
                    if ((parameters.reducedMask & (1u << uint(axis))) != 0u)
                    {
                        const uint coordinate = rest % parameters.dims[axis];
                        rest /= parameters.dims[axis];
                        index += uint64_t(coordinate) * strides[axis];
                    }
                }
                if (kFloatAccumulator)
                    combineFloat(accumulator, compensation, loadFloat(index, parameters));
                else if (kLogical)
                    combineInt(intAccumulator, logicalValue(index, parameters));
                else
                    combineInt(intAccumulator, loadInt(index, parameters.inputAddress));
            }
            storeResult(output, compensated(accumulator, compensation), intAccumulator, parameters);
        }
    }
}
