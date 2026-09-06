static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 136;
#include "half_bits.slangh"

[SpecializationConstant]
const uint kOp = 4;
[SpecializationConstant]
const uint kInputDType = 0;
[SpecializationConstant]
const uint kOutputDType = 0;

struct Parameters
{
    uint64_t lhsAddress;
    uint64_t rhsAddress;
    uint64_t outputAddress;
    uint lhsDims[8];
    uint rhsDims[8];
    uint outputDims[8];
    uint lhsRank;
    uint rhsRank;
    uint outputRank;
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

uint broadcastOffset(uint linear, uint rank, uint dims[8], uint outputRank,
                     uint outputDims[8])
{
    uint sourceOffset = 0;
    uint sourceStride = 1;
    for (uint reverse = 0; reverse < outputRank; ++reverse)
    {
        const uint outputDimension = outputRank - reverse - 1;
        const uint coordinate = linear % outputDims[outputDimension];
        linear /= outputDims[outputDimension];
        if (outputDimension + rank >= outputRank)
        {
            const uint sourceDimension = outputDimension + rank - outputRank;
            sourceOffset += (dims[sourceDimension] == 1u ? 0u : coordinate) * sourceStride;
            sourceStride *= dims[sourceDimension];
        }
    }
    return sourceOffset;
}

float ieeeMaximum(float lhs, float rhs)
{
    const uint lhsBits = asuint(lhs);
    const uint rhsBits = asuint(rhs);
    if ((lhsBits & 0x7fffffffu) > 0x7f800000u) return lhs;
    if ((rhsBits & 0x7fffffffu) > 0x7f800000u) return rhs;
    if ((lhsBits & 0x7fffffffu) == 0u && (rhsBits & 0x7fffffffu) == 0u)
        return asfloat((lhsBits & rhsBits) & 0x80000000u);
    return lhs < rhs ? rhs : lhs;
}

float ieeeMinimum(float lhs, float rhs)
{
    const uint lhsBits = asuint(lhs);
    const uint rhsBits = asuint(rhs);
    if ((lhsBits & 0x7fffffffu) > 0x7f800000u) return lhs;
    if ((rhsBits & 0x7fffffffu) > 0x7f800000u) return rhs;
    if ((lhsBits & 0x7fffffffu) == 0u && (rhsBits & 0x7fffffffu) == 0u)
        return asfloat((lhsBits | rhsBits) & 0x80000000u);
    return rhs < lhs ? rhs : lhs;
}

float floatOp(float lhs, float rhs)
{
    if (kOp == 4) return lhs + rhs;
    if (kOp == 5) return lhs - rhs;
    if (kOp == 6) return lhs * rhs;
    if (kOp == 7) return lhs / rhs;
    if (kOp == 25) return rhs == 2.0f ? lhs * lhs : pow(lhs, rhs);
    if (kOp == 26) return fmod(lhs, rhs);
    if (kOp == 27) return ieeeMaximum(lhs, rhs);
    if (kOp == 28) return ieeeMinimum(lhs, rhs);
    return lhs;
}

bool predicateFloat(float lhs, float rhs)
{
    if (kOp == 29) return lhs == rhs;
    if (kOp == 30) return lhs != rhs;
    if (kOp == 31) return lhs < rhs;
    if (kOp == 32) return lhs <= rhs;
    if (kOp == 33) return lhs > rhs;
    if (kOp == 34) return lhs >= rhs;
    if (kOp == 35) return lhs != 0.0f && rhs != 0.0f;
    if (kOp == 36) return lhs != 0.0f || rhs != 0.0f;
    return (lhs != 0.0f) != (rhs != 0.0f);
}

int intOp(int lhs, int rhs)
{
    if (kOp == 4) return lhs + rhs;
    if (kOp == 5) return lhs - rhs;
    if (kOp == 6) return lhs * rhs;
    if (kOp == 7) return rhs == 0 ? 0 : lhs / rhs;
    if (kOp == 25) return rhs == 2 ? lhs * lhs : int(pow(float(lhs), float(rhs)));
    if (kOp == 26) return rhs == 0 ? 0 : lhs % rhs;
    if (kOp == 27) return max(lhs, rhs);
    if (kOp == 28) return min(lhs, rhs);
    return lhs;
}

bool predicateInt64(int64_t lhs, int64_t rhs)
{
    if (kOp == 29) return lhs == rhs;
    if (kOp == 30) return lhs != rhs;
    if (kOp == 31) return lhs < rhs;
    if (kOp == 32) return lhs <= rhs;
    if (kOp == 33) return lhs > rhs;
    if (kOp == 34) return lhs >= rhs;
    if (kOp == 35) return lhs != 0 && rhs != 0;
    if (kOp == 36) return lhs != 0 || rhs != 0;
    return (lhs != 0) != (rhs != 0);
}

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID, uniform Parameters parameters)
{
    const uint3 numWorkgroups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    const uint stride = numWorkgroups.x * LFS_LOCAL_SIZE_X;
    for (uint index = dispatchThreadId.x; index < parameters.count; index += stride)
    {
        const uint lhsIndex = broadcastOffset(index, parameters.lhsRank, parameters.lhsDims,
                                              parameters.outputRank, parameters.outputDims);
        const uint rhsIndex = broadcastOffset(index, parameters.rhsRank, parameters.rhsDims,
                                              parameters.outputRank, parameters.outputDims);
        if (kInputDType == 0 || kInputDType == 1)
        {
            const float lhs = kInputDType == 0
                                  ? ((Ptr<float, Access.Read>)parameters.lhsAddress)[lhsIndex]
                                  : halfBitsToFloat(uint(((Ptr<uint16_t, Access.Read>)parameters.lhsAddress)[lhsIndex]));
            const float rhs = kInputDType == 0
                                  ? ((Ptr<float, Access.Read>)parameters.rhsAddress)[rhsIndex]
                                  : halfBitsToFloat(uint(((Ptr<uint16_t, Access.Read>)parameters.rhsAddress)[rhsIndex]));
            if (kOutputDType == 5)
                storeByte(parameters.outputAddress, index, predicateFloat(lhs, rhs) ? 1u : 0u);
            else if (kOutputDType == 0)
                ((Ptr<float, Access.ReadWrite>)parameters.outputAddress)[index] = floatOp(lhs, rhs);
            else
                ((Ptr<uint16_t, Access.ReadWrite>)parameters.outputAddress)[index] =
                    uint16_t(floatToHalfBits(floatOp(lhs, rhs)));
            continue;
        }
        if (kInputDType == 3)
        {
            const int64_t lhs = ((Ptr<int64_t, Access.Read>)parameters.lhsAddress)[lhsIndex];
            const int64_t rhs = ((Ptr<int64_t, Access.Read>)parameters.rhsAddress)[rhsIndex];
            if (kOutputDType == 5)
                storeByte(parameters.outputAddress, index, predicateInt64(lhs, rhs) ? 1u : 0u);
            else
            {
                int64_t result = lhs;
                if (kOp == 4) result = lhs + rhs;
                if (kOp == 5) result = lhs - rhs;
                if (kOp == 6) result = lhs * rhs;
                if (kOp == 7) result = rhs == 0 ? int64_t(0) : lhs / rhs;
                if (kOp == 26) result = rhs == 0 ? int64_t(0) : lhs % rhs;
                if (kOp == 27) result = lhs < rhs ? rhs : lhs;
                if (kOp == 28) result = rhs < lhs ? rhs : lhs;
                ((Ptr<int64_t, Access.ReadWrite>)parameters.outputAddress)[index] = result;
            }
            continue;
        }
        const uint lhs = kInputDType == 2
                             ? uint(((Ptr<int, Access.Read>)parameters.lhsAddress)[lhsIndex])
                             : loadByte(parameters.lhsAddress, lhsIndex);
        const uint rhs = kInputDType == 2
                             ? uint(((Ptr<int, Access.Read>)parameters.rhsAddress)[rhsIndex])
                             : loadByte(parameters.rhsAddress, rhsIndex);
        if (kOutputDType == 5)
            storeByte(parameters.outputAddress, index,
                      predicateInt64(int64_t(int(lhs)), int64_t(int(rhs))) ? 1u : 0u);
        else if (kInputDType == 2)
            ((Ptr<int, Access.ReadWrite>)parameters.outputAddress)[index] = intOp(int(lhs), int(rhs));
        else
            storeByte(parameters.outputAddress, index, uint(intOp(int(lhs), int(rhs))) & 255u);
    }
}
