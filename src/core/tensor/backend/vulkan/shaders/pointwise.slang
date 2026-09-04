static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 48;

[SpecializationConstant]
const uint kOp = 0;
[SpecializationConstant]
const uint kInputDType = 0;
[SpecializationConstant]
const uint kOutputDType = 0;
[SpecializationConstant]
const uint kArity = 1;
[SpecializationConstant]
const uint kVectorized = 0;
[SpecializationConstant]
const uint kNativeHalf = 0;

static const uint kFloat32 = 0;
static const uint kFloat16 = 1;
static const uint kInt32 = 2;
static const uint kInt64 = 3;
static const uint kUInt8 = 4;
static const uint kBool = 5;
static const uint kUInt32 = 6;

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

struct ChainOp
{
    uint kind;
    float scalar;
    uint64_t rhsAddress;
};

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

float accurateLog1p(float value)
{
    if (value == 0.0f)
        return value;
    if (value <= -1.0f || !isfinite(value))
        return log(1.0f + value);

    const float sum = 1.0f + value;
    const int exponentBits = (asint(sum) - 1061158912) & -8388608;
    const float reducedValue = asfloat(asint(value) - exponentBits);
    const float scale = asfloat(1082130432 - exponentBits);
    const float x = reducedValue + fma(0.25f, scale, -1.0f);
    const float exponent = float(exponentBits) * 1.1920928955078125e-7f;

    float polynomial = fma(-0.04534861445426941f, x, 0.10546888411045074f);
    polynomial = fma(polynomial, x, -0.13229703903198242f);
    polynomial = fma(polynomial, x, 0.14491446316242218f);
    polynomial = fma(polynomial, x, -0.16641564667224884f);
    polynomial = fma(polynomial, x, 0.199888676404953f);
    polynomial = fma(polynomial, x, -0.2500019669532776f);
    polynomial = fma(polynomial, x, 0.33333510160446167f);
    polynomial = fma(polynomial, x, -0.5f);
    const float reducedLog = fma(polynomial * x, x, x);
    return fma(exponent, 0.6931471824645996f, reducedLog);
}

float accurateAsin(float value)
{
    const float bounded = clamp(value, -1.0f, 1.0f);
    return atan2(bounded, sqrt(max(0.0f, 1.0f - bounded * bounded)));
}

float accurateAcos(float value)
{
    const float bounded = clamp(value, -1.0f, 1.0f);
    return atan2(sqrt(max(0.0f, 1.0f - bounded * bounded)), bounded);
}

float floatBinary(float lhs, float rhs)
{
    if (kOp == 0 || kOp == 4) return lhs + rhs;
    if (kOp == 1 || kOp == 5) return lhs - rhs;
    if (kOp == 2 || kOp == 6) return lhs * rhs;
    if (kOp == 3 || kOp == 7) return lhs / rhs;
    if (kOp == 8 || kOp == 25) return rhs == 2.0f ? lhs * lhs : pow(lhs, rhs);
    if (kOp == 9 || kOp == 26) return fmod(lhs, rhs);
    if (kOp == 27 || kOp == 57) return ieeeMaximum(lhs, rhs);
    if (kOp == 28 || kOp == 58) return ieeeMinimum(lhs, rhs);
    return lhs;
}

bool floatPredicate(float lhs, float rhs)
{
    if (kOp == 29 || kOp == 59) return lhs == rhs;
    if (kOp == 30 || kOp == 60) return lhs != rhs;
    if (kOp == 31 || kOp == 61) return lhs < rhs;
    if (kOp == 32 || kOp == 62) return lhs <= rhs;
    if (kOp == 33 || kOp == 63) return lhs > rhs;
    if (kOp == 34 || kOp == 64) return lhs >= rhs;
    if (kOp == 35) return lhs != 0.0f && rhs != 0.0f;
    if (kOp == 36) return lhs != 0.0f || rhs != 0.0f;
    if (kOp == 37) return (lhs != 0.0f) != (rhs != 0.0f);
    if (kOp == 53) return isnan(lhs);
    if (kOp == 54) return isinf(lhs);
    if (kOp == 55) return isfinite(lhs);
    if (kOp == 56) return lhs == 0.0f;
    return false;
}

float floatUnary(float value, float scalar, bool scalarOnRight)
{
    if (kOp <= 9 || (kOp >= 57 && kOp <= 64))
        return scalarOnRight ? floatBinary(value, scalar) : floatBinary(scalar, value);
    if (kOp == 10) return abs(value);
    if (kOp == 11) return -value;
    if (kOp == 12) return exp(value);
    if (kOp == 13) return log(value);
    if (kOp == 14) return sqrt(value);
    if (kOp == 15) return 1.0f / (1.0f + exp(-value));
    if (kOp == 16) return isnan(value) ? value : ieeeMaximum(value, 0.0f);
    if (kOp == 17) return value * value;
    if (kOp == 18) return tanh(value);
    if (kOp == 19) return rsqrt(value);
    if (kOp == 20) return float((value > 0.0f) - (value < 0.0f));
    if (kOp == 21) return 1.0f / value;
    if (kOp == 22) return floor(value);
    if (kOp == 23) return ceil(value);
    if (kOp == 24) return ieeeRound(value);
    if (kOp == 38) return exp2(value);
    if (kOp == 39) return log2(value);
    if (kOp == 40) return log10(value);
    if (kOp == 41) return accurateLog1p(value);
    if (kOp == 42) return sin(value);
    if (kOp == 43) return cos(value);
    if (kOp == 44) return tan(value);
    if (kOp == 45) return accurateAsin(value);
    if (kOp == 46) return accurateAcos(value);
    if (kOp == 47) return atan(value);
    if (kOp == 48) return sinh(value);
    if (kOp == 49) return cosh(value);
    if (kOp == 50)
    {
        const float inner = sqrt(2.0f / 3.14159265358979323846f) *
                            (value + 0.044715f * value * value * value);
        return 0.5f * value * (1.0f + tanh(inner));
    }
    if (kOp == 51) return value / (1.0f + exp(-value));
    if (kOp == 52) return trunc(value);
    if (kOp == 65) return exp(value) * scalar;
    if (kOp == 66) return abs(value * scalar);
    if (kOp == 67) return ieeeMaximum(value * scalar, 0.0f);
    return value;
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

int intBinary(int lhs, int rhs)
{
    if (kOp == 0 || kOp == 4) return lhs + rhs;
    if (kOp == 1 || kOp == 5) return lhs - rhs;
    if (kOp == 2 || kOp == 6) return lhs * rhs;
    if (kOp == 3 || kOp == 7) return rhs == 0 ? 0 : lhs / rhs;
    if (kOp == 8 || kOp == 25) return rhs == 2 ? lhs * lhs : int(pow(float(lhs), float(rhs)));
    if (kOp == 9 || kOp == 26) return rhs == 0 ? 0 : lhs % rhs;
    if (kOp == 27 || kOp == 57) return max(lhs, rhs);
    if (kOp == 28 || kOp == 58) return min(lhs, rhs);
    return lhs;
}

bool intPredicate(int lhs, int rhs)
{
    if (kOp == 29 || kOp == 59) return lhs == rhs;
    if (kOp == 30 || kOp == 60) return lhs != rhs;
    if (kOp == 31 || kOp == 61) return lhs < rhs;
    if (kOp == 32 || kOp == 62) return lhs <= rhs;
    if (kOp == 33 || kOp == 63) return lhs > rhs;
    if (kOp == 34 || kOp == 64) return lhs >= rhs;
    if (kOp == 35) return lhs != 0 && rhs != 0;
    if (kOp == 36) return lhs != 0 || rhs != 0;
    if (kOp == 37) return (lhs != 0) != (rhs != 0);
    if (kOp == 53 || kOp == 54) return false;
    if (kOp == 55) return true;
    if (kOp == 56) return lhs == 0;
    return false;
}

int intUnary(int value, int scalar, bool scalarOnRight)
{
    if (kOp <= 9 || (kOp >= 57 && kOp <= 64))
        return scalarOnRight ? intBinary(value, scalar) : intBinary(scalar, value);
    if (kOp == 10) return abs(value);
    if (kOp == 11) return -value;
    if (kOp == 16) return max(value, 0);
    if (kOp == 17) return value * value;
    if (kOp == 20) return (value > 0) - (value < 0);
    if (kOp == 22 || kOp == 23 || kOp == 24 || kOp == 52) return value;
    return int(floatUnary(float(value), float(scalar), scalarOnRight));
}

int64_t int64Binary(int64_t lhs, int64_t rhs)
{
    if (kOp == 4) return lhs + rhs;
    if (kOp == 5) return lhs - rhs;
    if (kOp == 6) return lhs * rhs;
    if (kOp == 7) return rhs == 0 ? int64_t(0) : lhs / rhs;
    if (kOp == 25) return rhs == 2 ? lhs * lhs : int64_t(pow(float(lhs), float(rhs)));
    if (kOp == 26) return rhs == 0 ? int64_t(0) : lhs % rhs;
    if (kOp == 27) return lhs < rhs ? rhs : lhs;
    if (kOp == 28) return rhs < lhs ? rhs : lhs;
    return lhs;
}

bool int64Predicate(int64_t lhs, int64_t rhs)
{
    if (kOp == 29) return lhs == rhs;
    if (kOp == 30) return lhs != rhs;
    if (kOp == 31) return lhs < rhs;
    if (kOp == 32) return lhs <= rhs;
    if (kOp == 33) return lhs > rhs;
    if (kOp == 34) return lhs >= rhs;
    if (kOp == 35) return lhs != 0 && rhs != 0;
    if (kOp == 36) return lhs != 0 || rhs != 0;
    if (kOp == 37) return (lhs != 0) != (rhs != 0);
    return false;
}

float loadFloat(uint64_t address, uint index)
{
    if (kInputDType == kFloat32)
        return ((Ptr<float, Access.Read>)address)[index];
    if (kInputDType == kFloat16)
        return f16tof32(uint(((Ptr<uint16_t, Access.Read>)address)[index]));
    return 0.0f;
}

void storeFloat(uint64_t address, uint index, float value)
{
    if (kOutputDType == kFloat32)
        ((Ptr<float, Access.ReadWrite>)address)[index] = value;
    else
        ((Ptr<uint16_t, Access.ReadWrite>)address)[index] = uint16_t(f32tof16(value));
}

void evaluateScalar(uint index, uniform Parameters parameters)
{
    const bool scalarOnRight = (parameters.flags & 1u) != 0u;
    if (kArity == 4)
    {
        float value = ((Ptr<float, Access.Read>)parameters.lhsAddress)[index];
        Ptr<ChainOp, Access.Read> chain = (Ptr<ChainOp, Access.Read>)parameters.rhsAddress;
        for (uint operation = 0; operation < parameters.scalarKind; ++operation)
            value = applyChainOp(value, chain[operation], index);
        ((Ptr<float, Access.ReadWrite>)parameters.outputAddress)[index] = value;
        return;
    }
    if (kArity == 5)
    {
        if (kInputDType == kFloat32)
        {
            const float value = ((Ptr<float, Access.Read>)parameters.lhsAddress)[index];
            const float maximum = asfloat(uint(parameters.scalarInt64));
            ((Ptr<float, Access.ReadWrite>)parameters.outputAddress)[index] =
                isnan(value) ? value : min(max(value, parameters.scalarFloat), maximum);
        }
        else
        {
            const int value = ((Ptr<int, Access.Read>)parameters.lhsAddress)[index];
            const int minimum = int(uint(parameters.scalarInt64));
            const int maximum = int(uint(parameters.scalarInt64 >> 32));
            ((Ptr<int, Access.ReadWrite>)parameters.outputAddress)[index] =
                min(max(value, minimum), maximum);
        }
        return;
    }
    if (kInputDType == kFloat32 || kInputDType == kFloat16)
    {
        const float lhs = loadFloat(parameters.lhsAddress, index);
        const float rhs = kArity == 2 ? loadFloat(parameters.rhsAddress, index)
                                     : parameters.scalarFloat;
        if (kOutputDType == kBool)
        {
            storeByte(parameters.outputAddress, index, floatPredicate(lhs, rhs) ? 1u : 0u);
            return;
        }
        storeFloat(parameters.outputAddress, index,
                   kArity == 2 ? floatBinary(lhs, rhs)
                               : floatUnary(lhs, rhs, scalarOnRight));
        return;
    }
    if (kInputDType == kInt32)
    {
        const int lhs = ((Ptr<int, Access.Read>)parameters.lhsAddress)[index];
        const int rhs = kArity == 2
                            ? ((Ptr<int, Access.Read>)parameters.rhsAddress)[index]
                            : int(parameters.scalarInt64);
        if (kOutputDType == kBool)
            storeByte(parameters.outputAddress, index, intPredicate(lhs, rhs) ? 1u : 0u);
        else
            ((Ptr<int, Access.ReadWrite>)parameters.outputAddress)[index] =
                kArity == 2 ? intBinary(lhs, rhs) : intUnary(lhs, rhs, scalarOnRight);
        return;
    }
    if (kInputDType == kInt64)
    {
        const int64_t lhs = ((Ptr<int64_t, Access.Read>)parameters.lhsAddress)[index];
        const int64_t rhs = ((Ptr<int64_t, Access.Read>)parameters.rhsAddress)[index];
        if (kOutputDType == kBool)
            storeByte(parameters.outputAddress, index, int64Predicate(lhs, rhs) ? 1u : 0u);
        else
            ((Ptr<int64_t, Access.ReadWrite>)parameters.outputAddress)[index] =
                int64Binary(lhs, rhs);
        return;
    }
    if (kInputDType == kUInt8 || kInputDType == kBool)
    {
        const uint lhs = loadByte(parameters.lhsAddress, index);
        const uint rhs = kArity == 2
                             ? loadByte(parameters.rhsAddress, index)
                             : uint(parameters.scalarInt64);
        if (kOutputDType == kBool)
            storeByte(parameters.outputAddress, index,
                      intPredicate(int(lhs), int(rhs)) ? 1u : 0u);
        else
            storeByte(parameters.outputAddress, index,
                      uint(intBinary(int(lhs), int(rhs))) & 255u);
        return;
    }
    if (kInputDType == kUInt32)
    {
        const uint lhs = ((Ptr<uint, Access.Read>)parameters.lhsAddress)[index];
        const uint rhs = ((Ptr<uint, Access.Read>)parameters.rhsAddress)[index];
        ((Ptr<uint, Access.ReadWrite>)parameters.outputAddress)[index] =
            kOp == 4 ? lhs + rhs : kOp == 5 ? lhs - rhs : kOp == 6 ? lhs * rhs
            : kOp == 7 ? (rhs == 0u ? 0u : lhs / rhs) : lhs;
    }
}

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID, uniform Parameters parameters)
{
    if (kVectorized != 0)
    {
        const uint base = dispatchThreadId.x * 4u;
        if (base >= parameters.count)
            return;
        const float4 lhs = ((Ptr<float4, Access.Read>)parameters.lhsAddress)[dispatchThreadId.x];
        float4 rhs = float4(parameters.scalarFloat);
        if (kArity == 2)
            rhs = ((Ptr<float4, Access.Read>)parameters.rhsAddress)[dispatchThreadId.x];
        if (kOutputDType == kFloat32)
        {
            float4 result;
            result.x = kArity == 2 ? floatBinary(lhs.x, rhs.x) : floatUnary(lhs.x, rhs.x, true);
            result.y = kArity == 2 ? floatBinary(lhs.y, rhs.y) : floatUnary(lhs.y, rhs.y, true);
            result.z = kArity == 2 ? floatBinary(lhs.z, rhs.z) : floatUnary(lhs.z, rhs.z, true);
            result.w = kArity == 2 ? floatBinary(lhs.w, rhs.w) : floatUnary(lhs.w, rhs.w, true);
            ((Ptr<float4, Access.ReadWrite>)parameters.outputAddress)[dispatchThreadId.x] = result;
        }
        else
        {
            const uint packed = (floatPredicate(lhs.x, rhs.x) ? 1u : 0u) |
                                ((floatPredicate(lhs.y, rhs.y) ? 1u : 0u) << 8u) |
                                ((floatPredicate(lhs.z, rhs.z) ? 1u : 0u) << 16u) |
                                ((floatPredicate(lhs.w, rhs.w) ? 1u : 0u) << 24u);
            ((Ptr<uint, Access.ReadWrite>)parameters.outputAddress)[dispatchThreadId.x] = packed;
        }
        return;
    }
    const uint index = dispatchThreadId.x;
    if (index < parameters.count)
        evaluateScalar(index, parameters);
}
