static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 40;

// 0 uniform [first, second), 2 randint [low, high), 3 normal(first, second).
[SpecializationConstant]
const uint kKind = 0;


struct Parameters
{
    uint64_t outputAddress;
    uint64_t seed;
    uint count;
    int low;
    int high;
    float first;
    float second;
    uint pad0;
};

static const float kUnitScale = 1.0f / 16777216.0f;
static const float kTwoPi = 6.28318530717958647692f;

// Philox4x32-10 (Salmon et al. 2011): the element index is the counter and
// the seed is the key, so every element draws an independent, reproducible
// 128-bit block.
void multiplyHighLow(uint lhs, uint rhs, out uint high, out uint low)
{
    const uint64_t product = uint64_t(lhs) * uint64_t(rhs);
    high = uint(product >> 32);
    low = uint(product & uint64_t(0xffffffffu));
}

uint4 philox(uint4 counter, uint2 key)
{
    for (uint round = 0; round < 10; ++round)
    {
        uint high0;
        uint low0;
        uint high1;
        uint low1;
        multiplyHighLow(0xD2511F53u, counter.x, high0, low0);
        multiplyHighLow(0xCD9E8D57u, counter.z, high1, low1);
        counter = uint4(high1 ^ counter.y ^ key.x, low1, high0 ^ counter.w ^ key.y, low0);
        key.x += 0x9E3779B9u;
        key.y += 0xBB67AE85u;
    }
    return counter;
}

uint4 draw(uint64_t index, uint64_t seed)
{
    const uint4 counter = uint4(uint(index & uint64_t(0xffffffffu)), uint(index >> 32), 0u, 0u);
    const uint2 key = uint2(uint(seed & uint64_t(0xffffffffu)), uint(seed >> 32));
    return philox(counter, key);
}

// 24-bit mantissa draw in [0, 1), the same construction as the CUDA kernels.
float unitInterval(uint word)
{
    return float(word >> 8) * kUnitScale;
}

// The float just below value, in the direction of smaller numbers.
float floatBelow(float value)
{
    const uint bits = asuint(value);
    if (value > 0.0f)
        return asfloat(bits - 1u);
    if (value == 0.0f)
        return asfloat(0x80000001u);
    return asfloat(bits + 1u);
}

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID, uint3 groupThreadId : SV_GroupThreadID,
          uniform Parameters parameters)
{
    const uint3 numWorkgroups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    const uint stride = numWorkgroups.x * LFS_LOCAL_SIZE_X;
    const uint total = parameters.count;
    for (uint index = dispatchThreadId.x; index < total; index += stride)
    {
        const uint4 words = draw(index, parameters.seed);
        if (kKind == 0)
        {
            float value = parameters.first;
            if (parameters.first != parameters.second)
            {
                value = fma(unitInterval(words.x), parameters.second - parameters.first, parameters.first);
                if (!(value < parameters.second))
                    value = floatBelow(parameters.second);
            }
            ((Ptr<float, Access.ReadWrite>)parameters.outputAddress)[index] = value;
        }
        else if (kKind == 2)
        {
            const uint64_t range = uint64_t(int64_t(parameters.high) - int64_t(parameters.low));
            const uint64_t offset = (uint64_t(words.x) * range) >> 32;
            ((Ptr<int, Access.ReadWrite>)parameters.outputAddress)[index] =
                int(int64_t(parameters.low) + int64_t(offset));
        }
        else if (kKind == 3)
        {
            const float u1 = float((words.x >> 8) + 1u) * kUnitScale;
            const float u2 = unitInterval(words.y);
            const float radius = sqrt(-2.0f * log(u1));
            const float z = radius * cos(kTwoPi * u2);
            ((Ptr<float, Access.ReadWrite>)parameters.outputAddress)[index] =
                parameters.first + parameters.second * z;
        }
    }
}
