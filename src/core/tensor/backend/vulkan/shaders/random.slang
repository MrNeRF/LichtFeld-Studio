static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 64;

// 0 uniform [first, second), 1 bernoulli(first), 2 randint [low, high),
// 3 normal(first, second), 4 multinomial with replacement over weights,
// 5 Gumbel keys for sampling without replacement, 6 rank selection of the
// sampleCount largest keys into the int64 output, 7 weight statistics
// (output[0] = sum as float bits, output[1] = 1 when a weight is not finite
// or negative), run by a single workgroup.
[SpecializationConstant]
const uint kKind = 0;

groupshared float sharedSum[LFS_LOCAL_SIZE_X];
groupshared uint sharedInvalid[LFS_LOCAL_SIZE_X];

struct Parameters
{
    uint64_t outputAddress;
    uint64_t weightsAddress;
    uint64_t keysAddress;
    uint64_t seed;
    uint count;
    uint sampleCount;
    int low;
    int high;
    float first;
    float second;
    float total;
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
    if (kKind == 7)
    {
        Ptr<float, Access.Read> weights = (Ptr<float, Access.Read>)parameters.weightsAddress;
        const uint thread = groupThreadId.x;
        float sum = 0.0f;
        uint invalid = 0u;
        for (uint index = thread; index < parameters.count; index += LFS_LOCAL_SIZE_X)
        {
            const float weight = weights[index];
            if (!(weight >= 0.0f) || isinf(weight))
                invalid = 1u;
            sum += weight;
        }
        sharedSum[thread] = sum;
        sharedInvalid[thread] = invalid;
        GroupMemoryBarrierWithGroupSync();
        for (uint span = LFS_LOCAL_SIZE_X / 2; span > 0; span >>= 1)
        {
            if (thread < span)
            {
                sharedSum[thread] += sharedSum[thread + span];
                sharedInvalid[thread] |= sharedInvalid[thread + span];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if (thread == 0)
        {
            ((Ptr<float, Access.ReadWrite>)parameters.outputAddress)[0] = sharedSum[0];
            ((Ptr<uint, Access.ReadWrite>)parameters.outputAddress)[1] = sharedInvalid[0];
        }
        return;
    }
    const uint total = kKind == 4 ? parameters.sampleCount : parameters.count;
    for (uint index = dispatchThreadId.x; index < total; index += stride)
    {
        if (kKind == 6)
        {
            Ptr<float, Access.Read> keys = (Ptr<float, Access.Read>)parameters.keysAddress;
            const float key = keys[index];
            uint rank = 0;
            for (uint other = 0; other < parameters.count; ++other)
            {
                const float candidate = keys[other];
                if (candidate > key || (candidate == key && other < index))
                    ++rank;
            }
            if (rank < parameters.sampleCount)
                ((Ptr<int64_t, Access.ReadWrite>)parameters.outputAddress)[rank] = int64_t(index);
            continue;
        }
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
        else if (kKind == 1)
        {
            ((Ptr<float, Access.ReadWrite>)parameters.outputAddress)[index] =
                unitInterval(words.x) < parameters.first ? 1.0f : 0.0f;
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
        else if (kKind == 4)
        {
            Ptr<float, Access.Read> weights = (Ptr<float, Access.Read>)parameters.weightsAddress;
            const float u = unitInterval(words.x) * parameters.total;
            float cumulative = 0.0f;
            int64_t sample = int64_t(parameters.count - 1);
            for (uint category = 0; category < parameters.count; ++category)
            {
                cumulative += weights[category];
                if (u < cumulative)
                {
                    sample = int64_t(category);
                    break;
                }
            }
            ((Ptr<int64_t, Access.ReadWrite>)parameters.outputAddress)[index] = sample;
        }
        else
        {
            Ptr<float, Access.Read> weights = (Ptr<float, Access.Read>)parameters.weightsAddress;
            float u = unitInterval(words.x);
            u = max(u, 1e-10f);
            u = min(u, 1.0f - 1e-10f);
            const float gumbel = -log(-log(u));
            const float logWeight = log(max(weights[index], 1e-10f));
            ((Ptr<float, Access.ReadWrite>)parameters.keysAddress)[index] = logWeight + gumbel;
        }
    }
}
