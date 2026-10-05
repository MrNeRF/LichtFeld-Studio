static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 64;

// 0 uniform [first, second), 1 bernoulli(first), 2 randint [low, high),
// 3 normal(first, second), 4 multinomial with replacement over the running
// sums of kinds 8 and 9, 5 Gumbel keys for sampling without replacement (the
// host sorts them), 7 weight statistics (output[0] = maximum, output[1] =
// invalid flag) in a single workgroup, 8 running sums of the scaled weights
// within blocks of kSumBlock, a thread per block, 9 the blocks' offsets and
// the total, in one thread. Metal's random_op adds the same sums in the same
// order, so both draw the same samples.
[SpecializationConstant]
const uint kKind = 0;

groupshared float sharedMaximum[LFS_LOCAL_SIZE_X];
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
static const uint kSumBlock = 1024;
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
        float maximum = 0.0f;
        uint invalid = 0u;
        for (uint index = thread; index < parameters.count; index += LFS_LOCAL_SIZE_X)
        {
            const float weight = weights[index];
            if (!(weight >= 0.0f) || isinf(weight))
                invalid = 1u;
            maximum = max(maximum, weight);
        }
        sharedMaximum[thread] = maximum;
        sharedInvalid[thread] = invalid;
        GroupMemoryBarrierWithGroupSync();
        for (uint span = LFS_LOCAL_SIZE_X / 2; span > 0; span >>= 1)
        {
            if (thread < span)
            {
                sharedMaximum[thread] = max(sharedMaximum[thread], sharedMaximum[thread + span]);
                sharedInvalid[thread] |= sharedInvalid[thread + span];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if (thread == 0)
        {
            ((Ptr<float, Access.ReadWrite>)parameters.outputAddress)[0] = sharedMaximum[0];
            ((Ptr<uint, Access.ReadWrite>)parameters.outputAddress)[1] = sharedInvalid[0];
        }
        return;
    }
    const uint blocks = (parameters.count + kSumBlock - 1) / kSumBlock;
    if (kKind == 8)
    {
        // The weights, scaled by a power of two (exact), summed in order
        // within the block.
        Ptr<float, Access.Read> weights = (Ptr<float, Access.Read>)parameters.weightsAddress;
        Ptr<float, Access.ReadWrite> sums = (Ptr<float, Access.ReadWrite>)parameters.keysAddress;
        for (uint block = dispatchThreadId.x; block < blocks; block += stride)
        {
            const uint end = min(parameters.count, (block + 1) * kSumBlock);
            float running = 0.0f;
            for (uint category = block * kSumBlock; category < end; ++category)
            {
                running += weights[category] * parameters.first;
                sums[category] = running;
            }
        }
        return;
    }
    if (kKind == 9)
    {
        // Each block's offset is the running sum of the block totals before
        // it; the running sum of category i is its block's offset plus its
        // sum within the block, which never decreases along the categories.
        if (dispatchThreadId.x != 0)
            return;
        Ptr<float, Access.ReadWrite> sums = (Ptr<float, Access.ReadWrite>)parameters.keysAddress;
        float offset = 0.0f;
        for (uint block = 0; block < blocks; ++block)
        {
            sums[parameters.count + block] = offset;
            offset += sums[min(parameters.count, (block + 1) * kSumBlock) - 1];
        }
        sums[parameters.count + blocks] = offset;
        return;
    }
    const uint total = kKind == 4 ? parameters.sampleCount : parameters.count;
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
            // The first category whose running sum exceeds the draw: a binary
            // search over the blocks' last sums, then within the block.
            Ptr<float, Access.Read> sums = (Ptr<float, Access.Read>)parameters.keysAddress;
            const float u = unitInterval(words.x) * sums[parameters.count + blocks];
            uint low = 0;
            uint high = blocks;
            while (low < high)
            {
                const uint middle = (low + high) / 2;
                if (u < sums[parameters.count + middle] + sums[min(parameters.count, (middle + 1) * kSumBlock) - 1])
                    high = middle;
                else
                    low = middle + 1;
            }
            uint sample = parameters.count - 1;
            if (low < blocks)
            {
                const float offset = sums[parameters.count + low];
                uint first = low * kSumBlock;
                uint last = min(parameters.count, (low + 1) * kSumBlock);
                while (first < last)
                {
                    const uint middle = (first + last) / 2;
                    if (u < offset + sums[middle])
                        last = middle;
                    else
                        first = middle + 1;
                }
                sample = min(first, parameters.count - 1);
            }
            ((Ptr<int64_t, Access.ReadWrite>)parameters.outputAddress)[index] = int64_t(sample);
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
