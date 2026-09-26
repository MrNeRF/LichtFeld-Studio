static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 56;

// Indexed extrema along the middle axis of an (outer, reduce, inner) view.
// Each candidate becomes a 64-bit key whose largest value is the CPU loop's
// choice: the high word orders values (inverted for the minimum), a NaN at
// position 0 sits above every number and later NaNs above that; the low word
// is the inverted position, so ties keep the first. With one chunk the first
// passes finish the output themselves.

// 1 picks the maximum, 0 the minimum.
[SpecializationConstant]
const uint kMaximum = 1;
// 0 a thread folds a chunk of one output's line, 1 a workgroup folds a chunk
// of one contiguous line, 2 a workgroup folds one output's chunk keys.
[SpecializationConstant]
const uint kPass = 0;

struct Parameters
{
    uint64_t inputAddress;
    uint64_t keysAddress;
    uint64_t valuesAddress;
    uint64_t indicesAddress;
    uint outer;
    uint reduce;
    uint inner;
    uint chunk;
    uint chunks;
    uint padding;
};

groupshared uint64_t sharedKeys[LFS_LOCAL_SIZE_X];

uint64_t extremeKey(float value, uint position)
{
    uint bits = asuint(value);
    uint order;
    if ((bits & 0x7fffffffu) > 0x7f800000u)
    {
        order = position == 0 ? 0xfffffffeu : 0xffffffffu;
    }
    else
    {
        if ((bits & 0x7fffffffu) == 0u)
            bits = 0u; // -0 ties +0, as the CPU's strict comparison does.
        order = (bits & 0x80000000u) != 0u ? ~bits : bits | 0x80000000u;
        if (kMaximum == 0)
            order = ~order;
    }
    return (uint64_t(order) << 32) | uint64_t(~position);
}

uint64_t largerKey(uint64_t a, uint64_t b)
{
    return a > b ? a : b;
}

void finish(uint output, uint64_t key, Parameters parameters)
{
    const uint position = ~uint(key & 0xffffffffull);
    const uint outerIndex = output / parameters.inner;
    const uint64_t base = uint64_t(outerIndex) * parameters.reduce * parameters.inner + (output - outerIndex * parameters.inner);
    Ptr<float, Access.Read> input = (Ptr<float, Access.Read>)parameters.inputAddress;
    ((Ptr<float, Access.ReadWrite>)parameters.valuesAddress)[output] = input[base + uint64_t(position) * parameters.inner];
    ((Ptr<int64_t, Access.ReadWrite>)parameters.indicesAddress)[output] = int64_t(position);
}

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint3 groupThreadId : SV_GroupThreadID, uniform Parameters parameters)
{
    Ptr<float, Access.Read> input = (Ptr<float, Access.Read>)parameters.inputAddress;
    Ptr<uint64_t, Access.ReadWrite> keys = (Ptr<uint64_t, Access.ReadWrite>)parameters.keysAddress;
    const uint thread = groupThreadId.x;
    const uint outputs = parameters.outer * parameters.inner;
    if (kPass >= 1)
    {
        // A workgroup folds a chunk of one contiguous line, or one output's chunk keys.
        const uint row = groupId.x;
        uint64_t best = 0;
        if (kPass == 1)
        {
            const uint64_t base = uint64_t(row) * parameters.reduce;
            const uint end = min(parameters.reduce, (groupId.y + 1) * parameters.chunk);
            for (uint position = groupId.y * parameters.chunk + thread; position < end; position += LFS_LOCAL_SIZE_X)
                best = largerKey(best, extremeKey(input[base + position], position));
        }
        else
        {
            for (uint chunk = thread; chunk < parameters.chunks; chunk += LFS_LOCAL_SIZE_X)
                best = largerKey(best, keys[uint64_t(chunk) * outputs + row]);
        }
        sharedKeys[thread] = best;
        GroupMemoryBarrierWithGroupSync();
        for (uint width = LFS_LOCAL_SIZE_X / 2; width > 0; width /= 2)
        {
            if (thread < width)
                sharedKeys[thread] = largerKey(sharedKeys[thread], sharedKeys[thread + width]);
            GroupMemoryBarrierWithGroupSync();
        }
        if (thread != 0)
            return;
        if (kPass == 2 || parameters.chunks == 1)
            finish(row, sharedKeys[0], parameters);
        else
            keys[uint64_t(groupId.y) * outputs + row] = sharedKeys[0];
        return;
    }
    const uint output = groupId.x * LFS_LOCAL_SIZE_X + thread;
    if (output >= outputs)
        return;
    uint64_t best = 0;
    const uint outerIndex = output / parameters.inner;
    const uint64_t base = uint64_t(outerIndex) * parameters.reduce * parameters.inner + (output - outerIndex * parameters.inner);
    const uint end = min(parameters.reduce, (groupId.y + 1) * parameters.chunk);
    for (uint position = groupId.y * parameters.chunk; position < end; ++position)
        best = largerKey(best, extremeKey(input[base + uint64_t(position) * parameters.inner], position));
    if (parameters.chunks == 1)
        finish(output, best, parameters);
    else
        keys[uint64_t(groupId.y) * outputs + output] = best;
}
