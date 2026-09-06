static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 48;

// 0 masked_fill, 1 and_live, 2 compact select (output[scan[i]] = data[i]),
// 3 compact scatter (data[i] = source[scan[i]]), 4 nonzero (output[scan[i]] = i),
// 5 exclusive scan of the predicate in one workgroup (scan[i] and the total).
[SpecializationConstant]
const uint kMode = 0;
// Element codes follow DataType: 0 Float32, 1 Float16, 2 Int32, 3 Int64, 4 UInt8, 5 Bool.
[SpecializationConstant]
const uint kDType = 0;
// Predicate source: 0 byte mask, 1 float element (nonzero).
[SpecializationConstant]
const uint kPredicate = 0;

static const bool kWide = kDType == 3;
static const bool kByte = kDType == 4 || kDType == 5;
static const bool kHalf = kDType == 1;

struct Parameters
{
    uint64_t dataAddress;
    uint64_t maskAddress;
    uint64_t sourceAddress;
    uint64_t scanAddress;
    uint count;
    uint fillLow;
    uint fillHigh;
    uint pad0;
};

groupshared uint sharedCount[LFS_LOCAL_SIZE_X];

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

void copyElement(uint64_t source, uint64_t sourceIndex, uint64_t destination, uint64_t destinationIndex)
{
    if (kWide)
        ((Ptr<uint64_t, Access.ReadWrite>)destination)[destinationIndex] =
            ((Ptr<uint64_t, Access.Read>)source)[sourceIndex];
    else if (!kByte && !kHalf)
        ((Ptr<uint, Access.ReadWrite>)destination)[destinationIndex] =
            ((Ptr<uint, Access.Read>)source)[sourceIndex];
    else if (kHalf)
        ((Ptr<uint16_t, Access.ReadWrite>)destination)[destinationIndex] =
            ((Ptr<uint16_t, Access.Read>)source)[sourceIndex];
    else
        storeByte(destination, destinationIndex, loadByte(source, sourceIndex));
}

void storeFill(uint64_t destination, uint64_t destinationIndex, uint low, uint high)
{
    if (kWide)
        ((Ptr<uint64_t, Access.ReadWrite>)destination)[destinationIndex] =
            (uint64_t(high) << 32) | uint64_t(low);
    else if (!kByte && !kHalf)
        ((Ptr<uint, Access.ReadWrite>)destination)[destinationIndex] = low;
    else if (kHalf)
        ((Ptr<uint16_t, Access.ReadWrite>)destination)[destinationIndex] = uint16_t(low);
    else
        storeByte(destination, destinationIndex, low);
}

bool predicate(uint64_t maskAddress, uint64_t index)
{
    if (kPredicate == 0)
        return loadByte(maskAddress, index) != 0u;
    return ((Ptr<float, Access.Read>)maskAddress)[index] != 0.0f;
}

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID, uint3 groupThreadId : SV_GroupThreadID,
          uniform Parameters parameters)
{
    const uint3 numWorkgroups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    const uint stride = numWorkgroups.x * LFS_LOCAL_SIZE_X;
    const uint thread = groupThreadId.x;
    Ptr<uint, Access.ReadWrite> scan = (Ptr<uint, Access.ReadWrite>)parameters.scanAddress;

    if (kMode == 5)
    {
        uint carry = 0;
        for (uint chunk = 0; chunk < parameters.count; chunk += LFS_LOCAL_SIZE_X)
        {
            const uint element = chunk + thread;
            const bool live = element < parameters.count;
            const uint flag = live && predicate(parameters.maskAddress, element) ? 1u : 0u;
            sharedCount[thread] = flag;
            GroupMemoryBarrierWithGroupSync();
            for (uint offset = 1; offset < LFS_LOCAL_SIZE_X; offset <<= 1)
            {
                const uint partial = thread >= offset ? sharedCount[thread - offset] : 0u;
                GroupMemoryBarrierWithGroupSync();
                sharedCount[thread] += partial;
                GroupMemoryBarrierWithGroupSync();
            }
            if (live)
                scan[element] = carry + sharedCount[thread] - flag;
            carry += sharedCount[LFS_LOCAL_SIZE_X - 1];
            GroupMemoryBarrierWithGroupSync();
        }
        if (thread == 0)
            ((Ptr<uint, Access.ReadWrite>)parameters.sourceAddress)[0] = carry;
        return;
    }

    for (uint64_t index = dispatchThreadId.x; index < parameters.count; index += stride)
    {
        if (kMode == 0)
        {
            if (loadByte(parameters.maskAddress, index) != 0u)
                storeFill(parameters.dataAddress, index, parameters.fillLow, parameters.fillHigh);
        }
        else if (kMode == 1)
        {
            if (loadByte(parameters.maskAddress, index) == 0u)
                storeByte(parameters.dataAddress, index, 0u);
        }
        else if (kMode == 2)
        {
            if (predicate(parameters.maskAddress, index))
                copyElement(parameters.dataAddress, index, parameters.sourceAddress, scan[index]);
        }
        else if (kMode == 3)
        {
            if (predicate(parameters.maskAddress, index))
                copyElement(parameters.sourceAddress, scan[index], parameters.dataAddress, index);
        }
        else
        {
            if (predicate(parameters.maskAddress, index))
                ((Ptr<int64_t, Access.ReadWrite>)parameters.sourceAddress)[scan[index]] = int64_t(index);
        }
    }
}
