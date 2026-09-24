static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 48;
#include "byte_word.slangh"

// 0 masked_fill, 1 and_live, 2 compact select, 3 compact scatter, 4 nonzero,
// 5 predicate values for an inclusive scan. Selected offsets are scan[i] - 1.
// 6 scalar where (data[i] = mask[i] ? fill : source[i]).
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

uint evaluateByte(uint index, uniform Parameters parameters)
{
    const bool selected = loadByte(parameters.maskAddress, index) != 0u;
    if (kMode == 0 && selected) return parameters.fillLow & 255u;
    if (kMode == 1 && !selected) return 0u;
    return loadByte(parameters.dataAddress, index);
}

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID,
          uniform Parameters parameters)
{
    if (kByte && kMode <= 1)
    {
        const uint64_t start = parameters.dataAddress;
        const uint64_t end = start + uint64_t(parameters.count);
        const uint64_t firstWord = start & ~uint64_t(3);
        const uint wordCount = uint((end - firstWord + uint64_t(3)) / uint64_t(4));
        const uint3 groups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
        for (uint wordIndex = dispatchThreadId.x; wordIndex < wordCount;
             wordIndex += groups.x * LFS_LOCAL_SIZE_X)
        {
            const uint64_t wordAddress = firstWord + uint64_t(wordIndex) * uint64_t(4);
            uint packed = 0;
            uint mask = 0;
            [unroll]
            for (uint lane = 0; lane < 4u; ++lane)
            {
                const uint64_t byteAddress = wordAddress + uint64_t(lane);
                if (byteAddress >= start && byteAddress < end)
                {
                    packed |= evaluateByte(uint(byteAddress - start), parameters) << (lane * 8u);
                    mask |= 255u << (lane * 8u);
                }
            }
            storeByteWord(wordAddress, packed, mask);
        }
        return;
    }
    const uint3 numWorkgroups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    const uint stride = numWorkgroups.x * LFS_LOCAL_SIZE_X;
    Ptr<uint, Access.ReadWrite> scan = (Ptr<uint, Access.ReadWrite>)parameters.scanAddress;

    if (kMode == 5)
    {
        for (uint64_t index = dispatchThreadId.x; index < parameters.count; index += stride)
            scan[index] = predicate(parameters.maskAddress, index) ? 1u : 0u;
        return;
    }

    for (uint64_t index = dispatchThreadId.x; index < parameters.count; index += stride)
    {
        if (kMode == 0)
        {
            if (loadByte(parameters.maskAddress, index) != 0u)
                storeFill(parameters.dataAddress, index, parameters.fillLow, parameters.fillHigh);
        }
        else if (kMode == 2)
        {
            if (predicate(parameters.maskAddress, index))
                copyElement(parameters.dataAddress, index, parameters.sourceAddress, scan[index] - 1u);
        }
        else if (kMode == 3)
        {
            if (predicate(parameters.maskAddress, index))
                copyElement(parameters.sourceAddress, scan[index] - 1u, parameters.dataAddress, index);
        }
        else if (kMode == 6)
        {
            if (predicate(parameters.maskAddress, index))
                storeFill(parameters.dataAddress, index, parameters.fillLow, parameters.fillHigh);
            else
                copyElement(parameters.sourceAddress, index, parameters.dataAddress, index);
        }
        else
        {
            if (predicate(parameters.maskAddress, index))
                ((Ptr<int64_t, Access.ReadWrite>)parameters.sourceAddress)[scan[index] - 1u] = int64_t(index);
        }
    }
}
