static const uint LFS_LOCAL_SIZE_X = 256;
static const uint LFS_PUSH_CONSTANT_SIZE = 160;

// 0 gather (output shaped like the index tensor), 1 take (flat clamped),
// 2 index_select, 3 scatter assign, 4 scatter add, 5 index_put (flat clamped),
// 6 index_fill (scatter a constant), 7 deterministic scatter over runs of
// indices sorted by target (keys and positions from radix.slang).
[SpecializationConstant]
const uint kMode = 0;
// Element codes follow DataType: 0 Float32, 1 Float16, 2 Int32, 3 Int64, 4 UInt8, 5 Bool.
[SpecializationConstant]
const uint kDType = 0;
// Fused gather unary: 0 none, 1 abs, 2 sqrt, 3 neg.
[SpecializationConstant]
const uint kUnary = 0;
// 0 assert (records a device fault and writes zero), 1 clamp, 2 wrap.
[SpecializationConstant]
const uint kBoundary = 0;
// Mode 7 applies every source element of a duplicate-index run in position
// order: 0 assign (last wins), 1 add, 2 multiply, 3 max, 4 min.
[SpecializationConstant]
const uint kRunOp = 0;

// Element width classes; the payload is moved as bits.
static const bool kWide = kDType == 3;
static const bool kByte = kDType == 4 || kDType == 5;
static const bool kHalf = kDType == 1;
static const uint kFaultIndexOutOfRange = 1;

struct Parameters
{
    uint64_t inputAddress;
    uint64_t indexAddress;
    uint64_t valueAddress;
    uint64_t faultAddress;
    uint64_t keysAddress;
    uint64_t positionsAddress;
    uint outer;
    uint dimSize;
    uint inner;
    uint indexSize;
    uint total;
    uint rank;
    uint indexRank;
    uint dim;
    uint fillLow;
    uint fillHigh;
    uint inputSize;
    uint opId;
    uint inputDims[8];
    uint indexDims[8];
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

void addByte(uint64_t address, uint64_t index, uint value)
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
        const uint sum = (((assumed & mask) >> shift) + value) & 255u;
        const uint replacement = (assumed & ~mask) | (sum << shift);
        InterlockedCompareExchange(word[0], assumed, replacement, observed);
        if (observed == assumed)
            return;
        assumed = observed;
    } while (true);
}

void addFloatCompareExchange(uint64_t address, uint64_t index, float value)
{
    Ptr<uint, Access.ReadWrite> word = (Ptr<uint, Access.ReadWrite>)address;
    uint assumed = word[index];
    uint observed;
    do
    {
        const uint replacement = asuint(asfloat(assumed) + value);
        InterlockedCompareExchange(word[index], assumed, replacement, observed);
        if (observed == assumed)
            return;
        assumed = observed;
    } while (true);
}

// Moves one element; the payload is never interpreted.
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

void storeZero(uint64_t destination, uint64_t destinationIndex)
{
    if (kWide)
        ((Ptr<uint64_t, Access.ReadWrite>)destination)[destinationIndex] = uint64_t(0);
    else if (!kByte && !kHalf)
        ((Ptr<uint, Access.ReadWrite>)destination)[destinationIndex] = 0u;
    else if (kHalf)
        ((Ptr<uint16_t, Access.ReadWrite>)destination)[destinationIndex] = uint16_t(0);
    else
        storeByte(destination, destinationIndex, 0u);
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

// Mirrors the CUDA index_clamp_op: negative indices count from the end, then
// the result is clamped into the extent.
uint clampFlat(int index, uint size)
{
    if (index < 0)
        index += int(size);
    if (index < 0)
        return 0u;
    if (index >= int(size))
        return size - 1u;
    return uint(index);
}

// Wraps an index into [0, extent) with unsigned arithmetic; signed remainder
// of a negative value is not portable across SPIR-V emitters.
int wrapIndex(int index, uint extent)
{
    if (index >= 0)
        return int(uint(index) % extent);
    const uint remainder = uint(-index) % extent;
    return remainder == 0u ? 0 : int(extent - remainder);
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

float combineFloat(float accumulator, float value)
{
    if (kRunOp == 1) return accumulator + value;
    if (kRunOp == 2) return accumulator * value;
    if (kRunOp == 3) return ieeeMaximum(accumulator, value);
    if (kRunOp == 4) return ieeeMinimum(accumulator, value);
    return value;
}

int combineInt(int accumulator, int value)
{
    if (kRunOp == 1) return accumulator + value;
    if (kRunOp == 2) return accumulator * value;
    if (kRunOp == 3) return max(accumulator, value);
    if (kRunOp == 4) return min(accumulator, value);
    return value;
}

float applyUnary(float value)
{
    if (kUnary == 1) return abs(value);
    if (kUnary == 2) return sqrt(value);
    if (kUnary == 3) return -value;
    return value;
}

// First fault wins; the host reads code, index and extent after the wait.
void recordFault(uint64_t faultAddress, int index, uint extent, uint opId)
{
    Ptr<uint, Access.ReadWrite> words = (Ptr<uint, Access.ReadWrite>)faultAddress;
    uint observed;
    InterlockedCompareExchange(words[0], 0u, kFaultIndexOutOfRange, observed);
    if (observed == 0u)
    {
        words[1] = asuint(index);
        words[2] = extent;
        words[3] = opId;
    }
}

[numthreads(LFS_LOCAL_SIZE_X, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID, uniform Parameters parameters)
{
    const uint3 numWorkgroups = spirv_asm { result:$$uint3 = OpLoad builtin(NumWorkgroups:uint3); };
    const uint stride = numWorkgroups.x * LFS_LOCAL_SIZE_X;
    Ptr<int, Access.Read> indices = (Ptr<int, Access.Read>)parameters.indexAddress;

    for (uint64_t tid = dispatchThreadId.x; tid < parameters.total; tid += stride)
    {
        if (kMode == 0)
        {
            uint inputStrides[8];
            uint outputStrides[8];
            uint strideAccumulator = 1;
            for (int axis = int(parameters.rank) - 1; axis >= 0; --axis)
            {
                inputStrides[axis] = strideAccumulator;
                strideAccumulator *= parameters.inputDims[axis];
            }
            strideAccumulator = 1;
            for (int axis = int(parameters.indexRank) - 1; axis >= 0; --axis)
            {
                outputStrides[axis] = strideAccumulator;
                strideAccumulator *= parameters.indexDims[axis];
            }
            uint outputCoordinates[8];
            uint remaining = uint(tid);
            for (uint axis = 0; axis < parameters.indexRank; ++axis)
            {
                outputCoordinates[axis] = remaining / outputStrides[axis];
                remaining -= outputCoordinates[axis] * outputStrides[axis];
            }
            int gatherIndex = indices[tid];
            const uint extent = parameters.inputDims[parameters.dim];
            if (kBoundary == 1)
                gatherIndex = max(0, min(int(extent) - 1, gatherIndex));
            else if (kBoundary == 2)
                gatherIndex = wrapIndex(gatherIndex, extent);
            else if (gatherIndex < 0 || gatherIndex >= int(extent))
            {
                if (parameters.faultAddress != 0)
                    recordFault(parameters.faultAddress, gatherIndex, extent, parameters.opId);
                storeZero(parameters.valueAddress, tid);
                continue;
            }
            uint64_t sourceIndex = 0;
            bool inRange = true;
            for (uint axis = 0; axis < parameters.rank; ++axis)
            {
                uint coordinate;
                if (axis == parameters.dim)
                    coordinate = uint(gatherIndex);
                else if (axis < parameters.indexRank)
                    coordinate = outputCoordinates[axis];
                else
                    coordinate = 0;
                if (coordinate >= parameters.inputDims[axis])
                    inRange = false;
                sourceIndex += uint64_t(coordinate) * inputStrides[axis];
            }
            if (!inRange)
            {
                storeZero(parameters.valueAddress, tid);
                continue;
            }
            copyElement(parameters.inputAddress, sourceIndex, parameters.valueAddress, tid);
        }
        else if (kMode == 1)
        {
            const uint sourceIndex = clampFlat(indices[tid], parameters.inputSize);
            if (kUnary == 0)
                copyElement(parameters.inputAddress, sourceIndex, parameters.valueAddress, tid);
            else
                ((Ptr<float, Access.ReadWrite>)parameters.valueAddress)[tid] =
                    applyUnary(((Ptr<float, Access.Read>)parameters.inputAddress)[sourceIndex]);
        }
        else if (kMode == 2)
        {
            const uint64_t segment = uint64_t(parameters.indexSize) * parameters.inner;
            const uint outerIndex = uint(tid / segment);
            const uint position = uint((tid / parameters.inner) % parameters.indexSize);
            const uint innerIndex = uint(tid % parameters.inner);
            int selected = indices[position];
            if (kBoundary == 1)
                selected = max(0, min(int(parameters.dimSize) - 1, selected));
            else if (kBoundary == 2)
                selected = wrapIndex(selected, parameters.dimSize);
            else if (selected < 0 || selected >= int(parameters.dimSize))
            {
                if (parameters.faultAddress != 0)
                    recordFault(parameters.faultAddress, selected, parameters.dimSize, parameters.opId);
                storeZero(parameters.valueAddress, tid);
                continue;
            }
            const uint64_t sourceIndex =
                uint64_t(outerIndex) * parameters.dimSize * parameters.inner +
                uint64_t(selected) * parameters.inner + innerIndex;
            copyElement(parameters.inputAddress, sourceIndex, parameters.valueAddress, tid);
        }
        else if (kMode == 3 || kMode == 4 || kMode == 6)
        {
            const uint64_t segment = uint64_t(parameters.indexSize) * parameters.inner;
            const uint outerIndex = uint(tid / segment);
            const uint position = uint((tid / parameters.inner) % parameters.indexSize);
            const uint innerIndex = uint(tid % parameters.inner);
            const int target = indices[position];
            if (target < 0 || target >= int(parameters.dimSize))
                continue;
            const uint64_t destinationIndex =
                uint64_t(outerIndex) * parameters.dimSize * parameters.inner +
                uint64_t(target) * parameters.inner + innerIndex;
            if (kMode == 3)
                copyElement(parameters.valueAddress, tid, parameters.inputAddress, destinationIndex);
            else if (kMode == 6)
                storeFill(parameters.inputAddress, destinationIndex, parameters.fillLow, parameters.fillHigh);
            else if (kDType == 0)
                addFloatCompareExchange(parameters.inputAddress, destinationIndex,
                                        ((Ptr<float, Access.Read>)parameters.valueAddress)[tid]);
            else if (kDType == 2)
            {
                int previous;
                InterlockedAdd(((Ptr<int, Access.ReadWrite>)parameters.inputAddress)[destinationIndex],
                               ((Ptr<int, Access.Read>)parameters.valueAddress)[tid], previous);
            }
            else
                addByte(parameters.inputAddress, destinationIndex, loadByte(parameters.valueAddress, tid));
        }
        else if (kMode == 7)
        {
            // One thread per (sorted slot, outer, inner); only run starts work.
            Ptr<uint, Access.Read> keys = (Ptr<uint, Access.Read>)parameters.keysAddress;
            Ptr<uint, Access.Read> positions = (Ptr<uint, Access.Read>)parameters.positionsAddress;
            const uint64_t lanes = uint64_t(parameters.outer) * parameters.inner;
            const uint slot = uint(tid / lanes);
            const uint lane = uint(tid - uint64_t(slot) * lanes);
            const uint outerIndex = lane / parameters.inner;
            const uint innerIndex = lane - outerIndex * parameters.inner;
            const uint key = keys[slot];
            if (slot > 0 && keys[slot - 1] == key)
                continue;
            if (key >= parameters.dimSize)
                continue;
            const uint64_t destinationIndex =
                uint64_t(outerIndex) * parameters.dimSize * parameters.inner +
                uint64_t(key) * parameters.inner + innerIndex;
            uint end = slot + 1;
            while (end < parameters.indexSize && keys[end] == key)
                ++end;
            const uint64_t sourceBase = uint64_t(outerIndex) * parameters.indexSize * parameters.inner + innerIndex;
            if (kRunOp == 0)
            {
                copyElement(parameters.valueAddress, sourceBase + uint64_t(positions[end - 1]) * parameters.inner,
                            parameters.inputAddress, destinationIndex);
            }
            else if (kDType == 0)
            {
                Ptr<float, Access.ReadWrite> destination = (Ptr<float, Access.ReadWrite>)parameters.inputAddress;
                Ptr<float, Access.Read> source = (Ptr<float, Access.Read>)parameters.valueAddress;
                float accumulator = destination[destinationIndex];
                for (uint run = slot; run < end; ++run)
                    accumulator = combineFloat(accumulator, source[sourceBase + uint64_t(positions[run]) * parameters.inner]);
                destination[destinationIndex] = accumulator;
            }
            else if (kDType == 2)
            {
                Ptr<int, Access.ReadWrite> destination = (Ptr<int, Access.ReadWrite>)parameters.inputAddress;
                Ptr<int, Access.Read> source = (Ptr<int, Access.Read>)parameters.valueAddress;
                int accumulator = destination[destinationIndex];
                for (uint run = slot; run < end; ++run)
                    accumulator = combineInt(accumulator, source[sourceBase + uint64_t(positions[run]) * parameters.inner]);
                destination[destinationIndex] = accumulator;
            }
            else
            {
                int accumulator = int(loadByte(parameters.inputAddress, destinationIndex));
                for (uint run = slot; run < end; ++run)
                    accumulator = combineInt(accumulator, int(loadByte(parameters.valueAddress, sourceBase + uint64_t(positions[run]) * parameters.inner)));
                storeByte(parameters.inputAddress, destinationIndex, uint(accumulator) & 255u);
            }
        }
        else
        {
            const uint destinationIndex = clampFlat(indices[tid], parameters.inputSize);
            copyElement(parameters.valueAddress, tid, parameters.inputAddress, destinationIndex);
        }
    }
}
