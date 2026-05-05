#define CUMSUM_PHASE_blockScan 1
#define CUMSUM_PHASE_scanBlockSums 2
#define CUMSUM_PHASE_addBlockOffsets 3
#define CUMSUM_PHASE_singlePassPrefixSum 0

#ifndef CUMSUM_PHASE
#define CUMSUM_PHASE -1
#endif

#if CUMSUM_PHASE == CUMSUM_PHASE_blockScan
#define blockScan main
#elif CUMSUM_PHASE == CUMSUM_PHASE_scanBlockSums
#define scanBlockSums main
#elif CUMSUM_PHASE == CUMSUM_PHASE_addBlockOffsets
#define addBlockOffsets main
#elif CUMSUM_PHASE == CUMSUM_PHASE_singlePassPrefixSum
#define singlePassPrefixSum main
#endif


#ifndef SUBGROUP_SIZE
#define SUBGROUP_SIZE 32
#endif

#ifndef BLOCK_SIZE
    static const uint BLOCK_SIZE_0 = 1024;
    #define BLOCK_SIZE min(BLOCK_SIZE_0, SUBGROUP_SIZE*SUBGROUP_SIZE*SUBGROUP_SIZE)
#else
    static const uint BLOCK_SIZE_0 = BLOCK_SIZE;
#endif


layout(binding=0) StructuredBuffer<int32_t> g_input;

layout(binding=1) RWStructuredBuffer<int32_t> g_output;
layout(binding=2) RWStructuredBuffer<int32_t> g_blockSums;


groupshared int32_t s_data[BLOCK_SIZE_0];


struct Uniforms {
    uint32_t numElements;
    // uint32_t stride;
};



// Utility function for subgroup-level scan
int32_t subgroupScan(int32_t val, uint laneId) {
    [ForceUnroll]
    for (uint offset = 1; offset < SUBGROUP_SIZE; offset <<= 1) {
        // int32_t temp = WaveReadLaneAt(val, (laneId - offset) & (SUBGROUP_SIZE - 1));
        int32_t temp = WaveReadLaneAt(val, laneId - offset);
        if (laneId >= offset)
            val += temp;
    }
    return val;
    // return val + WavePrefixSum(val);
}


// Phase 1: Block-level scan with block sum generation
[numthreads(BLOCK_SIZE_0, 1, 1)]
void blockScan(
    uint3 groupId : SV_GroupID,
    uint3 localId : SV_GroupThreadID,
    uniform Uniforms uniforms
) {
    uint tid = localId.x;
    if (tid >= BLOCK_SIZE)
        return;
    uint blockId = groupId.x;
    uint gid = blockId * BLOCK_SIZE + tid;

    // Load data into shared memory
    s_data[tid] = (gid < uniforms.numElements) ? g_input[gid] : 0;
    GroupMemoryBarrierWithGroupSync();

    // Subgroup-level scan
    uint subgroupId = tid / SUBGROUP_SIZE;
    uint laneId = tid % SUBGROUP_SIZE;

    int32_t subgroupSum = subgroupScan(s_data[tid], laneId);
    s_data[tid] = subgroupSum;
    GroupMemoryBarrierWithGroupSync();

    // Block-level scan of subgroup sums
    if (tid < (BLOCK_SIZE / SUBGROUP_SIZE)) {
        int32_t subgroupSumVal = s_data[tid * SUBGROUP_SIZE + SUBGROUP_SIZE - 1];
        int32_t blockScanResult = subgroupSumVal + WavePrefixSum(subgroupSumVal);
        s_data[tid * SUBGROUP_SIZE + SUBGROUP_SIZE - 1] = blockScanResult;
    }
    GroupMemoryBarrierWithGroupSync();

    // Distribute block scan results back to subgroups
    if (subgroupId > 0 && laneId != SUBGROUP_SIZE - 1) {
        int32_t subgroupOffset = s_data[subgroupId * SUBGROUP_SIZE - 1];
        s_data[tid] += subgroupOffset;
    }
    GroupMemoryBarrierWithGroupSync();

    // Write results and block sum
    if (gid < uniforms.numElements) {
        g_output[gid] = s_data[tid];
    }

    // Store block sum for next phase
    if (tid == BLOCK_SIZE - 1) {
        g_blockSums[blockId] = s_data[tid];
    }
}


// Phase 2: Scan the block sums (recursive for large arrays)
[numthreads(BLOCK_SIZE_0, 1, 1)]
void scanBlockSums(
    uint3 groupId : SV_GroupID,
    uint3 localId : SV_GroupThreadID,
    uniform Uniforms uniforms
) {
    uint tid = localId.x;
    if (tid >= BLOCK_SIZE)
        return;
    uint blockId = groupId.x;
    uint gid = blockId * BLOCK_SIZE + tid;

    // Load block sums
    s_data[tid] = (gid < uniforms.numElements) ? g_blockSums[gid] : 0;
    GroupMemoryBarrierWithGroupSync();

    // Perform scan similar to blockScan
    uint subgroupId = tid / SUBGROUP_SIZE;
    uint laneId = tid % SUBGROUP_SIZE;

    int32_t subgroupSum = subgroupScan(s_data[tid], laneId);
    s_data[tid] = subgroupSum;
    GroupMemoryBarrierWithGroupSync();

    if (tid < (BLOCK_SIZE / SUBGROUP_SIZE)) {
        int32_t subgroupSumVal = s_data[tid * SUBGROUP_SIZE + SUBGROUP_SIZE - 1];
        int32_t blockScanResult = subgroupSumVal + WavePrefixSum(subgroupSumVal);
        s_data[tid * SUBGROUP_SIZE + SUBGROUP_SIZE - 1] = blockScanResult;
    }
    GroupMemoryBarrierWithGroupSync();

    if (subgroupId > 0 && laneId != SUBGROUP_SIZE-1) {
        int32_t subgroupOffset = s_data[subgroupId * SUBGROUP_SIZE - 1];
        s_data[tid] += subgroupOffset;
    }
    GroupMemoryBarrierWithGroupSync();

    // Write scanned block sums back
    if (gid < uniforms.numElements) {
        g_blockSums[gid] = s_data[tid];
    }
}


// Phase 3: Add block offsets to final results
[numthreads(BLOCK_SIZE_0, 1, 1)]
void addBlockOffsets(
    uint3 groupId : SV_GroupID,
    uint3 localId : SV_GroupThreadID,
    uniform Uniforms uniforms
) {
    uint tid = localId.x;
    if (tid >= BLOCK_SIZE)
        return;
    uint blockId = groupId.x;
    uint gid = blockId * BLOCK_SIZE + tid;

    if (gid < uniforms.numElements && blockId > 0) {
        g_output[gid] += g_blockSums[blockId-1];
    }
}


// Alternative single-pass version for smaller arrays (up to BLOCK_SIZE_0 elements)
[numthreads(BLOCK_SIZE_0, 1, 1)]
void singlePassPrefixSum(
    uint3 localId : SV_GroupThreadID,
    uint3 globalId : SV_DispatchThreadID,
    uniform Uniforms uniforms
) {
    uint tid = localId.x;
    uint gid = globalId.x;

    // Load data
    int32_t v_in = (gid < uniforms.numElements) ? g_input[gid] : 0;
    s_data[tid] = v_in;
    GroupMemoryBarrierWithGroupSync();

    // Up-sweep (reduction) phase
    for (uint stride = 1; stride < BLOCK_SIZE_0; stride <<= 1) {
        if (tid % (stride * 2) == 0) {
            s_data[tid + stride * 2 - 1] += s_data[tid + stride - 1];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Clear the last element
    if (tid == 0) {
        s_data[BLOCK_SIZE_0 - 1] = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    // Down-sweep phase
    for (uint stride = BLOCK_SIZE_0 / 2; stride > 0; stride >>= 1) {
        if (tid % (stride * 2) == 0) {
            int32_t temp = s_data[tid + stride - 1];
            s_data[tid + stride - 1] = s_data[tid + stride * 2 - 1];
            s_data[tid + stride * 2 - 1] += temp;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Write exclusive prefix sum result
    if (gid < uniforms.numElements) {
        g_output[gid] = s_data[tid] + v_in;
    }
}
