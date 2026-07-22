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


// CUMSUM_INDIRECT=1: the element count is GPU-resident (per-level counts
// written by prepare_visible_chain) and the dispatch is indirect, so the host
// never needs to know the visible count. numElements carries the
// level index instead of the count.
#ifndef CUMSUM_INDIRECT
#define CUMSUM_INDIRECT 0
#endif

layout(binding=0) StructuredBuffer<int32_t> g_input;

layout(binding=1) RWStructuredBuffer<int32_t> g_output;
layout(binding=2) RWStructuredBuffer<int32_t> g_blockSums;
#if CUMSUM_INDIRECT
layout(binding=3) StructuredBuffer<int32_t> g_counts;
#endif


groupshared int32_t s_data[BLOCK_SIZE_0];


struct Uniforms {
    uint32_t numElements;
    // uint32_t stride;
};

[ForceInline]
int32_t sat_add(int32_t a, int32_t b) {
    const uint32_t sum = uint32_t(a) + uint32_t(b);
    return int32_t(min(sum, 0x7fffffffu));
}

#if CUMSUM_INDIRECT
#define RESOLVE_NUM_ELEMENTS(u) uint32_t(max(g_counts[(u).numElements], 0))
#else
#define RESOLVE_NUM_ELEMENTS(u) (u).numElements
#endif



// Utility function for subgroup-level scan
int32_t subgroupScan(int32_t val, uint laneId) {
    [ForceUnroll]
    for (uint offset = 1; offset < SUBGROUP_SIZE; offset <<= 1) {
        // int32_t temp = WaveReadLaneAt(val, (laneId - offset) & (SUBGROUP_SIZE - 1));
        int32_t temp = WaveReadLaneAt(val, laneId - offset);
        if (laneId >= offset)
            val = sat_add(val, temp);
    }
    return val;
}


// Phase 1: Block-level scan with block sum generation
[numthreads(BLOCK_SIZE_0, 1, 1)]
void blockScan(
    uint3 groupId : SV_GroupID,
    uint3 localId : SV_GroupThreadID,
    uniform Uniforms uniforms
) {
    const uint32_t numElements = RESOLVE_NUM_ELEMENTS(uniforms);
    uint tid = localId.x;
    // The workgroup always launches BLOCK_SIZE_0 threads (numthreads is compile-time),
    // but only BLOCK_SIZE of them do work. Every launched thread must reach each
    // GroupMemoryBarrierWithGroupSync — a barrier hit by only a subset of the workgroup
    // is undefined behavior. So gate the shared-memory/wave work with `active` instead of
    // returning early. BLOCK_SIZE is always a whole multiple of SUBGROUP_SIZE, so each
    // subgroup is uniformly active or inactive and the wave ops keep their semantics.
    const bool active = (tid < BLOCK_SIZE);
    uint blockId = groupId.x;
    uint gid = blockId * BLOCK_SIZE + tid;

    // Load data into shared memory
    if (active)
        s_data[tid] = (gid < numElements) ? g_input[gid] : 0;
    GroupMemoryBarrierWithGroupSync();

    // Subgroup-level scan
    uint subgroupId = tid / SUBGROUP_SIZE;
    uint laneId = tid % SUBGROUP_SIZE;

    if (active) {
        int32_t subgroupSum = subgroupScan(s_data[tid], laneId);
        s_data[tid] = subgroupSum;
    }
    GroupMemoryBarrierWithGroupSync();

    // Block-level scan of subgroup sums
    if (active && tid < (BLOCK_SIZE / SUBGROUP_SIZE)) {
        int32_t subgroupSumVal = s_data[tid * SUBGROUP_SIZE + SUBGROUP_SIZE - 1];
        int32_t blockScanResult = subgroupScan(subgroupSumVal, laneId);
        s_data[tid * SUBGROUP_SIZE + SUBGROUP_SIZE - 1] = blockScanResult;
    }
    GroupMemoryBarrierWithGroupSync();

    // Distribute block scan results back to subgroups
    if (active && subgroupId > 0 && laneId != SUBGROUP_SIZE - 1) {
        int32_t subgroupOffset = s_data[subgroupId * SUBGROUP_SIZE - 1];
        s_data[tid] = sat_add(s_data[tid], subgroupOffset);
    }
    GroupMemoryBarrierWithGroupSync();

    // Write results and block sum
    if (active && gid < numElements) {
        g_output[gid] = s_data[tid];
    }

    // Store block sum for next phase
    if (active && tid == BLOCK_SIZE - 1) {
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
    const uint32_t numElements = RESOLVE_NUM_ELEMENTS(uniforms);
    uint tid = localId.x;
    // See blockScan: gate work with `active` so all launched threads reach every barrier.
    const bool active = (tid < BLOCK_SIZE);
    uint blockId = groupId.x;
    uint gid = blockId * BLOCK_SIZE + tid;

    // Load block sums
    if (active)
        s_data[tid] = (gid < numElements) ? g_blockSums[gid] : 0;
    GroupMemoryBarrierWithGroupSync();

    // Perform scan similar to blockScan
    uint subgroupId = tid / SUBGROUP_SIZE;
    uint laneId = tid % SUBGROUP_SIZE;

    if (active) {
        int32_t subgroupSum = subgroupScan(s_data[tid], laneId);
        s_data[tid] = subgroupSum;
    }
    GroupMemoryBarrierWithGroupSync();

    if (active && tid < (BLOCK_SIZE / SUBGROUP_SIZE)) {
        int32_t subgroupSumVal = s_data[tid * SUBGROUP_SIZE + SUBGROUP_SIZE - 1];
        int32_t blockScanResult = subgroupScan(subgroupSumVal, laneId);
        s_data[tid * SUBGROUP_SIZE + SUBGROUP_SIZE - 1] = blockScanResult;
    }
    GroupMemoryBarrierWithGroupSync();

    if (active && subgroupId > 0 && laneId != SUBGROUP_SIZE-1) {
        int32_t subgroupOffset = s_data[subgroupId * SUBGROUP_SIZE - 1];
        s_data[tid] = sat_add(s_data[tid], subgroupOffset);
    }
    GroupMemoryBarrierWithGroupSync();

    // Write scanned block sums back
    if (active && gid < numElements) {
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
    const uint32_t numElements = RESOLVE_NUM_ELEMENTS(uniforms);
    uint tid = localId.x;
    if (tid >= BLOCK_SIZE)
        return;
    uint blockId = groupId.x;
    uint gid = blockId * BLOCK_SIZE + tid;

    if (gid < numElements && blockId > 0) {
        g_output[gid] = sat_add(g_output[gid], g_blockSums[blockId-1]);
    }
}


// Alternative single-pass version for smaller arrays (up to BLOCK_SIZE_0 elements)
[numthreads(BLOCK_SIZE_0, 1, 1)]
void singlePassPrefixSum(
    uint3 localId : SV_GroupThreadID,
    uint3 globalId : SV_DispatchThreadID,
    uniform Uniforms uniforms
) {
    const uint32_t numElements = RESOLVE_NUM_ELEMENTS(uniforms);
    uint tid = localId.x;
    uint gid = globalId.x;

    // Load data
    int32_t v_in = (gid < numElements) ? g_input[gid] : 0;
    s_data[tid] = v_in;
    GroupMemoryBarrierWithGroupSync();

    // Up-sweep (reduction) phase
    for (uint stride = 1; stride < BLOCK_SIZE_0; stride <<= 1) {
        if (tid % (stride * 2) == 0) {
            s_data[tid + stride * 2 - 1] = sat_add(
                s_data[tid + stride * 2 - 1], s_data[tid + stride - 1]);
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
            s_data[tid + stride * 2 - 1] = sat_add(s_data[tid + stride * 2 - 1], temp);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Write exclusive prefix sum result
    if (gid < numElements) {
        g_output[gid] = sat_add(s_data[tid], v_in);
    }
}
