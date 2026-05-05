#ifndef BLOCK_SIZE
static const uint BLOCK_SIZE = 1024;
#endif

layout(binding=0) StructuredBuffer<int32_t> g_input;  // (N,)
layout(binding=1) RWStructuredBuffer<int32_t> g_output;  // (1,)

groupshared int32_t s_data[BLOCK_SIZE];


[numthreads(BLOCK_SIZE, 1, 1)]
void main(
    uint3 groupID : SV_GroupID,
    uint3 groupThreadID : SV_GroupThreadID,
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uint groupIndex : SV_GroupIndex,
    uniform uint32_t arraySize
) {
    uint tid = groupThreadID.x;
    uint gid = dispatchThreadID.x;
    
    // Load data into shared memory
    // Each thread loads one element (with bounds checking)
    if (gid < arraySize) {
        s_data[tid] = g_input[gid];
    } else {
        s_data[tid] = 0;  // Pad with zeros for out-of-bounds
    }
    
    GroupMemoryBarrierWithGroupSync();
    
    // Perform parallel reduction in shared memory
    for (uint stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    
    // Thread 0 writes the result for this block
    if (tid == 0) {
        // If this is a single block reduction (array fits in one block)
        if (arraySize <= BLOCK_SIZE) {
            g_output[0] = s_data[0];
        } else {
            // For multi-block reduction, accumulate results atomically
            InterlockedAdd(g_output[0], s_data[0]);
        }
    }
}
