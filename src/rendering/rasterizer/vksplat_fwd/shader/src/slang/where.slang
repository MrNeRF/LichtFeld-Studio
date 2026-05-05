#define BLOCK_SIZE 256

layout(binding=0) StructuredBuffer<int32_t> mask;  // (N,)
layout(binding=1) StructuredBuffer<int32_t> mask_cumsum;  // (N,)
layout(binding=2) RWStructuredBuffer<int32_t> out_indices;  // (M,)


[numthreads(BLOCK_SIZE, 1, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uniform uint numElements
) {
    uint gid = dispatchThreadID.x;
    if (gid >= numElements)
        return;
    
    if (mask[gid] > 0) {
        int32_t idx = gid == 0 ? 0 : mask_cumsum[gid-1];
        out_indices[idx] = gid;
    }
}
