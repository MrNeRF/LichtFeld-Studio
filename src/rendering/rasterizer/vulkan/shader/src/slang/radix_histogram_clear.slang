// Predicated replacement for the radix workspaces' vkCmdFillBuffer clear.
// One workgroup loops over both bounded buffers so conditional rendering can
// skip the transfer-free operation for inactive depth-wave slots.

layout(binding=0) RWStructuredBuffer<int32_t> global_histogram;
layout(binding=1) RWStructuredBuffer<int32_t> partition_histogram;

struct Uniforms {
    uint global_words;
    uint partition_words;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 groupThreadID : SV_GroupThreadID, uniform Uniforms uniforms) {
    for (uint i = groupThreadID.x; i < uniforms.global_words; i += 256u)
        global_histogram[i] = 0;
    for (uint i = groupThreadID.x; i < uniforms.partition_words; i += 256u)
        partition_histogram[i] = 0;
}
