static const float FAR_DEPTH = 1.0e10f;

layout(binding=0) RWStructuredBuffer<float> pixel_depth;
layout(binding=1) StructuredBuffer<float> pixel_depth_weight;

struct Uniforms {
    uint num_pixels;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uniform Uniforms uniforms) {
    const uint pixel = dispatchThreadID.x;
    if (pixel >= uniforms.num_pixels)
        return;
    const float weight = pixel_depth_weight[pixel];
    pixel_depth[pixel] = weight > 1.0e-4f ? pixel_depth[pixel] / weight : FAR_DEPTH;
}
