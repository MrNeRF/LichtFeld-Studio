// SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later
// One source for every backend. Resources are tensor bindings, patched into
// the parameter block by core; callers never manufacture device addresses.
struct Parameters {
    Ptr<float, Access.ReadWrite> first;
    Ptr<float, Access.ReadWrite> second;
    uint count;
    float scale;
    float bias;
    uint padding;
};

[[vk::push_constant]] ConstantBuffer<Parameters> p;

[shader("compute")]
[numthreads(64, 1, 1)]
void transform(uint3 tid : SV_DispatchThreadID) {
    if (tid.x < p.count)
        p.second[tid.x] = p.first[tid.x] * p.scale + p.bias;
}

struct Vertex {
    float4 position : SV_Position;
    float4 color : COLOR0;
};

[shader("vertex")]
Vertex vertexMain(uint index : SV_VertexID) {
    Vertex v;
    v.position = ((Ptr<float4>)p.first)[index];
    v.color = ((Ptr<float4>)p.second)[index];
    return v;
}

[shader("fragment")]
float4 fragmentMain(Vertex v) : SV_Target0 {
    return v.color;
}
