#version 450

layout(triangles) in;
layout(triangle_strip, max_vertices = 6) out;

layout(location = 0) in vec3 inWorldPos[];
layout(location = 1) in vec3 inNormal[];
layout(location = 2) in vec2 inTexcoord[];
layout(location = 3) in vec4 inColor[];
layout(location = 4) in mat3 inTBN[];

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec2 outTexcoord;
layout(location = 3) out vec4 outColor;
layout(location = 4) out mat3 outTBN;

layout(set = 0, binding = 0) uniform LightUbo {
    vec4 camera_pos;
    vec4 light_dir;
    vec4 params;
    vec4 selection; // z = panorama projection
    mat4 light_vp;
    vec4 camera_forward;
    vec4 depth_view;
    mat4 view;
    mat4 projection;
} u_light;

void emitVertex(int index, float longitude_offset) {
    gl_Position = gl_in[index].gl_Position;
    gl_Position.x += longitude_offset * gl_Position.w;
    outWorldPos = inWorldPos[index];
    outNormal = inNormal[index];
    outTexcoord = inTexcoord[index];
    outColor = inColor[index];
    outTBN = inTBN[index];
    EmitVertex();
}

void emitTriangle(float minority_offset, float all_offset, bool unwrap) {
    for (int i = 0; i < 3; ++i) {
        float offset = all_offset;
        if (unwrap && ((gl_in[i].gl_Position.x / gl_in[i].gl_Position.w < 0.0) == (minority_offset > 0.0))) {
            offset += minority_offset;
        }
        emitVertex(i, offset);
    }
    EndPrimitive();
}

void main() {
    if (u_light.selection.z <= 0.5) {
        emitTriangle(0.0, 0.0, false);
        return;
    }

    float longitude[3];
    int positive_count = 0;
    float minimum_longitude = 1e30;
    float maximum_longitude = -1e30;
    for (int i = 0; i < 3; ++i) {
        longitude[i] = gl_in[i].gl_Position.x / gl_in[i].gl_Position.w;
        minimum_longitude = min(minimum_longitude, longitude[i]);
        maximum_longitude = max(maximum_longitude, longitude[i]);
        positive_count += longitude[i] >= 0.0 ? 1 : 0;
    }

    bool seam_crossing = maximum_longitude - minimum_longitude > 1.0;
    if (!seam_crossing) {
        emitTriangle(0.0, 0.0, false);
        return;
    }

    // Move the minority side beside the majority, then draw the unwrapped
    // triangle at both panorama edges. Viewport clipping keeps each half local.
    float minority_offset = positive_count <= 1 ? -2.0 : 2.0;
    emitTriangle(minority_offset, 0.0, true);
    emitTriangle(minority_offset, -minority_offset, true);
}
