#version 430 core

in vec2 v_texcoord;

uniform sampler2D u_splat_color;
uniform sampler2D u_splat_depth;
uniform sampler2D u_mesh_color;
uniform sampler2D u_mesh_depth;

uniform float u_near_plane;
uniform float u_far_plane;
uniform bool u_flip_splat_y;

layout(location = 0) out vec4 frag_color;

float view_depth_to_ndc(float z) {
    if (z > 1e9) return 1.0;
    float A = (u_far_plane + u_near_plane) / (u_far_plane - u_near_plane);
    float B = (2.0 * u_far_plane * u_near_plane) / (u_far_plane - u_near_plane);
    float ndc = A - B / z;
    return ndc * 0.5 + 0.5;
}

void main() {
    vec2 splat_uv = u_flip_splat_y ? vec2(v_texcoord.x, 1.0 - v_texcoord.y) : v_texcoord;
    vec4 splat_color = texture(u_splat_color, splat_uv);
    float splat_depth = texture(u_splat_depth, splat_uv).r;

    vec4 mesh_color = texture(u_mesh_color, v_texcoord);
    float mesh_depth = texture(u_mesh_depth, v_texcoord).r;

    if (mesh_color.a < 0.001) {
        frag_color = splat_color;
        gl_FragDepth = view_depth_to_ndc(splat_depth);
        return;
    }

    if (splat_color.a < 0.001 || splat_depth > 1e9) {
        frag_color = mesh_color;
        gl_FragDepth = view_depth_to_ndc(mesh_depth);
        return;
    }

    if (mesh_depth < splat_depth) {
        frag_color = vec4(mesh_color.rgb * mesh_color.a + splat_color.rgb * (1.0 - mesh_color.a),
                          max(mesh_color.a, splat_color.a));
        gl_FragDepth = view_depth_to_ndc(mesh_depth);
    } else {
        frag_color = vec4(splat_color.rgb * splat_color.a + mesh_color.rgb * (1.0 - splat_color.a),
                          max(splat_color.a, mesh_color.a));
        gl_FragDepth = view_depth_to_ndc(splat_depth);
    }
}
