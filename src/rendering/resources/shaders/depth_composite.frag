#version 430 core

in vec2 v_texcoord;

uniform sampler2D u_splat_color;
uniform sampler2D u_splat_depth;
uniform sampler2D u_mesh_color;
uniform sampler2D u_mesh_depth;

uniform float u_near_plane;
uniform float u_far_plane;
uniform bool u_flip_splat_y;
uniform vec2 u_splat_texcoord_scale;
uniform bool u_splat_depth_is_ndc;

layout(location = 0) out vec4 frag_color;

float view_depth_to_ndc(float z) {
    float A = (u_far_plane + u_near_plane) / (u_far_plane - u_near_plane);
    float B = (2.0 * u_far_plane * u_near_plane) / (u_far_plane - u_near_plane);
    float ndc = A - B / z;
    return ndc * 0.5 + 0.5;
}

float ndc_to_view_depth(float ndc_z) {
    float z_ndc = ndc_z * 2.0 - 1.0;
    float A = (u_far_plane + u_near_plane) / (u_far_plane - u_near_plane);
    float B = (2.0 * u_far_plane * u_near_plane) / (u_far_plane - u_near_plane);
    return B / (A - z_ndc);
}

void main() {
    vec2 splat_uv = v_texcoord * u_splat_texcoord_scale;
    if (u_flip_splat_y)
        splat_uv.y = u_splat_texcoord_scale.y - splat_uv.y;
    vec4 splat_color = texture(u_splat_color, splat_uv);
    float splat_depth = texture(u_splat_depth, splat_uv).r;
    if (u_splat_depth_is_ndc)
        splat_depth = ndc_to_view_depth(splat_depth);

    vec4 mesh_color = texture(u_mesh_color, v_texcoord);
    float mesh_ndc = texture(u_mesh_depth, v_texcoord).r;

    bool splat_has_depth = splat_depth >= u_near_plane && splat_depth < 1e9;
    bool mesh_has_depth = mesh_ndc < 1.0;

    if (!mesh_has_depth) {
        frag_color = splat_color;
        gl_FragDepth = splat_has_depth ? view_depth_to_ndc(splat_depth) : 1.0;
        return;
    }

    float mesh_view_depth = ndc_to_view_depth(mesh_ndc);

    if (!splat_has_depth) {
        frag_color = mesh_color;
        gl_FragDepth = mesh_ndc;
        return;
    }

    if (mesh_view_depth < splat_depth) {
        frag_color = mesh_color;
        gl_FragDepth = mesh_ndc;
    } else {
        frag_color = splat_color;
        gl_FragDepth = view_depth_to_ndc(splat_depth);
    }
}
