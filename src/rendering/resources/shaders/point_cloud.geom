#version 430 core

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in vec3 v_color[];
in vec3 v_normal[];
in float v_ndc_x[];
in int v_equirectangular[];

out vec3 g_color;
out vec3 g_normal;

void main() {
    // In equirectangular mode, cull triangles that cross the ±PI seam
    if (v_equirectangular[0] == 1) {
        float ndc0 = v_ndc_x[0];
        float ndc1 = v_ndc_x[1];
        float ndc2 = v_ndc_x[2];

        // If any two vertices differ by > 1.0 in ndc_x, the triangle crosses the seam
        bool crosses_seam = abs(ndc0 - ndc1) > 1.0 ||
                            abs(ndc1 - ndc2) > 1.0 ||
                            abs(ndc2 - ndc0) > 1.0;

        if (crosses_seam)
            return;
    }

    // Emit the triangle
    for (int i = 0; i < 3; i++) {
        gl_Position = gl_in[i].gl_Position;
        g_color = v_color[i];
        g_normal = v_normal[i];
        EmitVertex();
    }
    EndPrimitive();
}
