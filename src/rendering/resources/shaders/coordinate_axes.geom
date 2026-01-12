#version 330 core

layout(lines) in;
layout(line_strip, max_vertices = 2) out;

in vec3 v_color[];
in float v_ndc_x[];
in int v_equi[];

out vec3 g_color;

void main() {
    // In equirectangular mode, cull lines that cross the ±PI seam
    if (v_equi[0] == 1) {
        float ndc0 = v_ndc_x[0];
        float ndc1 = v_ndc_x[1];

        if (abs(ndc0 - ndc1) > 1.0)
            return;
    }

    // Emit the line
    for (int i = 0; i < 2; i++) {
        gl_Position = gl_in[i].gl_Position;
        g_color = v_color[i];
        EmitVertex();
    }
    EndPrimitive();
}
