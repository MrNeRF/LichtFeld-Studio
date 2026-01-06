#version 430 core

in vec3 v_color;
in vec3 v_normal;
in vec3 v_frag_pos;

out vec4 frag_color;

void main() {
    // Simple diffuse lighting with stronger ambient
    vec3 light_dir = normalize(vec3(0.5, 0.5, 1.0));
    float diff = max(dot(v_normal, light_dir), 0.0);
    
    // Stronger ambient light to preserve colors
    vec3 ambient = 0.6 * v_color;
    vec3 diffuse = 0.4 * diff * v_color;
    
    vec3 result = ambient + diffuse;
    
    // Ensure colors don't get too dark
    result = max(result, v_color * 0.5);
    
    frag_color = vec4(result, 1.0);
}
