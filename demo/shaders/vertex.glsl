#version 300 es

layout(location = 0) in vec3 vertex_pos_ws;
layout(location = 1) in vec3 wibble;

uniform mat4 view;
uniform mat4 projection;
uniform float horiz_offset;

out lowp float distance;
out lowp float reflectance;

out lowp float argh;

void main()
{
    argh = 0.0;
    vec4 vertex_pos_vs_h = view * vec4(vertex_pos_ws, 1.0);
    vec4 vertex_norm_vs_h = view * vec4(vertex_pos_ws, 0.0);

    vec3 light_pos_vs = vec3(0.0);
    vec3 vertex_norm_vs = vertex_norm_vs_h.xyz;
    vec3 vertex_to_light_vs = light_pos_vs - vertex_pos_vs_h.xyz;


    reflectance = clamp(dot(normalize(vertex_to_light_vs), normalize(vertex_norm_vs)), -1, 1);
//    if (length(vertex_pos_vs_h.xyz) > 2.0) argh = 1.0;

    distance = -vertex_pos_vs_h.z;

    vertex_pos_vs_h.x += horiz_offset;
    gl_Position = projection * vertex_pos_vs_h;
 }
