#version 450

#extension GL_ARB_separate_shader_objects: enable

layout (location = 0) in vec3 fragNormal;
layout (location = 1) in vec3 fragColor;

layout (location = 0) out vec4 outColor;

// Hardcoded directional light + ambient term. Suits the rotating-cube test;
// games that want more (multiple lights, configurable direction, materials
// with specular) should register their own shader rather than grow this one.
const vec3 LIGHT_DIR = normalize(vec3(0.4, -1.0, 0.6));
const float AMBIENT = 0.25;

void main() {
    float diffuse = max(dot(fragNormal, -LIGHT_DIR), 0.0);
    float lighting = AMBIENT + (1.0 - AMBIENT) * diffuse;
    outColor = vec4(fragColor * lighting, 1.0);
}
