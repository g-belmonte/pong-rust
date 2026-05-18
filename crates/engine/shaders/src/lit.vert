#version 450

#extension GL_ARB_separate_shader_objects: enable

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

// Per-vertex (binding 0): pos@0, normal@1, uv@2.
// Per-instance (binding 1): mat4 model at locations 3..6, vec3 color at 7.
layout (location = 0) in vec3 inPosition;
layout (location = 1) in vec3 inNormal;
layout (location = 2) in vec2 inUV;
layout (location = 3) in mat4 inModel;
layout (location = 7) in vec3 inColor;

layout (location = 0) out vec3 fragNormal;
layout (location = 1) out vec3 fragColor;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    gl_Position = ubo.proj * ubo.view * inModel * vec4(inPosition, 1.0);
    // Pass world-space normal. mat3(model) is the correct transform as long
    // as model has no non-uniform scale; the rotating-cube test has uniform
    // scale so this is fine. Lit materials needing arbitrary scale should
    // switch to transpose(inverse(mat3(model))).
    fragNormal = normalize(mat3(inModel) * inNormal);
    fragColor = inColor;
}
