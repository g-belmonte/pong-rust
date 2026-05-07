#version 450

#extension GL_ARB_separate_shader_objects: enable

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

// Per-vertex (binding 0) and per-instance (binding 1). The mat4 model matrix
// at location 1 consumes locations 1..4, so the colour sits at location 5.
layout (location = 0) in vec2 inPosition;
layout (location = 1) in mat4 inModel;
layout (location = 5) in vec3 inColor;

layout (location = 0) out vec3 fragColor;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    gl_Position = ubo.proj * ubo.view * inModel * vec4(inPosition, 0.0, 1.0);
    fragColor = inColor;
}
