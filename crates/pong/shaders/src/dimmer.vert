#version 450

#extension GL_ARB_separate_shader_objects: enable

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

// Per-vertex 2D position (binding 0) and per-instance model matrix +
// straight RGBA tint (binding 1). The alpha channel of `inColor` is what
// the engine's `BlendMode::Alpha` blends against the framebuffer.
layout (location = 0) in vec2 inPosition;
layout (location = 1) in mat4 inModel;
layout (location = 5) in vec4 inColor;

layout (location = 0) out vec4 fragColor;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    gl_Position = ubo.proj * ubo.view * inModel * vec4(inPosition, 0.0, 1.0);
    fragColor = inColor;
}
