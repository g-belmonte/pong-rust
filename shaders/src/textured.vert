#version 450

#extension GL_ARB_separate_shader_objects: enable

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

layout(push_constant) uniform PushConstants {
    mat4 model;
} pc;

layout (location = 0) in vec2 inPosition;
layout (location = 1) in vec2 inUV;

layout (location = 0) out vec2 fragUV;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    gl_Position = ubo.proj * ubo.view * pc.model * vec4(inPosition, 0.0, 1.0);
    fragUV = inUV;
}
