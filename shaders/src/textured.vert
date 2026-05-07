#version 450

#extension GL_ARB_separate_shader_objects: enable

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

// Per-vertex (binding 0) is the unit quad in [-0.5..0.5]^2.
// Per-instance (binding 1): mat4 model at locations 1..4, then uv_offset/uv_scale.
// uv = uv_offset + (pos + 0.5) * uv_scale  →  pos at -0.5 maps to uv_offset,
// pos at +0.5 maps to uv_offset + uv_scale.
layout (location = 0) in vec2 inPosition;
layout (location = 1) in mat4 inModel;
layout (location = 5) in vec2 inUVOffset;
layout (location = 6) in vec2 inUVScale;

layout (location = 0) out vec2 fragUV;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    gl_Position = ubo.proj * ubo.view * inModel * vec4(inPosition, 0.0, 1.0);
    fragUV = inUVOffset + (inPosition + vec2(0.5)) * inUVScale;
}
