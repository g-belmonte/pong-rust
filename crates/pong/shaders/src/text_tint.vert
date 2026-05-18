#version 450

#extension GL_ARB_separate_shader_objects: enable

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

// Same vertex/instance layout as textured.vert with one extra instance
// attribute: a vec3 tint colour at location 7. Used by scene_menu's tinted
// glyph instances to render selected (white) vs. unselected (grey) menu options.
layout (location = 0) in vec2 inPosition;
layout (location = 1) in mat4 inModel;
layout (location = 5) in vec2 inUVOffset;
layout (location = 6) in vec2 inUVScale;
layout (location = 7) in vec3 inColor;

layout (location = 0) out vec2 fragUV;
layout (location = 1) out vec3 fragColor;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    gl_Position = ubo.proj * ubo.view * inModel * vec4(inPosition, 0.0, 1.0);
    fragUV = inUVOffset + (inPosition + vec2(0.5)) * inUVScale;
    fragColor = inColor;
}
