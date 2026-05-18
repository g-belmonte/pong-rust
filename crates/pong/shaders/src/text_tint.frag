#version 450

#extension GL_ARB_separate_shader_objects: enable

layout(set = 0, binding = 1) uniform sampler2D texSampler;

layout (location = 0) in vec2 fragUV;
layout (location = 1) in vec3 fragColor;

layout (location = 0) out vec4 outColor;

// Same alpha-discard as the engine's textured.frag, with the rgb modulated
// by the per-instance tint. The atlas is white-RGB + alpha-mask, so
// `sampled.rgb * fragColor` reduces to `fragColor` for lit pixels.
void main() {
    vec4 sampled = texture(texSampler, fragUV);
    if (sampled.a < 0.5) discard;
    outColor = vec4(sampled.rgb * fragColor, 1.0);
}
