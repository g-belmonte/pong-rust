#version 450

#extension GL_ARB_separate_shader_objects: enable

layout(set = 0, binding = 1) uniform sampler2D texSampler;

layout (location = 0) in vec2 fragUV;

layout (location = 0) out vec4 outColor;

void main() {
    vec4 sampled = texture(texSampler, fragUV);
    // Alpha test: pipeline has no blend, so anything we don't discard becomes
    // fully opaque. Suits font-atlas glyphs (white RGB + alpha mask) and any
    // 1-bit-mask sprite. For real translucent textures, switch to alpha blending.
    if (sampled.a < 0.5) discard;
    outColor = sampled;
}
