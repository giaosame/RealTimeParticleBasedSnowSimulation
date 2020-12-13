#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) in vec4 inPosition;
layout(location = 1) in vec4 inVelocity;
layout(location = 2) in vec4 inAttr1;
layout(location = 3) in vec4 inAttr2;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;

out gl_PerVertex
{
	vec4 gl_Position;
	float gl_PointSize;
};

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * inPosition;
    gl_PointSize = 3.0;
    //fragColor = normalize(inVelocity.xyz);
    fragColor = vec3(1.0, 1.0, 1.0);
    //fragTexCoord = inTexCoord;
}
