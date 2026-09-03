#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

struct Gpu_Instance {
    mat4 model_mat;
    uint color;
    uint texture_id;
    uint type;
};

layout(buffer_reference, scalar) readonly buffer InstanceBuffer {
    Gpu_Instance instances[];
};

layout(push_constant, scalar) uniform PushConstants {
    mat4 view_proj;
    InstanceBuffer instances;
} u_push;

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;
layout(location = 3) in vec4 in_color;

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec3 out_normal;
layout(location = 2) out vec2 out_uv;
layout(location = 3) out vec3 out_world_pos;
layout(location = 4) flat out uint out_texture_id;
layout(location = 5) flat out uint out_type;

void main() {
    Gpu_Instance inst = u_push.instances.instances[gl_InstanceIndex];
    out_type = inst.type;
    out_texture_id = inst.texture_id;
    out_color = in_color * unpackUnorm4x8(inst.color);
    out_uv = in_uv;

    if (inst.type == 1) {
        out_world_pos = (inst.model_mat * vec4(in_pos.xy, 0.0, 1.0)).xyz;
        gl_Position = vec4(in_pos.xy, 1.0, 1.0);
    } else {
        vec4 world_pos = inst.model_mat * vec4(in_pos, 1.0);
        out_world_pos = world_pos.xyz;
        out_normal = mat3(inst.model_mat) * in_normal;
        gl_Position = u_push.view_proj * world_pos;
    }
}

