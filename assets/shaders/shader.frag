#version 460
#extension GL_EXT_ray_query : require
#extension GL_EXT_nonuniform_qualifier : enable

layout(early_fragment_tests) in;

layout(location = 0) in vec4 in_color;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;
layout(location = 3) in vec3 in_world_pos;
layout(location = 4) flat in uint in_texture_id;
layout(location = 5) flat in uint in_type;

layout(set = 0, binding = 0) uniform sampler2D g_textures[];
layout(set = 0, binding = 1) uniform accelerationStructureEXT u_tlas;

layout(location = 0) out vec4 out_color;

const float PI = 3.14159265358979323846;

vec3 aces_film(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    if (in_type == 1) {
        vec3 dir = normalize(in_world_pos);
        vec2 uv = vec2(atan(-dir.x, dir.z) / (2.0 * PI) + 0.5, clamp(acos(clamp(dir.y, -1.0, 1.0)) / PI, 0.0001, 0.9999));
        out_color = textureLod(g_textures[nonuniformEXT(in_texture_id)], uv, 0.0);
        return;
    }

    vec3 N = normalize(in_normal);
    vec3 sun_color = vec3(1.4, 1.3, 1.1);
    vec3 sun_dir = normalize(vec3(-0.5167, 0.4850, 0.7055));
    float sun_diffuse = max(dot(N, sun_dir), 0.0);

    if (sun_diffuse > 0.0) {
        rayQueryEXT ray_query;
        vec3 origin = in_world_pos + N * 0.08;
        rayQueryInitializeEXT(
            ray_query,
            u_tlas,
            gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT,
            0xFF,
            origin,
            0.01,
            sun_dir,
            10000.0
        );

        while (rayQueryProceedEXT(ray_query)) {}

        if (rayQueryGetIntersectionTypeEXT(ray_query, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
            sun_diffuse = 0.0;
        }
    }

    vec4 tex = texture(g_textures[nonuniformEXT(in_texture_id)], in_uv);
    vec3 albedo = in_color.rgb * tex.rgb;

    vec3 sky_color = vec3(0.28, 0.38, 0.55);
    vec3 ground_color = vec3(0.14, 0.11, 0.09);
    vec3 ambient = mix(ground_color, sky_color, N.y * 0.5 + 0.5);

    vec3 lit_color = albedo * (sun_diffuse * sun_color + ambient);
    vec3 mapped = aces_film(lit_color);

    out_color = vec4(mapped, in_color.a * tex.a);
}

