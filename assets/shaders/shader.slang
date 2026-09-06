struct Gpu_Instance {
    column_major float4x4 model_mat;
    uint texture_id;
    uint type;
};

struct PushConstants {
    column_major float4x4 view_proj;
    Gpu_Instance* instances;
};

[vk::push_constant] PushConstants u_push;

[vk::binding(0, 0)] Sampler2D g_textures[];
[vk::binding(1, 0)] RaytracingAccelerationStructure u_tlas;

struct VertexInput {
    float3 pos : POSITION;
    float3 normal : NORMAL;
    float2 uv : TEXCOORD;
    float4 color : COLOR;
};

struct VertexOutput {
    float4 pos : SV_Position;
    float4 color;
    float3 normal;
    float2 uv;
    float3 world_pos;
    nointerpolation uint texture_id;
    nointerpolation uint type;
};

[shader("vertex")]
VertexOutput vs_main(VertexInput vin, uint instance_id : SV_VulkanInstanceID) {
    Gpu_Instance inst = u_push.instances[instance_id];

    VertexOutput vout;
    vout.type = inst.type;
    vout.texture_id = inst.texture_id;
    vout.color = vin.color;
    vout.uv = vin.uv;

    if (inst.type == 1) {
        vout.world_pos = mul(inst.model_mat, float4(vin.pos.xy, 0.0, 1.0)).xyz;
        vout.pos = float4(vin.pos.xy, 1.0, 1.0);
        vout.normal = float3(0, 0, 0);
    } else {
        float4 world_pos = mul(inst.model_mat, float4(vin.pos, 1.0));
        vout.world_pos = world_pos.xyz;
        vout.normal = mul((float3x3)inst.model_mat, vin.normal);
        vout.pos = mul(u_push.view_proj, world_pos);
    }

    return vout;
}

static const float PI = 3.14159265358979323846;

float3 aces_film(float3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

[earlydepthstencil]
[shader("fragment")]
float4 fs_main(VertexOutput pin) : SV_Target {
    if (pin.type == 1) {
        float3 dir = normalize(pin.world_pos);
        float2 uv = float2(atan2(-dir.x, dir.z) / (2.0 * PI) + 0.5, clamp(acos(clamp(dir.y, -1.0, 1.0)) / PI, 0.0001, 0.9999));
        return g_textures[NonUniformResourceIndex(pin.texture_id)].SampleLevel(uv, 0.0);
    }

    float3 N = normalize(pin.normal);
    float3 sun_color = float3(1.4, 1.3, 1.1);
    float3 sun_dir = normalize(float3(-0.5167, 0.4850, 0.7055));
    float sun_diffuse = max(dot(N, sun_dir), 0.0);

    if (sun_diffuse > 0.0) {
        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER> ray_query;
        RayDesc ray;
        ray.Origin = pin.world_pos + N * 0.08;
        ray.Direction = sun_dir;
        ray.TMin = 0.01;
        ray.TMax = 10000.0;
        ray_query.TraceRayInline(u_tlas, RAY_FLAG_NONE, 0xFF, ray);
        while (ray_query.Proceed()) {}
        if (ray_query.CommittedStatus() != COMMITTED_NOTHING) {
            sun_diffuse = 0.0;
        }
    }

    float4 tex = g_textures[NonUniformResourceIndex(pin.texture_id)].Sample(pin.uv);
    float3 albedo = pin.color.rgb * tex.rgb;

    float3 sky_color = float3(0.28, 0.38, 0.55);
    float3 ground_color = float3(0.14, 0.11, 0.09);
    float3 ambient = lerp(ground_color, sky_color, N.y * 0.5 + 0.5);

    float3 lit_color = albedo * (sun_diffuse * sun_color + ambient);
    float3 mapped = aces_film(lit_color);

    return float4(mapped, pin.color.a * tex.a);
}
