package main

import "core:math/linalg"
import "vendor:cgltf"
import "vendor:box3d"

import "gfx"

Model :: struct {
	mesh:    gfx.GPU_Mesh,
	texture: gfx.GPU_Texture,
}

load_model :: proc(path: cstring, world_id: box3d.WorldId) -> (models: [dynamic]Model, ok: bool) {
	gltf_data, parse_res := cgltf.parse_file({}, path)
	if parse_res != .success {
		panic("Failed to parse GLTF file")
	}
	defer cgltf.free(gltf_data)

	buf_res := cgltf.load_buffers({}, gltf_data, path)
	if buf_res != .success {
		panic("Failed to load GLTF buffers")
	}

	image_textures := make([]gfx.GPU_Texture, len(gltf_data.images), context.temp_allocator)
	image_loaded   := make([]bool, len(gltf_data.images), context.temp_allocator)

	get_or_load_image_texture :: proc(img: ^cgltf.image, gltf_data: ^cgltf.data, image_textures: []gfx.GPU_Texture, image_loaded: []bool) -> gfx.GPU_Texture {
		if img == nil do return {}
		img_idx := int(cgltf.image_index(gltf_data, img))
		if img_idx < 0 || img_idx >= len(image_textures) do return {}

		if !image_loaded[img_idx] {
			image_loaded[img_idx] = true
			if img.buffer_view != nil {
				buf_ptr := cgltf.buffer_view_data(img.buffer_view)
				buf_size := img.buffer_view.size
				if buf_ptr != nil && buf_size > 0 {
					img_bytes := buf_ptr[:buf_size]
					tex_id, tex_ok := gfx.texture_create_from_memory(img_bytes, true)
					if tex_ok {
						image_textures[img_idx] = tex_id
					}
				}
			}
		}

		return image_textures[img_idx]
	}

	get_primitive_texture :: proc(prim: ^cgltf.primitive, gltf_data: ^cgltf.data, image_textures: []gfx.GPU_Texture, image_loaded: []bool) -> gfx.GPU_Texture {
		if prim.material == nil do return {}
		mat := prim.material

		img: ^cgltf.image
		if mat.has_pbr_metallic_roughness && mat.pbr_metallic_roughness.base_color_texture.texture != nil {
			tex := mat.pbr_metallic_roughness.base_color_texture.texture
			img = tex.has_basisu ? tex.basisu_image : tex.image_
		} else if mat.has_pbr_specular_glossiness && mat.pbr_specular_glossiness.diffuse_texture.texture != nil {
			tex := mat.pbr_specular_glossiness.diffuse_texture.texture
			img = tex.has_basisu ? tex.basisu_image : tex.image_
		}

		if img != nil {
			return get_or_load_image_texture(img, gltf_data, image_textures, image_loaded)
		}
		return {}
	}

	col_vertices := make([dynamic]box3d.Vec3, context.temp_allocator)
	col_indices  := make([dynamic]i32, context.temp_allocator)

	for &node in gltf_data.nodes {
		if node.mesh == nil do continue

		world_mat: gfx.Mat4
		cgltf.node_transform_world(&node, cast([^]f32)&world_mat)

		normal_mat3 := linalg.matrix3_from_matrix4(world_mat)

		for &prim in node.mesh.primitives {
			var_pos_acc: ^cgltf.accessor
			var_norm_acc: ^cgltf.accessor
			var_uv_acc: ^cgltf.accessor
			var_col_acc: ^cgltf.accessor

			for &attr in prim.attributes {
				switch attr.type {
				case .position:
					var_pos_acc = attr.data
				case .normal:
					var_norm_acc = attr.data
				case .texcoord:
					if attr.index == 0 do var_uv_acc = attr.data
				case .color:
					if attr.index == 0 do var_col_acc = attr.data
				case .invalid, .tangent, .joints, .weights, .custom:
				}
			}

			if var_pos_acc == nil do continue
			v_count := int(var_pos_acc.count)
			if v_count == 0 do continue

			default_color := gfx.Color{255, 255, 255, 255}
			if prim.material != nil && prim.material.has_pbr_metallic_roughness {
				bcf := prim.material.pbr_metallic_roughness.base_color_factor
				default_color = {
					u8(clamp(bcf[0] * 255.0, 0, 255)),
					u8(clamp(bcf[1] * 255.0, 0, 255)),
					u8(clamp(bcf[2] * 255.0, 0, 255)),
					u8(clamp(bcf[3] * 255.0, 0, 255)),
				}
			}

			verts := make([]gfx.Vertex, v_count, context.temp_allocator)

			// Positions
			pos_floats := make([]f32, v_count * 3, context.temp_allocator)
			_ = cgltf.accessor_unpack_floats(var_pos_acc, raw_data(pos_floats), uint(v_count * 3))
			for i in 0..<v_count {
				local_pos := gfx.Vec3{pos_floats[i*3+0], pos_floats[i*3+1], pos_floats[i*3+2]}
				p4 := world_mat * gfx.Vec4{local_pos.x, local_pos.y, local_pos.z, 1.0}
				verts[i].pos = p4.xyz / p4.w
				verts[i].color = default_color
				verts[i].normal = {0, 1, 0}
			}

			// Normals
			if var_norm_acc != nil {
				norm_floats := make([]f32, v_count * 3, context.temp_allocator)
				_ = cgltf.accessor_unpack_floats(var_norm_acc, raw_data(norm_floats), uint(v_count * 3))
				for i in 0..<v_count {
					local_norm := gfx.Vec3{norm_floats[i*3+0], norm_floats[i*3+1], norm_floats[i*3+2]}
					verts[i].normal = linalg.normalize(normal_mat3 * local_norm)
				}
			}

			// UVs
			if var_uv_acc != nil {
				uv_floats := make([]f32, v_count * 2, context.temp_allocator)
				_ = cgltf.accessor_unpack_floats(var_uv_acc, raw_data(uv_floats), uint(v_count * 2))
				for i in 0..<v_count {
					verts[i].uv = {uv_floats[i*2+0], uv_floats[i*2+1]}
				}
			}

			// Vertex Colors
			if var_col_acc != nil {
				if var_col_acc.type == .vec4 {
					col_floats := make([]f32, v_count * 4, context.temp_allocator)
					_ = cgltf.accessor_unpack_floats(var_col_acc, raw_data(col_floats), uint(v_count * 4))
					for i in 0..<v_count {
						verts[i].color = {
							u8(clamp(col_floats[i*4+0] * 255.0, 0, 255)),
							u8(clamp(col_floats[i*4+1] * 255.0, 0, 255)),
							u8(clamp(col_floats[i*4+2] * 255.0, 0, 255)),
							u8(clamp(col_floats[i*4+3] * 255.0, 0, 255)),
						}
					}
				} else if var_col_acc.type == .vec3 {
					col_floats := make([]f32, v_count * 3, context.temp_allocator)
					_ = cgltf.accessor_unpack_floats(var_col_acc, raw_data(col_floats), uint(v_count * 3))
					for i in 0..<v_count {
						verts[i].color = {
							u8(clamp(col_floats[i*3+0] * 255.0, 0, 255)),
							u8(clamp(col_floats[i*3+1] * 255.0, 0, 255)),
							u8(clamp(col_floats[i*3+2] * 255.0, 0, 255)),
							255,
						}
					}
				}
			}

			// Indices
			var_indices: []u32
			if prim.indices != nil {
				idx_count := int(prim.indices.count)
				var_indices = make([]u32, idx_count, context.temp_allocator)
				_ = cgltf.accessor_unpack_indices(prim.indices, raw_data(var_indices), size_of(u32), uint(idx_count))
			} else {
				var_indices = make([]u32, v_count, context.temp_allocator)
				for i in 0..<v_count do var_indices[i] = u32(i)
			}

			base_vertex_idx := i32(len(col_vertices))
			for v in verts {
				append(&col_vertices, box3d.Vec3{v.pos.x, v.pos.y, v.pos.z})
			}
			for idx in var_indices {
				append(&col_indices, base_vertex_idx + i32(idx))
			}

			gpu_mesh := gfx.mesh_create(verts, var_indices)
			tex := get_primitive_texture(&prim, gltf_data, image_textures, image_loaded)

			append(&models, Model{
				mesh = gpu_mesh,
				texture = tex,
			})
		}
	}

	if len(col_vertices) >= 3 && len(col_indices) >= 3 {
		body_def := box3d.DefaultBodyDef()
		body_def.type = .staticBody
		body_def.position = {0, 0, 0}
		static_body := box3d.CreateBody(world_id, body_def)

		mesh_def := box3d.MeshDef{
			vertices       = raw_data(col_vertices),
			indices        = raw_data(col_indices),
			vertexCount    = i32(len(col_vertices)),
			triangleCount  = i32(len(col_indices) / 3),
			useMedianSplit = true,
			identifyEdges  = true,
		}

		mesh_data := box3d.CreateMesh(mesh_def, nil, 0)
		shape_def := box3d.DefaultShapeDef()
		shape_def.baseMaterial.friction = 0.7
		shape_def.baseMaterial.restitution = 0.1
		_ = box3d.CreateMeshShape(static_body, shape_def, mesh_data, {1, 1, 1})
	}

	return models, true
}

create_box_mesh :: proc() -> gfx.GPU_Mesh {
	verts := [24]gfx.Vertex{
		{ pos = {-0.5, -0.5,  0.5}, normal = { 0,  0,  1}, uv = {0, 1}, color = {255, 255, 255, 255} },
		{ pos = { 0.5, -0.5,  0.5}, normal = { 0,  0,  1}, uv = {1, 1}, color = {255, 255, 255, 255} },
		{ pos = { 0.5,  0.5,  0.5}, normal = { 0,  0,  1}, uv = {1, 0}, color = {255, 255, 255, 255} },
		{ pos = {-0.5,  0.5,  0.5}, normal = { 0,  0,  1}, uv = {0, 0}, color = {255, 255, 255, 255} },

		{ pos = { 0.5, -0.5, -0.5}, normal = { 0,  0, -1}, uv = {0, 1}, color = {255, 255, 255, 255} },
		{ pos = {-0.5, -0.5, -0.5}, normal = { 0,  0, -1}, uv = {1, 1}, color = {255, 255, 255, 255} },
		{ pos = {-0.5,  0.5, -0.5}, normal = { 0,  0, -1}, uv = {1, 0}, color = {255, 255, 255, 255} },
		{ pos = { 0.5,  0.5, -0.5}, normal = { 0,  0, -1}, uv = {0, 0}, color = {255, 255, 255, 255} },

		{ pos = {-0.5,  0.5,  0.5}, normal = { 0,  1,  0}, uv = {0, 1}, color = {255, 255, 255, 255} },
		{ pos = { 0.5,  0.5,  0.5}, normal = { 0,  1,  0}, uv = {1, 1}, color = {255, 255, 255, 255} },
		{ pos = { 0.5,  0.5, -0.5}, normal = { 0,  1,  0}, uv = {1, 0}, color = {255, 255, 255, 255} },
		{ pos = {-0.5,  0.5, -0.5}, normal = { 0,  1,  0}, uv = {0, 0}, color = {255, 255, 255, 255} },

		{ pos = {-0.5, -0.5, -0.5}, normal = { 0, -1,  0}, uv = {0, 1}, color = {255, 255, 255, 255} },
		{ pos = { 0.5, -0.5, -0.5}, normal = { 0, -1,  0}, uv = {1, 1}, color = {255, 255, 255, 255} },
		{ pos = { 0.5, -0.5,  0.5}, normal = { 0, -1,  0}, uv = {1, 0}, color = {255, 255, 255, 255} },
		{ pos = {-0.5, -0.5,  0.5}, normal = { 0, -1,  0}, uv = {0, 0}, color = {255, 255, 255, 255} },

		{ pos = { 0.5, -0.5,  0.5}, normal = { 1,  0,  0}, uv = {0, 1}, color = {255, 255, 255, 255} },
		{ pos = { 0.5, -0.5, -0.5}, normal = { 1,  0,  0}, uv = {1, 1}, color = {255, 255, 255, 255} },
		{ pos = { 0.5,  0.5, -0.5}, normal = { 1,  0,  0}, uv = {1, 0}, color = {255, 255, 255, 255} },
		{ pos = { 0.5,  0.5,  0.5}, normal = { 1,  0,  0}, uv = {0, 0}, color = {255, 255, 255, 255} },

		{ pos = {-0.5, -0.5, -0.5}, normal = {-1,  0,  0}, uv = {0, 1}, color = {255, 255, 255, 255} },
		{ pos = {-0.5, -0.5,  0.5}, normal = {-1,  0,  0}, uv = {1, 1}, color = {255, 255, 255, 255} },
		{ pos = {-0.5,  0.5,  0.5}, normal = {-1,  0,  0}, uv = {1, 0}, color = {255, 255, 255, 255} },
		{ pos = {-0.5,  0.5, -0.5}, normal = {-1,  0,  0}, uv = {0, 0}, color = {255, 255, 255, 255} },
	}

	indices := [36]u32{
		 0,  1,  2,  0,  2,  3,
		 4,  5,  6,  4,  6,  7,
		 8,  9, 10,  8, 10, 11,
		12, 13, 14, 12, 14, 15,
		16, 17, 18, 16, 18, 19,
		20, 21, 22, 20, 22, 23,
	}

	return gfx.mesh_create(verts[:], indices[:])
}