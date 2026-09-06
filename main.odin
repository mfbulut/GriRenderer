package main

import "base:runtime"
import "core:math"
import "core:math/linalg"
import "vendor:box3d"
import "gfx"

Player :: struct {
	pos:         box3d.Pos,
	vel:         box3d.Vec3,
	yaw:         f32,
	pitch:       f32,
	is_grounded: bool,
	mover:       box3d.Capsule,
}

plane_callback :: proc "c" (shapeId: box3d.ShapeId, planes: [^]box3d.PlaneResult, planeCount: i32, ctx: rawptr) -> bool {
	context = runtime.default_context()
	collision_planes := cast(^[dynamic]box3d.CollisionPlane)ctx
	for idx in 0..<int(planeCount) {
		append(collision_planes, box3d.CollisionPlane{
			plane        = planes[idx].plane,
			pushLimit    = 1e10,
			clipVelocity = true,
		})
	}
	return true
}

main :: proc() {
	gfx.init("Playground", {1280, 720})
	gfx.mouse_set_locked(true)

	world_def := box3d.DefaultWorldDef()
	world_def.gravity = {0, -16.0, 0}
	world := box3d.CreateWorld(world_def)

	gfx.upload_begin()
	box_mesh := create_box_mesh()
	box_tex := gfx.texture_create_from_file("assets/textures/box.png") or_else panic("Failed to load skybox")
	models := load_model("assets/models/nuke.glb", world) or_else panic("Failed to load model")
	sky_tex := gfx.texture_create_from_file("assets/textures/sky.jpg") or_else panic("Failed to load skybox")
	gfx.skybox_set(sky_tex)
	gfx.upload_end()

	player := Player{
		pos = {-40, 0, -20},
		mover = {
			center1 = {0, 0.35, 0},
			center2 = {0, 1.45, 0},
			radius  = 0.35,
		},
	}

	collision_planes := make([dynamic]box3d.CollisionPlane, 0, 128)
	boxes := make([dynamic]box3d.BodyId, 0, 64)

	held_body: box3d.BodyId
	hold_dist: f32 = 3.5
	rel_rot:   box3d.Quat

	for gfx.update() {
		free_all(context.temp_allocator)
		dt := gfx.frame_time()

		if gfx.key_is_pressed(.Esc) do gfx.mouse_set_locked(false)
		if gfx.key_is_pressed(.Mouse_Left) do gfx.mouse_set_locked(true)

		if gfx.window.mouse_locked {
			m_delta := gfx.mouse_delta()
			player.yaw += m_delta.x * 0.12
			player.pitch -= m_delta.y * 0.12
			player.pitch = clamp(player.pitch, -89.0, 89.0)
		}

		pitch_rad := math.to_radians_f32(player.pitch)
		yaw_rad   := math.to_radians_f32(player.yaw)

		cos_pitch := math.cos(pitch_rad)
		sin_pitch := math.sin(pitch_rad)
		cos_yaw   := math.cos(yaw_rad)
		sin_yaw   := math.sin(yaw_rad)

		cam_dir := linalg.normalize(gfx.Vec3{
			cos_pitch * sin_yaw,
			sin_pitch,
			-cos_pitch * cos_yaw,
		})
		cam_eye := gfx.Vec3{player.pos.x, player.pos.y + 1.6, player.pos.z}

		q_yaw   := linalg.quaternion_angle_axis_f32(-yaw_rad, gfx.Vec3{0, 1, 0})
		q_pitch := linalg.quaternion_angle_axis_f32(pitch_rad, gfx.Vec3{1, 0, 0})
		q_cam   := q_yaw * q_pitch

		move_forward := linalg.normalize(gfx.Vec3{sin_yaw, 0, -cos_yaw})
		move_right   := linalg.normalize(gfx.Vec3{cos_yaw, 0, sin_yaw})

		input_dir := gfx.Vec3{0, 0, 0}
		if gfx.key_is_down(.W) do input_dir += move_forward
		if gfx.key_is_down(.S) do input_dir -= move_forward
		if gfx.key_is_down(.D) do input_dir += move_right
		if gfx.key_is_down(.A) do input_dir -= move_right

		target_hvel := gfx.Vec3{0, 0, 0}
		if linalg.length(input_dir) > 0.001 {
			wish_dir := linalg.normalize(input_dir)
			speed: f32 = 9.0 if gfx.key_is_down(.Shift) else 5.0
			target_hvel = wish_dir * speed
		}

		current_hvel := gfx.Vec3{player.vel.x, 0, player.vel.z}
		accel_rate: f32 = 20.0 if player.is_grounded else 4.0
		new_hvel := math.lerp(current_hvel, target_hvel, clamp(accel_rate * dt, 0.0, 1.0))
		player.vel.x = new_hvel.x
		player.vel.z = new_hvel.z

		if player.is_grounded {
			if gfx.key_is_down(.Space) {
				player.vel.y = 5.5
				player.is_grounded = false
			}
		} else {
			player.vel.y -= 16.0 * dt
			player.vel.y = max(player.vel.y, -50.0)
		}

		target_delta := player.vel * dt

		clear(&collision_planes)
		box3d.World_CollideMover(world, player.pos, player.mover, box3d.DefaultQueryFilter(), plane_callback, &collision_planes)

		if len(collision_planes) > 0 {
			res := box3d.SolvePlanes(target_delta, raw_data(collision_planes[:]), i32(len(collision_planes)))
			player.pos = box3d.OffsetPos(player.pos, res.delta)

			player.is_grounded = false
			for cp in collision_planes {
				if cp.plane.normal.y > 0.4 {
					player.is_grounded = true
				}
			}

			player.vel = box3d.ClipVector(player.vel, raw_data(collision_planes[:]), i32(len(collision_planes)))
		} else {
			player.pos = box3d.OffsetPos(player.pos, target_delta)
			player.is_grounded = false
		}

		gfx.camera_set(cam_eye, cam_eye + cam_dir)

		if gfx.key_is_down(.F) {
			b_def := box3d.DefaultBodyDef()
			b_def.type = .dynamicBody
			b_def.position = cam_eye + cam_dir * 3.0
			b_def.rotation = 1
			body := box3d.CreateBody(world, b_def)

			box_hull := box3d.MakeBoxHull(0.5, 0.5, 0.5)
			s_def := box3d.DefaultShapeDef()
			s_def.baseMaterial.friction = 0.6
			s_def.baseMaterial.restitution = 0.2
			_ = box3d.CreateHullShape(body, s_def, &box_hull.base)

			append(&boxes, body)
			held_body = body
		}

		if gfx.window.mouse_locked {
			if gfx.key_is_pressed(.Mouse_Left) {
				ray := box3d.World_CastRayClosest(world, cam_eye, cam_dir * 100.0, box3d.DefaultQueryFilter())
				if box3d.IS_NON_NULL(ray.shapeId) {
					hit_body := box3d.Shape_GetBody(ray.shapeId)
					if box3d.Body_IsValid(hit_body) && box3d.Body_GetType(hit_body) == .dynamicBody {
						held_body = hit_body
						hit_pos := gfx.Vec3(ray.point)
						hold_dist = clamp(linalg.length(hit_pos - cam_eye), 2.0, 25.0)
					}
				}

				if box3d.Body_IsValid(held_body) {
					rel_rot = conj(q_cam) * box3d.Body_GetRotation(held_body)
				}
			}

			hold_dist = clamp(hold_dist + gfx.mouse_scroll().y * 0.5, 1.5, 25.0)

			if gfx.key_is_down(.Mouse_Left) && box3d.Body_IsValid(held_body) {
				target_pos := cam_eye + cam_dir * hold_dist
				target_rot := q_cam * rel_rot

				cur_pos := gfx.Vec3(box3d.Body_GetPosition(held_body))
				cur_rot := box3d.Body_GetRotation(held_body)

				to_target := target_pos - cur_pos
				dist := linalg.length(to_target)
				speed := clamp(dist * 14.0, 0.0, 35.0)
				lin_vel := (to_target / dist) * speed if dist > 0.001 else gfx.Vec3{0, 0, 0}
				box3d.Body_SetLinearVelocity(held_body, lin_vel)

				delta_q := target_rot * conj(cur_rot)
				if delta_q.w < 0 do delta_q = -delta_q
				v := gfx.Vec3{delta_q.x, delta_q.y, delta_q.z}
				v_len := linalg.length(v)
				ang_vel := gfx.Vec3{0, 0, 0}
				if v_len > 1e-4 {
					angle := 2.0 * math.atan2(v_len, delta_q.w)
					axis := v / v_len
					ang_vel = axis * clamp(angle * 25.0, 0.0, 35.0)
				}
				box3d.Body_SetAngularVelocity(held_body, ang_vel)

				box3d.Body_SetAwake(held_body, true)
			}
		}

		box3d.World_Step(world, dt, 4)

		for m in models {
			gfx.mesh_draw(m.mesh, texture = m.texture)
		}

		for b in boxes {
			b_pos := box3d.Body_GetPosition(b)
			b_rot := box3d.Body_GetRotation(b)
			model_mat := linalg.matrix4_from_trs_f32(b_pos, b_rot, {1, 1, 1})
			gfx.mesh_draw(box_mesh, model_mat, texture = box_tex)
		}
	}
}
