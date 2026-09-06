#+build !windows
package gfx

import "core:time"
import "core:strings"
import sdl "vendor:sdl3"
import vk "vendor:vulkan"

window_handle: ^sdl.Window

vk_platform_instance_extensions :: proc() -> []cstring {
	count: u32
	exts := sdl.Vulkan_GetInstanceExtensions(&count)
	return exts[:count]
}

vk_create_platform_surface :: proc(instance: vk.Instance) -> (vk.SurfaceKHR, bool) {
	surface: vk.SurfaceKHR
	if !sdl.Vulkan_CreateSurface(window_handle, instance, nil, &surface) {
		return 0, false
	}
	return surface, true
}

init :: proc(title: string, size := [2]int{1280, 720}) {
	_ = sdl.Init({.VIDEO})

	title_cstr := strings.clone_to_cstring(title, context.temp_allocator)
	window_handle = sdl.CreateWindow(title_cstr, i32(size.x), i32(size.y), {.VULKAN, .RESIZABLE})

	w, h: i32
	sdl.GetWindowSizeInPixels(window_handle, &w, &h)
	window.size = {int(w), int(h)}
	window.prev_time = time.now()

	vk_init()
}

scancode_to_key :: proc(sc: sdl.Scancode) -> Key {
	#partial switch sc {
	case .A: return .A
	case .B: return .B
	case .C: return .C
	case .D: return .D
	case .E: return .E
	case .F: return .F
	case .G: return .G
	case .H: return .H
	case .I: return .I
	case .J: return .J
	case .K: return .K
	case .L: return .L
	case .M: return .M
	case .N: return .N
	case .O: return .O
	case .P: return .P
	case .Q: return .Q
	case .R: return .R
	case .S: return .S
	case .T: return .T
	case .U: return .U
	case .V: return .V
	case .W: return .W
	case .X: return .X
	case .Y: return .Y
	case .Z: return .Z

	case ._0: return .N0
	case ._1: return .N1
	case ._2: return .N2
	case ._3: return .N3
	case ._4: return .N4
	case ._5: return .N5
	case ._6: return .N6
	case ._7: return .N7
	case ._8: return .N8
	case ._9: return .N9

	case .RETURN, .RETURN2, .KP_ENTER: return .Enter
	case .ESCAPE: return .Esc
	case .BACKSPACE, .KP_BACKSPACE: return .Backspace
	case .TAB, .KP_TAB: return .Tab
	case .SPACE, .KP_SPACE: return .Space

	case .MINUS: return .Minus
	case .EQUALS, .KP_EQUALS, .KP_EQUALSAS400: return .Equal
	case .LEFTBRACKET: return .LeftBracket
	case .RIGHTBRACKET: return .RightBracket
	case .BACKSLASH, .NONUSBACKSLASH: return .BackSlash
	case .SEMICOLON: return .Semicolon
	case .APOSTROPHE: return .Quote
	case .GRAVE: return .Backtick
	case .COMMA, .KP_COMMA: return .Comma
	case .PERIOD, .KP_PERIOD: return .Period
	case .SLASH: return .Slash

	case .F1: return .F1
	case .F2: return .F2
	case .F3: return .F3
	case .F4: return .F4
	case .F5: return .F5
	case .F6: return .F6
	case .F7: return .F7
	case .F8: return .F8
	case .F9: return .F9
	case .F10: return .F10
	case .F11: return .F11
	case .F12: return .F12
	case .F13: return .F13
	case .F14: return .F14
	case .F15: return .F15
	case .F16: return .F16
	case .F17: return .F17
	case .F18: return .F18
	case .F19: return .F19
	case .F20: return .F20
	case .F21: return .F21
	case .F22: return .F22
	case .F23: return .F23
	case .F24: return .F24

	case .HOME, .AC_HOME: return .Home
	case .END: return .End
	case .PAGEUP: return .PageUp
	case .PAGEDOWN: return .PageDown
	case .DELETE: return .Delete

	case .RIGHT: return .Right
	case .LEFT: return .Left
	case .DOWN: return .Down
	case .UP: return .Up

	case .KP_DIVIDE: return .NumSlash
	case .KP_MULTIPLY: return .NumStar
	case .KP_MINUS: return .NumMinus
	case .KP_PLUS: return .NumPlus
	case .KP_0: return .P0
	case .KP_1: return .P1
	case .KP_2: return .P2
	case .KP_3: return .P3
	case .KP_4: return .P4
	case .KP_5: return .P5
	case .KP_6: return .P6
	case .KP_7: return .P7
	case .KP_8: return .P8
	case .KP_9: return .P9

	case .LCTRL, .RCTRL: return .Ctrl
	case .LSHIFT, .RSHIFT: return .Shift
	case .LALT, .RALT: return .Alt
	case .LGUI: return .Left_Super
	case .RGUI: return .Right_Super

	case: return .Null
	}
}

update :: proc() -> bool {
	window.mouse_scroll = {0, 0}
	window.mouse_delta = {0, 0}

	for &state in window.key_state {
		state -= {.Pressed, .Released, .Repeat}
	}

	event: sdl.Event
	for sdl.PollEvent(&event) {
		#partial switch event.type {
		case .QUIT:
			window.should_close = true

		case .WINDOW_CLOSE_REQUESTED, .WINDOW_DESTROYED:
			window.should_close = true

		case .WINDOW_RESIZED, .WINDOW_PIXEL_SIZE_CHANGED:
			w, h: i32
			sdl.GetWindowSizeInPixels(window_handle, &w, &h)
			if int(w) != window.size.x || int(h) != window.size.y {
				window.size.x = int(w)
				window.size.y = int(h)
				window.is_resized = true
			}

		case .WINDOW_FOCUS_LOST:
			window.mouse_locked = false
			for k in Key do button_update(k, false)

		case .MOUSE_BUTTON_DOWN, .MOUSE_BUTTON_UP:
			btn_down := event.button.down
			switch event.button.button {
			case sdl.BUTTON_LEFT:
				button_update(.Mouse_Left, btn_down)
			case sdl.BUTTON_RIGHT:
				button_update(.Mouse_Right, btn_down)
			case sdl.BUTTON_MIDDLE:
				button_update(.Mouse_Middle, btn_down)
			}

		case .MOUSE_MOTION:
			window.mouse_pos.x = event.motion.x
			window.mouse_pos.y = event.motion.y
			if window.mouse_locked {
				window.mouse_delta.x += event.motion.xrel
				window.mouse_delta.y += event.motion.yrel
			}

		case .MOUSE_WHEEL:
			x := event.wheel.x
			y := event.wheel.y
			if event.wheel.direction == .FLIPPED {
				x = -x
				y = -y
			}
			window.mouse_scroll.x += x
			window.mouse_scroll.y += y

		case .KEY_DOWN, .KEY_UP:
			key := scancode_to_key(event.key.scancode)
			if key != .Null {
				button_update(key, event.key.down)
				if event.key.repeat {
					window.key_state[key] += {.Repeat}
				}
			}
		}
	}

	@(static) prev_mouse_locked: bool
	if window.mouse_locked != prev_mouse_locked {
		_ = sdl.SetWindowRelativeMouseMode(window_handle, window.mouse_locked)
		prev_mouse_locked = window.mouse_locked
	}

	if !window.should_close && window.size.x > 0 && window.size.y > 0 {
		if window.is_resized {
			vk_swapchain_create()
			window.is_resized = false
		}
		vk_render()
	}

	cur_time := time.now()
	window.frame_time = cast(f32)time.duration_seconds(time.diff(window.prev_time, cur_time))
	window.prev_time = cur_time

	return !window.should_close
}
