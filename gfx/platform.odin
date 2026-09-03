package gfx

import "core:time"

Mat4  :: matrix[4, 4]f32
Vec2  :: [2]f32
Vec3  :: [3]f32
Vec4  :: [4]f32
Color :: [4]u8

Key_State :: enum { Held, Pressed, Released, Repeat }

Key :: enum u8 {
	Null          = 0,
	Mouse_Left    = 0x01,
	Mouse_Right   = 0x02,
	Mouse_Middle  = 0x04,

	N0 = '0', N1 = '1', N2 = '2', N3 = '3', N4 = '4',
	N5 = '5', N6 = '6', N7 = '7', N8 = '8', N9 = '9',

	A = 'A', B = 'B', C = 'C', D = 'D', E = 'E', F = 'F',
	G = 'G', H = 'H', I = 'I', J = 'J', K = 'K', L = 'L',
	M = 'M', N = 'N', O = 'O', P = 'P', Q = 'Q', R = 'R',
	S = 'S', T = 'T', U = 'U', V = 'V', W = 'W', X = 'X',
	Y = 'Y', Z = 'Z',

	Backspace     = 0x08,
	Tab           = 0x09,
	Enter         = 0x0D,
	Shift         = 0x10,
	Ctrl          = 0x11,
	Alt           = 0x12,
	Esc           = 0x1B,
	Space         = 0x20,

	End           = 0x23,
	Home          = 0x24,
	Left          = 0x25,
	Up            = 0x26,
	Right         = 0x27,
	Down          = 0x28,
	Delete        = 0x2E,

	Left_Super    = 0x5B,
	Right_Super   = 0x5C,

	P0 = 0x60, P1 = 0x61, P2 = 0x62, P3 = 0x63, P4 = 0x64,
	P5 = 0x65, P6 = 0x66, P7 = 0x67, P8 = 0x68, P9 = 0x69,

	NumStar   = 0x6A, NumPlus  = 0x6B, NumMinus = 0x6D,
	NumPeriod = 0x6E, NumSlash = 0x6F,

	F1  = 0x70, F2  = 0x71, F3  = 0x72, F4  = 0x73,
	F5  = 0x74, F6  = 0x75, F7  = 0x76, F8  = 0x77,
	F9  = 0x78, F10 = 0x79, F11 = 0x7A, F12 = 0x7B,
	F13 = 0x7C, F14 = 0x7D, F15 = 0x7E, F16 = 0x7F,
	F17 = 0x80, F18 = 0x81, F19 = 0x82, F20 = 0x83,
	F21 = 0x84, F22 = 0x85, F23 = 0x86, F24 = 0x87,

	Semicolon    = 0xBA, Equal        = 0xBB,
	Comma        = 0xBC, Minus        = 0xBD,
	Period       = 0xBE, Slash        = 0xBF,
	Backtick     = 0xC0,

	PageUp       = 0x21, PageDown     = 0x22,
	LeftBracket  = 0xDB, RightBracket = 0xDD,
	BackSlash    = 0xDC, Quote        = 0xDE,
}

window: struct {
	size:           [2]int,
	is_resized:     bool,
	should_close:   bool,
	key_state:      [256]bit_set[Key_State],
	mouse_pos:      Vec2,
	mouse_delta:    Vec2,
	mouse_scroll:   Vec2,
	mouse_locked:   bool,
	prev_time:      time.Time,
	frame_time:     f32,
}

frame_time :: proc() -> f32 {
	return min(window.frame_time, 1.0 / 60.0)
}

window_size :: proc() -> Vec2 {
	return Vec2(window.size)
}

mouse_pos :: proc() -> Vec2 {
	return window.mouse_pos
}

mouse_delta :: proc() -> Vec2 {
	return window.mouse_delta
}

mouse_set_locked :: proc(locked: bool) {
	window.mouse_locked = locked
}

mouse_scroll :: proc() -> Vec2 {
	return window.mouse_scroll
}

key_is_down :: proc(key: Key) -> bool {
	return .Held in window.key_state[key]
}

key_is_pressed :: proc(key: Key) -> bool {
	return .Pressed in window.key_state[key]
}

key_is_released :: proc(key: Key) -> bool {
	return .Released in window.key_state[key]
}

key_is_pressed_repeat :: proc(key: Key) -> bool {
	return .Repeat in window.key_state[key]
}

button_update :: proc(button: Key, down_up: bool) {
	was_down := .Held in window.key_state[button]
	if was_down != down_up {
		if down_up {
			window.key_state[button] += {.Held, .Pressed}
		} else {
			window.key_state[button] -= {.Held}
			window.key_state[button] += {.Released}
		}
	}
}