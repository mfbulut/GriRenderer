#+build windows
package gfx

import "base:runtime"
import "core:time"
import "core:sys/windows"
import vk "vendor:vulkan"

hwnd: windows.HWND

vk_platform_instance_extensions :: proc() -> []cstring {
	@(static) ext := []cstring {
		vk.KHR_SURFACE_EXTENSION_NAME,
		vk.KHR_WIN32_SURFACE_EXTENSION_NAME,
	}

	return ext
}

vk_create_platform_surface :: proc(instance: vk.Instance) -> (vk.SurfaceKHR, bool) {
	hinstance := cast(windows.HINSTANCE)windows.GetModuleHandleW(nil)
	surface_info := vk.Win32SurfaceCreateInfoKHR{
		sType     = .WIN32_SURFACE_CREATE_INFO_KHR,
		hinstance = hinstance,
		hwnd      = hwnd,
	}
	surface: vk.SurfaceKHR
	if vk.CreateWin32SurfaceKHR(instance, &surface_info, nil, &surface) != .SUCCESS {
		return 0, false
	}
	return surface, true
}

init :: proc(title: string, size := [2]int{1280, 720}) {
	windows.SetProcessDPIAware()
	hinstance := cast(windows.HINSTANCE)windows.GetModuleHandleW(nil)
	wndclass := windows.WNDCLASSW{
		lpfnWndProc   = window_proc,
		style         = windows.CS_OWNDC,
		hInstance     = hinstance,
		hIcon         = windows.LoadIconW(hinstance, cast(windows.LPCWSTR)windows.MAKEINTRESOURCEW(1)),
		hCursor       = windows.LoadCursorA(nil, windows.IDC_ARROW),
		hbrBackground = cast(windows.HBRUSH)windows.GetStockObject(windows.BLACK_BRUSH),
		lpszClassName = "Renderer",
	}

	windows.RegisterClassW(&wndclass)

	window_rect := windows.RECT{
		right  = i32(size.x),
		bottom = i32(size.y),
	}

	ex_style := windows.WS_EX_APPWINDOW
	dw_style := windows.WS_OVERLAPPEDWINDOW | windows.WS_CLIPCHILDREN
	windows.AdjustWindowRectEx(&window_rect, dw_style, false, ex_style)

	window_w := window_rect.right - window_rect.left
	window_h := window_rect.bottom - window_rect.top

	x_pos := (windows.GetSystemMetrics(windows.SM_CXSCREEN) - window_w) / 2
	y_pos := (windows.GetSystemMetrics(windows.SM_CYSCREEN) - window_h) / 2

	title16 := windows.utf8_to_wstring(title, context.temp_allocator)
	hwnd = windows.CreateWindowExW(ex_style, "Renderer", title16, dw_style, x_pos, y_pos, window_w, window_h, nil, nil, hinstance, nil)
	window.prev_time = time.now()
	window.size = size

	value := windows.TRUE
	windows.DwmSetWindowAttribute(hwnd, u32(windows.DWMWINDOWATTRIBUTE.DWMWA_USE_IMMERSIVE_DARK_MODE), &value, size_of(value))
	windows.ShowWindow(hwnd, windows.SW_SHOW)
	windows.UpdateWindow(hwnd)

	vk_init()
}

update :: proc() -> bool {
	for &state in window.key_state {
		state -= {.Pressed, .Released, .Repeat}
	}

	window.mouse_scroll = {0, 0}
	prev_mouse := window.mouse_pos

	msg: windows.MSG
	for windows.PeekMessageW(&msg, nil, 0, 0, windows.PM_REMOVE) {
		windows.TranslateMessage(&msg)
		windows.DispatchMessageW(&msg)
	}

	window.mouse_delta = window.mouse_pos - prev_mouse

	@(static) prev_mouse_locked: bool
	if window.mouse_locked != prev_mouse_locked {
		if window.mouse_locked {
			rect: windows.RECT
			windows.GetClientRect(hwnd, &rect)
			p1 := windows.POINT{rect.left, rect.top}
			p2 := windows.POINT{rect.right, rect.bottom}
			windows.ClientToScreen(hwnd, &p1)
			windows.ClientToScreen(hwnd, &p2)
			clip_rect := windows.RECT{p1.x, p1.y, p2.x, p2.y}
			windows.ClipCursor(&clip_rect)
			for windows.ShowCursor(false) >= 0 {}
			center_screen := windows.POINT{i32(window.size.x / 2), i32(window.size.y / 2)}
			windows.ClientToScreen(hwnd, &center_screen)
			windows.SetCursorPos(center_screen.x, center_screen.y)
		} else {
			windows.ClipCursor(nil)
			for windows.ShowCursor(true) < 0 {}
		}
		prev_mouse_locked = window.mouse_locked
	}

	if window.mouse_locked {
		cur_pos: windows.POINT
		windows.GetCursorPos(&cur_pos)

		center_screen := windows.POINT{i32(window.size.x / 2), i32(window.size.y / 2)}
		windows.ClientToScreen(hwnd, &center_screen)

		window.mouse_delta = {f32(cur_pos.x - center_screen.x), f32(cur_pos.y - center_screen.y)}
		windows.SetCursorPos(center_screen.x, center_screen.y)
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

window_proc :: proc "system" (hwnd: windows.HWND, msg: windows.UINT, wparam: windows.WPARAM, lparam: windows.LPARAM) -> windows.LRESULT {
	context = runtime.default_context()

	switch msg {
	case windows.WM_DESTROY, windows.WM_CLOSE:
		window.should_close = true
		windows.PostQuitMessage(0)

	case windows.WM_SIZE:
		window.size.x = cast(int)windows.LOWORD(lparam)
		window.size.y = cast(int)windows.HIWORD(lparam)
		window.is_resized = true

	case windows.WM_SETFOCUS:
	case windows.WM_KILLFOCUS:
		window.mouse_locked = false
		for k in Key do button_update(k, false)

	case windows.WM_LBUTTONUP:
		button_update(.Mouse_Left, false)
		windows.ReleaseCapture()
	case windows.WM_LBUTTONDOWN:
		button_update(.Mouse_Left, true)
		windows.SetCapture(hwnd)
	case windows.WM_MBUTTONUP:
		button_update(.Mouse_Middle, false)
		windows.ReleaseCapture()
	case windows.WM_MBUTTONDOWN:
		button_update(.Mouse_Middle, true)
		windows.SetCapture(hwnd)
	case windows.WM_RBUTTONUP:
		button_update(.Mouse_Right, false)
		windows.ReleaseCapture()
	case windows.WM_RBUTTONDOWN:
		button_update(.Mouse_Right, true)
		windows.SetCapture(hwnd)

	case windows.WM_MOUSEMOVE:
		window.mouse_pos.x = cast(f32)windows.GET_X_LPARAM(lparam)
		window.mouse_pos.y = cast(f32)windows.GET_Y_LPARAM(lparam)

	case windows.WM_MOUSEWHEEL:
		window.mouse_scroll.y += cast(f32)windows.GET_WHEEL_DELTA_WPARAM(wparam) / windows.WHEEL_DELTA
	case windows.WM_MOUSEHWHEEL:
		window.mouse_scroll.x += cast(f32)windows.GET_WHEEL_DELTA_WPARAM(wparam) / windows.WHEEL_DELTA

	case windows.WM_SYSKEYDOWN:
		if wparam == windows.VK_F4 {
			window.should_close = true
			break
		}
		fallthrough
	case windows.WM_SYSKEYUP, windows.WM_KEYUP, windows.WM_KEYDOWN:
		is_down := (lparam & (1 << 31)) == 0
		button_update(cast(Key)wparam, is_down)
	}

	return windows.DefWindowProcW(hwnd, msg, wparam, lparam)
}