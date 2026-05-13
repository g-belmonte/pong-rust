use ash::vk;

#[cfg(target_os = "windows")]
use ash::khr::win32_surface;
#[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
use ash::khr::xlib_surface;
#[cfg(target_os = "macos")]
use ash::mvk::macos_surface;

use ash::ext::debug_utils;
use ash::khr::surface;

#[cfg(target_os = "macos")]
use cocoa::appkit::{NSView, NSWindow};
#[cfg(target_os = "macos")]
use cocoa::base::id as cocoa_id;
#[cfg(target_os = "macos")]
use metal::CoreAnimationLayer;
#[cfg(target_os = "macos")]
use objc::runtime::YES;

// required extension ------------------------------------------------------
#[cfg(target_os = "macos")]
pub fn required_extension_names() -> Vec<*const i8> {
    vec![
        surface::NAME.as_ptr(),
        macos_surface::NAME.as_ptr(),
        debug_utils::NAME.as_ptr(),
    ]
}

#[cfg(windows)]
pub fn required_extension_names() -> Vec<*const i8> {
    vec![
        surface::NAME.as_ptr(),
        win32_surface::NAME.as_ptr(),
        debug_utils::NAME.as_ptr(),
    ]
}

#[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
pub fn required_extension_names() -> Vec<*const i8> {
    vec![
        surface::NAME.as_ptr(),
        xlib_surface::NAME.as_ptr(),
        debug_utils::NAME.as_ptr(),
    ]
}
// ------------------------------------------------------------------------

// create surface ---------------------------------------------------------
#[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
pub unsafe fn create_surface(
    entry: &ash::Entry,
    instance: &ash::Instance,
    window: &winit::window::Window,
) -> Result<vk::SurfaceKHR, vk::Result> {
    use winit::platform::unix::WindowExtUnix;

    let x11_display = window.xlib_display().unwrap();
    let x11_window = window.xlib_window().unwrap();
    let x11_create_info = vk::XlibSurfaceCreateInfoKHR::default()
        .window(x11_window as vk::Window)
        .dpy(x11_display as *mut vk::Display);
    let xlib_surface_loader = xlib_surface::Instance::new(entry, instance);
    xlib_surface_loader.create_xlib_surface(&x11_create_info, None)
}

#[cfg(target_os = "macos")]
pub unsafe fn create_surface(
    entry: &ash::Entry,
    instance: &ash::Instance,
    window: &winit::window::Window,
) -> Result<vk::SurfaceKHR, vk::Result> {
    use std::mem;
    use std::os::raw::c_void;
    use winit::platform::macos::WindowExtMacOS;

    let wnd: cocoa_id = mem::transmute(window.ns_window());

    let layer = CoreAnimationLayer::new();

    layer.set_edge_antialiasing_mask(0);
    layer.set_presents_with_transaction(false);
    layer.remove_all_animations();

    let view = wnd.contentView();

    layer.set_contents_scale(view.backingScaleFactor());
    view.setLayer(mem::transmute(layer.as_ref()));
    view.setWantsLayer(YES);

    let create_info = vk::MacOSSurfaceCreateInfoMVK::default()
        .view(window.ns_view() as *const c_void);

    let macos_surface_loader = macos_surface::Instance::new(entry, instance);
    macos_surface_loader.create_mac_os_surface(&create_info, None)
}

#[cfg(target_os = "windows")]
pub unsafe fn create_surface(
    entry: &ash::Entry,
    instance: &ash::Instance,
    window: &winit::window::Window,
) -> Result<vk::SurfaceKHR, vk::Result> {
    use std::os::raw::c_void;
    use std::ptr;
    use winapi::shared::windef::HWND;
    use winapi::um::libloaderapi::GetModuleHandleW;
    use winit::platform::windows::WindowExtWindows;

    let hwnd = window.hwnd() as HWND;
    let hinstance = GetModuleHandleW(ptr::null()) as *const c_void;
    let win32_create_info = vk::Win32SurfaceCreateInfoKHR::default()
        .hinstance(hinstance as isize)
        .hwnd(hwnd as isize);
    let win32_surface_loader = win32_surface::Instance::new(entry, instance);
    win32_surface_loader.create_win32_surface(&win32_create_info, None)
}
// ------------------------------------------------------------------------
