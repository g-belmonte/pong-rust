use winit::dpi::LogicalSize;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowAttributes};

/// Create the engine's window. winit 0.30 split `EventLoop` (long-lived) from
/// `ActiveEventLoop` (the per-callback handle that owns window creation), so
/// this is called from `ApplicationHandler::resumed`, not from the static
/// startup path.
pub fn init_window(
    event_loop: &ActiveEventLoop,
    title: &str,
    width: u32,
    height: u32,
) -> Window {
    let attrs = WindowAttributes::default()
        .with_title(title)
        .with_inner_size(LogicalSize::new(width, height));
    event_loop
        .create_window(attrs)
        .expect("Failed to create window.")
}
