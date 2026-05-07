use crate::graphics_manager::debug::ValidationInfo;
use crate::graphics_manager::structures::DeviceExtension;
use ash::vk_make_version;

use std::os::raw::c_char;

pub const APPLICATION_VERSION: u32 = vk_make_version!(1, 0, 0);
pub const ENGINE_VERSION: u32 = vk_make_version!(1, 0, 0);
pub const API_VERSION: u32 = vk_make_version!(1, 0, 92);

pub const WINDOW_TITLE: &str = "Pong-rust";
pub const WINDOW_WIDTH: u32 = 800;
pub const WINDOW_HEIGHT: u32 = 600;
pub const VALIDATION: ValidationInfo = ValidationInfo {
    is_enable: true,
    required_validation_layers: ["VK_LAYER_KHRONOS_validation"],
};
pub const DEVICE_EXTENSIONS: DeviceExtension = DeviceExtension {
    names: ["VK_KHR_swapchain"],
};
pub const MAX_FRAMES_IN_FLIGHT: usize = 2;
pub const IS_PAINT_FPS_COUNTER: bool = true;

// Caps how many textured models can be live simultaneously: bounds the
// descriptor-pool size set up in `share::create_descriptor_pool`. Picked with
// step 9 (per-glyph quads for text labels) in mind — bump if you ever hit the
// allocation failure on `register_textured_model_with`.
pub const MAX_TEXTURED_MODELS: usize = 256;

impl DeviceExtension {
    pub fn get_extensions_raw_names(&self) -> [*const c_char; 1] {
        [ash::extensions::khr::Swapchain::name().as_ptr()]
    }
}
