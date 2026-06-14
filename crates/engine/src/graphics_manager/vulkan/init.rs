//! Vulkan instance, surface, and device creation.
//!
//! Three steps in one module because they're tightly sequenced at startup and
//! share supporting types (`SurfaceStuff`, `QueueFamilyIndices`,
//! `DeviceExtension`). Splitting them further would just thread the same
//! handles through more boundaries.

use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use std::ffi::CString;
use std::os::raw::c_void;
use std::ptr;

use crate::graphics_manager::constants::*;
use crate::graphics_manager::debug;
use crate::graphics_manager::structures::*;

/// Vulkan instance extensions required to present to a window on the current
/// platform (e.g. `VK_KHR_surface` + `VK_KHR_xlib_surface` on Linux/X11).
/// Delegates to `ash-window`, plus `VK_EXT_debug_utils` for the validation
/// layer's messenger.
fn required_extension_names(window: &winit::window::Window) -> Vec<*const i8> {
    let display = window.display_handle().expect("display handle");
    let mut names: Vec<*const i8> = ash_window::enumerate_required_extensions(display.as_raw())
        .expect("ash_window::enumerate_required_extensions failed")
        .to_vec();
    names.push(ash::ext::debug_utils::NAME.as_ptr());
    names
}

/// Create a `VkSurfaceKHR` for the given window. `ash-window` picks the right
/// `VK_KHR_*_surface` extension based on the window's handle variant.
unsafe fn create_raw_surface(
    entry: &ash::Entry,
    instance: &ash::Instance,
    window: &winit::window::Window,
) -> Result<vk::SurfaceKHR, vk::Result> {
    let display = window.display_handle().expect("display handle");
    let window_handle = window.window_handle().expect("window handle");
    ash_window::create_surface(entry, instance, display.as_raw(), window_handle.as_raw(), None)
}

pub fn create_instance(
    entry: &ash::Entry,
    window: &winit::window::Window,
    window_title: &str,
    is_enable_debug: bool,
    required_validation_layers: &[&str],
) -> ash::Instance {
    if is_enable_debug && !debug::check_validation_layer_support(entry, required_validation_layers)
    {
        panic!("Validation layers requested, but not available!");
    }

    let app_name = CString::new(window_title).unwrap();
    let engine_name = CString::new("Vulkan Engine").unwrap();
    let app_info = vk::ApplicationInfo {
        p_application_name: app_name.as_ptr(),
        s_type: vk::StructureType::APPLICATION_INFO,
        p_next: ptr::null(),
        application_version: APPLICATION_VERSION,
        p_engine_name: engine_name.as_ptr(),
        engine_version: ENGINE_VERSION,
        api_version: API_VERSION,
        ..Default::default()
    };

    // This create info used to debug issues in vk::createInstance and vk::destroyInstance.
    let debug_utils_create_info = debug::populate_debug_messenger_create_info();

    // VK_EXT debug report has been requested here.
    let extension_names = required_extension_names(window);

    let requred_validation_layer_raw_names: Vec<CString> = required_validation_layers
        .iter()
        .map(|layer_name| CString::new(*layer_name).unwrap())
        .collect();
    let layer_names: Vec<*const i8> = requred_validation_layer_raw_names
        .iter()
        .map(|layer_name| layer_name.as_ptr())
        .collect();

    let create_info = vk::InstanceCreateInfo {
        s_type: vk::StructureType::INSTANCE_CREATE_INFO,
        p_next: if VALIDATION.is_enable {
            &debug_utils_create_info as *const vk::DebugUtilsMessengerCreateInfoEXT as *const c_void
        } else {
            ptr::null()
        },
        flags: vk::InstanceCreateFlags::empty(),
        p_application_info: &app_info,
        pp_enabled_layer_names: if is_enable_debug {
            layer_names.as_ptr()
        } else {
            ptr::null()
        },
        enabled_layer_count: if is_enable_debug {
            layer_names.len()
        } else {
            0
        } as u32,
        pp_enabled_extension_names: extension_names.as_ptr(),
        enabled_extension_count: extension_names.len() as u32,
        ..Default::default()
    };

    let instance: ash::Instance = unsafe {
        entry
            .create_instance(&create_info, None)
            .expect("Failed to create instance!")
    };

    instance
}

pub fn create_surface(
    entry: &ash::Entry,
    instance: &ash::Instance,
    window: &winit::window::Window,
) -> SurfaceStuff {
    let surface = unsafe {
        create_raw_surface(entry, instance, window).expect("Failed to create surface.")
    };
    let surface_loader = ash::khr::surface::Instance::new(entry, instance);

    SurfaceStuff {
        surface_loader,
        surface,
    }
}

pub fn pick_physical_device(
    instance: &ash::Instance,
    surface_stuff: &SurfaceStuff,
    required_device_extensions: &DeviceExtension,
) -> vk::PhysicalDevice {
    let physical_devices = unsafe {
        instance
            .enumerate_physical_devices()
            .expect("Failed to enumerate Physical Devices!")
    };

    let result = physical_devices.iter().find(|physical_device| {
        is_physical_device_suitable(
            instance,
            **physical_device,
            surface_stuff,
            required_device_extensions,
        )
    });

    match result {
        Some(p_physical_device) => *p_physical_device,
        None => panic!("Failed to find a suitable GPU!"),
    }
}

pub fn is_physical_device_suitable(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    surface_stuff: &SurfaceStuff,
    required_device_extensions: &DeviceExtension,
) -> bool {
    let device_features = unsafe { instance.get_physical_device_features(physical_device) };

    let indices = find_queue_family(instance, physical_device, surface_stuff);

    let is_queue_family_supported = indices.is_complete();
    let is_device_extension_supported =
        check_device_extension_support(instance, physical_device, required_device_extensions);
    let is_swapchain_supported = if is_device_extension_supported {
        let swapchain_support = super::swapchain::query_swapchain_support(physical_device, surface_stuff);
        !swapchain_support.formats.is_empty() && !swapchain_support.present_modes.is_empty()
    } else {
        false
    };
    let is_support_sampler_anisotropy = device_features.sampler_anisotropy == 1;

    is_queue_family_supported
        && is_device_extension_supported
        && is_swapchain_supported
        && is_support_sampler_anisotropy
}

pub fn create_logical_device(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    validation: &crate::graphics_manager::debug::ValidationInfo,
    device_extensions: &DeviceExtension,
    surface_stuff: &SurfaceStuff,
) -> (ash::Device, QueueFamilyIndices) {
    let indices = find_queue_family(instance, physical_device, surface_stuff);

    use std::collections::HashSet;
    let unique_queue_families = HashSet::from([
        indices.graphics_family.unwrap(),
        indices.present_family.unwrap(),
    ]);

    let queue_priorities = [1.0_f32];
    let queue_create_infos: Vec<vk::DeviceQueueCreateInfo> = unique_queue_families
        .iter()
        .map(|&queue_family| vk::DeviceQueueCreateInfo {
            s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
            p_next: ptr::null(),
            flags: vk::DeviceQueueCreateFlags::empty(),
            queue_family_index: queue_family,
            p_queue_priorities: queue_priorities.as_ptr(),
            queue_count: queue_priorities.len() as u32,
            ..Default::default()
        })
        .collect();

    let physical_device_features = vk::PhysicalDeviceFeatures {
        sampler_anisotropy: vk::TRUE, // enable anisotropy device feature from Chapter-24.
        ..Default::default()
    };

    // Per-device validation layers were deprecated and ignored by modern Vulkan
    // loaders — validation is enabled at the instance level only. We no longer
    // wire `enabled_layer_*` here; the `validation` arg is kept for the call-site
    // signature but unused at device creation.
    let _ = validation;

    let enable_extension_names = device_extensions.get_extensions_raw_names();

    let device_create_info = vk::DeviceCreateInfo {
        s_type: vk::StructureType::DEVICE_CREATE_INFO,
        p_next: ptr::null(),
        flags: vk::DeviceCreateFlags::empty(),
        queue_create_info_count: queue_create_infos.len() as u32,
        p_queue_create_infos: queue_create_infos.as_ptr(),
        enabled_extension_count: enable_extension_names.len() as u32,
        pp_enabled_extension_names: enable_extension_names.as_ptr(),
        p_enabled_features: &physical_device_features,
        ..Default::default()
    };

    let device: ash::Device = unsafe {
        instance
            .create_device(physical_device, &device_create_info, None)
            .expect("Failed to create logical Device!")
    };

    (device, indices)
}

pub fn find_queue_family(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    surface_stuff: &SurfaceStuff,
) -> QueueFamilyIndices {
    let queue_families =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

    let mut queue_family_indices = QueueFamilyIndices::new();

    for (index, queue_family) in queue_families.iter().enumerate() {
        if queue_family.queue_count > 0
            && queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
        {
            queue_family_indices.graphics_family = Some(index as u32);
        }

        let is_present_support = unsafe {
            surface_stuff
                .surface_loader
                .get_physical_device_surface_support(
                    physical_device,
                    index as u32,
                    surface_stuff.surface,
                )
                .unwrap_or(false)
        };
        if queue_family.queue_count > 0 && is_present_support {
            queue_family_indices.present_family = Some(index as u32);
        }

        if queue_family_indices.is_complete() {
            break;
        }
    }

    queue_family_indices
}

pub fn check_device_extension_support(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    device_extensions: &DeviceExtension,
) -> bool {
    let available_extensions = unsafe {
        instance
            .enumerate_device_extension_properties(physical_device)
            .expect("Failed to get device extension properties.")
    };

    let available_extension_names: Vec<String> = available_extensions
        .iter()
        .map(|extension| crate::graphics_manager::tools::vk_to_string(&extension.extension_name))
        .collect();

    use std::collections::HashSet;
    let mut required_extensions = HashSet::new();
    for extension in device_extensions.names.iter() {
        required_extensions.insert(extension.to_string());
    }

    for extension_name in available_extension_names.iter() {
        required_extensions.remove(extension_name);
    }

    required_extensions.is_empty()
}
