// ---------- Module layout ----------
//
// The submodules below split the renderer's surface area by concern:
//
// - `vulkan` (formerly `share.rs`): low-level Vulkan helpers (instance,
//   swapchain, buffers, images, render pass, commands, descriptors, shader).
// - `material`: pipeline / descriptor-set layout for one material.
// - `registry`: `register_*` / `unregister_*` / `set_instance_extra` —
//   meshes, textures, materials, per-instance state.
// - `camera`: per-slot camera UBO management + `set_camera`.
// - `draw`: `draw_frame` and the per-material batching that feeds it.
// - `swapchain_recreate`: swapchain teardown + rebuild on resize.
// - `hot_reload` (feature-gated): file-watcher plus shader / texture / mesh
//   reload paths.
//
// This file owns the `GraphicsManager` struct definition, the constructor
// (`new`), `Drop`, simple accessors, and the disk-backed pipeline cache.
pub mod constants;
pub mod debug;
#[cfg(feature = "hot-reload")]
pub mod hot_reload;
pub mod material;
pub mod structures;
pub mod tools;
pub mod vulkan;
pub mod window;

mod camera;
mod draw;
mod registry;
mod swapchain_recreate;

use glam::Mat4;
use constants::*;
use structures::QueueFamilyIndices;

use ash::vk;

use std::collections::HashMap;

pub use self::material::{
    Binding, BlendMode, DepthMode, MaterialDesc, MaterialHandle, VertexAttr,
};
use self::material::Material;

// Two distinct handle types: MeshHandle identifies a piece of geometry that
// can be reused across many instances; ModelHandle identifies a single
// drawable instance (of any material — solid or textured).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct MeshHandle(u32);

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct ModelHandle(u32);

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct TextureHandle(u32);

pub struct TextureResources {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    sampler: vk::Sampler,
}

struct MeshBuffers {
    vertex_buffer: vk::Buffer,
    vertex_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_memory: vk::DeviceMemory,
    index_count: u32,
}

/// One registered drawable instance. `material` selects the pipeline and
/// layout; `mesh` is populated for non-sampler materials (each instance brings
/// its own geometry); `texture` is populated for sampler materials (the
/// renderer batches instances by texture and draws against the engine's shared
/// unit quad). `extra` holds the per-instance bytes after the model matrix —
/// e.g. `[f32; 3]` colour for the solid material, `[f32; 2] + [f32; 2]` UV
/// rect for the textured material.
struct MaterialInstanceData {
    material: MaterialHandle,
    mesh: Option<MeshHandle>,
    texture: Option<TextureHandle>,
    extra: Vec<u8>,
    last_model: Mat4,
}

fn destroy_mesh(device: &ash::Device, mesh: &MeshBuffers) {
    unsafe {
        device.destroy_buffer(mesh.index_buffer, None);
        device.free_memory(mesh.index_memory, None);
        device.destroy_buffer(mesh.vertex_buffer, None);
        device.free_memory(mesh.vertex_memory, None);
    }
}

fn destroy_texture_resources(device: &ash::Device, tex: &TextureResources) {
    unsafe {
        device.destroy_sampler(tex.sampler, None);
        device.destroy_image_view(tex.view, None);
        device.destroy_image(tex.image, None);
        device.free_memory(tex.memory, None);
    }
}

/// Cap on simultaneously-registered materials. Bounds the descriptor pool
/// sizing in [`vulkan::create_descriptor_pool`].
pub const MAX_MATERIALS: usize = 16;

/// Field groups below mark which state survives `recreate_swapchain` and
/// which is rebuilt. Changing this layout means revisiting that method.
pub struct GraphicsManager {
    // ----- Window -----
    window: winit::window::Window,

    // ----- Vulkan core (created once, destroyed at shutdown) -----
    _entry: ash::Entry,
    instance: ash::Instance,
    surface_loader: ash::khr::surface::Instance,
    surface: vk::SurfaceKHR,
    debug_utils_loader: ash::ext::debug_utils::Instance,
    debug_merssager: vk::DebugUtilsMessengerEXT,
    physical_device: vk::PhysicalDevice,
    physical_device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    device: ash::Device,
    queue_family: QueueFamilyIndices,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,

    // ----- Swapchain + dependent state (rebuilt on resize / OUT_OF_DATE) -----
    swapchain_loader: ash::khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain_imageviews: Vec<vk::ImageView>,
    swapchain_framebuffers: Vec<vk::Framebuffer>,

    // ----- Depth attachment (rebuilt on resize). One shared depth image
    // referenced by every framebuffer; cross-frame writes are serialised by
    // the render pass's external→subpass-0 dependency on the depth stages.
    depth_format: vk::Format,
    depth_image: vk::Image,
    depth_image_memory: vk::DeviceMemory,
    depth_image_view: vk::ImageView,

    // ----- Render pass (survives recreation unless swapchain format changes) -----
    render_pass: vk::RenderPass,
    pipeline_cache: vk::PipelineCache,

    // ----- Material registry -----
    materials: HashMap<MaterialHandle, Material>,
    next_material_handle: u32,
    /// Built-in solid-colour material (vertex pos, instance = mat4 model +
    /// vec3 color, bindings = [CameraUbo]). Registered in `new`.
    solid_material_handle: MaterialHandle,
    /// Built-in textured material (vertex pos = unit quad, instance = mat4
    /// model + vec2 uv_offset + vec2 uv_scale, bindings = [CameraUbo,
    /// Sampler2d]). Registered in `new`.
    textured_material_handle: MaterialHandle,

    // ----- Camera UBO + descriptor pool (survive swapchain recreation) -----
    //
    // Per-slot: each slot has `swapchain_image_count` UBO buffers (one per
    // swapchain image). Slots are allocated lazily — on `register_material`
    // for the slot it declares, and on `set_camera(slot, …)` for any slot a
    // scene populates that no material samples yet. Slot 0 is created
    // eagerly in `new` so the two built-in materials can wire up against it.
    camera_uniform_buffers: HashMap<u32, Vec<vk::Buffer>>,
    camera_uniform_buffers_memory: HashMap<u32, Vec<vk::DeviceMemory>>,
    descriptor_pool: vk::DescriptorPool,
    last_camera_views: HashMap<u32, Mat4>,
    last_camera_projs: HashMap<u32, Mat4>,

    // ----- Registered geometry, instances, and textures (survive recreation) -----
    meshes: HashMap<MeshHandle, MeshBuffers>,
    next_mesh_handle: u32,
    instances: HashMap<ModelHandle, MaterialInstanceData>,
    next_handle: u32,
    textures: HashMap<TextureHandle, TextureResources>,
    next_texture_handle: u32,

    // ----- Shared unit quad for textured (sampler-material) instances -----
    // Built once at construction. The textured vertex shader maps pos
    // `[-0.5..0.5]^2` to UV via the per-instance uv_offset/uv_scale, so any
    // sprite size and any UV rect is supported.
    textured_quad_vertex_buffer: vk::Buffer,
    textured_quad_vertex_memory: vk::DeviceMemory,
    textured_quad_index_buffer: vk::Buffer,
    textured_quad_index_memory: vk::DeviceMemory,
    textured_quad_index_count: u32,

    // ----- Command buffers + sync (survive recreation) -----
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    current_frame: usize,

    is_framebuffer_resized: bool,

    // ----- Shader hot-reload (dev only) -----
    #[cfg(feature = "hot-reload")]
    hot_reload: Option<hot_reload::HotReload>,
}

impl GraphicsManager {
    pub fn new(event_loop: &winit::event_loop::ActiveEventLoop) -> GraphicsManager {
        let window = window::init_window(event_loop, WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT);

        let entry = unsafe { ash::Entry::load().expect("Failed to load Vulkan library") };
        let instance = vulkan::create_instance(
            &entry,
            &window,
            WINDOW_TITLE,
            VALIDATION.is_enable,
            &VALIDATION.required_validation_layers,
        );
        let surface_stuff = vulkan::create_surface(&entry, &instance, &window);
        let (debug_utils_loader, debug_merssager) =
            debug::setup_debug_utils(VALIDATION.is_enable, &entry, &instance);
        let physical_device =
            vulkan::pick_physical_device(&instance, &surface_stuff, &DEVICE_EXTENSIONS);
        let physical_device_memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let (device, queue_family) = vulkan::create_logical_device(
            &instance,
            physical_device,
            &VALIDATION,
            &DEVICE_EXTENSIONS,
            &surface_stuff,
        );
        let graphics_queue =
            unsafe { device.get_device_queue(queue_family.graphics_family.unwrap(), 0) };
        let present_queue =
            unsafe { device.get_device_queue(queue_family.present_family.unwrap(), 0) };
        let swapchain_stuff = vulkan::create_swapchain(
            &instance,
            &device,
            physical_device,
            &window,
            &surface_stuff,
            &queue_family,
        );
        let swapchain_imageviews = vulkan::create_image_views(
            &device,
            swapchain_stuff.swapchain_format,
            &swapchain_stuff.swapchain_images,
        );
        let depth_format = vulkan::pick_depth_format(&instance, physical_device);
        let render_pass =
            vulkan::create_render_pass(&device, swapchain_stuff.swapchain_format, depth_format);
        let pipeline_cache = create_pipeline_cache(&device);
        let (depth_image, depth_image_memory, depth_image_view) = vulkan::create_depth_attachment(
            &device,
            &physical_device_memory_properties,
            depth_format,
            swapchain_stuff.swapchain_extent,
        );
        let swapchain_framebuffers = vulkan::create_framebuffers(
            &device,
            render_pass,
            &swapchain_imageviews,
            depth_image_view,
            swapchain_stuff.swapchain_extent,
        );
        let command_pool = vulkan::create_command_pool(&device, &queue_family);

        let swapchain_image_count = swapchain_stuff.swapchain_images.len();
        // Slot 0 is created eagerly so the two built-in materials (registered
        // immediately after) have a UBO to bind. Any additional slot a game
        // declares lands lazily inside `ensure_camera_slot`.
        let (slot0_buffers, slot0_memories) = vulkan::create_uniform_buffers(
            &device,
            &physical_device_memory_properties,
            swapchain_image_count,
        );
        let mut camera_uniform_buffers: HashMap<u32, Vec<vk::Buffer>> = HashMap::new();
        camera_uniform_buffers.insert(0, slot0_buffers);
        let mut camera_uniform_buffers_memory: HashMap<u32, Vec<vk::DeviceMemory>> = HashMap::new();
        camera_uniform_buffers_memory.insert(0, slot0_memories);
        let descriptor_pool = vulkan::create_descriptor_pool(
            &device,
            swapchain_image_count,
            MAX_MATERIALS,
            MAX_TEXTURED_MODELS,
        );

        // Shared unit-quad VBO/IBO. Used by every sampler-using material's
        // instances; non-sampler materials bring their own meshes. Layout is
        // `vec2 pos` per vertex (matches the textured material's
        // `vertex_attrs: [F32x2]`).
        let unit_quad_positions: [[f32; 2]; 4] = [
            [-0.5, -0.5],
            [ 0.5, -0.5],
            [ 0.5,  0.5],
            [-0.5,  0.5],
        ];
        let unit_quad_indices: [u32; 6] = [0, 1, 2, 2, 3, 0];
        let (textured_quad_vertex_buffer, textured_quad_vertex_memory) =
            vulkan::create_vertex_buffer(
                &device,
                &physical_device_memory_properties,
                command_pool,
                graphics_queue,
                &unit_quad_positions,
            );
        let (textured_quad_index_buffer, textured_quad_index_memory) =
            vulkan::create_index_buffer(
                &device,
                &physical_device_memory_properties,
                command_pool,
                graphics_queue,
                &unit_quad_indices,
            );

        let command_buffers =
            vulkan::allocate_command_buffers(&device, command_pool, MAX_FRAMES_IN_FLIGHT as u32);
        let sync_ojbects =
            vulkan::create_sync_objects(&device, MAX_FRAMES_IN_FLIGHT, swapchain_image_count);

        let mut gm = GraphicsManager {
            window,

            _entry: entry,
            instance,
            surface: surface_stuff.surface,
            surface_loader: surface_stuff.surface_loader,
            debug_utils_loader,
            debug_merssager,

            physical_device,
            physical_device_memory_properties,
            device,

            queue_family,
            graphics_queue,
            present_queue,

            swapchain_loader: swapchain_stuff.swapchain_loader,
            swapchain: swapchain_stuff.swapchain,
            swapchain_format: swapchain_stuff.swapchain_format,
            swapchain_images: swapchain_stuff.swapchain_images,
            swapchain_extent: swapchain_stuff.swapchain_extent,
            swapchain_imageviews,
            swapchain_framebuffers,

            depth_format,
            depth_image,
            depth_image_memory,
            depth_image_view,

            render_pass,
            pipeline_cache,

            materials: HashMap::new(),
            next_material_handle: 0,
            // Filled in immediately below; sentinel values until register_material runs.
            solid_material_handle: MaterialHandle(0),
            textured_material_handle: MaterialHandle(0),

            camera_uniform_buffers,
            camera_uniform_buffers_memory,
            descriptor_pool,

            meshes: HashMap::new(),
            next_mesh_handle: 0,
            instances: HashMap::new(),
            next_handle: 0,
            textured_quad_vertex_buffer,
            textured_quad_vertex_memory,
            textured_quad_index_buffer,
            textured_quad_index_memory,
            textured_quad_index_count: unit_quad_indices.len() as u32,
            textures: HashMap::new(),
            next_texture_handle: 0,
            last_camera_views: HashMap::new(),
            last_camera_projs: HashMap::new(),

            command_pool,
            command_buffers,

            image_available_semaphores: sync_ojbects.image_available_semaphores,
            render_finished_semaphores: sync_ojbects.render_finished_semaphores,
            in_flight_fences: sync_ojbects.inflight_fences,
            current_frame: 0,

            is_framebuffer_resized: false,

            #[cfg(feature = "hot-reload")]
            hot_reload: hot_reload::HotReload::try_new(&hot_reload::engine_spv_dir()),
        };

        // Built-in materials. Both run depth-disabled — the render pass now
        // always has a depth attachment, but Pong layered everything via
        // material registration order before depth existed and games still
        // depend on that contract for HUDs / overlays. Order matters: solid
        // registers first, so its draws land before textured ones in
        // `draw_frame`.
        let solid_handle = gm.register_material(&MaterialDesc {
            vertex_spv: include_bytes!("../shaders/spv/main.vert.spv"),
            fragment_spv: include_bytes!("../shaders/spv/main.frag.spv"),
            vertex_attrs: &[VertexAttr::F32x2],
            instance_attrs: &[VertexAttr::Mat4, VertexAttr::F32x3],
            bindings: &[Binding::CameraUbo(0)],
            depth: DepthMode::Disabled,
            blend: BlendMode::Opaque,
        });
        gm.solid_material_handle = solid_handle;

        let textured_handle = gm.register_material(&MaterialDesc {
            vertex_spv: include_bytes!("../shaders/spv/textured.vert.spv"),
            fragment_spv: include_bytes!("../shaders/spv/textured.frag.spv"),
            vertex_attrs: &[VertexAttr::F32x2],
            instance_attrs: &[
                VertexAttr::Mat4,
                VertexAttr::F32x2,
                VertexAttr::F32x2,
            ],
            bindings: &[Binding::CameraUbo(0), Binding::Sampler2d],
            depth: DepthMode::Disabled,
            blend: BlendMode::Opaque,
        });
        gm.textured_material_handle = textured_handle;

        // Register the built-in shader paths with the hot-reload watcher so
        // edits → `compile-shaders.sh` → SPV bytes on disk → pipeline rebuild
        // round-trip without restarting the binary.
        #[cfg(feature = "hot-reload")]
        if let Some(hr) = &mut gm.hot_reload {
            let dir = hot_reload::engine_spv_dir();
            hr.register(&dir.join("main.vert.spv"), solid_handle, hot_reload::ShaderStage::Vertex);
            hr.register(&dir.join("main.frag.spv"), solid_handle, hot_reload::ShaderStage::Fragment);
            hr.register(
                &dir.join("textured.vert.spv"),
                textured_handle,
                hot_reload::ShaderStage::Vertex,
            );
            hr.register(
                &dir.join("textured.frag.spv"),
                textured_handle,
                hot_reload::ShaderStage::Fragment,
            );
        }

        // Persist the cache now: pipeline creation above is what populates it
        // (compiling shaders on a cold start, or filling in any state combos
        // missing on a warm start). The engine bridges winit 0.30's
        // `Result`-returning `run_app` via `process::exit` to keep `App::run() -> !`,
        // so `Drop` is not a reliable place to save — writing here guarantees
        // the disk copy is up to date for next launch.
        save_pipeline_cache(&gm.device, gm.pipeline_cache);

        gm
    }

    pub fn window_request_redraw(&mut self) {
        self.window.request_redraw();
    }

    pub fn device_wait_idle(&mut self) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed to wait device idle!")
        };
    }

    /// Current swapchain extent in physical pixels (`[width, height]`).
    /// Useful for converting between window-space and world-space coordinates
    /// (e.g. mouse hit-testing via [`crate::camera::Camera2D::screen_to_world`]).
    pub fn extent(&self) -> [u32; 2] {
        [self.swapchain_extent.width, self.swapchain_extent.height]
    }

    /// Built-in solid-colour material handle. Per-instance extra payload is
    /// `[f32; 3]` (the colour).
    pub fn solid_material(&self) -> MaterialHandle {
        self.solid_material_handle
    }

    /// Built-in textured material handle. Per-instance extra payload is
    /// `[f32; 2] uv_offset` followed by `[f32; 2] uv_scale` (16 bytes total).
    pub fn textured_material(&self) -> MaterialHandle {
        self.textured_material_handle
    }
}


impl Drop for GraphicsManager {
    fn drop(&mut self) {
        unsafe {
            for i in 0..MAX_FRAMES_IN_FLIGHT {
                self.device
                    .destroy_semaphore(self.image_available_semaphores[i], None);
                self.device.destroy_fence(self.in_flight_fences[i], None);
            }
            for &s in self.render_finished_semaphores.iter() {
                self.device.destroy_semaphore(s, None);
            }

            self.cleanup_swapchain();
            self.device.destroy_image_view(self.depth_image_view, None);
            self.device.destroy_image(self.depth_image, None);
            self.device.free_memory(self.depth_image_memory, None);
            self.device.destroy_render_pass(self.render_pass, None);

            save_pipeline_cache(&self.device, self.pipeline_cache);
            self.device.destroy_pipeline_cache(self.pipeline_cache, None);

            self.device
                .free_command_buffers(self.command_pool, &self.command_buffers);

            for mesh in self.meshes.values() {
                destroy_mesh(&self.device, mesh);
            }
            for tex in self.textures.values() {
                destroy_texture_resources(&self.device, tex);
            }

            // Tear down each registered material: pipeline, layout,
            // descriptor-set layout, per-frame instance buffers. Descriptor
            // sets allocated from `descriptor_pool` (camera + per-texture) get
            // freed implicitly when the pool is destroyed.
            for mat in self.materials.values() {
                self.device.destroy_pipeline(mat.pipeline, None);
                self.device
                    .destroy_pipeline_layout(mat.pipeline_layout, None);
                self.device
                    .destroy_descriptor_set_layout(mat.descriptor_set_layout, None);
                for &b in mat.instance_buffers.iter() {
                    self.device.destroy_buffer(b, None);
                }
                for &m in mat.instance_buffer_memories.iter() {
                    self.device.free_memory(m, None);
                }
            }

            self.device
                .destroy_buffer(self.textured_quad_index_buffer, None);
            self.device
                .free_memory(self.textured_quad_index_memory, None);
            self.device
                .destroy_buffer(self.textured_quad_vertex_buffer, None);
            self.device
                .free_memory(self.textured_quad_vertex_memory, None);

            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            // Tear down per-slot camera UBO buffers + memories. The descriptor
            // sets allocated against these were already freed when the
            // descriptor pool went away.
            for buffers in self.camera_uniform_buffers.values() {
                for &b in buffers {
                    self.device.destroy_buffer(b, None);
                }
            }
            for memories in self.camera_uniform_buffers_memory.values() {
                for &m in memories {
                    self.device.free_memory(m, None);
                }
            }

            self.device.destroy_command_pool(self.command_pool, None);

            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);

            if VALIDATION.is_enable {
                self.debug_utils_loader
                    .destroy_debug_utils_messenger(self.debug_merssager, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

// Pipeline cache persisted to disk so the driver can skip shader compilation
// on subsequent launches. Vulkan validates the cache header (vendor/device/UUID)
// and ignores incompatible blobs, so a best-effort load is safe.
fn pipeline_cache_path() -> Option<std::path::PathBuf> {
    dirs::data_dir().map(|d| d.join("pong-rust").join("pipeline.cache"))
}

fn create_pipeline_cache(device: &ash::Device) -> vk::PipelineCache {
    let initial = pipeline_cache_path()
        .and_then(|p| std::fs::read(p).ok())
        .unwrap_or_default();
    let create_info = vk::PipelineCacheCreateInfo::default().initial_data(&initial);
    unsafe {
        device
            .create_pipeline_cache(&create_info, None)
            .expect("Failed to create pipeline cache")
    }
}

fn save_pipeline_cache(device: &ash::Device, cache: vk::PipelineCache) {
    let Some(path) = pipeline_cache_path() else { return };
    let data = match unsafe { device.get_pipeline_cache_data(cache) } {
        Ok(data) => data,
        Err(_) => return,
    };
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(path, data);
}
