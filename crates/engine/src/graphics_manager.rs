pub mod constants;
pub mod debug;
#[cfg(feature = "hot-reload")]
pub mod hot_reload;
pub mod material;
pub mod share;
pub mod structures;
pub mod tools;
pub mod window;

use glam::Mat4;
use constants::*;
use structures::{QueueFamilyIndices, SurfaceStuff};

use crate::camera::Camera;

use ash::vk;

use std::collections::HashMap;
use std::ptr;

pub use self::material::{Binding, DepthMode, MaterialDesc, MaterialHandle, VertexAttr};
use self::material::{
    allocate_camera_descriptor_sets, allocate_textured_descriptor_sets,
    create_descriptor_set_layout as create_material_descriptor_set_layout,
    create_pipeline as create_material_pipeline, Material, MAX_INSTANCES_PER_MATERIAL,
};
use self::structures::{ModelMesh, UniformBufferObject};

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

fn write_camera_ubos(
    device: &ash::Device,
    memories: &[vk::DeviceMemory],
    view: Mat4,
    proj: Mat4,
) {
    let ubo = UniformBufferObject { view, proj };
    let buffer_size = ::std::mem::size_of::<UniformBufferObject>() as u64;
    for &memory in memories {
        unsafe {
            let data_ptr = device
                .map_memory(memory, 0, buffer_size, vk::MemoryMapFlags::empty())
                .expect("Failed to Map Memory") as *mut UniformBufferObject;
            data_ptr.copy_from_nonoverlapping(&ubo, 1);
            device.unmap_memory(memory);
        }
    }
}

/// Cap on simultaneously-registered materials. Bounds the descriptor pool
/// sizing in [`share::create_descriptor_pool`].
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
    camera_uniform_buffers: Vec<vk::Buffer>,
    camera_uniform_buffers_memory: Vec<vk::DeviceMemory>,
    descriptor_pool: vk::DescriptorPool,
    last_camera_view: Option<Mat4>,
    last_camera_proj: Option<Mat4>,

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
        let instance = share::create_instance(
            &entry,
            &window,
            WINDOW_TITLE,
            VALIDATION.is_enable,
            &VALIDATION.required_validation_layers,
        );
        let surface_stuff = share::create_surface(&entry, &instance, &window);
        let (debug_utils_loader, debug_merssager) =
            debug::setup_debug_utils(VALIDATION.is_enable, &entry, &instance);
        let physical_device =
            share::pick_physical_device(&instance, &surface_stuff, &DEVICE_EXTENSIONS);
        let physical_device_memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let (device, queue_family) = share::create_logical_device(
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
        let swapchain_stuff = share::create_swapchain(
            &instance,
            &device,
            physical_device,
            &window,
            &surface_stuff,
            &queue_family,
        );
        let swapchain_imageviews = share::create_image_views(
            &device,
            swapchain_stuff.swapchain_format,
            &swapchain_stuff.swapchain_images,
        );
        let depth_format = share::pick_depth_format(&instance, physical_device);
        let render_pass =
            share::create_render_pass(&device, swapchain_stuff.swapchain_format, depth_format);
        let pipeline_cache = create_pipeline_cache(&device);
        let (depth_image, depth_image_memory, depth_image_view) = share::create_depth_attachment(
            &device,
            &physical_device_memory_properties,
            depth_format,
            swapchain_stuff.swapchain_extent,
        );
        let swapchain_framebuffers = share::create_framebuffers(
            &device,
            render_pass,
            &swapchain_imageviews,
            depth_image_view,
            swapchain_stuff.swapchain_extent,
        );
        let command_pool = share::create_command_pool(&device, &queue_family);

        let swapchain_image_count = swapchain_stuff.swapchain_images.len();
        let (camera_uniform_buffers, camera_uniform_buffers_memory) = share::create_uniform_buffers(
            &device,
            &physical_device_memory_properties,
            swapchain_image_count,
        );
        let descriptor_pool = share::create_descriptor_pool(
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
            share::create_vertex_buffer(
                &device,
                &physical_device_memory_properties,
                command_pool,
                graphics_queue,
                &unit_quad_positions,
            );
        let (textured_quad_index_buffer, textured_quad_index_memory) =
            share::create_index_buffer(
                &device,
                &physical_device_memory_properties,
                command_pool,
                graphics_queue,
                &unit_quad_indices,
            );

        let command_buffers =
            share::allocate_command_buffers(&device, command_pool, MAX_FRAMES_IN_FLIGHT as u32);
        let sync_ojbects =
            share::create_sync_objects(&device, MAX_FRAMES_IN_FLIGHT, swapchain_image_count);

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
            last_camera_view: None,
            last_camera_proj: None,

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
            bindings: &[Binding::CameraUbo],
            depth: DepthMode::Disabled,
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
            bindings: &[Binding::CameraUbo, Binding::Sampler2d],
            depth: DepthMode::Disabled,
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

    /// Register a new material. The renderer copies the SPIR-V bytes and the
    /// attribute / binding lists; the caller's references are no longer needed
    /// after the call returns.
    ///
    /// The first instance attribute **must** be [`VertexAttr::Mat4`] — the
    /// engine writes the per-instance model matrix at that offset. Anything
    /// after it is material-defined and is supplied as raw bytes through
    /// [`Self::register_material_instance`].
    pub fn register_material(&mut self, desc: &MaterialDesc) -> MaterialHandle {
        assert!(
            !desc.instance_attrs.is_empty()
                && matches!(desc.instance_attrs[0], VertexAttr::Mat4),
            "register_material: first instance attribute must be VertexAttr::Mat4 (the model matrix)"
        );
        assert!(
            self.materials.len() < MAX_MATERIALS,
            "register_material: hit MAX_MATERIALS cap (bump in graphics_manager::MAX_MATERIALS)"
        );

        let handle = MaterialHandle(self.next_material_handle);
        self.next_material_handle += 1;

        let descriptor_set_layout =
            create_material_descriptor_set_layout(&self.device, desc.bindings);
        let (pipeline, pipeline_layout, vertex_stride, instance_stride) = create_material_pipeline(
            &self.device,
            self.render_pass,
            self.swapchain_extent,
            descriptor_set_layout,
            self.pipeline_cache,
            desc.vertex_spv,
            desc.fragment_spv,
            desc.vertex_attrs,
            desc.instance_attrs,
            desc.depth,
        );
        // Per-instance buffer: host-visible+coherent, rewritten each frame.
        let buffer_size =
            (MAX_INSTANCES_PER_MATERIAL * instance_stride as usize) as vk::DeviceSize;
        let mut instance_buffers = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        let mut instance_buffer_memories = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let (buffer, memory) = share::create_buffer(
                &self.device,
                buffer_size,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                &self.physical_device_memory_properties,
            );
            instance_buffers.push(buffer);
            instance_buffer_memories.push(memory);
        }

        let has_sampler = Material::has_sampler_binding(desc.bindings);
        // Pre-allocate camera-only descriptor sets for non-sampler materials
        // (the textured-batch materials get per-(material, texture) sets on
        // demand when a textured instance is first registered).
        let camera_descriptor_sets = if has_sampler {
            Vec::new()
        } else {
            allocate_camera_descriptor_sets(
                &self.device,
                self.descriptor_pool,
                descriptor_set_layout,
                &self.camera_uniform_buffers,
            )
        };

        // Mat4 is always first in the instance stream (64 bytes); anything
        // after it is the material-specific "extra" payload.
        let extra_size = instance_stride - 64;

        let _ = vertex_stride; // currently unused outside pipeline creation
        self.materials.insert(
            handle,
            Material {
                vertex_spv: desc.vertex_spv.to_vec(),
                fragment_spv: desc.fragment_spv.to_vec(),
                vertex_attrs: desc.vertex_attrs.to_vec(),
                instance_attrs: desc.instance_attrs.to_vec(),
                instance_stride,
                extra_size,
                depth: desc.depth,
                descriptor_set_layout,
                pipeline_layout,
                pipeline,
                instance_buffers,
                instance_buffer_memories,
                has_sampler,
                camera_descriptor_sets,
                texture_descriptor_sets: HashMap::new(),
            },
        );

        handle
    }

    pub(crate) fn register_mesh(&mut self, mesh: &ModelMesh) -> MeshHandle {
        let handle = MeshHandle(self.next_mesh_handle);
        self.next_mesh_handle += 1;
        // `mesh.vertex_bytes` is a raw byte buffer; the material that draws
        // against this mesh owns the layout via its `vertex_attrs`. Stride
        // isn't passed to the buffer creator (the GPU only sees raw bytes),
        // but the material's pipeline computes the binding stride from the
        // same `vertex_attrs` so the two agree by construction.
        let _ = mesh.vertex_stride;
        let (vertex_buffer, vertex_memory) = share::create_vertex_buffer(
            &self.device,
            &self.physical_device_memory_properties,
            self.command_pool,
            self.graphics_queue,
            &mesh.vertex_bytes,
        );
        let (index_buffer, index_memory) = share::create_index_buffer(
            &self.device,
            &self.physical_device_memory_properties,
            self.command_pool,
            self.graphics_queue,
            &mesh.indices,
        );
        self.meshes.insert(
            handle,
            MeshBuffers {
                vertex_buffer,
                vertex_memory,
                index_buffer,
                index_memory,
                index_count: mesh.indices.len() as u32,
            },
        );
        handle
    }

    /// Register an instance of a material. The variant is selected by
    /// `mesh` / `texture`:
    ///
    /// - Non-sampler material: pass `mesh = Some(_)`, `texture = None`.
    /// - Sampler material: pass `mesh = None`, `texture = Some(_)` (the
    ///   engine's shared unit quad is used as the geometry).
    ///
    /// `extra` must be exactly the size declared by the material's
    /// `instance_attrs` after the leading [`VertexAttr::Mat4`].
    pub fn register_material_instance(
        &mut self,
        material: MaterialHandle,
        mesh: Option<MeshHandle>,
        texture: Option<TextureHandle>,
        extra: &[u8],
    ) -> ModelHandle {
        let mat = self
            .materials
            .get_mut(&material)
            .expect("register_material_instance: unknown MaterialHandle");
        assert_eq!(
            extra.len() as u32,
            mat.extra_size,
            "register_material_instance: extra payload size mismatch"
        );
        if mat.has_sampler {
            assert!(
                texture.is_some(),
                "register_material_instance: sampler material requires a TextureHandle"
            );
        } else {
            assert!(
                mesh.is_some(),
                "register_material_instance: non-sampler material requires a MeshHandle"
            );
        }
        if let Some(m) = mesh {
            assert!(
                self.meshes.contains_key(&m),
                "register_material_instance: unknown MeshHandle"
            );
        }

        // Lazily allocate (material, texture) descriptor sets the first time
        // we see this pairing.
        if let Some(t) = texture {
            let tex = self
                .textures
                .get(&t)
                .expect("register_material_instance: unknown TextureHandle");
            if !mat.texture_descriptor_sets.contains_key(&t) {
                let sets = allocate_textured_descriptor_sets(
                    &self.device,
                    self.descriptor_pool,
                    mat.descriptor_set_layout,
                    &self.camera_uniform_buffers,
                    tex.view,
                    tex.sampler,
                );
                mat.texture_descriptor_sets.insert(t, sets);
            }
        }

        let handle = ModelHandle(self.next_handle);
        self.next_handle += 1;
        self.instances.insert(
            handle,
            MaterialInstanceData {
                material,
                mesh,
                texture,
                extra: extra.to_vec(),
                last_model: Mat4::IDENTITY,
            },
        );
        handle
    }

    /// Convenience wrapper for the built-in solid material. Equivalent to
    /// `register_material_instance(solid_material(), Some(mesh), None,
    /// bytemuck-cast(color))`.
    pub fn register_instance(&mut self, mesh: MeshHandle, color: [f32; 3]) -> ModelHandle {
        let bytes: [u8; 12] = unsafe { ::std::mem::transmute(color) };
        let material = self.solid_material_handle;
        self.register_material_instance(material, Some(mesh), None, &bytes)
    }

    /// Convenience wrapper for the built-in textured material.
    pub fn register_textured_instance(
        &mut self,
        texture: TextureHandle,
        uv_offset: [f32; 2],
        uv_scale: [f32; 2],
    ) -> ModelHandle {
        let mut bytes = [0u8; 16];
        bytes[0..8].copy_from_slice(unsafe {
            &::std::mem::transmute::<[f32; 2], [u8; 8]>(uv_offset)
        });
        bytes[8..16].copy_from_slice(unsafe {
            &::std::mem::transmute::<[f32; 2], [u8; 8]>(uv_scale)
        });
        let material = self.textured_material_handle;
        self.register_material_instance(material, None, Some(texture), &bytes)
    }

    pub(crate) fn unregister_mesh(&mut self, mesh: MeshHandle) {
        if let Some(buffers) = self.meshes.remove(&mesh) {
            unsafe {
                self.device
                    .device_wait_idle()
                    .expect("device_wait_idle failed in unregister_mesh");
            }
            destroy_mesh(&self.device, &buffers);
        }
    }

    pub fn unregister_instance(&mut self, handle: ModelHandle) {
        self.instances.remove(&handle);
    }

    /// Alias kept for source compatibility with pre-Phase-7 game code. Today
    /// every instance — solid or textured — lives in one `instances` map keyed
    /// by [`ModelHandle`], so the two unregister paths are identical.
    pub fn unregister_textured_instance(&mut self, handle: ModelHandle) {
        self.unregister_instance(handle);
    }

    pub(crate) fn register_texture(&mut self, png_bytes: &[u8]) -> TextureHandle {
        let handle = TextureHandle(self.next_texture_handle);
        self.next_texture_handle += 1;

        let (image, memory, _w, _h) = share::load_texture_image(
            &self.device,
            &self.physical_device_memory_properties,
            self.command_pool,
            self.graphics_queue,
            png_bytes,
        );
        let view = share::create_image_view(
            &self.device,
            image,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageAspectFlags::COLOR,
            1,
        );
        let sampler = share::create_texture_sampler(&self.device);

        self.textures.insert(
            handle,
            TextureResources { image, memory, view, sampler },
        );
        handle
    }

    pub(crate) fn register_texture_rgba(
        &mut self,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) -> TextureHandle {
        let handle = TextureHandle(self.next_texture_handle);
        self.next_texture_handle += 1;

        let (image, memory) = share::upload_rgba_image(
            &self.device,
            &self.physical_device_memory_properties,
            self.command_pool,
            self.graphics_queue,
            width,
            height,
            rgba,
        );
        let view = share::create_image_view(
            &self.device,
            image,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageAspectFlags::COLOR,
            1,
        );
        let sampler = share::create_texture_sampler(&self.device);

        self.textures.insert(
            handle,
            TextureResources { image, memory, view, sampler },
        );
        handle
    }

    pub(crate) fn unregister_texture(&mut self, handle: TextureHandle) {
        if let Some(tex) = self.textures.remove(&handle) {
            unsafe {
                self.device
                    .device_wait_idle()
                    .expect("device_wait_idle failed in unregister_texture");
            }
            // Every material that has allocated descriptor sets against this
            // texture must release them back to the pool.
            for mat in self.materials.values_mut() {
                if let Some(sets) = mat.texture_descriptor_sets.remove(&handle) {
                    if !sets.is_empty() {
                        unsafe {
                            let _ = self
                                .device
                                .free_descriptor_sets(self.descriptor_pool, &sets);
                        }
                    }
                }
            }
            destroy_texture_resources(&self.device, &tex);
        }
    }

    /// Push the active camera's matrices to the shared UBOs. Accepts any
    /// `&dyn Camera`; cache-checked so a static camera (Pong) only pays
    /// `device_wait_idle` once.
    pub fn set_camera(&mut self, camera: &dyn Camera) {
        let aspect =
            self.swapchain_extent.width as f32 / self.swapchain_extent.height as f32;
        let view = camera.view();
        let proj = camera.proj(aspect);
        if self.last_camera_view == Some(view) && self.last_camera_proj == Some(proj) {
            return;
        }
        unsafe {
            self.device
                .device_wait_idle()
                .expect("device_wait_idle failed in set_camera");
        }
        write_camera_ubos(
            &self.device,
            &self.camera_uniform_buffers_memory,
            view,
            proj,
        );
        self.last_camera_view = Some(view);
        self.last_camera_proj = Some(proj);
    }

    pub fn draw_frame(&mut self, transforms: &[(ModelHandle, Mat4)]) {
        #[cfg(feature = "hot-reload")]
        self.process_hot_reloads();

        let wait_fences = [self.in_flight_fences[self.current_frame]];

        unsafe {
            self.device
                .wait_for_fences(&wait_fences, true, std::u64::MAX)
                .expect("Failed to wait for Fence!");
        }

        let (image_index, _is_sub_optimal) = unsafe {
            let result = self.swapchain_loader.acquire_next_image(
                self.swapchain,
                std::u64::MAX,
                self.image_available_semaphores[self.current_frame],
                vk::Fence::null(),
            );
            match result {
                Ok(image_index) => image_index,
                Err(vk_result) => match vk_result {
                    vk::Result::ERROR_OUT_OF_DATE_KHR => {
                        self.recreate_swapchain();
                        return;
                    }
                    _ => panic!("Failed to acquire Swap Chain Image!"),
                },
            }
        };

        // A handle in `transforms` that isn't registered is a programming error → panic.
        // A registered handle missing from `transforms` keeps last frame's matrix.
        for (handle, transform) in transforms {
            match self.instances.get_mut(handle) {
                Some(inst) => inst.last_model = *transform,
                None => panic!("draw_frame received transform for unknown ModelHandle"),
            }
        }

        // Per-material batching. Iterate materials in registration order
        // (MaterialHandle order) — this preserves today's "solid first,
        // textured second" paint order because the engine registers them in
        // that order during `new`.
        let mut batches: Vec<MaterialBatch> = Vec::new();
        let mut material_handles: Vec<MaterialHandle> = self.materials.keys().copied().collect();
        material_handles.sort();

        for mh in material_handles {
            let mat = &self.materials[&mh];
            // Gather instances owned by this material.
            let mut owned: Vec<(ModelHandle, &MaterialInstanceData)> = self
                .instances
                .iter()
                .filter(|(_, ii)| ii.material == mh)
                .map(|(h, ii)| (*h, ii))
                .collect();
            if owned.is_empty() {
                continue;
            }
            owned.sort_by_key(|(h, _)| *h);

            // Pack instance bytes (model + extra) contiguously, ordered by
            // group (mesh for non-sampler, texture for sampler).
            let mut packed: Vec<u8> = Vec::with_capacity(owned.len() * mat.instance_stride as usize);
            let mut draws: Vec<MaterialDraw> = Vec::new();

            if mat.has_sampler {
                // Group by texture.
                let mut by_texture: HashMap<TextureHandle, Vec<(ModelHandle, &MaterialInstanceData)>> =
                    HashMap::new();
                for (h, ii) in owned.iter() {
                    if let Some(t) = ii.texture {
                        by_texture.entry(t).or_default().push((*h, ii));
                    }
                }
                let mut tex_handles: Vec<TextureHandle> = by_texture.keys().copied().collect();
                tex_handles.sort_by_key(|t| t.0);
                for th in tex_handles {
                    let group = by_texture.get_mut(&th).unwrap();
                    group.sort_by_key(|(h, _)| *h);
                    let offset_bytes = packed.len() as u64;
                    let count = group.len() as u32;
                    for (_, ii) in group.iter() {
                        append_instance_bytes(&mut packed, mat.instance_stride, ii);
                    }
                    let sets = mat
                        .texture_descriptor_sets
                        .get(&th)
                        .expect("draw_frame: missing per-texture descriptor sets");
                    draws.push(MaterialDraw {
                        descriptor_set: sets[image_index as usize],
                        mesh_vertex_buffer: self.textured_quad_vertex_buffer,
                        mesh_index_buffer: self.textured_quad_index_buffer,
                        mesh_index_count: self.textured_quad_index_count,
                        instance_offset: offset_bytes,
                        instance_count: count,
                    });
                }
            } else {
                // Group by mesh.
                let mut by_mesh: HashMap<MeshHandle, Vec<(ModelHandle, &MaterialInstanceData)>> =
                    HashMap::new();
                for (h, ii) in owned.iter() {
                    if let Some(m) = ii.mesh {
                        by_mesh.entry(m).or_default().push((*h, ii));
                    }
                }
                let mut mesh_handles: Vec<MeshHandle> = by_mesh.keys().copied().collect();
                mesh_handles.sort();
                let camera_set = mat.camera_descriptor_sets[image_index as usize];
                for mhk in mesh_handles {
                    let group = by_mesh.get_mut(&mhk).unwrap();
                    group.sort_by_key(|(h, _)| *h);
                    let mesh = &self.meshes[&mhk];
                    let offset_bytes = packed.len() as u64;
                    let count = group.len() as u32;
                    for (_, ii) in group.iter() {
                        append_instance_bytes(&mut packed, mat.instance_stride, ii);
                    }
                    draws.push(MaterialDraw {
                        descriptor_set: camera_set,
                        mesh_vertex_buffer: mesh.vertex_buffer,
                        mesh_index_buffer: mesh.index_buffer,
                        mesh_index_count: mesh.index_count,
                        instance_offset: offset_bytes,
                        instance_count: count,
                    });
                }
            }

            if !packed.is_empty() {
                let buffer_size = packed.len() as vk::DeviceSize;
                unsafe {
                    let data_ptr = self
                        .device
                        .map_memory(
                            mat.instance_buffer_memories[self.current_frame],
                            0,
                            buffer_size,
                            vk::MemoryMapFlags::empty(),
                        )
                        .expect("Failed to map material instance buffer memory")
                        as *mut u8;
                    data_ptr.copy_from_nonoverlapping(packed.as_ptr(), packed.len());
                    self.device
                        .unmap_memory(mat.instance_buffer_memories[self.current_frame]);
                }
            }

            batches.push(MaterialBatch {
                pipeline: mat.pipeline,
                pipeline_layout: mat.pipeline_layout,
                instance_buffer: mat.instance_buffers[self.current_frame],
                draws,
            });
        }

        let command_buffer = self.command_buffers[self.current_frame];
        unsafe {
            self.device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .expect("Failed to reset command buffer");
        }
        // Draw order is load-bearing: the render pass has no depth attachment
        // and the pipelines don't blend, so command order *is* paint order.
        // Materials are iterated in registration order so that game code can
        // control overdraw layering by the order it registers materials in.
        record_material_command_buffer(
            &self.device,
            command_buffer,
            self.swapchain_framebuffers[image_index as usize],
            self.render_pass,
            self.swapchain_extent,
            &batches,
        );

        let wait_semaphores = [self.image_available_semaphores[self.current_frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [self.render_finished_semaphores[image_index as usize]];

        let submit_infos = [vk::SubmitInfo {
            s_type: vk::StructureType::SUBMIT_INFO,
            p_next: ptr::null(),
            wait_semaphore_count: wait_semaphores.len() as u32,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            p_wait_dst_stage_mask: wait_stages.as_ptr(),
            command_buffer_count: 1,
            p_command_buffers: &command_buffer,
            signal_semaphore_count: signal_semaphores.len() as u32,
            p_signal_semaphores: signal_semaphores.as_ptr(),
            ..Default::default()
        }];

        unsafe {
            self.device
                .reset_fences(&wait_fences)
                .expect("Failed to reset Fence!");

            self.device
                .queue_submit(
                    self.graphics_queue,
                    &submit_infos,
                    self.in_flight_fences[self.current_frame],
                )
                .expect("Failed to execute queue submit.");
        }

        let swapchains = [self.swapchain];

        let present_info = vk::PresentInfoKHR {
            s_type: vk::StructureType::PRESENT_INFO_KHR,
            p_next: ptr::null(),
            wait_semaphore_count: 1,
            p_wait_semaphores: signal_semaphores.as_ptr(),
            swapchain_count: 1,
            p_swapchains: swapchains.as_ptr(),
            p_image_indices: &image_index,
            p_results: ptr::null_mut(),
            ..Default::default()
        };

        let result = unsafe {
            self.swapchain_loader
                .queue_present(self.present_queue, &present_info)
        };

        let is_resized = match result {
            Ok(_) => self.is_framebuffer_resized,
            Err(vk_result) => match vk_result {
                vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR => true,
                _ => panic!("Failed to execute queue present."),
            },
        };
        if is_resized {
            self.is_framebuffer_resized = false;
            self.recreate_swapchain();
        }

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    fn recreate_swapchain(&mut self) {
        let surface_suff = SurfaceStuff {
            surface_loader: self.surface_loader.clone(),
            surface: self.surface,
        };

        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed to wait device idle!")
        };
        self.cleanup_swapchain();

        let swapchain_stuff = share::create_swapchain(
            &self.instance,
            &self.device,
            self.physical_device,
            &self.window,
            &surface_suff,
            &self.queue_family,
        );
        self.swapchain_loader = swapchain_stuff.swapchain_loader;
        self.swapchain = swapchain_stuff.swapchain;
        self.swapchain_images = swapchain_stuff.swapchain_images;
        let new_format = swapchain_stuff.swapchain_format;
        let format_changed = new_format != self.swapchain_format;
        self.swapchain_format = new_format;
        self.swapchain_extent = swapchain_stuff.swapchain_extent;

        let new_image_count = self.swapchain_images.len();
        if new_image_count != self.render_finished_semaphores.len() {
            unsafe {
                for &s in self.render_finished_semaphores.iter() {
                    self.device.destroy_semaphore(s, None);
                }
            }
            self.render_finished_semaphores =
                share::create_render_finished_semaphores(&self.device, new_image_count);
        }

        self.last_camera_view = None;
        self.last_camera_proj = None;

        self.swapchain_imageviews =
            share::create_image_views(&self.device, self.swapchain_format, &self.swapchain_images);
        if format_changed {
            unsafe {
                self.device.destroy_render_pass(self.render_pass, None);
            }
            self.render_pass = share::create_render_pass(
                &self.device,
                self.swapchain_format,
                self.depth_format,
            );
        }

        // Rebuild the depth attachment to match the new swapchain extent.
        unsafe {
            self.device.destroy_image_view(self.depth_image_view, None);
            self.device.destroy_image(self.depth_image, None);
            self.device.free_memory(self.depth_image_memory, None);
        }
        let (depth_image, depth_image_memory, depth_image_view) = share::create_depth_attachment(
            &self.device,
            &self.physical_device_memory_properties,
            self.depth_format,
            self.swapchain_extent,
        );
        self.depth_image = depth_image;
        self.depth_image_memory = depth_image_memory;
        self.depth_image_view = depth_image_view;

        // Rebuild every material's pipeline (viewport/scissor are baked in,
        // so a swapchain-extent change invalidates them). Descriptor set
        // layouts and instance buffers survive — only the pipeline + pipeline
        // layout are recreated.
        for mat in self.materials.values_mut() {
            unsafe {
                self.device.destroy_pipeline(mat.pipeline, None);
                self.device
                    .destroy_pipeline_layout(mat.pipeline_layout, None);
            }
            let (pipeline, pipeline_layout, _vs, _is) = create_material_pipeline(
                &self.device,
                self.render_pass,
                self.swapchain_extent,
                mat.descriptor_set_layout,
                self.pipeline_cache,
                &mat.vertex_spv,
                &mat.fragment_spv,
                &mat.vertex_attrs,
                &mat.instance_attrs,
                mat.depth,
            );
            mat.pipeline = pipeline;
            mat.pipeline_layout = pipeline_layout;
        }

        self.swapchain_framebuffers = share::create_framebuffers(
            &self.device,
            self.render_pass,
            &self.swapchain_imageviews,
            self.depth_image_view,
            self.swapchain_extent,
        );
    }

    fn cleanup_swapchain(&self) {
        unsafe {
            for &framebuffer in self.swapchain_framebuffers.iter() {
                self.device.destroy_framebuffer(framebuffer, None);
            }
            for &image_view in self.swapchain_imageviews.iter() {
                self.device.destroy_image_view(image_view, None);
            }
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
        }
    }
}

/// Internal per-material draw record produced inside `draw_frame`.
struct MaterialBatch {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    instance_buffer: vk::Buffer,
    draws: Vec<MaterialDraw>,
}

struct MaterialDraw {
    descriptor_set: vk::DescriptorSet,
    mesh_vertex_buffer: vk::Buffer,
    mesh_index_buffer: vk::Buffer,
    mesh_index_count: u32,
    instance_offset: u64,
    instance_count: u32,
}

fn append_instance_bytes(out: &mut Vec<u8>, instance_stride: u32, inst: &MaterialInstanceData) {
    let stride = instance_stride as usize;
    let start = out.len();
    out.resize(start + stride, 0);
    let dst = &mut out[start..start + stride];
    // Model matrix: 64 bytes at offset 0. glam's Mat4 is #[repr(C)] of 16 f32
    // columns; a raw byte copy is layout-equivalent to the previous typed copy.
    let m_bytes: &[u8] = unsafe {
        ::std::slice::from_raw_parts(
            (&inst.last_model as *const Mat4) as *const u8,
            ::std::mem::size_of::<Mat4>(),
        )
    };
    dst[0..64].copy_from_slice(m_bytes);
    if !inst.extra.is_empty() {
        dst[64..64 + inst.extra.len()].copy_from_slice(&inst.extra);
    }
}

fn record_material_command_buffer(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    framebuffer: vk::Framebuffer,
    render_pass: vk::RenderPass,
    surface_extent: vk::Extent2D,
    batches: &[MaterialBatch],
) {
    let begin_info = vk::CommandBufferBeginInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
        p_next: ptr::null(),
        p_inheritance_info: ptr::null(),
        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        ..Default::default()
    };

    // Attachment order must match `share::create_render_pass`: colour at 0,
    // depth at 1.
    let clear_values = [
        vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        },
        vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        },
    ];

    let render_pass_begin_info = vk::RenderPassBeginInfo {
        s_type: vk::StructureType::RENDER_PASS_BEGIN_INFO,
        p_next: ptr::null(),
        render_pass,
        framebuffer,
        render_area: vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: surface_extent,
        },
        clear_value_count: clear_values.len() as u32,
        p_clear_values: clear_values.as_ptr(),
        ..Default::default()
    };

    unsafe {
        device
            .begin_command_buffer(command_buffer, &begin_info)
            .expect("Failed to begin recording Command Buffer!");
        device.cmd_begin_render_pass(
            command_buffer,
            &render_pass_begin_info,
            vk::SubpassContents::INLINE,
        );

        for batch in batches.iter() {
            if batch.draws.is_empty() {
                continue;
            }
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                batch.pipeline,
            );
            let mut current_set = vk::DescriptorSet::null();
            for d in batch.draws.iter() {
                if d.descriptor_set != current_set {
                    let sets = [d.descriptor_set];
                    device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        batch.pipeline_layout,
                        0,
                        &sets,
                        &[],
                    );
                    current_set = d.descriptor_set;
                }
                let vertex_buffers = [d.mesh_vertex_buffer, batch.instance_buffer];
                let offsets = [0_u64, d.instance_offset];
                device.cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &offsets);
                device.cmd_bind_index_buffer(
                    command_buffer,
                    d.mesh_index_buffer,
                    0,
                    vk::IndexType::UINT32,
                );
                device.cmd_draw_indexed(
                    command_buffer,
                    d.mesh_index_count,
                    d.instance_count,
                    0,
                    0,
                    0,
                );
            }
        }

        device.cmd_end_render_pass(command_buffer);

        device
            .end_command_buffer(command_buffer)
            .expect("Failed to record Command Buffer at Ending!");
    }
}

#[cfg(feature = "hot-reload")]
impl GraphicsManager {
    /// Drain the hot-reload watcher and rebuild any built-in material pipeline
    /// whose SPV bytes changed on disk. Called from `draw_frame` once per
    /// frame. Errors (missing file, IO error, Vulkan pipeline-creation
    /// failure) are logged to stderr and leave the old pipeline running.
    fn process_hot_reloads(&mut self) {
        let Some(hr) = self.hot_reload.as_mut() else {
            return;
        };
        let changes = hr.drain();
        if changes.is_empty() {
            return;
        }
        for (handle, stage, path) in changes {
            let new_bytes = match std::fs::read(&path) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("hot-reload: read {} failed: {e}", path.display());
                    continue;
                }
            };
            let Some(mat) = self.materials.get_mut(&handle) else {
                continue;
            };
            // Stash old SPV so we can roll back on Vulkan failure.
            let backup = match stage {
                hot_reload::ShaderStage::Vertex => {
                    std::mem::replace(&mut mat.vertex_spv, new_bytes)
                }
                hot_reload::ShaderStage::Fragment => {
                    std::mem::replace(&mut mat.fragment_spv, new_bytes)
                }
            };
            // Build new pipeline against the same descriptor-set layout and
            // current swapchain extent. `material::create_pipeline` is the
            // same code path `recreate_swapchain` uses.
            //
            // Wait for the GPU to be idle before destroying the old pipeline:
            // a frame already in flight may still be bound to it.
            unsafe {
                self.device
                    .device_wait_idle()
                    .expect("device_wait_idle failed in process_hot_reloads");
            }
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                create_material_pipeline(
                    &self.device,
                    self.render_pass,
                    self.swapchain_extent,
                    mat.descriptor_set_layout,
                    self.pipeline_cache,
                    &mat.vertex_spv,
                    &mat.fragment_spv,
                    &mat.vertex_attrs,
                    &mat.instance_attrs,
                    mat.depth,
                )
            }));
            match result {
                Ok((pipeline, pipeline_layout, _vs, _is)) => {
                    unsafe {
                        self.device.destroy_pipeline(mat.pipeline, None);
                        self.device
                            .destroy_pipeline_layout(mat.pipeline_layout, None);
                    }
                    mat.pipeline = pipeline;
                    mat.pipeline_layout = pipeline_layout;
                    eprintln!("hot-reload: reloaded {}", path.display());
                }
                Err(_) => {
                    // Pipeline creation panicked (e.g. bad SPV header). Roll
                    // back the SPV bytes so the next save can be detected as
                    // a real change.
                    match stage {
                        hot_reload::ShaderStage::Vertex => mat.vertex_spv = backup,
                        hot_reload::ShaderStage::Fragment => mat.fragment_spv = backup,
                    }
                    eprintln!(
                        "hot-reload: pipeline rebuild failed for {} — kept old pipeline",
                        path.display()
                    );
                }
            }
        }
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
            for i in 0..self.camera_uniform_buffers.len() {
                self.device
                    .destroy_buffer(self.camera_uniform_buffers[i], None);
                self.device
                    .free_memory(self.camera_uniform_buffers_memory[i], None);
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
