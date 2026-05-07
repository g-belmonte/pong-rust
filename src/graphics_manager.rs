pub mod constants;
pub mod debug;
pub mod fps_limiter;
pub mod platforms;
pub mod share;
pub mod structures;
pub mod tools;
pub mod window;

use cgmath::Deg;
use cgmath::Matrix4;
use cgmath::Point3;
use cgmath::Vector3;
use constants::*;
use structures::{QueueFamilyIndices, SurfaceStuff};

use ash::version::DeviceV1_0;
use ash::version::InstanceV1_0;
use ash::vk;

use std::collections::HashMap;
use std::ptr;

use self::structures::{
    Instance, ModelMesh, TexturedInstance, TexturedVertex, UniformBufferObject,
};

// Two distinct handle types: MeshHandle identifies a piece of geometry that
// can be reused across many instances; ModelHandle identifies a single
// drawable (an instance of a mesh, or a textured model). Solid-colour
// rendering goes through the mesh+instance path; textured models stay
// per-model because each owns its own descriptor set.
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
    // One descriptor set per swapchain image (camera UBO[i] + this texture's
    // view+sampler). Shared across every textured instance that uses this
    // texture — that's the whole point of the textured-instance refactor.
    descriptor_sets: Vec<vk::DescriptorSet>,
}

struct MeshBuffers {
    vertex_buffer: vk::Buffer,
    vertex_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_memory: vk::DeviceMemory,
    index_count: u32,
}

struct InstanceData {
    mesh: MeshHandle,
    color: [f32; 3],
    last_model: Matrix4<f32>,
}

struct TexturedInstanceData {
    texture: TextureHandle,
    uv_offset: [f32; 2],
    uv_scale: [f32; 2],
    last_model: Matrix4<f32>,
}

fn destroy_mesh(device: &ash::Device, mesh: &MeshBuffers) {
    unsafe {
        device.destroy_buffer(mesh.index_buffer, None);
        device.free_memory(mesh.index_memory, None);
        device.destroy_buffer(mesh.vertex_buffer, None);
        device.free_memory(mesh.vertex_memory, None);
    }
}

fn destroy_texture_resources(
    device: &ash::Device,
    descriptor_pool: vk::DescriptorPool,
    tex: &TextureResources,
) {
    unsafe {
        // Pool was created with FREE_DESCRIPTOR_SET so this is legal.
        if !tex.descriptor_sets.is_empty() {
            device.free_descriptor_sets(descriptor_pool, &tex.descriptor_sets);
        }
        device.destroy_sampler(tex.sampler, None);
        device.destroy_image_view(tex.view, None);
        device.destroy_image(tex.image, None);
        device.free_memory(tex.memory, None);
    }
}

fn write_camera_ubos(
    device: &ash::Device,
    memories: &[vk::DeviceMemory],
    view: Matrix4<f32>,
    proj: Matrix4<f32>,
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

/// Field groups below mark which state survives `recreate_swapchain` and
/// which is rebuilt. Changing this layout means revisiting that method.
pub struct GraphicsManager {
    // ----- Window -----
    window: winit::window::Window,

    // ----- Vulkan core (created once, destroyed at shutdown) -----
    _entry: ash::Entry,
    instance: ash::Instance,
    surface_loader: ash::extensions::khr::Surface,
    surface: vk::SurfaceKHR,
    debug_utils_loader: ash::extensions::ext::DebugUtils,
    debug_merssager: vk::DebugUtilsMessengerEXT,
    physical_device: vk::PhysicalDevice,
    physical_device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    device: ash::Device,
    queue_family: QueueFamilyIndices,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,

    // ----- Swapchain + dependent state (rebuilt on resize / OUT_OF_DATE) -----
    swapchain_loader: ash::extensions::khr::Swapchain,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain_imageviews: Vec<vk::ImageView>,
    swapchain_framebuffers: Vec<vk::Framebuffer>,

    // ----- Pipelines + render pass (rebuilt on swapchain recreation) -----
    render_pass: vk::RenderPass,
    ubo_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    graphics_pipeline: vk::Pipeline,
    // Second pipeline for textured (sampled image) draws. Coexists with the
    // solid-colour pipeline in the same render pass.
    textured_ubo_layout: vk::DescriptorSetLayout,
    textured_pipeline_layout: vk::PipelineLayout,
    textured_pipeline: vk::Pipeline,
    // Persistent across runs: loaded from / saved to disk so the driver can
    // skip shader compilation on subsequent launches.
    pipeline_cache: vk::PipelineCache,

    // ----- Camera UBO + descriptor pool/sets (survive swapchain recreation) -----
    // Camera UBO `(view, proj)` is shared across all models — one buffer per
    // swapchain image, one descriptor set per swapchain image. Contents are
    // rewritten only when the camera changes (init, swapchain recreation).
    camera_uniform_buffers: Vec<vk::Buffer>,
    camera_uniform_buffers_memory: Vec<vk::DeviceMemory>,
    descriptor_pool: vk::DescriptorPool,
    camera_descriptor_sets: Vec<vk::DescriptorSet>,

    // ----- Registered geometry, instances, and textures (survive recreation) -----
    meshes: HashMap<MeshHandle, MeshBuffers>,
    next_mesh_handle: u32,
    instances: HashMap<ModelHandle, InstanceData>,
    textured_instances: HashMap<ModelHandle, TexturedInstanceData>,
    next_handle: u32,
    // Texture lifetime is independent of model lifetime — multiple textured
    // models may share one texture (e.g. a font atlas with one glyph per quad).
    textures: HashMap<TextureHandle, TextureResources>,
    next_texture_handle: u32,

    // ----- Per-frame instance buffers (survive recreation) -----
    // Per-frame solid-colour instance buffer (host-visible + coherent). Layout
    // each frame is "all instances of mesh A, then all of mesh B, ..." so each
    // per-mesh instanced draw can bind it at the right offset.
    instance_buffers: Vec<vk::Buffer>,
    instance_buffer_memories: Vec<vk::DeviceMemory>,
    // Per-frame textured instance buffer, same pattern but for the textured
    // pipeline. Layout each frame is "all instances of texture A, then B, ...".
    textured_instance_buffers: Vec<vk::Buffer>,
    textured_instance_buffer_memories: Vec<vk::DeviceMemory>,

    // ----- Shared unit quad for textured instances (survives recreation) -----
    // Built once at construction. The vertex shader maps pos `[-0.5..0.5]^2`
    // to UV via the per-instance uv_offset/uv_scale, so any sprite size and
    // any UV rect is supported.
    textured_quad_vertex_buffer: vk::Buffer,
    textured_quad_vertex_memory: vk::DeviceMemory,
    textured_quad_index_buffer: vk::Buffer,
    textured_quad_index_memory: vk::DeviceMemory,
    textured_quad_index_count: u32,

    // ----- Camera matrices (survive recreation; proj is rewritten on resize) -----
    current_view: Matrix4<f32>,
    current_proj: Matrix4<f32>,

    // ----- Command buffers + sync (survive recreation) -----
    command_pool: vk::CommandPool,
    // MAX_FRAMES_IN_FLIGHT command buffers, indexed by current_frame. Re-recorded
    // each frame in draw_frame so per-instance/per-draw matrices reflect Scene state.
    command_buffers: Vec<vk::CommandBuffer>,
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    current_frame: usize,

    is_framebuffer_resized: bool,
}

impl GraphicsManager {
    pub fn new(event_loop: &winit::event_loop::EventLoop<()>) -> GraphicsManager {
        let window = window::init_window(event_loop, WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT);

        let entry = ash::Entry::new().unwrap();
        let instance = share::create_instance(
            &entry,
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
        let render_pass = share::create_render_pass(&device, swapchain_stuff.swapchain_format);
        let ubo_layout = share::create_descriptor_set_layout(&device);
        let textured_ubo_layout = share::create_textured_descriptor_set_layout(&device);
        let pipeline_cache = create_pipeline_cache(&device);
        let (graphics_pipeline, pipeline_layout) = share::create_graphics_pipeline(
            &device,
            render_pass,
            swapchain_stuff.swapchain_extent,
            ubo_layout,
            pipeline_cache,
        );
        let (textured_pipeline, textured_pipeline_layout) = share::create_textured_graphics_pipeline(
            &device,
            render_pass,
            swapchain_stuff.swapchain_extent,
            textured_ubo_layout,
            pipeline_cache,
        );
        // Persist the cache now: pipeline creation above is what populates it
        // (compiling shaders on a cold start, or filling in any state combos
        // missing on a warm start). winit 0.20's `EventLoop::run` is `-> !` and
        // skips destructors, so `Drop` is not a reliable place to save —
        // writing here guarantees the disk copy is up to date for next launch.
        save_pipeline_cache(&device, pipeline_cache);
        let swapchain_framebuffers = share::create_framebuffers(
            &device,
            render_pass,
            &swapchain_imageviews,
            swapchain_stuff.swapchain_extent,
        );
        let command_pool = share::create_command_pool(&device, &queue_family);

        // Camera lives in the renderer. View/proj are hardcoded; projection is recomputed
        // on swapchain recreation. Models are registered by Scene after `new()` returns.
        let current_view = Matrix4::look_at(
            Point3::new(0.0, 0.0, 10.0),
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        );
        let current_proj = cgmath::perspective(
            Deg(45.0),
            swapchain_stuff.swapchain_extent.width as f32
                / swapchain_stuff.swapchain_extent.height as f32,
            0.1,
            10.0,
        );

        let swapchain_image_count = swapchain_stuff.swapchain_images.len();
        let (camera_uniform_buffers, camera_uniform_buffers_memory) = share::create_uniform_buffers(
            &device,
            &physical_device_memory_properties,
            swapchain_image_count,
        );
        let descriptor_pool =
            share::create_descriptor_pool(&device, swapchain_image_count, MAX_TEXTURED_MODELS);
        let camera_descriptor_sets = share::create_descriptor_sets(
            &device,
            descriptor_pool,
            ubo_layout,
            &camera_uniform_buffers,
            swapchain_image_count,
        );

        write_camera_ubos(
            &device,
            &camera_uniform_buffers_memory,
            current_view,
            current_proj,
        );

        let (instance_buffers, instance_buffer_memories) = create_instance_buffers(
            &device,
            &physical_device_memory_properties,
            MAX_FRAMES_IN_FLIGHT,
        );

        let (textured_instance_buffers, textured_instance_buffer_memories) =
            create_textured_instance_buffers(
                &device,
                &physical_device_memory_properties,
                MAX_FRAMES_IN_FLIGHT,
            );

        // Shared unit-quad VBO/IBO. Front-facing winding matches the solid-colour
        // quads (TRIANGLE_LIST, CW: 0,1,2 / 2,3,0).
        let unit_quad_vertices: [TexturedVertex; 4] = [
            TexturedVertex { pos: [-0.5, -0.5] },
            TexturedVertex { pos: [ 0.5, -0.5] },
            TexturedVertex { pos: [ 0.5,  0.5] },
            TexturedVertex { pos: [-0.5,  0.5] },
        ];
        let unit_quad_indices: [u32; 6] = [0, 1, 2, 2, 3, 0];
        let (textured_quad_vertex_buffer, textured_quad_vertex_memory) =
            share::create_vertex_buffer(
                &device,
                &physical_device_memory_properties,
                command_pool,
                graphics_queue,
                &unit_quad_vertices,
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

        GraphicsManager {
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

            pipeline_layout,
            render_pass,
            graphics_pipeline,
            ubo_layout,
            textured_ubo_layout,
            textured_pipeline_layout,
            textured_pipeline,
            pipeline_cache,

            camera_uniform_buffers,
            camera_uniform_buffers_memory,
            descriptor_pool,
            camera_descriptor_sets,

            meshes: HashMap::new(),
            next_mesh_handle: 0,
            instances: HashMap::new(),
            textured_instances: HashMap::new(),
            next_handle: 0,
            instance_buffers,
            instance_buffer_memories,
            textured_instance_buffers,
            textured_instance_buffer_memories,
            textured_quad_vertex_buffer,
            textured_quad_vertex_memory,
            textured_quad_index_buffer,
            textured_quad_index_memory,
            textured_quad_index_count: unit_quad_indices.len() as u32,
            textures: HashMap::new(),
            next_texture_handle: 0,
            current_view,
            current_proj,

            command_pool,
            command_buffers,

            image_available_semaphores: sync_ojbects.image_available_semaphores,
            render_finished_semaphores: sync_ojbects.render_finished_semaphores,
            in_flight_fences: sync_ojbects.inflight_fences,
            current_frame: 0,

            is_framebuffer_resized: false,
        }
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

    pub fn register_mesh(&mut self, mesh: &ModelMesh) -> MeshHandle {
        let handle = MeshHandle(self.next_mesh_handle);
        self.next_mesh_handle += 1;
        let (vertex_buffer, vertex_memory) = share::create_vertex_buffer(
            &self.device,
            &self.physical_device_memory_properties,
            self.command_pool,
            self.graphics_queue,
            &mesh.vertices,
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

    pub fn register_instance(&mut self, mesh: MeshHandle, color: [f32; 3]) -> ModelHandle {
        assert!(
            self.meshes.contains_key(&mesh),
            "register_instance received unknown MeshHandle"
        );
        let handle = ModelHandle(self.next_handle);
        self.next_handle += 1;
        self.instances.insert(
            handle,
            InstanceData {
                mesh,
                color,
                last_model: Matrix4::from_scale(1.0),
            },
        );
        handle
    }

    #[allow(dead_code)]
    pub fn unregister_mesh(&mut self, mesh: MeshHandle) {
        if let Some(buffers) = self.meshes.remove(&mesh) {
            unsafe {
                self.device
                    .device_wait_idle()
                    .expect("device_wait_idle failed in unregister_mesh");
            }
            destroy_mesh(&self.device, &buffers);
        }
    }

    #[allow(dead_code)]
    pub fn unregister_instance(&mut self, handle: ModelHandle) {
        self.instances.remove(&handle);
    }

    pub fn register_texture(&mut self, png_bytes: &[u8]) -> TextureHandle {
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
        let descriptor_sets = share::create_textured_descriptor_sets(
            &self.device,
            self.descriptor_pool,
            self.textured_ubo_layout,
            &self.camera_uniform_buffers,
            view,
            sampler,
        );

        self.textures.insert(
            handle,
            TextureResources { image, memory, view, sampler, descriptor_sets },
        );
        handle
    }

    pub fn register_texture_rgba(
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
        let descriptor_sets = share::create_textured_descriptor_sets(
            &self.device,
            self.descriptor_pool,
            self.textured_ubo_layout,
            &self.camera_uniform_buffers,
            view,
            sampler,
        );

        self.textures.insert(
            handle,
            TextureResources { image, memory, view, sampler, descriptor_sets },
        );
        handle
    }

    #[allow(dead_code)]
    pub fn unregister_texture(&mut self, handle: TextureHandle) {
        if let Some(tex) = self.textures.remove(&handle) {
            unsafe {
                self.device
                    .device_wait_idle()
                    .expect("device_wait_idle failed in unregister_texture");
            }
            destroy_texture_resources(&self.device, self.descriptor_pool, &tex);
        }
    }

    pub fn register_textured_instance(
        &mut self,
        texture: TextureHandle,
        uv_offset: [f32; 2],
        uv_scale: [f32; 2],
    ) -> ModelHandle {
        assert!(
            self.textures.contains_key(&texture),
            "register_textured_instance received unknown TextureHandle"
        );
        let handle = ModelHandle(self.next_handle);
        self.next_handle += 1;
        self.textured_instances.insert(
            handle,
            TexturedInstanceData {
                texture,
                uv_offset,
                uv_scale,
                last_model: Matrix4::from_scale(1.0),
            },
        );
        handle
    }

    #[allow(dead_code)]
    pub fn unregister_textured_instance(&mut self, handle: ModelHandle) {
        self.textured_instances.remove(&handle);
    }

    pub fn draw_frame(&mut self, transforms: &[(ModelHandle, Matrix4<f32>)]) {
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
            if let Some(inst) = self.instances.get_mut(handle) {
                inst.last_model = *transform;
            } else if let Some(tinst) = self.textured_instances.get_mut(handle) {
                tinst.last_model = *transform;
            } else {
                panic!("draw_frame received transform for unknown ModelHandle");
            }
        }

        // SOLID-COLOUR: group instances by mesh and pack them into the per-frame
        // instance buffer. Iterate meshes in MeshHandle order so the layout is
        // deterministic; iterate instances inside each mesh by ModelHandle order.
        let mut instances_by_mesh: HashMap<MeshHandle, Vec<(ModelHandle, &InstanceData)>> =
            HashMap::new();
        for (h, inst) in self.instances.iter() {
            instances_by_mesh
                .entry(inst.mesh)
                .or_insert_with(Vec::new)
                .push((*h, inst));
        }

        let mut packed: Vec<Instance> = Vec::with_capacity(self.instances.len());
        let mut solid_draws: Vec<share::SolidDraw> = Vec::new();
        let mut mesh_handles: Vec<MeshHandle> = self.meshes.keys().copied().collect();
        mesh_handles.sort();
        for mh in mesh_handles {
            let group = match instances_by_mesh.get_mut(&mh) {
                Some(g) => g,
                None => continue,
            };
            group.sort_by_key(|(h, _)| *h);
            let mesh = &self.meshes[&mh];
            let offset_bytes = (packed.len() * ::std::mem::size_of::<Instance>()) as u64;
            let count = group.len() as u32;
            for (_, inst) in group.iter() {
                packed.push(Instance {
                    model: inst.last_model,
                    color: inst.color,
                });
            }
            solid_draws.push(share::SolidDraw {
                mesh_vertex_buffer: mesh.vertex_buffer,
                mesh_index_buffer: mesh.index_buffer,
                mesh_index_count: mesh.index_count,
                instance_buffer: self.instance_buffers[self.current_frame],
                instance_offset: offset_bytes,
                instance_count: count,
            });
        }

        if !packed.is_empty() {
            let buffer_size =
                (packed.len() * ::std::mem::size_of::<Instance>()) as vk::DeviceSize;
            unsafe {
                let data_ptr = self
                    .device
                    .map_memory(
                        self.instance_buffer_memories[self.current_frame],
                        0,
                        buffer_size,
                        vk::MemoryMapFlags::empty(),
                    )
                    .expect("Failed to map instance buffer memory")
                    as *mut Instance;
                data_ptr.copy_from_nonoverlapping(packed.as_ptr(), packed.len());
                self.device
                    .unmap_memory(self.instance_buffer_memories[self.current_frame]);
            }
        }

        // TEXTURED: same idea, grouped by texture. Iterate textures in
        // TextureHandle order; iterate instances inside each texture by
        // ModelHandle order.
        let mut tinstances_by_texture: HashMap<
            TextureHandle,
            Vec<(ModelHandle, &TexturedInstanceData)>,
        > = HashMap::new();
        for (h, ti) in self.textured_instances.iter() {
            tinstances_by_texture
                .entry(ti.texture)
                .or_insert_with(Vec::new)
                .push((*h, ti));
        }

        let mut tpacked: Vec<TexturedInstance> =
            Vec::with_capacity(self.textured_instances.len());
        let mut textured_draws: Vec<share::TexturedDraw> = Vec::new();
        let mut texture_handles: Vec<TextureHandle> =
            self.textures.keys().copied().collect();
        texture_handles.sort_by_key(|h| h.0);
        let camera_set = self.camera_descriptor_sets[image_index as usize];
        let _ = camera_set; // camera UBO is bound through each texture's descriptor set.
        for th in texture_handles {
            let group = match tinstances_by_texture.get_mut(&th) {
                Some(g) => g,
                None => continue,
            };
            group.sort_by_key(|(h, _)| *h);
            let tex = &self.textures[&th];
            let offset_bytes =
                (tpacked.len() * ::std::mem::size_of::<TexturedInstance>()) as u64;
            let count = group.len() as u32;
            for (_, ti) in group.iter() {
                tpacked.push(TexturedInstance {
                    model: ti.last_model,
                    uv_offset: ti.uv_offset,
                    uv_scale: ti.uv_scale,
                });
            }
            textured_draws.push(share::TexturedDraw {
                descriptor_set: tex.descriptor_sets[image_index as usize],
                instance_buffer: self.textured_instance_buffers[self.current_frame],
                instance_offset: offset_bytes,
                instance_count: count,
            });
        }

        if !tpacked.is_empty() {
            let buffer_size =
                (tpacked.len() * ::std::mem::size_of::<TexturedInstance>()) as vk::DeviceSize;
            unsafe {
                let data_ptr = self
                    .device
                    .map_memory(
                        self.textured_instance_buffer_memories[self.current_frame],
                        0,
                        buffer_size,
                        vk::MemoryMapFlags::empty(),
                    )
                    .expect("Failed to map textured instance buffer memory")
                    as *mut TexturedInstance;
                data_ptr.copy_from_nonoverlapping(tpacked.as_ptr(), tpacked.len());
                self.device
                    .unmap_memory(self.textured_instance_buffer_memories[self.current_frame]);
            }
        }

        let solid_pipeline = share::SolidPipeline {
            pipeline: self.graphics_pipeline,
            layout: self.pipeline_layout,
            camera_set,
        };
        let textured_pipeline = share::TexturedPipeline {
            pipeline: self.textured_pipeline,
            layout: self.textured_pipeline_layout,
            mesh_vertex_buffer: self.textured_quad_vertex_buffer,
            mesh_index_buffer: self.textured_quad_index_buffer,
            mesh_index_count: self.textured_quad_index_count,
        };

        let command_buffer = self.command_buffers[self.current_frame];
        unsafe {
            self.device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .expect("Failed to reset command buffer");
        }
        // Draw order is load-bearing: the render pass has no depth attachment
        // and neither pipeline blends, so command order *is* paint order.
        // Solids first (paddles, walls, digit segments), textured second
        // (ball sprite, label glyphs) so on-top sprites appear on top.
        // If a future feature ever needs interleaving (e.g. textured
        // background → solid HUD → textured tooltip), the per-pipeline
        // batching below has to grow into a more general per-draw ordering.
        share::record_command_buffer(
            &self.device,
            command_buffer,
            self.swapchain_framebuffers[image_index as usize],
            self.render_pass,
            self.swapchain_extent,
            &solid_pipeline,
            &solid_draws,
            &textured_pipeline,
            &textured_draws,
        );

        let wait_semaphores = [self.image_available_semaphores[self.current_frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        // Per-image, not per-frame: presentation may still be using this
        // semaphore on whichever swapchain image was previously paired with
        // current_frame. Indexing by image_index avoids that collision.
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
        // parameters -------------
        let surface_suff = SurfaceStuff {
            surface_loader: self.surface_loader.clone(),
            surface: self.surface,
        };
        // ------------------------

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

        // render_finished_semaphores are indexed by swapchain image; rebuild
        // them if the new swapchain has a different image count. device_wait_idle
        // above guarantees no submission still references the old semaphores.
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

        // Aspect ratio changed — push the new projection through the shared UBO.
        self.current_proj = cgmath::perspective(
            Deg(45.0),
            self.swapchain_extent.width as f32 / self.swapchain_extent.height as f32,
            0.1,
            10.0,
        );
        write_camera_ubos(
            &self.device,
            &self.camera_uniform_buffers_memory,
            self.current_view,
            self.current_proj,
        );

        self.swapchain_imageviews =
            share::create_image_views(&self.device, self.swapchain_format, &self.swapchain_images);
        // Render pass only depends on swapchain format, which rarely changes —
        // skip the destroy + recreate when the format is unchanged.
        if format_changed {
            unsafe {
                self.device.destroy_render_pass(self.render_pass, None);
            }
            self.render_pass = share::create_render_pass(&self.device, self.swapchain_format);
        }
        let (graphics_pipeline, pipeline_layout) = share::create_graphics_pipeline(
            &self.device,
            self.render_pass,
            swapchain_stuff.swapchain_extent,
            self.ubo_layout,
            self.pipeline_cache,
        );
        self.graphics_pipeline = graphics_pipeline;
        self.pipeline_layout = pipeline_layout;

        let (textured_pipeline, textured_pipeline_layout) =
            share::create_textured_graphics_pipeline(
                &self.device,
                self.render_pass,
                swapchain_stuff.swapchain_extent,
                self.textured_ubo_layout,
                self.pipeline_cache,
            );
        self.textured_pipeline = textured_pipeline;
        self.textured_pipeline_layout = textured_pipeline_layout;

        self.swapchain_framebuffers = share::create_framebuffers(
            &self.device,
            self.render_pass,
            &self.swapchain_imageviews,
            self.swapchain_extent,
        );
    }

    fn cleanup_swapchain(&self) {
        unsafe {
            for &framebuffer in self.swapchain_framebuffers.iter() {
                self.device.destroy_framebuffer(framebuffer, None);
            }
            self.device.destroy_pipeline(self.graphics_pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_pipeline(self.textured_pipeline, None);
            self.device
                .destroy_pipeline_layout(self.textured_pipeline_layout, None);
            for &image_view in self.swapchain_imageviews.iter() {
                self.device.destroy_image_view(image_view, None);
            }
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
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
            self.device.destroy_render_pass(self.render_pass, None);

            // Save before destroying — the cache is what we want persisted, not
            // the file. Best-effort: any IO error is silently ignored.
            save_pipeline_cache(&self.device, self.pipeline_cache);
            self.device.destroy_pipeline_cache(self.pipeline_cache, None);

            self.device
                .free_command_buffers(self.command_pool, &self.command_buffers);

            for mesh in self.meshes.values() {
                destroy_mesh(&self.device, mesh);
            }
            for tex in self.textures.values() {
                destroy_texture_resources(&self.device, self.descriptor_pool, tex);
            }

            self.device
                .destroy_buffer(self.textured_quad_index_buffer, None);
            self.device
                .free_memory(self.textured_quad_index_memory, None);
            self.device
                .destroy_buffer(self.textured_quad_vertex_buffer, None);
            self.device
                .free_memory(self.textured_quad_vertex_memory, None);

            for &b in self.instance_buffers.iter() {
                self.device.destroy_buffer(b, None);
            }
            for &m in self.instance_buffer_memories.iter() {
                self.device.free_memory(m, None);
            }
            for &b in self.textured_instance_buffers.iter() {
                self.device.destroy_buffer(b, None);
            }
            for &m in self.textured_instance_buffer_memories.iter() {
                self.device.free_memory(m, None);
            }

            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            for i in 0..self.camera_uniform_buffers.len() {
                self.device
                    .destroy_buffer(self.camera_uniform_buffers[i], None);
                self.device
                    .free_memory(self.camera_uniform_buffers_memory[i], None);
            }

            self.device
                .destroy_descriptor_set_layout(self.ubo_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.textured_ubo_layout, None);

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
    let create_info = vk::PipelineCacheCreateInfo::builder()
        .initial_data(&initial)
        .build();
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

// Persistent host-visible+coherent instance buffers (one per frame in flight).
// Each frame's buffer is rewritten in draw_frame before being read by the GPU.
// Coherent memory means we don't need explicit flushes.
fn create_instance_buffers(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    count: usize,
) -> (Vec<vk::Buffer>, Vec<vk::DeviceMemory>) {
    let buffer_size =
        (MAX_INSTANCES * ::std::mem::size_of::<Instance>()) as vk::DeviceSize;
    create_per_frame_vertex_buffers(device, mem_props, count, buffer_size)
}

fn create_textured_instance_buffers(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    count: usize,
) -> (Vec<vk::Buffer>, Vec<vk::DeviceMemory>) {
    let buffer_size =
        (MAX_TEXTURED_INSTANCES * ::std::mem::size_of::<TexturedInstance>()) as vk::DeviceSize;
    create_per_frame_vertex_buffers(device, mem_props, count, buffer_size)
}

fn create_per_frame_vertex_buffers(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    count: usize,
    buffer_size: vk::DeviceSize,
) -> (Vec<vk::Buffer>, Vec<vk::DeviceMemory>) {
    let mut buffers = Vec::with_capacity(count);
    let mut memories = Vec::with_capacity(count);
    for _ in 0..count {
        let (buffer, memory) = share::create_buffer(
            device,
            buffer_size,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            mem_props,
        );
        buffers.push(buffer);
        memories.push(memory);
    }
    (buffers, memories)
}
