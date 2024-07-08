use crate::buffer::Buffer;

use glam::U16Vec2;
use pollster::FutureExt as _;
use shun_winput::window::Window;
use wgpu::util::DeviceExt as _;
use wgpu::{
    Adapter, AddressMode, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType,
    ColorTargetState, ColorWrites, CommandBuffer, CommandEncoder, CommandEncoderDescriptor,
    CompositeAlphaMode, Device, DeviceDescriptor, Extent3d, Features, FilterMode, FragmentState,
    ImageCopyTexture, ImageDataLayout, Instance, InstanceDescriptor, InstanceFlags, Limits, LoadOp,
    MultisampleState, Operations, PipelineCompilationOptions, PipelineLayoutDescriptor,
    PowerPreference, PresentMode, PrimitiveState, Queue, RenderPassColorAttachment,
    RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor, RequestAdapterOptions, Sampler,
    SamplerBindingType, SamplerDescriptor, ShaderModuleDescriptor, ShaderSource, ShaderStages,
    StoreOp, Surface, SurfaceConfiguration, SurfaceError, SurfaceTargetUnsafe, SurfaceTexture,
    TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages,
    TextureView, TextureViewDescriptor, TextureViewDimension, VertexState,
};

use std::num::NonZeroU16;

//= RENDERER =================================================================

pub struct Renderer {
    surface: Surface<'static>,
    surface_config: SurfaceConfiguration,
    device: Device,
    queue: Queue,

    result_texture: TextureHandler,
    output: Option<SurfaceTexture>,

    screen_shader: ScreenShader,
}

impl Renderer {
    pub fn new(window: &Window) -> Result<Self, String> {
        let instance = create_instance();

        let surface = create_surface(window, &instance)?;

        let adapter = request_adapter(instance, &surface)?;

        let surface_config = create_surface_config(&surface, &adapter, window)?;

        let (device, queue) = request_device(adapter)?;

        surface.configure(&device, &surface_config);

        let result_texture = {
            TextureHandler::new(
                &device,
                U16Vec2::new(window.inner_width(), window.inner_height()),
            )
        };

        let screen_shader = ScreenShader::new(&device, &result_texture, surface_config.format);

        Ok(Self {
            surface,
            surface_config,
            device,
            queue,
            result_texture,
            output: None,
            screen_shader,
        })
    }

    pub fn resize(&mut self, width: NonZeroU16, height: NonZeroU16) {
        let width = width.get();
        let height = height.get();
        if self.surface_config.width as u16 != width || self.surface_config.height as u16 != height
        {
            self.surface_config.width = width as u32;
            self.surface_config.height = height as u32;
            self.surface.configure(&self.device, &self.surface_config);

            let new_size = U16Vec2::new(width, height);
            self.result_texture = TextureHandler::new(&self.device, new_size);

            self.screen_shader
                .recreate_bind_group(&self.device, &self.result_texture);
        }
    }

    pub fn update(&mut self, buffer: &Buffer) -> Result<(), String> {
        let (output, view) = self.get_output().map_err(|e| e.to_string())?;
        let mut encoder = self.create_command_encoder();

        self.screen_shader.encode_pass(&mut encoder, &view);

        let temp_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Temporary Buffer"),
                contents: bytemuck::cast_slice(buffer.data.as_slice()),
                usage: wgpu::BufferUsages::COPY_SRC,
            });
        encoder.copy_buffer_to_texture(
            wgpu::ImageCopyBuffer {
                buffer: &temp_buf,
                layout: ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * buffer.width as u32),
                    rows_per_image: Some(buffer.height as u32),
                },
            },
            ImageCopyTexture {
                texture: &self.result_texture.handle,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            Extent3d {
                width: self.result_texture.size().x as u32,
                height: self.result_texture.size().y as u32,
                depth_or_array_layers: 1,
            },
        );

        self.submit_once(encoder.finish(), output);

        Ok(())
    }

    fn create_command_encoder(&self) -> CommandEncoder {
        self.device
            .create_command_encoder(&CommandEncoderDescriptor::default())
    }

    fn submit_once(&mut self, command_buffer: CommandBuffer, output: SurfaceTexture) {
        self.queue.submit(std::iter::once(command_buffer));
        self.output = Some(output);
    }

    pub fn surface_size(&self) -> U16Vec2 {
        U16Vec2::new(
            self.surface_config.width as u16,
            self.surface_config.height as u16,
        )
    }

    pub fn get_output(&self) -> Result<(SurfaceTexture, TextureView), SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&TextureViewDescriptor::default());
        Ok((output, view))
    }

    pub fn present(&mut self) {
        if self.output.is_some() {
            self.output.take().unwrap().present();
        }
    }
}

//= RENDERER SETUP ===========================================================

/// The backends are in order of support, the greater the first.
fn supported_backends() -> wgpu::Backends {
    #[cfg(target_os = "windows")]
    return wgpu::Backends::DX12 | wgpu::Backends::VULKAN;

    #[cfg(target_os = "linux")]
    return wgpu::Backends::VULKAN;

    #[cfg(target_os = "macos")]
    return wgpu::Backends::METAL | wgpu::Backends::VULKAN;
}

fn create_instance() -> Instance {
    let desc = InstanceDescriptor {
        backends: supported_backends(),
        flags: InstanceFlags::default(),
        ..Default::default()
    };
    Instance::new(desc)
}

fn create_surface(window: &Window, instance: &Instance) -> Result<Surface<'static>, String> {
    let raw_display_handle = window.raw_display_handle();
    if raw_display_handle.is_err() {
        return Err("Raw display handle error on surface creation".to_string());
    };
    let raw_window_handle = window.raw_window_handle();
    if raw_window_handle.is_err() {
        return Err("Raw window handle error on surface creation".to_string());
    }

    let surface_target = SurfaceTargetUnsafe::RawHandle {
        raw_display_handle: raw_display_handle.unwrap(),
        raw_window_handle: raw_window_handle.unwrap(),
    };
    let surface = match unsafe { instance.create_surface_unsafe(surface_target) } {
        Ok(s) => s,
        Err(e) => return Err(e.to_string()),
    };

    Ok(surface)
}

fn request_adapter(instance: Instance, surface: &Surface<'static>) -> Result<Adapter, String> {
    log_possible_adapters(supported_backends(), &instance);

    let a = async {
        instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: Some(surface),
                force_fallback_adapter: false,
            })
            .await
    }
    .block_on();

    let Some(a) = a else {
        return Err("No adapters were found with requested options.".to_string());
    };

    log_picked_adapter(&a);
    Ok(a)
}

/// Log all the adapters' info.
fn log_possible_adapters(backends: wgpu::Backends, wgpu_instance: &Instance) {
    for (i, adapter) in wgpu_instance
        .enumerate_adapters(backends)
        .iter()
        .enumerate()
    {
        log::debug!("Possible Adapter #{}: {}", i, get_adapter_info(&adapter))
    }
}

/// Log the picked adapter info.
fn log_picked_adapter(adapter: &Adapter) {
    log::info!("Picked Adapter: {}", get_adapter_info(&adapter));
    log::debug!("Its Features: {:?}", adapter.features());
}

/// Return an adapter info pretty printed.
fn get_adapter_info(adapter: &Adapter) -> String {
    format!("{:?}", adapter.get_info())
        .replace("AdapterInfo { name: ", "")
        .replace(" }", "")
}

fn request_device(adapter: Adapter) -> Result<(Device, Queue), String> {
    let dq = async {
        adapter
            .request_device(
                &DeviceDescriptor {
                    required_features: Features::default(),
                    required_limits: Limits::default(),
                    label: None,
                },
                None,
            )
            .await
    }
    .block_on();

    let Ok(dq) = dq else {
        return Err(format!("{:?}", dq.err()));
    };

    Ok(dq)
}

fn create_surface_config(
    surface: &Surface<'static>,
    adapter: &Adapter,
    window: &Window,
) -> Result<SurfaceConfiguration, String> {
    if window.inner_width() == 0 {
        return Err(
            "Impossible to create a surface configuration with zero width size".to_string(),
        );
    }
    if window.inner_height() == 0 {
        return Err(
            "Impossible to create a surface configuration with zero height size".to_string(),
        );
    }

    let texture_formats = surface.get_capabilities(adapter).formats;
    let Some(texture_format) = texture_formats.first() else {
        return Err("A valid surface texture format isn't supported by this adapter.".to_string());
    };

    Ok(SurfaceConfiguration {
        usage: TextureUsages::RENDER_ATTACHMENT,
        format: *texture_format,
        width: window.inner_width() as u32,
        height: window.inner_height() as u32,
        desired_maximum_frame_latency: 2,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Auto,
        view_formats: vec![],
    })
}

//= TEXTURE HANDLER ==========================================================

struct TextureHandler {
    pub handle: wgpu::Texture,
    pub sampler: Sampler,
    pub view: TextureView,
}

impl TextureHandler {
    pub fn new(device: &Device, size: U16Vec2) -> Self {
        let handle = device.create_texture(&TextureDescriptor {
            label: None,
            size: Extent3d {
                width: size.x as u32,
                height: size.y as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8Unorm,
            view_formats: &[],
            usage: TextureUsages::COPY_DST
                .union(TextureUsages::COPY_SRC)
                .union(TextureUsages::STORAGE_BINDING)
                .union(TextureUsages::TEXTURE_BINDING),
        });
        let view = handle.create_view(&TextureViewDescriptor::default());

        let sampler = device.create_sampler(&SamplerDescriptor {
            label: None,
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Nearest,
            lod_min_clamp: 1.0,
            lod_max_clamp: 1.0,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        });
        Self {
            handle,
            view,
            sampler,
        }
    }

    pub fn size(&self) -> U16Vec2 {
        let size = self.handle.size();
        U16Vec2::new(size.width as u16, size.height as u16)
    }
}

//= SCREEN SHADER ============================================================

static SCREEN_SHADER_SRC: &str = include_str!("../shaders/screen_shader.wgsl");

struct ScreenShader {
    pub pipeline: RenderPipeline,
    pub bind_group_layout: BindGroupLayout,
    pub bind_group: BindGroup,
}

impl ScreenShader {
    fn new(device: &Device, texture: &TextureHandler, surface_format: TextureFormat) -> Self {
        let shader_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("screen_shader_module"),
            source: ShaderSource::Wgsl(SCREEN_SHADER_SRC.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("screen_shader_bind_group_layout"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: TextureSampleType::default(),
                        view_dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let bind_group = Self::create_bind_group(device, &bind_group_layout, texture);

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("screen_shader_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("screen_shader_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader_module,
                entry_point: "vs_main",
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module: &shader_module,
                entry_point: "fs_main",
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some(ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview: None,
        });

        Self {
            pipeline,
            bind_group,
            bind_group_layout,
        }
    }

    pub fn create_bind_group(
        device: &Device,
        layout: &BindGroupLayout,
        tex: &TextureHandler,
    ) -> BindGroup {
        device.create_bind_group(&BindGroupDescriptor {
            label: Some("screen_shader_bind_group"),
            layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&tex.view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Sampler(&tex.sampler),
                },
            ],
        })
    }

    pub fn recreate_bind_group(&mut self, device: &Device, tex: &TextureHandler) {
        self.bind_group = Self::create_bind_group(device, &self.bind_group_layout, tex);
    }

    pub fn encode_pass(&self, encoder: &mut CommandEncoder, view: &TextureView) {
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("screen_shader_pass"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..6, 0..1);
    }
}
