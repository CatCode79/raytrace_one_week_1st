//= MODS =====================================================================

mod scene;
mod renderer;

//= IMPORTS ==================================================================

use crate::scene::{Camera, Scene};
use crate::renderer::Renderer;

use shun_winput::mapping::InputMapping;
use shun_winput::window::{Event, Window};

//= CONSTANTS ================================================================

// The additional values are used to obtain a padding of 256 relative to the buffer.
const WIDTH: u16 = 1280 + 16;
const HEIGHT: u16 = 720 + 39;

//= MAIN STUFF! ==============================================================

fn main() -> Result<(), String> {
    let scene = Scene::new();
    let mut camera = Camera::new(WIDTH, HEIGHT);

    let input_mapping = InputMapping::new();
    let mut window = Window::new(
        "Raytrace One Week 1st".to_string(),
        WIDTH,
        HEIGHT,
        input_mapping,
    )?;

    let mut renderer = Renderer::new(&window)?;

    loop {
        let events = window.process_events();
        for event in events {
            match event {
                Event::Resize { width, height } => {
                    camera.width = width.get();
                    camera.height = height.get();
                    camera.data = vec![0_u32; camera.width as usize * camera.height as usize];
                    renderer.resize(width, height);
                }
            }
        }

        camera.render(&scene);
        renderer.update(&camera)?;
        renderer.present();

        profiling::finish_frame!();
    }
}
