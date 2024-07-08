mod buffer;
mod renderer;

use crate::buffer::Buffer;
use crate::renderer::Renderer;

use shun_winput::mapping::InputMapping;
use shun_winput::window::{Event, Window};

// The additional values are used to obtain a padding of 256 relative to the buffer.
const WIDTH: u16 = 1280 + 16;
const HEIGHT: u16 = 720 + 39;

fn main() -> Result<(), String> {
    let mut buffer = Buffer {
        width: WIDTH,
        height: HEIGHT,
        data: vec![0_u32; WIDTH as usize * HEIGHT as usize],
    };

    let input_mapping = InputMapping::new();
    let mut window = Window::new(
        "Raytrace One Week".to_string(),
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
                    buffer.width = width.get();
                    buffer.height = height.get();
                    buffer.data = vec![0_u32; buffer.width as usize * buffer.height as usize];

                    renderer.resize(width, height);
                }
            }
        }

        buffer.compute();
        renderer.update(&buffer)?;

        renderer.present();
        profiling::finish_frame!();
    }

    Ok(())
}
