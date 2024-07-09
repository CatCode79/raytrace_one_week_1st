//= IMPORTS ==================================================================

use glam::Vec3;

//= SCENE ====================================================================

pub struct Scene {
    pub(crate) width: u16,
    pub(crate) height: u16,
    pub(crate) data: Vec<u32>,
}

impl Scene {
    pub(crate) fn compute(&mut self) {
        profiling::scope!("render");

        for (i, pixel) in self.data.iter_mut().enumerate() {
            let width = i % self.width as usize;
            let height = i / self.width as usize;

            let r = width as f32 / self.width as f32;
            let g = height as f32 / self.height as f32;

            *pixel = u32::from_ne_bytes([(255.9999 * r) as u8, (255.9999 * g) as u8, 0_u8, 255_u8]);
        }
    }
}

//= COLOR ====================================================================

type Color = Vec3;

//= RAY ======================================================================

struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    fn ray_color(ray: &Ray) -> Color {
        return Color::new(0.0, 0.0, 0.0);
    }
}
