//= IMPORTS ==================================================================

use glam::{dvec3, DVec3, uvec3, Vec3, vec3};

//= TYPES ====================================================================

type Color = DVec3;
type Point = DVec3;

//= BUFFER ===================================================================

pub struct Buffer {
    pub(crate) width: u16,
    pub(crate) height: u16,
    pub(crate) data: Vec<u32>,
}

impl Buffer {
    pub(crate) fn compute(&mut self) {
        profiling::scope!("render");

        // Camera
        let focal_length = 1.0;
        let viewport_height = 2.0;
        let viewport_width = viewport_height * (self.width as f64 / self.height as f64);
        let camera_center = dvec3(0.0, 0.0, 0.0);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        let viewport_u = dvec3(viewport_width, 0.0, 0.0);
        let viewport_v = dvec3(0.0, -viewport_height, 0.0);

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        let pixel_delta_u = viewport_u / self.width as f64;
        let pixel_delta_v = viewport_v / self.height as f64;

        // Calculate the location of the upper left pixel.
        let viewport_upper_left = camera_center
            - dvec3(0.0, 0.0, focal_length) - viewport_u/2.0 - viewport_v/2.0;
        let pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

        for (i, pixel) in self.data.iter_mut().enumerate() {
            let width = i % self.width as usize;
            let height = i / self.width as usize;

            let pixel_center = pixel00_loc + (width as f64 * pixel_delta_u) + (height as f64 * pixel_delta_v);
            let ray_direction = pixel_center - camera_center;
            let ray = Ray::new(camera_center, ray_direction);

            let color = ray.color();
            *pixel = u32::from_ne_bytes([
                (255.9999 * color.x) as u8,
                (255.9999 * color.y) as u8,
                (255.9999 * color.z) as u8,
                255_u8]);
        }
    }
}

//= RAY ======================================================================

struct Ray {
    pub origin: Point,
    pub direction: DVec3,
}

impl Ray {
    fn new(origin: Point, direction: DVec3) -> Self {
        Self {
            origin,
            direction,
        }
    }

    fn at(&self, t: f64) -> Point {
        return self.origin + t*self.direction;
    }

    fn color(self: &Ray) -> Color {
        let t = self.hit_sphere(&Point::new(0.0, 0.0, -1.0), 0.5);
        if t > 0.0 {
            let n = (self.at(t) - dvec3(0.0, 0.0, -1.0)).normalize();
            return 0.5 * Color::new(n.x+1.0, n.y+1.0, n.z+1.0);
        }

        let unit_direction = self.direction.normalize();
        let a = 0.5 * (unit_direction.y + 1.0);
        return (1.0 - a) * Color::new(1.0, 1.0, 1.0) + (a * Color::new(0.5, 0.7, 1.0));
    }

    fn hit_sphere(&self, center: &Point, radius: f64) -> f64 {
        let oc = *center - self.origin;
        let a = self.direction.length_squared();
        let h = self.direction.dot(oc);
        let c = oc.length_squared() - radius*radius;
        let discriminant = h*h - a*c;

        return if discriminant < 0.0 {
            -1.0
        } else {
            (h - discriminant.sqrt()) / a
        }
    }
}
