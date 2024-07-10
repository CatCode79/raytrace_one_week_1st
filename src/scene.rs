//= IMPORTS ==================================================================

use glam::{dvec3, DVec3, uvec3, Vec3, vec3};

use std::f64::consts::PI;
use rand::Rng;

//= UTILITY FUNCTIONS ========================================================

#[inline(always)]
fn degrees_to_radians(degrees: f64) -> f64 {
    return degrees * PI / 180.0;
}

// Returns a random real in [0,1).
#[inline(always)]
fn random_double() -> f64 {
    rand::random()
}

// Returns a random real in [min,max).
#[inline(always)]
fn random_double2(min: f64, max: f64) -> f64 {
    rand::thread_rng().gen_range(min..max)
}
//= TYPES ====================================================================

type Color = DVec3;
type Point = DVec3;

//= SCENE ====================================================================

pub struct Scene {
    pub hittables: Vec<Sphere>,
}

impl Scene {
    pub fn new() -> Self {
        let mut hittables = Vec::new();
        hittables.push(Sphere::new(Point::new(0.0, 0.0, -1.0), 0.5));
        hittables.push(Sphere::new(Point::new(0.0, -100.5, -1.0), 100.0));

        Self {
            hittables
        }
    }
}

impl Hittable for Scene {
    fn hit(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord> {
        let mut closest_so_far = ray_t.max;
        let mut hit_anything = None;

        for hittable in &self.hittables {
            let temp_rec = hittable.hit(ray, Interval::new(ray_t.min, closest_so_far));
            if let Some(temp_rec) = temp_rec {
                closest_so_far = temp_rec.t;
                hit_anything = Some(temp_rec);
            }
        }

        return hit_anything;
    }
}

//= CAMERA ===================================================================

pub struct Camera {
    pub(crate) width: u16,
    pub(crate) height: u16,
    pub(crate) data: Vec<u32>,

    center: Point,         // Camera center
    pixel00_loc: Point,    // Location of pixel 0, 0
    pixel_delta_u: DVec3,  // Offset to pixel to the right
    pixel_delta_v: DVec3,  // Offset to pixel below
}

impl Camera {
    pub(crate) fn new(width: u16, height: u16) -> Self {
        // Camera
        let focal_length = 1.0;
        let viewport_height = 2.0;
        let viewport_width = viewport_height * (width as f64 / height as f64);
        let center = dvec3(0.0, 0.0, 0.0);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        let viewport_u = dvec3(viewport_width, 0.0, 0.0);
        let viewport_v = dvec3(0.0, -viewport_height, 0.0);

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        let pixel_delta_u = viewport_u / width as f64;
        let pixel_delta_v = viewport_v / height as f64;

        // Calculate the location of the upper left pixel.
        let viewport_upper_left = center
            - dvec3(0.0, 0.0, focal_length) - viewport_u/2.0 - viewport_v/2.0;
        let pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

        Self {
            width,
            height,
            data: vec![0_u32; width as usize * height as usize],

            center,
            pixel00_loc,
            pixel_delta_u,
            pixel_delta_v,
        }
    }

    pub(crate) fn render(&mut self, scene: &Scene) {
        profiling::scope!("render");

        for i in 0..self.data.len() {
            let width = i % self.width as usize;
            let height = i / self.width as usize;

            let pixel_center = self.pixel00_loc + (width as f64 * self.pixel_delta_u) + (height as f64 * self.pixel_delta_v);
            let ray_direction = pixel_center - self.center;
            let ray = Ray::new(self.center, ray_direction);

            let color = self.ray_color(&ray, scene);
            self.data[i] = u32::from_ne_bytes([
                (255.9999 * color.x) as u8,
                (255.9999 * color.y) as u8,
                (255.9999 * color.z) as u8,
                255_u8]);
        }
    }

    fn ray_color(&self, ray: &Ray, scene: &Scene) -> Color {
        let rec = scene.hit(ray, Interval::new(0.0, f64::INFINITY));
        if let Some(rec) = rec {
            return 0.5 * (rec.normal + Color::new(1.0, 1.0, 1.0));
        }

        let unit_direction = ray.direction.normalize();
        let a = 0.5 * (unit_direction.y + 1.0);
        return (1.0 - a) * Color::new(1.0, 1.0, 1.0) + (a * Color::new(0.5, 0.7, 1.0));
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
}

//= HITTABLE =================================================================

#[derive(Default)]
struct HitRecord {
    p: Point,
    normal: DVec3,
    t: f64,
    front_face: bool,
}

impl HitRecord {
    fn new(t: f64, p: Point) -> Self {
        Self {
            p,
            normal: Default::default(),
            t,
            front_face: false,
        }
    }

    fn set_face_normal(&mut self, ray: &Ray, outward_normal: &DVec3) {
        // Sets the hit record normal vector.
        // NOTE: the parameter `outward_normal` is assumed to have unit length.

        self.front_face = ray.direction.dot(*outward_normal) < 0.0;
        self.normal = if self.front_face { *outward_normal } else { -(*outward_normal) };
    }
}

trait Hittable {
    fn hit(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord>;
}

//= INTERVAL =================================================================

struct Interval {
    min: f64,
    max: f64,
}

impl Interval {
    fn new(min: f64, max: f64) -> Self {
        Self {
            min,
            max,
        }
    }

    fn size(&self) -> f64 {
        return self.max - self.min;
    }

    fn contains(&self, x: f64) -> bool {
        return self.min <= x && x <= self.max;
    }

    fn surrounds(&self, x: f64) -> bool {
        return self.min < x && x < self.max;
    }

    fn empty() -> Self {
        Self::new(f64::INFINITY, f64::NEG_INFINITY)
    }

    fn universe() -> Self {
        Self::new(f64::NEG_INFINITY, f64::INFINITY)
    }
}

impl Default for Interval {
    fn default() -> Self {
        Interval::empty()
    }
}

//= SPHERE ===================================================================

struct Sphere {
    center: Point,
    radius: f64,
}

impl Sphere {
    fn new(center: Point, radius: f64) -> Self {
        Self {
            center,
            radius,
        }
    }
}

impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, ray_t: Interval) -> Option<HitRecord> {
        let oc = self.center - ray.origin;
        let a = ray.direction.length_squared();
        let h = ray.direction.dot(oc);
        let c = oc.length_squared() - self.radius*self.radius;

        let discriminant = h*h - a*c;
        if discriminant < 0.0 {
            return None;
        }

        let sqrtd = discriminant.sqrt();

        // Find the nearest root that lies in the acceptable range.
        let mut root = (h - sqrtd) / a;
        if !ray_t.surrounds(root) {
            root = (h + sqrtd) / a;
            if !ray_t.surrounds(root) {
                return None;
            }
        }

        let mut rec = HitRecord::new(root, ray.at(root));
        let outward_normal = (rec.p - self.center) / self.radius;
        rec.set_face_normal(ray, &outward_normal);

        return Some(rec);
    }
}