//= IMPORTS ==================================================================

use glam::{dvec3, DVec3, uvec3, Vec3, vec3};

use std::f64::consts::PI;

//= UTILITY FUNCTIONS ========================================================

#[inline(always)]
fn degrees_to_radians(degrees: f64) -> f64 {
    return degrees * PI / 180.0;
}

//= TYPES ====================================================================

type Color = DVec3;
type Point = DVec3;

//= SCENE ====================================================================

pub struct Scene {
    pub width: u16,
    pub height: u16,
    pub data: Vec<u32>,
    pub hittables: Vec<Sphere>,
}

impl Scene {
    pub fn new(width: u16, height: u16) -> Self {
        let mut hittables = Vec::new();
        hittables.push(Sphere::new(Point::new(0.0, 0.0, -1.0), 0.5));
        hittables.push(Sphere::new(Point::new(0.0, -100.5, -1.0), 100.0));

        Self {
            width,
            height,
            data: vec![0_u32; width as usize * height as usize],
            hittables
        }
    }

    pub fn compute(&mut self) {
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

        for i in 0..self.data.len() {
            let width = i % self.width as usize;
            let height = i / self.width as usize;

            let pixel_center = pixel00_loc + (width as f64 * pixel_delta_u) + (height as f64 * pixel_delta_v);
            let ray_direction = pixel_center - camera_center;
            let ray = Ray::new(camera_center, ray_direction);

            let color = ray.color(&self);
            self.data[i] = u32::from_ne_bytes([
                (255.9999 * color.x) as u8,
                (255.9999 * color.y) as u8,
                (255.9999 * color.z) as u8,
                255_u8]);
        }
    }
}

impl Hittable for Scene {
    fn hit(&self, ray: &Ray, ray_tmin: f64, ray_tmax: f64) -> Option<HitRecord> {
        let mut closest_so_far = ray_tmax;
        let mut hit_anything = None;

        for hittable in &self.hittables {
            let temp_rec = hittable.hit(ray, ray_tmin, closest_so_far);
            if let Some(temp_rec) = temp_rec {
                closest_so_far = temp_rec.t;
                hit_anything = Some(temp_rec);
            }
        }

        return hit_anything;
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

    fn color(self: &Ray, scene: &Scene) -> Color {
        let rec = scene.hit(self, 0.0, f64::INFINITY);
        if let Some(rec) = rec {
            return 0.5 * (rec.normal + Color::new(1.0, 1.0, 1.0));
        }

        let unit_direction = self.direction.normalize();
        let a = 0.5 * (unit_direction.y + 1.0);
        return (1.0 - a) * Color::new(1.0, 1.0, 1.0) + (a * Color::new(0.5, 0.7, 1.0));
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
    fn hit(&self, ray: &Ray, ray_tmin: f64, ray_tmax: f64) -> Option<HitRecord>;
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
    fn hit(&self, ray: &Ray, ray_tmin: f64, ray_tmax: f64) -> Option<HitRecord> {
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
        if root <= ray_tmin || ray_tmax <= root {
            root = (h + sqrtd) / a;
            if root <= ray_tmin || ray_tmax <= root {
                return None;
            }
        }

        let mut rec = HitRecord::new(root, ray.at(root));
        let outward_normal = (rec.p - self.center) / self.radius;
        rec.set_face_normal(ray, &outward_normal);

        return Some(rec);
    }
}