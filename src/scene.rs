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
fn random_double_range(min: f64, max: f64) -> f64 {
    rand::thread_rng().gen_range(min..max)
}

#[inline(always)]
fn random_vec3() -> DVec3 {
    dvec3(random_double(), random_double(), random_double())
}

#[inline(always)]
fn random_vec3_range(min: f64, max: f64) -> DVec3 {
    dvec3(random_double_range(min, max), random_double_range(min, max), random_double_range(min, max))
}

#[inline(always)]
fn random_vec3_in_unit_sphere() -> DVec3 {
    loop {
        let p = random_vec3_range(-1.0, 1.0);
        if p.length_squared() < 1.0 {
            return p;
        }
    }
}

#[inline(always)]
fn random_vec3_normalized() -> DVec3 {
    random_vec3_in_unit_sphere().normalize()
}

#[inline(always)]
fn random_vec3_on_hemisphere(normal: DVec3) -> DVec3 {
    let on_unit_sphere = random_vec3_normalized();
    // In the same hemisphere as the normal
    if on_unit_sphere.dot(normal) > 0.0 {
        on_unit_sphere
    } else {
        -on_unit_sphere
    }
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

    samples_per_pixel: u8,   // Count of random samples for each pixel
    pixel_samples_scale: f64,  // Color scale factor for a sum of pixel samples
    max_depth: u8,   // Maximum number of ray bounces into scene
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

        let samples_per_pixel = 6;
        Self {
            width,
            height,
            data: vec![0_u32; width as usize * height as usize],

            samples_per_pixel,
            pixel_samples_scale: 1.0 / samples_per_pixel as f64,
            max_depth: 10,
            center,
            pixel00_loc,
            pixel_delta_u,
            pixel_delta_v,
        }
    }

    pub(crate) fn render(&mut self, scene: &Scene) {
        profiling::scope!("render");

        const INTENSITY: Interval = Interval::new(0.000, 0.999999);

        for i in 0..self.data.len() {
            let w = i % self.width as usize;
            let h = i / self.width as usize;

            let mut color = Color::new(0.0, 0.0, 0.0);
            for _ in 0..self.samples_per_pixel {
                let ray = self.get_ray(w as f64, h as f64);
                color += self.ray_color(&ray, self.max_depth, scene);
            }

            self.data[i] = u32::from_ne_bytes([
                (255.9999 * INTENSITY.clamp(self.pixel_samples_scale * color.x)) as u8,
                (255.9999 * INTENSITY.clamp(self.pixel_samples_scale * color.y)) as u8,
                (255.9999 * INTENSITY.clamp(self.pixel_samples_scale * color.z)) as u8,
                255_u8]);
        }
    }

    fn ray_color(&self, ray: &Ray, depth: u8, scene: &Scene) -> Color {
        // If we've exceeded the ray bounce limit, no more light is gathered.
        if depth <= 0 {
            return Color::new(0.0,0.0,0.0);
        }

        let rec = scene.hit(ray, Interval::new(0.001, f64::INFINITY));
        if let Some(rec) = rec {
            let direction = rec.normal + random_vec3_normalized();
            return 0.5 * self.ray_color(&Ray::new(rec.p, direction), depth-1, scene);
        }

        let unit_direction = ray.direction.normalize();
        let a = 0.5 * (unit_direction.y + 1.0);
        return (1.0 - a) * Color::new(1.0, 1.0, 1.0) + (a * Color::new(0.5, 0.7, 1.0));
    }

    // Construct a camera ray originating from the origin and directed at randomly sampled
    // point around the pixel location i, j.
    fn get_ray(&self, w: f64, h: f64) -> Ray {
        let offset = self.sample_square();
        let pixel_sample = self.pixel00_loc
            + ((w + offset.x) * self.pixel_delta_u)
            + ((h + offset.y) * self.pixel_delta_v);

        return Ray::new(self.center,  pixel_sample - self.center);
    }

    // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
    fn sample_square(&self) -> DVec3 {
        return dvec3(random_double() - 0.5, random_double() - 0.5, 0.0);
    }
}

//= RAY ======================================================================

#[derive(Debug)]
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
    const fn new(min: f64, max: f64) -> Self {
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

    fn clamp(&self, x: f64) -> f64 {
        if x < self.min {
            return self.min
        };
        if x > self.max {
            return self.max
        };
        return x;
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