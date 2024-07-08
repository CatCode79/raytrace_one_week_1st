pub struct Buffer {
    pub(crate) width: u16,
    pub(crate) height: u16,
    pub(crate) data: Vec<u32>,
}

impl Buffer {
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
