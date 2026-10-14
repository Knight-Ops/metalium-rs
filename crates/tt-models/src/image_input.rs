//! The pinned test photographs, decoded and normalised on the host.
//!
//! Both backends of a comparison are given the same `f32` tensor, so how it was made does not
//! matter to the result; it only has to be deterministic and to look like what the network saw in
//! training. This is the torchvision recipe (resize the short side, centre crop, scale to 0..1,
//! subtract the mean, divide by the deviation), with the `image` crate's Catmull-Rom filter standing
//! in for PIL's bicubic, so a top-1 here is close to, not identical to, timm's published pipeline.

use std::path::{Path, PathBuf};

use image::imageops::FilterType;

/// The pinned photographs, in the order the gates use them.
pub const IMAGES: &[&str] = &[
    "cats.jpg",
    "dog.jpg",
    "bear.jpg",
    "living_room.jpg",
    "bedroom.jpg",
];

pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];
pub const HALF_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
pub const HALF_STD: [f32; 3] = [0.5, 0.5, 0.5];

pub fn image_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../vendor/weights/images")
        .join(name)
}

/// `name` as a normalised CHW `f32` image of `size` x `size`: the short side resized to
/// `size * 256 / 224` and the centre cropped.
pub fn load_chw(name: &str, size: u32, mean: [f32; 3], std: [f32; 3]) -> Result<Vec<f32>, String> {
    let path = image_path(name);
    if !path.exists() {
        return Err(format!(
            "{} does not exist: run `cargo xtask fetch-weights --model images` first",
            path.display()
        ));
    }
    let img = image::open(&path)
        .map_err(|e| format!("decoding {}: {e}", path.display()))?
        .to_rgb8();
    let short = (size * 256).div_ceil(224);
    let (w, h) = img.dimensions();
    let scale = short as f32 / w.min(h) as f32;
    let (rw, rh) = (
        ((w as f32 * scale).round() as u32).max(size),
        ((h as f32 * scale).round() as u32).max(size),
    );
    let resized = image::imageops::resize(&img, rw, rh, FilterType::CatmullRom);
    let (x0, y0) = ((rw - size) / 2, (rh - size) / 2);
    let cropped = image::imageops::crop_imm(&resized, x0, y0, size, size).to_image();

    let plane = (size * size) as usize;
    let mut out = vec![0f32; 3 * plane];
    for (x, y, px) in cropped.enumerate_pixels() {
        for c in 0..3 {
            out[c * plane + (y * size + x) as usize] = (px[c] as f32 / 255.0 - mean[c]) / std[c];
        }
    }
    Ok(out)
}
