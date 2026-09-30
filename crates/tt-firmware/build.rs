//! Pick each image's link script.
//!
//! Every image uses `link.x` (T0's default reset PC) except the T1 and T2 role
//! images, which live at their own cores' default reset PCs so that one tile can
//! hold all three role images at once. Per-binary link arguments are the only
//! way to vary this within one crate.

fn main() {
    let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    println!("cargo:rustc-link-arg-bins=-L{dir}");
    for bin in ["heartbeat", "sfpu_mul", "corpus", "corpus_t0", "role_t0"] {
        println!("cargo:rustc-link-arg-bin={bin}=-Tlink.x");
    }
    println!("cargo:rustc-link-arg-bin=role_t1=-Tlink_t1.x");
    println!("cargo:rustc-link-arg-bin=role_t2=-Tlink_t2.x");
    for f in ["link.x", "link_t1.x", "link_t2.x", "sections.x"] {
        println!("cargo:rerun-if-changed={f}");
    }
}
