//! Pick each image's link script, and tell it where its mailbox is.
//!
//! Every Tensix image uses `link.x` (T0's default reset PC) except the T1 and T2
//! role images, which live at their own cores' default reset PCs so that one tile
//! can hold all three role images at once. The Ethernet image uses `link_e1.x`.
//! Per-binary link arguments are the only way to vary this within one crate.
//!
//! The mailbox base is a `--defsym` taken from `tt-isa` (a build-dependency), not
//! a number in a link script: the host polls the same constant, so the two cannot
//! drift.

fn main() {
    let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    println!("cargo:rustc-link-arg-bins=-L{dir}");
    let tensix = [
        ("heartbeat", "link.x"),
        ("sfpu_mul", "link.x"),
        ("corpus", "link.x"),
        ("corpus_t0", "link.x"),
        ("role_t0", "link.x"),
        ("role_t1", "link_t1.x"),
        ("role_t2", "link_t2.x"),
    ];
    for (bin, script) in tensix {
        link(bin, script, tt_isa::mailbox::MAILBOX_BASE);
    }
    link("eth_e1", "link_e1.x", tt_isa::eth::MAILBOX_BASE);
    link("dm_b", "link_b.x", tt_isa::dm::MAILBOX_BASE);
    for f in [
        "link.x",
        "link_t1.x",
        "link_t2.x",
        "link_e1.x",
        "link_b.x",
        "sections.x",
    ] {
        println!("cargo:rerun-if-changed={f}");
    }
}

fn link(bin: &str, script: &str, mailbox: u64) {
    println!("cargo:rustc-link-arg-bin={bin}=-T{script}");
    println!("cargo:rustc-link-arg-bin={bin}=--defsym=__mailbox={mailbox:#x}");
}
