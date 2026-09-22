//! The transcription check for `tt_kmd::abi`.
//!
//! `abi.rs` is hand-written. That is only safe if something independent confirms
//! the layout, so this compiles the *real* `vendor/ioctl.h` with the *real* C
//! compiler and compares its `sizeof`/`offsetof` against Rust's. A field added,
//! reordered, or given the wrong width upstream fails here rather than producing
//! an ioctl that returns success and the wrong answer.
//!
//! The same shape as `crates/tt-tests/tests/fma_oracle.rs`: compile the authority
//! and differential-test the port, rather than trusting a careful reading.
//!
//! Skipped with a warning if the header has not been fetched
//! (`cargo xtask fetch-kmd`) or there is no C compiler, so a checkout without
//! either still builds.

use std::mem::{align_of, offset_of, size_of};
use std::path::PathBuf;
use std::process::Command;

use tt_kmd::abi;

fn header() -> Option<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let h = root.join("vendor/ioctl.h");
    h.exists().then_some(h)
}

/// `(what, value)` pairs the C program prints, in the order it prints them.
const PROBE: &str = r#"
#include <stdio.h>
#include <stddef.h>
#include "ioctl.h"

#define S(name, ty)        printf("%s sizeof %zu\n",  name, sizeof(ty))
#define A(name, ty)        printf("%s alignof %zu\n", name, _Alignof(ty))
#define O(name, ty, f)     printf("%s %s %zu\n", name, #f, offsetof(ty, f))
/* fields inside an in/out sub-struct: offset of the sub-struct plus the field */
#define O2(name, ty, sub, subty, f) \
    printf("%s %s %zu\n", name, #f, offsetof(ty, sub) + offsetof(subty, f))

int main(void) {
    S("get_device_info", struct tenstorrent_get_device_info);
    A("get_device_info", struct tenstorrent_get_device_info);
    O2("get_device_info", struct tenstorrent_get_device_info, in,  struct tenstorrent_get_device_info_in,  output_size_bytes);
    O2("get_device_info", struct tenstorrent_get_device_info, out, struct tenstorrent_get_device_info_out, vendor_id);
    O2("get_device_info", struct tenstorrent_get_device_info, out, struct tenstorrent_get_device_info_out, device_id);
    O2("get_device_info", struct tenstorrent_get_device_info, out, struct tenstorrent_get_device_info_out, bus_dev_fn);
    O2("get_device_info", struct tenstorrent_get_device_info, out, struct tenstorrent_get_device_info_out, pci_domain);

    S("get_driver_info", struct tenstorrent_get_driver_info);
    A("get_driver_info", struct tenstorrent_get_driver_info);
    O2("get_driver_info", struct tenstorrent_get_driver_info, out, struct tenstorrent_get_driver_info_out, driver_version);
    O2("get_driver_info", struct tenstorrent_get_driver_info, out, struct tenstorrent_get_driver_info_out, driver_version_major);

    S("mapping", struct tenstorrent_mapping);
    A("mapping", struct tenstorrent_mapping);
    O("mapping", struct tenstorrent_mapping, mapping_id);
    O("mapping", struct tenstorrent_mapping, mapping_base);
    O("mapping", struct tenstorrent_mapping, mapping_size);

    S("query_mappings_in", struct tenstorrent_query_mappings_in);
    A("query_mappings_in", struct tenstorrent_query_mappings_in);

    S("allocate_tlb", struct tenstorrent_allocate_tlb);
    A("allocate_tlb", struct tenstorrent_allocate_tlb);
    O2("allocate_tlb", struct tenstorrent_allocate_tlb, in,  struct tenstorrent_allocate_tlb_in,  size);
    O2("allocate_tlb", struct tenstorrent_allocate_tlb, out, struct tenstorrent_allocate_tlb_out, id);
    O2("allocate_tlb", struct tenstorrent_allocate_tlb, out, struct tenstorrent_allocate_tlb_out, mmap_offset_uc);
    O2("allocate_tlb", struct tenstorrent_allocate_tlb, out, struct tenstorrent_allocate_tlb_out, mmap_offset_wc);

    S("noc_tlb_config", struct tenstorrent_noc_tlb_config);
    A("noc_tlb_config", struct tenstorrent_noc_tlb_config);
    O("noc_tlb_config", struct tenstorrent_noc_tlb_config, addr);
    O("noc_tlb_config", struct tenstorrent_noc_tlb_config, x_end);
    O("noc_tlb_config", struct tenstorrent_noc_tlb_config, y_end);
    O("noc_tlb_config", struct tenstorrent_noc_tlb_config, x_start);
    O("noc_tlb_config", struct tenstorrent_noc_tlb_config, y_start);
    O("noc_tlb_config", struct tenstorrent_noc_tlb_config, noc);
    O("noc_tlb_config", struct tenstorrent_noc_tlb_config, mcast);
    O("noc_tlb_config", struct tenstorrent_noc_tlb_config, ordering);
    O("noc_tlb_config", struct tenstorrent_noc_tlb_config, linked);
    O("noc_tlb_config", struct tenstorrent_noc_tlb_config, static_vc);

    S("configure_tlb", struct tenstorrent_configure_tlb);
    A("configure_tlb", struct tenstorrent_configure_tlb);
    O2("configure_tlb", struct tenstorrent_configure_tlb, in, struct tenstorrent_configure_tlb_in, id);
    O2("configure_tlb", struct tenstorrent_configure_tlb, in, struct tenstorrent_configure_tlb_in, config);

    S("set_noc_cleanup", struct tenstorrent_set_noc_cleanup);
    A("set_noc_cleanup", struct tenstorrent_set_noc_cleanup);
    O("set_noc_cleanup", struct tenstorrent_set_noc_cleanup, argsz);
    O("set_noc_cleanup", struct tenstorrent_set_noc_cleanup, enabled);
    O("set_noc_cleanup", struct tenstorrent_set_noc_cleanup, x);
    O("set_noc_cleanup", struct tenstorrent_set_noc_cleanup, y);
    O("set_noc_cleanup", struct tenstorrent_set_noc_cleanup, noc);
    O("set_noc_cleanup", struct tenstorrent_set_noc_cleanup, addr);
    O("set_noc_cleanup", struct tenstorrent_set_noc_cleanup, data);

    S("lock_ctl", struct tenstorrent_lock_ctl);
    A("lock_ctl", struct tenstorrent_lock_ctl);
    O2("lock_ctl", struct tenstorrent_lock_ctl, in,  struct tenstorrent_lock_ctl_in,  flags);
    O2("lock_ctl", struct tenstorrent_lock_ctl, in,  struct tenstorrent_lock_ctl_in,  index);
    O2("lock_ctl", struct tenstorrent_lock_ctl, out, struct tenstorrent_lock_ctl_out, value);

    printf("IOCTL GET_DEVICE_INFO %lu\n", (unsigned long)TENSTORRENT_IOCTL_GET_DEVICE_INFO);
    printf("IOCTL QUERY_MAPPINGS %lu\n",  (unsigned long)TENSTORRENT_IOCTL_QUERY_MAPPINGS);
    printf("IOCTL GET_DRIVER_INFO %lu\n", (unsigned long)TENSTORRENT_IOCTL_GET_DRIVER_INFO);
    printf("IOCTL LOCK_CTL %lu\n",        (unsigned long)TENSTORRENT_IOCTL_LOCK_CTL);
    printf("IOCTL ALLOCATE_TLB %lu\n",    (unsigned long)TENSTORRENT_IOCTL_ALLOCATE_TLB);
    printf("IOCTL FREE_TLB %lu\n",        (unsigned long)TENSTORRENT_IOCTL_FREE_TLB);
    printf("IOCTL CONFIGURE_TLB %lu\n",   (unsigned long)TENSTORRENT_IOCTL_CONFIGURE_TLB);
    printf("IOCTL SET_NOC_CLEANUP %lu\n", (unsigned long)TENSTORRENT_IOCTL_SET_NOC_CLEANUP);
    printf("IOCTL DRIVER_VERSION %lu\n",  (unsigned long)TENSTORRENT_DRIVER_VERSION);
    return 0;
}
"#;

/// Run the C probe and return its `"key" -> value` lines.
fn c_layout(header: &std::path::Path) -> Option<std::collections::HashMap<String, u64>> {
    let dir = std::env::temp_dir().join(format!("tt-kmd-abi-{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok()?;
    let src = dir.join("probe.c");
    let bin = dir.join("probe");
    std::fs::write(&src, PROBE).ok()?;

    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let status = Command::new(&cc)
        .arg("-I")
        .arg(header.parent().unwrap())
        .arg("-o")
        .arg(&bin)
        .arg(&src)
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }
    let out = Command::new(&bin).output().ok()?;
    let text = String::from_utf8(out.stdout).ok()?;
    let mut map = std::collections::HashMap::new();
    for line in text.lines() {
        let mut it = line.rsplitn(2, ' ');
        let v: u64 = it.next()?.parse().ok()?;
        map.insert(it.next()?.to_string(), v);
    }
    let _ = std::fs::remove_dir_all(&dir);
    Some(map)
}

#[test]
fn the_rust_abi_matches_the_c_header() {
    let Some(h) = header() else {
        eprintln!("skipping: vendor/ioctl.h absent — run `cargo xtask fetch-kmd`");
        return;
    };
    let Some(c) = c_layout(&h) else {
        eprintln!("skipping: no working C compiler");
        return;
    };

    let mut checked = 0usize;
    let mut check = |key: &str, rust: usize| {
        let want = *c
            .get(key)
            .unwrap_or_else(|| panic!("C probe printed nothing for `{key}`"))
            as usize;
        assert_eq!(
            rust, want,
            "`{key}`: Rust says {rust}, the C header says {want}"
        );
        checked += 1;
    };

    check("get_device_info sizeof", size_of::<abi::GetDeviceInfo>());
    check("get_device_info alignof", align_of::<abi::GetDeviceInfo>());
    check(
        "get_device_info output_size_bytes",
        offset_of!(abi::GetDeviceInfo, in_output_size_bytes),
    );
    check(
        "get_device_info vendor_id",
        offset_of!(abi::GetDeviceInfo, vendor_id),
    );
    check(
        "get_device_info device_id",
        offset_of!(abi::GetDeviceInfo, device_id),
    );
    check(
        "get_device_info bus_dev_fn",
        offset_of!(abi::GetDeviceInfo, bus_dev_fn),
    );
    check(
        "get_device_info pci_domain",
        offset_of!(abi::GetDeviceInfo, pci_domain),
    );

    check("get_driver_info sizeof", size_of::<abi::GetDriverInfo>());
    check("get_driver_info alignof", align_of::<abi::GetDriverInfo>());
    check(
        "get_driver_info driver_version",
        offset_of!(abi::GetDriverInfo, driver_version),
    );
    check(
        "get_driver_info driver_version_major",
        offset_of!(abi::GetDriverInfo, driver_version_major),
    );

    check("mapping sizeof", size_of::<abi::Mapping>());
    check("mapping alignof", align_of::<abi::Mapping>());
    check("mapping mapping_id", offset_of!(abi::Mapping, mapping_id));
    check(
        "mapping mapping_base",
        offset_of!(abi::Mapping, mapping_base),
    );
    check(
        "mapping mapping_size",
        offset_of!(abi::Mapping, mapping_size),
    );

    check(
        "query_mappings_in sizeof",
        size_of::<abi::QueryMappingsIn>(),
    );
    check(
        "query_mappings_in alignof",
        align_of::<abi::QueryMappingsIn>(),
    );

    check("allocate_tlb sizeof", size_of::<abi::AllocateTlb>());
    check("allocate_tlb alignof", align_of::<abi::AllocateTlb>());
    check("allocate_tlb size", offset_of!(abi::AllocateTlb, in_size));
    check("allocate_tlb id", offset_of!(abi::AllocateTlb, out_id));
    check(
        "allocate_tlb mmap_offset_uc",
        offset_of!(abi::AllocateTlb, out_mmap_offset_uc),
    );
    check(
        "allocate_tlb mmap_offset_wc",
        offset_of!(abi::AllocateTlb, out_mmap_offset_wc),
    );

    check("noc_tlb_config sizeof", size_of::<abi::NocTlbConfig>());
    check("noc_tlb_config alignof", align_of::<abi::NocTlbConfig>());
    check("noc_tlb_config addr", offset_of!(abi::NocTlbConfig, addr));
    check("noc_tlb_config x_end", offset_of!(abi::NocTlbConfig, x_end));
    check("noc_tlb_config y_end", offset_of!(abi::NocTlbConfig, y_end));
    check(
        "noc_tlb_config x_start",
        offset_of!(abi::NocTlbConfig, x_start),
    );
    check(
        "noc_tlb_config y_start",
        offset_of!(abi::NocTlbConfig, y_start),
    );
    check("noc_tlb_config noc", offset_of!(abi::NocTlbConfig, noc));
    check("noc_tlb_config mcast", offset_of!(abi::NocTlbConfig, mcast));
    check(
        "noc_tlb_config ordering",
        offset_of!(abi::NocTlbConfig, ordering),
    );
    check(
        "noc_tlb_config linked",
        offset_of!(abi::NocTlbConfig, linked),
    );
    check(
        "noc_tlb_config static_vc",
        offset_of!(abi::NocTlbConfig, static_vc),
    );

    check("configure_tlb sizeof", size_of::<abi::ConfigureTlb>());
    check("configure_tlb alignof", align_of::<abi::ConfigureTlb>());
    check("configure_tlb id", offset_of!(abi::ConfigureTlb, in_id));
    check(
        "configure_tlb config",
        offset_of!(abi::ConfigureTlb, in_config),
    );

    check("set_noc_cleanup sizeof", size_of::<abi::SetNocCleanup>());
    check("set_noc_cleanup alignof", align_of::<abi::SetNocCleanup>());
    check(
        "set_noc_cleanup argsz",
        offset_of!(abi::SetNocCleanup, argsz),
    );
    check(
        "set_noc_cleanup enabled",
        offset_of!(abi::SetNocCleanup, enabled),
    );
    check("set_noc_cleanup x", offset_of!(abi::SetNocCleanup, x));
    check("set_noc_cleanup y", offset_of!(abi::SetNocCleanup, y));
    check("set_noc_cleanup noc", offset_of!(abi::SetNocCleanup, noc));
    check("set_noc_cleanup addr", offset_of!(abi::SetNocCleanup, addr));
    check("set_noc_cleanup data", offset_of!(abi::SetNocCleanup, data));

    check("lock_ctl sizeof", size_of::<abi::LockCtl>());
    check("lock_ctl alignof", align_of::<abi::LockCtl>());
    check("lock_ctl flags", offset_of!(abi::LockCtl, in_flags));
    check("lock_ctl index", offset_of!(abi::LockCtl, in_index));
    check("lock_ctl value", offset_of!(abi::LockCtl, out_value));

    check("IOCTL GET_DEVICE_INFO", abi::GET_DEVICE_INFO as usize);
    check("IOCTL QUERY_MAPPINGS", abi::QUERY_MAPPINGS as usize);
    check("IOCTL GET_DRIVER_INFO", abi::GET_DRIVER_INFO as usize);
    check("IOCTL LOCK_CTL", abi::LOCK_CTL as usize);
    check("IOCTL ALLOCATE_TLB", abi::ALLOCATE_TLB as usize);
    check("IOCTL FREE_TLB", abi::FREE_TLB as usize);
    check("IOCTL CONFIGURE_TLB", abi::CONFIGURE_TLB as usize);
    check("IOCTL SET_NOC_CLEANUP", abi::SET_NOC_CLEANUP as usize);
    check("IOCTL DRIVER_VERSION", tt_kmd::PINNED_API_VERSION as usize);

    // The self-check the disassembly gate in `crates/tt-tests/build.rs` learned to
    // carry: a comparison loop that matches nothing passes vacuously.
    assert!(
        checked > 50,
        "only {checked} layout facts checked — the C probe's output is not being read"
    );
}
