//! Deterministic test data and small helpers the gates share.

/// A 64-bit LCG (Knuth's MMIX constants); the value is its top 31 bits.
pub fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *s >> 33
}

/// The same generator as [`lcg`], returning its top 32 bits.
pub fn lcg_word(seed: &mut u64) -> u32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*seed >> 32) as u32
}

/// A xorshift64 stream from `seed` (forced odd, so never the zero state).
pub fn xorshift(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed | 1;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    }
}

/// The `f32` values whose raw bits are `bits`.
pub fn float_data(bits: &[u32]) -> Vec<f32> {
    bits.iter().map(|&b| f32::from_bits(b)).collect()
}

/// The message of the panic `f` raises; fails if `f` does not panic.
pub fn panic_message(f: impl FnOnce()) -> String {
    let e = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
        .expect_err("the operation must refuse");
    e.downcast_ref::<String>()
        .cloned()
        .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default()
}
