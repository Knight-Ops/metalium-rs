//! The role runner with guarded runs (`tt_firmware::corpus::run_guarded_role`)
//! on T2 / Tensix thread 2: a probe image at T2's default reset PC, which
//! `tt_kernels::atomics` loads for the Sync Unit and L1 atomics gates
//! (`step105_mutex`, `step106_l1_atomics`). The production role image
//! (`role_t2`) leaves guarded runs out to keep its slot small. Never loaded
//! beside the mover.

#![no_std]
#![no_main]

use tt_firmware::corpus;
use tt_isa::tensix::{RiscvT2, Thread2};

#[no_mangle]
pub extern "Rust" fn firmware_main() -> ! {
    corpus::run_guarded_role::<RiscvT2, Thread2>()
}
