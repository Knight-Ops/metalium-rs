//! The role runner with guarded runs (`tt_firmware::corpus::run_guarded_role`)
//! on T1 / Tensix thread 1: a probe image at T1's default reset PC, which
//! `tt_kernels::atomics` loads for the Sync Unit and L1 atomics gates
//! (`step105_mutex`, `step106_l1_atomics`). The production role image
//! (`role_t1`) leaves guarded runs out to keep its slot small. Never loaded
//! beside the mover.

#![no_std]
#![no_main]

use tt_firmware::corpus;
use tt_isa::tensix::{RiscvT1, Thread1};

#[no_mangle]
pub extern "Rust" fn firmware_main() -> ! {
    corpus::run_guarded_role::<RiscvT1, Thread1>()
}
