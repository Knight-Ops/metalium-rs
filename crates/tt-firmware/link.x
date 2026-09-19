/* Link script for a single firmware image on any of T0, T1 or T2.
 *
 * The image is placed at L1 offset 0x6000, which is T0's default reset PC
 * (SoftReset.md:118-123), so it runs on T0 with no further setup. Any other T-core
 * runs the same image by pointing its reset-PC override here, which is what
 * `Device::load_and_start` does. Only RISCV B cannot: its entry point is hardwired
 * to L1 offset 0 and it has no override register.
 *
 * LENGTH stops at 0xA000, which is T1's default reset PC. That is not a guess at
 * how much code will fit -- it is a hard boundary: an image that overran it would
 * silently overwrite T1's entry point, and a link error is a much better way to
 * discover that than a hung T1 three steps later.
 */
MEMORY
{
  L1    (rwx) : ORIGIN = 0x00006000, LENGTH = 0x4000
  LOCAL (rw)  : ORIGIN = 0xFFB00000, LENGTH = 4K
}

ENTRY(_start)

SECTIONS
{
  .text :
  {
    KEEP(*(.text.start))
    *(.text .text.*)
  } > L1

  .rodata : ALIGN(4)
  {
    *(.rodata .rodata.*)
    *(.srodata .srodata.*)
  } > L1

  /* Initialised data lives in L1 alongside the code, so the host stages it as
   * part of the single image write. Nothing is copied at run time, and nothing is
   * staged into local data RAM -- which is what keeps the 2048-cycle local-RAM
   * zeroing window (BabyRISCV/README.md:152) from being a hazard here. */
  .data : ALIGN(4)
  {
    *(.data .data.*)
    *(.sdata .sdata.*)
  } > L1

  .bss (NOLOAD) : ALIGN(4)
  {
    __bss_start = .;
    *(.bss .bss.*)
    *(.sbss .sbss.*)
    *(COMMON)
    . = ALIGN(4);
    __bss_end = .;
  } > L1

  /* Stack in local data RAM: 2-cycle access, and the core's own accesses stall
   * automatically during the post-reset zeroing window, so setting it up in
   * _start is safe. */
  _stack_top = ORIGIN(LOCAL) + LENGTH(LOCAL);

  /* `.riscv.attributes` records which extensions the image was built for. It is
   * not loaded (objcopy leaves it out of the flat binary), but discarding it
   * makes every disassembler fall back to base RV32I, where an `m`-extension
   * instruction decodes as `<unknown>` -- which would quietly blind the
   * instruction-set gate in crates/tt-tests/build.rs. Keep it. */
  .riscv.attributes 0 : { *(.riscv.attributes) }

  /DISCARD/ :
  {
    *(.eh_frame)
    *(.eh_frame_hdr)
    *(.comment)
  }
}
