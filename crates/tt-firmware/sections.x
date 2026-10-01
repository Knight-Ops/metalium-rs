/* Everything but the MEMORY block, shared by every image's link script.
 *
 * The images differ only in where their code lives in L1 -- see link.x (T0's
 * default reset PC, used by every single-core image) and link_t1.x/link_t2.x
 * (the role images, each at its own core's default reset PC, so three cores can
 * hold three different images at once). */
ENTRY(_start)

SECTIONS
{
  .text :
  {
    KEEP(*(.text.start))
    /* A mover's per-entry path (`#[link_section = ".text.hot"]` in dm_b.rs),
     * together and first, then the NoC issue it calls (by its v0-mangled
     * function section): RISCV B's instruction cache is 2 KiB, and with the
     * cold paths out of line this takes a WAIT entry from 0.31 to 0.29 us
     * (`silicon_perf::mover_read_shapes`). */
    *(.text.hot .text.hot.*)
    *(.text.*11tt_firmware3noc5issue)
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
