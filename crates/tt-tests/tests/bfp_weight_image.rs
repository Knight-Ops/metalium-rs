//! A stored BFP weight downloaded as its GDDR extents and uploaded again decodes to the same
//! values. The copy uses the card's DMA when a staging buffer is pinned, and the BAR otherwise.
use tt_kernels::{
    bfp::BfpFormat,
    session::{Session, TileChoice},
    tensor::Layout,
};
use tt_ttsim::fork_scope;
#[cfg(not(feature = "silicon"))]
type Transport<'a> = tt_ttsim::LibTtsim<'a>;
#[cfg(feature = "silicon")]
type Transport<'a> = tt_kmd::Kmd;

fn with_session(f: impl FnOnce(&mut Session<Transport<'_>>)) {
    if let Err(e) = fork_scope(|| {
        #[cfg(not(feature = "silicon"))]
        let mut sim = tt_ttsim::Simulator::open().unwrap();
        #[cfg(not(feature = "silicon"))]
        let dev = tt_device::Device::open(sim.transport()).unwrap();
        #[cfg(not(feature = "silicon"))]
        let mut s = Session::open(
            dev,
            tt_firmware_images::ROLES,
            TileChoice::Count(2),
            |_, _| Ok(None),
        )
        .unwrap();
        #[cfg(feature = "silicon")]
        let mut s = Session::open_card(
            tt_tests::backend::device_index(),
            tt_firmware_images::ROLES,
            TileChoice::Count(2),
        )
        .unwrap();
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .unwrap();
        f(&mut s);
    }) {
        panic!("{e}");
    }
}

fn round_trip(s: &mut Session<Transport<'_>>, rows: usize, cols: usize, format: BfpFormat) {
    let values: Vec<f32> = (0..rows * cols).map(|i| ((i % 13) as f32) - 6.0).collect();
    let input = s.upload(&values, rows, cols).unwrap();
    let stored = s.store_bfp(&input, format).unwrap();
    let image = s.download_bfp_image(&stored.as_bfp().unwrap()).unwrap();
    let decoded = s.download_bfp(&stored.as_bfp().unwrap()).unwrap();
    let restored = s.upload_bfp_image(&image).unwrap();
    let again = s.download_bfp(&restored.as_bfp().unwrap()).unwrap();
    assert_eq!(decoded, again);
    let back = s.download_bfp_image(&restored.as_bfp().unwrap()).unwrap();
    assert_eq!(image.channels, back.channels);
    assert_eq!(image.layout, back.layout);
}

#[test]
fn a_weight_image_round_trips_in_the_layout_it_was_stored_in() {
    with_session(|s| {
        let (channels, columns) = s.weight_layout().unwrap();
        assert!(columns);
        // A column count that is a multiple of the channels fits by column.
        round_trip(s, 64, 32 * channels, BfpFormat::Bfp8);
        let wide = s
            .upload(&vec![0.0; 64 * 32 * channels], 64, 32 * channels)
            .unwrap();
        let stored = s.store_bfp(&wide, BfpFormat::Bfp4).unwrap();
        let image = s.download_bfp_image(&stored.as_bfp().unwrap()).unwrap();
        assert_eq!(image.layout, Layout::ByColumn);
        assert_eq!(image.channels.len(), channels);
        // Fewer columns than channels: by column would waste a column a channel, so the
        // weight stays interleaved. 128 rows is four tile rows, 64 columns is two.
        if channels > 2 {
            let narrow = s.upload(&vec![1.0; 128 * 64], 128, 64).unwrap();
            let stored = s.store_bfp(&narrow, BfpFormat::Bfp8).unwrap();
            let image = s.download_bfp_image(&stored.as_bfp().unwrap()).unwrap();
            assert_eq!(image.layout, Layout::Interleaved);
        }
        round_trip(s, 32, 32, BfpFormat::Bfp4);
    });
}
