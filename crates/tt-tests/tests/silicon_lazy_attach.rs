//! B2 gate on real hardware: Burn code with no `attach` runs on the card.
//!
//! `TtDevice::new(card)` is all a caller writes: the first op opens
//! `/dev/tenstorrent/{card}` through the registry (the silicon default, one
//! card on its first surviving tile) and the product is computed on the device
//! and matches burn-flex. While burn-tt holds the card a raw session is refused
//! (`CardBusy`), and `detach` hands it back.

#![cfg(feature = "silicon")]

use burn_flex::{Flex, FlexDevice};
use burn_tensor::{Tensor, TensorData, TensorPrimitive};
use burn_tt::{detach, TtBackend, TtDevice};
use tt_kernels::session::{Session, SessionError, TileChoice};
use tt_tests::backend::{device_index, GATE_TILE};
use tt_tests::harness::assert_on_silicon;

/// Small integers, so the TF32 product is exact and Flex is the answer.
fn ints(shape: &[usize], seed: usize) -> TensorData {
    let n: usize = shape.iter().product();
    TensorData::new(
        (0..n)
            .map(|i| ((i * 7 + seed) % 9) as f32 - 4.0)
            .collect::<Vec<f32>>(),
        shape.to_vec(),
    )
}

#[test]
fn the_first_op_opens_the_card_and_matches_flex() {
    assert_on_silicon();
    let card = device_index();
    let device = TtDevice::new(card);
    assert!(
        burn_tt::TtDevice::enumerate().contains(&device),
        "/dev/tenstorrent/{card} is listed without being opened"
    );

    let a = Tensor::<TtBackend, 2>::from_data(ints(&[64, 96], 1), &device);
    let b = Tensor::<TtBackend, 2>::from_data(ints(&[96, 32], 2), &device);
    let c = a.matmul(b);
    match c.clone().into_primitive() {
        TensorPrimitive::Float(t) => assert!(t.computed_on_device(), "computed on the card"),
        _ => panic!("unexpected quantized tensor"),
    }
    let want = Tensor::<Flex, 2>::from_data(ints(&[64, 96], 1), &FlexDevice)
        .matmul(Tensor::<Flex, 2>::from_data(
            ints(&[96, 32], 2),
            &FlexDevice,
        ))
        .into_data();
    assert_eq!(c.into_data().as_bytes(), want.as_bytes());

    // burn-tt owns the card now: nothing else may open it.
    let open = || {
        Session::open_card(
            card,
            tt_firmware_images::ROLES,
            TileChoice::Exactly(GATE_TILE.0, GATE_TILE.1),
        )
    };
    match open() {
        Err(SessionError::CardBusy { card: busy }) => assert_eq!(busy, card),
        Err(other) => panic!("expected CardBusy, got: {other}"),
        Ok(_) => panic!("the card was opened under burn-tt"),
    }
    detach(device).unwrap();
    drop(open().unwrap_or_else(|e| panic!("after detach: {e}")));
}
