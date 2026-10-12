//! CTC loss and its gradient, natively, in F32.
//!
//! Burn's default `ctc_loss` composes a recursion of up to a dozen steps of element-wise ops, and
//! under autodiff the gradient is the derivative of that composition. In a half-precision tensor
//! every one of those operations rounds to the half type, and the vendored `autodiff_f16` CTC
//! gradient tests, which compare with an F32 reference at `rel_abs(1e-3, 1e-3)`, are missed by up to
//! 0.235. Reporting `has_ctc_loss_backward` makes autodiff call the two methods here instead, and the
//! generated wrappers widen a half-precision operand to F32 and round the result once, as for every
//! other operation.
//!
//! The forward pass is Burn's own `ctc_loss_default`. The gradient is the one Burn composes from the
//! two recursions, `ctc_grad_from_alpha_beta_default`, which also zeroes an unreachable target's
//! gradient; the recursions are here because Burn keeps them private. The alpha recursion and its
//! helpers below are Burn's (burn-backend 0.21.0, `ops/modules/ctc.rs`, MIT or Apache-2.0), generic
//! over the backend and run on this one in F32. Beta is alpha of the reversed problem: the time steps
//! of each sample reversed within its input length, and its blank-inserted label sequence reversed
//! within `2 * target_length + 1`, which is the blank-inserted sequence of the reversed targets; the
//! skip rule (`l'[s] != blank` and `l'[s] != l'[s - 2]`) is symmetric under that reversal.
//!
//! A gradient cannot be had by differentiating the composition under an inner autodiff graph: the
//! outer `backward()` holds autodiff's global graph lock while it runs this method, and a nested
//! graph takes the same lock.
use super::*;
use burn_backend::ops::ctc::{ctc_grad_from_alpha_beta_default, ctc_loss_default};
use burn_backend::{get_device_settings, Backend, BoolDType, FloatDType, IntDType, Shape, Slice};

/// This backend implements [`ctc_loss_backward`].
pub fn has_ctc_loss_backward() -> bool {
    true
}

/// Burn's default CTC loss, in F32.
pub fn ctc_loss(
    log_probs: FloatTensor<TtBackend>,
    targets: IntTensor<TtBackend>,
    input_lengths: IntTensor<TtBackend>,
    target_lengths: IntTensor<TtBackend>,
    blank: usize,
) -> FloatTensor<TtBackend> {
    ctc_loss_default::<TtBackend>(log_probs, targets, input_lengths, target_lengths, blank)
}

/// The gradient of `sum(grad_loss * ctc_loss)` with respect to `log_probs`, in F32.
pub fn ctc_loss_backward(
    log_probs: FloatTensor<TtBackend>,
    targets: IntTensor<TtBackend>,
    input_lengths: IntTensor<TtBackend>,
    target_lengths: IntTensor<TtBackend>,
    grad_loss: FloatTensor<TtBackend>,
    blank: usize,
) -> FloatTensor<TtBackend> {
    backward::<TtBackend>(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        grad_loss,
        blank,
    )
}

fn backward<B: Backend>(
    log_probs: FloatTensor<B>,
    targets: IntTensor<B>,
    input_lengths: IntTensor<B>,
    target_lengths: IntTensor<B>,
    grad_loss: FloatTensor<B>,
    blank: usize,
) -> FloatTensor<B> {
    let [max_input_length, batch_size, num_classes] = log_probs.shape().dims::<3>();
    let max_target_len = targets.shape().dims::<2>()[1];
    let max_l_prime_len = 2 * max_target_len + 1;
    let device = B::float_device(&log_probs);
    let int_dtype: IntDType = targets.dtype().into();

    let alpha = AlphaCtx::<B>::compute(
        log_probs.clone(),
        &targets,
        input_lengths.clone(),
        target_lengths.clone(),
        blank,
    );
    let nll = extract_loss::<B>(&alpha, target_lengths.clone());

    // The reversed problem. Time: `t -> input_length - 1 - t`, clamped (the steps past a sample's
    // input length are masked everywhere and read any valid row).
    let t_index = B::int_expand(
        B::int_reshape(
            B::int_arange(0..max_input_length as i64, &device, int_dtype),
            Shape::new([max_input_length, 1]),
        ),
        Shape::new([max_input_length, batch_size]),
    );
    let il_tn = B::int_expand(
        B::int_reshape(input_lengths.clone(), Shape::new([1, batch_size])),
        Shape::new([max_input_length, batch_size]),
    );
    let reversed_t = B::int_clamp(
        B::int_sub(B::int_sub_scalar(il_tn, 1.into()), t_index),
        0.into(),
        (max_input_length as i64 - 1).max(0).into(),
    );
    let reversed_log_probs = B::float_gather(
        0,
        log_probs.clone(),
        B::int_expand(
            B::int_reshape(
                reversed_t.clone(),
                Shape::new([max_input_length, batch_size, 1]),
            ),
            Shape::new([max_input_length, batch_size, num_classes]),
        ),
    );
    // Labels: `j -> target_length - 1 - j`, clamped.
    let reversed_targets = if max_target_len == 0 {
        targets.clone()
    } else {
        let j = B::int_expand(
            B::int_reshape(
                B::int_arange(0..max_target_len as i64, &device, int_dtype),
                Shape::new([1, max_target_len]),
            ),
            Shape::new([batch_size, max_target_len]),
        );
        let tl = B::int_expand(
            B::int_reshape(target_lengths.clone(), Shape::new([batch_size, 1])),
            Shape::new([batch_size, max_target_len]),
        );
        let index = B::int_clamp(
            B::int_sub(B::int_sub_scalar(tl, 1.into()), j),
            0.into(),
            (max_target_len as i64 - 1).into(),
        );
        B::int_gather(1, targets.clone(), index)
    };
    let alpha_reversed = AlphaCtx::<B>::compute(
        reversed_log_probs,
        &reversed_targets,
        input_lengths.clone(),
        target_lengths.clone(),
        blank,
    );

    // `beta[t, n, s] = alpha_reversed[input_length - 1 - t, n, 2 target_length - s]`.
    let by_time = B::float_gather(
        0,
        alpha_reversed.full,
        B::int_expand(
            B::int_reshape(reversed_t, Shape::new([max_input_length, batch_size, 1])),
            Shape::new([max_input_length, batch_size, max_l_prime_len]),
        ),
    );
    let s_index = B::int_expand(
        B::int_reshape(
            B::int_arange(0..max_l_prime_len as i64, &device, int_dtype),
            Shape::new([1, max_l_prime_len]),
        ),
        Shape::new([batch_size, max_l_prime_len]),
    );
    let last = B::int_expand(
        B::int_reshape(
            B::int_add_scalar(B::int_mul_scalar(target_lengths, 2.into()), 0.into()),
            Shape::new([batch_size, 1]),
        ),
        Shape::new([batch_size, max_l_prime_len]),
    );
    // Past a sample's last position (`s > 2 target_length`) beta is `-inf`: the clamped read below
    // would return some other position's value, and alpha's start (`alpha[0, :, 1]`) is set whether
    // or not the sample has a label, so neither is `-inf` there.
    let beyond = B::bool_expand(
        B::bool_reshape(
            B::int_greater(
                s_index.clone(),
                last.clone(),
                get_device_settings::<B>(&device).bool_dtype,
            ),
            Shape::new([1, batch_size, max_l_prime_len]),
        ),
        Shape::new([max_input_length, batch_size, max_l_prime_len]),
    );
    let reversed_s = B::int_clamp(
        B::int_sub(last, s_index),
        0.into(),
        (max_l_prime_len as i64 - 1).into(),
    );
    let beta = B::float_gather(
        2,
        by_time,
        B::int_expand(
            B::int_reshape(reversed_s, Shape::new([1, batch_size, max_l_prime_len])),
            Shape::new([max_input_length, batch_size, max_l_prime_len]),
        ),
    );
    let beta = B::float_mask_fill(beta, beyond, f32::NEG_INFINITY.into());

    ctc_grad_from_alpha_beta_default::<B>(
        log_probs,
        targets,
        input_lengths,
        grad_loss,
        alpha.full,
        beta,
        nll,
        blank,
    )
}

/// The state of the alpha recursion that the gradient reads.
struct AlphaCtx<B: Backend> {
    /// `log_alpha[T, N, 2S+1]` (full history).
    full: FloatTensor<B>,
    /// `log_alpha[T-1, :, :]`, the last step.
    last: FloatTensor<B>,
}

impl<B: Backend> AlphaCtx<B> {
    /// Burn's `AlphaCtx::compute`, unchanged but for the fields it keeps.
    fn compute(
        log_probs: FloatTensor<B>,
        targets: &IntTensor<B>,
        input_lengths: IntTensor<B>,
        target_lengths: IntTensor<B>,
        blank: usize,
    ) -> Self {
        let [max_input_length, batch_size, num_classes] = log_probs.shape().dims::<3>();
        let max_target_len = targets.shape().dims::<2>()[1];
        let device = B::float_device(&log_probs);
        let float_dtype: FloatDType = log_probs.dtype().into();
        let int_dtype: IntDType = targets.dtype().into();
        let settings = get_device_settings::<B>(&device);

        let max_l_prime_len = 2 * max_target_len + 1;
        let blank_inserted_targets = insert_blanks::<B>(
            targets,
            batch_size,
            max_target_len,
            max_l_prime_len,
            blank,
            &device,
            int_dtype,
        );

        // The full alpha tensor `[T, N, 2S+1]`, `-inf` to begin with.
        let mut alpha_full = B::float_full(
            Shape::new([max_input_length, batch_size, max_l_prime_len]),
            f32::NEG_INFINITY.into(),
            &device,
            float_dtype,
        );

        // `alpha[0, :, 0] = log_probs[0, :, blank]` and `alpha[0, :, 1] = log_probs[0, :, l'[1]]`.
        let log_probs_t0 = B::float_slice(
            log_probs.clone(),
            &[Slice::new(0, Some(1), 1), Slice::full(), Slice::full()],
        );
        let log_probs_t0 = B::float_reshape(log_probs_t0, Shape::new([batch_size, num_classes]));

        let first_blank = B::int_slice(
            blank_inserted_targets.clone(),
            &[Slice::full(), Slice::new(0, Some(1), 1)],
        );
        let log_prob_blank = B::float_gather(1, log_probs_t0.clone(), first_blank);
        let log_prob_blank_3d = B::float_reshape(log_prob_blank, Shape::new([1, batch_size, 1]));
        alpha_full = B::float_slice_assign(
            alpha_full,
            &[
                Slice::new(0, Some(1), 1),
                Slice::full(),
                Slice::new(0, Some(1), 1),
            ],
            log_prob_blank_3d,
        );

        if max_l_prime_len > 1 {
            let first_label = B::int_slice(
                blank_inserted_targets.clone(),
                &[Slice::full(), Slice::new(1, Some(2), 1)],
            );
            let log_prob_first = B::float_gather(1, log_probs_t0, first_label);
            let log_prob_first_3d =
                B::float_reshape(log_prob_first, Shape::new([1, batch_size, 1]));
            alpha_full = B::float_slice_assign(
                alpha_full,
                &[
                    Slice::new(0, Some(1), 1),
                    Slice::full(),
                    Slice::new(1, Some(2), 1),
                ],
                log_prob_first_3d,
            );
        }

        // The latest row, kept apart for the recursion.
        let mut log_alpha = B::float_slice(
            alpha_full.clone(),
            &[Slice::new(0, Some(1), 1), Slice::full(), Slice::full()],
        );
        log_alpha = B::float_reshape(log_alpha, Shape::new([batch_size, max_l_prime_len]));

        let l_prime_mask = create_l_prime_mask::<B>(
            &blank_inserted_targets,
            batch_size,
            max_l_prime_len,
            blank,
            &device,
            int_dtype,
            settings.bool_dtype,
        );
        let s_mask = create_s_mask::<B>(
            &target_lengths,
            batch_size,
            max_l_prime_len,
            &device,
            int_dtype,
            settings.bool_dtype,
        );

        // Hoisted out of the T-loop: the `-inf` padding of `right_shift`, and the one gather of
        // `log_probs` at `l'` for every step.
        let pad_1 = B::float_full(
            Shape::new([batch_size, 1]),
            f32::NEG_INFINITY.into(),
            &device,
            float_dtype,
        );
        let pad_2 = B::float_full(
            Shape::new([batch_size, 2]),
            f32::NEG_INFINITY.into(),
            &device,
            float_dtype,
        );
        let indices_3d = B::int_expand(
            B::int_reshape(
                blank_inserted_targets,
                Shape::new([1, batch_size, max_l_prime_len]),
            ),
            Shape::new([max_input_length, batch_size, max_l_prime_len]),
        );
        let log_probs_at_l_full = B::float_gather(2, log_probs, indices_3d);

        // `combined_mask_all[t, n, s] = (input_lengths[n] > t) AND s_mask[n, s]`, for every step.
        let t_indices_2d = B::int_expand(
            B::int_reshape(
                B::int_arange(0..max_input_length as i64, &device, int_dtype),
                Shape::new([max_input_length, 1]),
            ),
            Shape::new([max_input_length, batch_size]),
        );
        let il_tn = B::int_expand(
            B::int_reshape(input_lengths, Shape::new([1, batch_size])),
            Shape::new([max_input_length, batch_size]),
        );
        let t_mask_all = B::bool_expand(
            B::bool_reshape(
                B::int_greater(il_tn, t_indices_2d, settings.bool_dtype),
                Shape::new([max_input_length, batch_size, 1]),
            ),
            Shape::new([max_input_length, batch_size, max_l_prime_len]),
        );
        let s_mask_bcast = B::bool_expand(
            B::bool_reshape(s_mask, Shape::new([1, batch_size, max_l_prime_len])),
            Shape::new([max_input_length, batch_size, max_l_prime_len]),
        );
        let combined_mask_all = B::bool_and(t_mask_all, s_mask_bcast);

        for t in 1..max_input_length {
            let combined_mask = B::bool_reshape(
                B::bool_slice(
                    combined_mask_all.clone(),
                    &[
                        Slice::new(t as isize, Some(t as isize + 1), 1),
                        Slice::full(),
                        Slice::full(),
                    ],
                ),
                Shape::new([batch_size, max_l_prime_len]),
            );

            let log_alpha_s = log_alpha.clone();
            let log_alpha_s_m1 = right_shift::<B>(&log_alpha, &pad_1, max_l_prime_len, 1);
            let log_alpha_s_m2 = right_shift::<B>(&log_alpha, &pad_2, max_l_prime_len, 2);

            let bar = log_sum_exp::<B>(log_alpha_s, log_alpha_s_m1, settings.bool_dtype);
            let bar_with_skip = log_sum_exp::<B>(bar.clone(), log_alpha_s_m2, settings.bool_dtype);
            let log_alpha_combined = B::float_mask_where(bar, l_prime_mask.clone(), bar_with_skip);

            let log_probs_at_l = B::float_reshape(
                B::float_slice(
                    log_probs_at_l_full.clone(),
                    &[
                        Slice::new(t as isize, Some(t as isize + 1), 1),
                        Slice::full(),
                        Slice::full(),
                    ],
                ),
                Shape::new([batch_size, max_l_prime_len]),
            );
            let new_alpha = B::float_add(log_alpha_combined, log_probs_at_l);
            log_alpha = B::float_mask_where(log_alpha, combined_mask, new_alpha);

            let log_alpha_3d = B::float_reshape(
                log_alpha.clone(),
                Shape::new([1, batch_size, max_l_prime_len]),
            );
            alpha_full = B::float_slice_assign(
                alpha_full,
                &[
                    Slice::new(t as isize, Some(t as isize + 1), 1),
                    Slice::full(),
                    Slice::full(),
                ],
                log_alpha_3d,
            );
        }

        Self {
            full: alpha_full,
            last: log_alpha,
        }
    }
}

/// The per-sample loss from the last alpha row.
fn extract_loss<B: Backend>(alpha: &AlphaCtx<B>, target_lengths: IntTensor<B>) -> FloatTensor<B> {
    let [batch_size, _] = alpha.last.shape().dims::<2>();
    let device = B::float_device(&alpha.last);
    let settings = get_device_settings::<B>(&device);

    let last_blank_idx = B::int_mul_scalar(target_lengths.clone(), 2.into());
    let last_blank_idx = B::int_reshape(last_blank_idx, Shape::new([batch_size, 1]));
    let last_label_idx = B::int_clamp_min(
        B::int_sub_scalar(last_blank_idx.clone(), 1.into()),
        0.into(),
    );

    let log_alpha_last_blank = B::float_gather(1, alpha.last.clone(), last_blank_idx);
    let log_alpha_last_blank = B::float_reshape(log_alpha_last_blank, Shape::new([batch_size]));

    let log_alpha_last_label = B::float_gather(1, alpha.last.clone(), last_label_idx);
    let log_alpha_last_label = B::float_reshape(log_alpha_last_label, Shape::new([batch_size]));

    // For `target_lengths == 0` the last label is meaningless: `-inf`.
    let target_len_zero = B::int_equal_elem(target_lengths, 0.into(), settings.bool_dtype);
    let log_alpha_last_label = B::float_mask_fill(
        log_alpha_last_label,
        target_len_zero,
        f32::NEG_INFINITY.into(),
    );

    let log_likelihood = log_sum_exp::<B>(
        log_alpha_last_blank,
        log_alpha_last_label,
        settings.bool_dtype,
    );
    B::float_neg(log_likelihood)
}

/// The blank-inserted targets: `[b, l1, b, l2, ..., b]`.
fn insert_blanks<B: Backend>(
    targets: &IntTensor<B>,
    batch_size: usize,
    max_target_len: usize,
    max_l_prime_len: usize,
    blank: usize,
    device: &B::Device,
    int_dtype: IntDType,
) -> IntTensor<B> {
    let result = B::int_full(
        Shape::new([batch_size, max_l_prime_len]),
        (blank as i64).into(),
        device,
        int_dtype,
    );
    if max_target_len == 0 {
        return result;
    }
    // Every target label at the odd columns, `result[:, 1::2] = targets`.
    B::int_slice_assign(
        result,
        &[Slice::full(), Slice::new(1, None, 2)],
        targets.clone(),
    )
}

/// Right-shift a 2D float tensor by `shift` positions, prepending the `-inf` `padding` (shape
/// `[batch_size, shift]`).
fn right_shift<B: Backend>(
    tensor: &FloatTensor<B>,
    padding: &FloatTensor<B>,
    cols: usize,
    shift: usize,
) -> FloatTensor<B> {
    // Shifting by more than the columns pushes every value off the right (`max_target_len == 0`).
    if cols < shift {
        return B::float_slice(
            padding.clone(),
            &[Slice::full(), Slice::new(0, Some(cols as isize), 1)],
        );
    }
    let shortened = B::float_slice(
        tensor.clone(),
        &[
            Slice::full(),
            Slice::new(0, Some((cols - shift) as isize), 1),
        ],
    );
    B::float_cat(vec![padding.clone(), shortened], 1)
}

/// `log(exp(a) + exp(b))`, stable: `max(a, b) + log1p(exp(-|a - b|))`, with `a = b = -inf` kept at
/// `-inf`. The inputs are log-probabilities (`<= 0`).
fn log_sum_exp<B: Backend>(
    a: FloatTensor<B>,
    b: FloatTensor<B>,
    bool_dtype: BoolDType,
) -> FloatTensor<B> {
    let a_is_neg_inf = B::float_equal_elem(a.clone(), f32::NEG_INFINITY.into(), bool_dtype);
    let b_is_neg_inf = B::float_equal_elem(b.clone(), f32::NEG_INFINITY.into(), bool_dtype);
    let either_neg_inf = B::bool_or(a_is_neg_inf.clone(), b_is_neg_inf.clone());

    let a_safe = B::float_mask_fill(a.clone(), a_is_neg_inf, 0.0.into());
    let b_safe = B::float_mask_fill(b.clone(), b_is_neg_inf, 0.0.into());

    let lt_mask = B::float_lower(a.clone(), b.clone(), bool_dtype);
    let mx = B::float_mask_where(a, lt_mask, b);

    let diff_safe = B::float_neg(B::float_abs(B::float_sub(a_safe, b_safe)));
    let diff_final = B::float_mask_fill(diff_safe, either_neg_inf, f32::NEG_INFINITY.into());

    B::float_add(mx, B::float_log1p(B::float_exp(diff_final)))
}

/// The alpha skip transition: `l'[s] != blank AND l'[s] != l'[s-2] AND s >= 2`.
fn create_l_prime_mask<B: Backend>(
    blank_inserted_targets: &IntTensor<B>,
    batch_size: usize,
    max_l_prime_len: usize,
    blank: usize,
    device: &B::Device,
    int_dtype: IntDType,
    bool_dtype: BoolDType,
) -> BoolTensor<B> {
    // `s >= 2` is unsatisfiable when `max_l_prime_len < 2` (targets of shape `[N, 0]`).
    if max_l_prime_len < 2 {
        return B::bool_zeros(
            Shape::new([batch_size, max_l_prime_len]),
            device,
            bool_dtype,
        );
    }
    let l_prime = blank_inserted_targets.clone();

    let not_blank = B::int_not_equal_elem(l_prime.clone(), (blank as i64).into(), bool_dtype);

    let l_prime_shifted = {
        let padding = B::int_full(
            Shape::new([batch_size, 2]),
            (blank as i64).into(),
            device,
            int_dtype,
        );
        let shortened = B::int_slice(
            l_prime.clone(),
            &[
                Slice::full(),
                Slice::new(0, Some((max_l_prime_len - 2) as isize), 1),
            ],
        );
        B::int_cat(vec![padding, shortened], 1)
    };
    let not_equal_s_m2 = B::int_not_equal(l_prime, l_prime_shifted, bool_dtype);

    let col_indices = B::int_arange(0..max_l_prime_len as i64, device, int_dtype);
    let col_indices = B::int_reshape(col_indices, Shape::new([1, max_l_prime_len]));
    let col_indices = B::int_expand(col_indices, Shape::new([batch_size, max_l_prime_len]));
    let s_ge_2 = B::int_greater_equal_elem(col_indices, 2.into(), bool_dtype);

    B::bool_and(B::bool_and(not_blank, not_equal_s_m2), s_ge_2)
}

/// The valid `s` positions: `s < 2 * target_length + 1`.
fn create_s_mask<B: Backend>(
    target_lengths: &IntTensor<B>,
    batch_size: usize,
    max_l_prime_len: usize,
    device: &B::Device,
    int_dtype: IntDType,
    bool_dtype: BoolDType,
) -> BoolTensor<B> {
    let col_indices = B::int_arange(0..max_l_prime_len as i64, device, int_dtype);
    let col_indices = B::int_reshape(col_indices, Shape::new([1, max_l_prime_len]));
    let col_indices = B::int_expand(col_indices, Shape::new([batch_size, max_l_prime_len]));

    let lengths = B::int_mul_scalar(target_lengths.clone(), 2.into());
    let lengths = B::int_add_scalar(lengths, 1.into());
    let lengths = B::int_reshape(lengths, Shape::new([batch_size, 1]));
    let lengths = B::int_expand(lengths, Shape::new([batch_size, max_l_prime_len]));

    B::int_lower(col_indices, lengths, bool_dtype)
}
