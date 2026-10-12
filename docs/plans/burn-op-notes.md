# Burn op notes

Hand-written per-method **Device** and **Item** notes for the pinned Burn 0.21 operation
traits (extracted from the Phase 10 tracker, archived as
[hardware-coverage.md](../completed-plans/hardware-coverage.md); the item ids below, such as
S4 and R1, are rows in that file). The generated status table, which is the source of truth
for what is overridden, composed or unsupported, is [burn-op-coverage.md](burn-op-coverage.md);
these notes carry the evidence and limits behind it. Open work is in [roadmap.md](roadmap.md).

The compute methods of the pinned Burn 0.21 op traits, with the item that brings each to
the device. Bookkeeping methods (`*_device`, `*_to_device`, `*_into_data`, `*_from_data`,
`*_reshape`, autodiff flags) are left out. **Device** is `x` when the method has a device
path today, `~` when only some shapes do.

## `FloatTensorOps`

| Methods | Device | Item |
|---|:-:|---|
| `float_matmul` | `~` F32 resident: 2-D; rank-N against an unbatched rhs folded to 2-D; direct aligned batches or native ragged materialization (views/broadcast included); new ragged F32 path simulator-gated. A K whose block needs more than a circular buffer's page counters hold (`MAX_CB_PAGES` = 0x7FFF) is split into K-blocks by the chunk search (`PlanError::TooManyPages`, unit-tested in `tt-kernels`); `[408, 261120] @ [261120, 408]` runs in 13 s on card 0 and is too slow for ttsim (found by `burn-tt-conformance`) **Accumulation mode** (2026-10-11, `step159_exact_matmul`): the default is the Matrix Unit matmul (TF32 operands, `MVMUL`'s group adder: error bounded as in `step158`); `MatmulMode::Exact`, opt-in by `TtConfig::matmul`, `TT_EXACT_MATMUL=1` or the `with_matmul_mode`/`exact_matmul` scope, computes F32 SFPU products summed in `k` order (a gather into `[k, outputs]`, `float_mul`, `float_sum_dim` over dim 0; chunked over outputs, refused by name past 2^20 products a chunk for `k` or 1024 chunks, and on a mesh), equal to a sequential-SFPU oracle bit for bit; it is 1,000x to 22,000x slower (performance-log, "Exact matmul cost"). `linear`, `conv`, `attention`, gradients and BF16 (widened) all take the mode. | P1b |
| `float_add`, `float_sub`, `float_mul` (incl. row and column broadcasts; any rank, P1a), `float_mul_scalar` | x (SFPU) | S1, P1a |
| `float_sum_dim` | all F32 axes/layouts on SFPU; both-card validated; BF16 via native adapters | R1 |
| `float_mean_dim` | `~` native sum_dim plus scaling on supported axes | R1 |
| `float_sum`, `float_mean` | x native nonempty F32, bounded full reductions; uploads host inputs,  unsupported inputs fail | R1b |
| `float_slice`, `float_flip` | x nonempty resident F32/BF16, arbitrary steps/axes via bit-preserving copies; both-card validated | D4 |
| `float_transpose`, `float_swap_dims`, `float_permute` | native rank-N views and device materialization; BFP shares exponent-preserving views, regrouping returns decoded F32; arbitrary payload-preserving Src transpose remains partial | M3, D2 |
| `float_add_scalar`, `float_sub_scalar` | x (SFPU) | S1 |
| `float_div{,_scalar}`, `float_recip` | x (SFPU, within 1 ulp) | S3 |
| `float_remainder{,_scalar}` | x exact SFPU fmod (Flex's `((a%b)+b)%b` bit for bit; NaN payload canonical); 2362 words, long division on integers; card 0 (step133-134) | S6 (T4) |
| `float_neg`, `float_abs`, `float_sign`, `float_clamp{,_min,_max}` | x (SFPU, exact) | S2 |
| comparisons (`float_equal`.. `float_lower_equal_elem`), `float_mask_where`, `float_mask_fill`, `float_is_nan`, `float_is_inf` | x (SFPU, exact; `Bool` results resident). `float_mask_where` broadcasts any operand (a `[1, t, t]` score against a `[batch, t, t]` mask) by device copies, then selects; bit-exact against Flex, card 0 (step151, found by `burn-tt-conformance`) | S2 |
| `float_cast` | x same dtype or native F32↔BF16; other reduced floats refused | D1, S6 |
| `float_exp`, `float_log` | x (SFPU, derived bounds) | S4 |
| `float_log1p`, `float_sqrt`, `float_powf*`, `float_powi*` | x (SFPU, derived bounds; `pow` one op). `powf`/`powi` of a tensor broadcast both operands by device copies, within `pow_bound` of Flex, card 0 (step151). Open: `powf(inf, subnormal)` is NaN on the device and `inf` in Flex (probably the SFPU flushing the exponent) | S4 |
| `float_erf`, `float_tanh`, `float_sinh`, `float_cosh`, `float_asinh`, `float_acosh`, `float_atanh` | x (SFPU, derived bounds) | S4 |
| `float_sin`, `float_cos`, `float_tan` | x (SFPU, derived bounds, every finite input) | S4 (10.2f) |
| `float_atan`, `float_asin`, `float_acos`, `float_atan2` | x (SFPU, derived bounds; `atan2` same-shape operands only, a broadcast refused) | S4 (10.2f) |
| `float_round`, `float_floor`, `float_ceil`, `float_trunc`, `float_into_int` | x F32 raw-bit rounding and saturating I32 conversion; both-card validated | S6 |
| `float_random`, `int_random` | x native seeded resident draws (Default, Uniform, Bernoulli, Normal); host construction only for a device with no GDDR; F16/F64 fail explicitly; refused inside a trace; cards 0 and 1 (step142) | S7 |
| `float_max_dim` | all F32 axes/layouts on SFPU; both-card validated | R1 |
| `float_argmax`, `float_argmin` | x resident rank-N F32, I32 output, first tie/NaN, axis up to 2^23; both-card validated | R1 |
| `float_any*`, `float_all*` | `~` Burn defaults over native comparisons, Boolean-to-F32 and supported sum axes/full sums | R1 |
| `float_max`, `float_max_abs*` | `~` Burn defaults over reshape/abs/max_dim; existing layout and size limits | R1 |
| `float_min*` | `~` Burn defaults over argmin/gather; existing gather axes and signed-zero limits | R1 |
| `float_prod{,_dim}` | x direct SFPU products on all resident F32 axes; both-card validated | R1 |
| `float_cumsum`, `float_cumprod` | x inclusive logical-order resident F32 scans on all axes; both-card validated | R1 |
| `float_cummin`, `float_cummax` | x resident Flex-order scans (NaN propagates, earlier equal element kept); both-card gates pending, card 0 passes (step130) | R1 |
| `float_sort*`, `float_argsort`, `float_topk`, `float_argtopk`, and the `int_` equivalents | x resident bitonic sort, F32/I32, any rank and axis up to 1024, stable (ties by original index), F32 in `total_cmp` order, trace-replayable; BF16 values widen, sort and narrow (flushing BF16 subnormals); longer axes, other dtypes and empty inputs refused by name; card 0 (step135 and step136, runs `1791561979`-`1791567677`) | R1 (T5) |
| `float_gather`, `float_scatter_add` | `~` resident arbitrary-axis multi-index raw gather; deterministic duplicate F32/BF16 additions, step86 | D4 |
| `float_select`, `float_select_add` | `~` arbitrary-axis resident indices and ordered additions, step86 | D4 |
| `float_expand`, `int_expand`, `bool_expand` | x nonempty stored dtypes; native byte-preserving gathers/transposes | D4 |
| `float_slice_assign`, `float_cat`, `float_repeat_dim`, `float_unfold` | `~` native raw copies/compositions, step84/85 | D4, M3 |
| `float_gather_nd`, `float_scatter_nd`, `int_gather_nd`, `int_scatter_nd` | x resident F32/I32 (BF16 gather_nd); every coordinate checked against its own axis (DOMAIN 11); scatter `Add` folds duplicates in index order, `Assign` is last writer, and `Mul`/`Min`/`Max` run the same per-tuple loop (Burn leaves their duplicates undefined; the order here is Flex's sequential one): `Mul` selects `current * value` (the `SFPMAD` multiply for F32, wrapping for I32), `Min`/`Max` select `value` where it compares lower (greater) than `current`, so a NaN never replaces and `-0.0` against `+0.0` keeps what is there. `Min`/`Max` equal Flex bit for bit over every special; F32 `Mul` equals the specification's multiply chained in index order, and Flex's IEEE product on ordinary normals. Card 0 (step127, runs `1791567843`-`1791567861`; `Mul`/`Min`/`Max`: step156, run `1791741849`, and Burn's 26 tests in the smoke tier, 858 of 858, run `1791742419`) | D4 (T2) |
| `float_cross` | x composition of slice/mul/sub/cat, device-rounding oracle; card 0 (step128) | D4 (T2) |
| `float_grid_sample_2d` | `[-]` out of scope: not planned until a model needs it | -- |

## `ActivationOps`

| Methods | Device | Item |
|---|:-:|---|
| `relu`, `relu_backward` | x (SFPU) | S1 |
| `leaky_relu`, `prelu`, `hard_sigmoid` | x (SFPU, exact; a one-element weight is expanded on the device by `device_op_ungated`, residency gate step129) | S2 |
| `sigmoid{,_backward}`, `gelu{,_backward}` | x (SFPU, derived bounds; `sigmoid_backward` exact) | S4 |
| `log_sigmoid{,_backward}` | x (SFPU, derived bounds) | S4 |
| `softmax`, `log_softmax`, `softmin` | x (device composition, derived bound; every size). Any rank and any axis: the axis is moved last, the rest flattened into rows, the same composition runs on the `[rows, len]` matrix and is put back, so an attention score `[batch, heads, t, t]` stays on the card; transposed inputs are copied to a plain matrix first. The bound is `step32`'s with `n` the axis length; card 0 (step149, found by `burn-tt-conformance`) | R2 |
| `softmin` | x (device composition, derived bound; every supported size) | R2 |

## `ModuleOps`

| Methods | Device | Item |
|---|:-:|---|
| `linear` and its three backwards | `~` over `float_matmul`; a rank-N input folds its batch into the rows (forward, `x` grad), `linear_{weight,bias}_backward` hand-written likewise | B6 |
| `embedding{,_backward}` | x (Burn's default over `select`/`select_add`, on the mover) | D4 |
| `conv1d`, `conv2d`, `conv3d`, `conv_transpose2d`, `conv_transpose3d`, their backwards, `unfold4d` | `~` native F32/BF16, grouped/ragged/dilated forward/backward, trace/training/mesh gates; acceptance open. `conv_transpose` accepts any output padding (Burn's definition; PyTorch's `< max(stride, dilation)` limit is not imposed), forward and all gradients equal Flex's, card 0 (step154) **3-D** (2026-10-11): the same patch tables, generic over the number of spatial dimensions (`Geometry<D>`, `ops/conv.rs`), no new kernel; forward, all three gradients and `padding_out` equal Flex bit for bit on small integers (step157, card 0; BF16 forward silicon-only, PACR `0x105`). 18 of Burn's 20 3-D tests pass; the other two miss their tolerance because the Matrix Unit rounds the small terms of a group of eight (see silicon-operating-notes, "Matmul accumulation: the Matrix Unit's group adder"; step158). | D6 |
| `avg_pool*`, `adaptive_avg_pool*`, `max_pool*` and backwards | x NCHW 2-D and Burn's 1-D compositions; all-padding windows refused; BF16 averages use GAPOOL | M2 |
| `layer_norm` | x Burn composition over native primitives; F32 and BF16 numerical/gradient gates | R3 |
| `attention` | `~` native F32/BF16 scale/softcap/masks/bias, zero masked rows, gradients/training, large-K views and replay; both-card forward/all-gradient references (`1791249486`) | R4 |
| `interpolate`, `interpolate_backward` (nearest, bilinear, bicubic, Lanczos3; `align_corners` either way) | x device composition, no new kernel (`ops/interpolate.rs`): every index and weight depends only on the shapes, so the host builds the tables (metadata, as pooling windows) and the card runs a word-copy gather of each output's taps, one SFPU multiply per weight factor (bilinear/bicubic: x then y factor, as Flex applies them; Lanczos3: one renormalized factor; nearest: none) and the in-order SFPU row sum. The gradient is the transpose of the forward map, each input listing its contributors in Flex's scatter order. **Nearest and bilinear equal Flex bit for bit**, forward and backward (the same f32 arithmetic in the same order; nearest moves words, so NaN payloads, `-0.0` and subnormals survive); bicubic and Lanczos3 forward are within a derived bound of Flex (`(γ_{2f} + γ_{n-1}(1 + γ_{2f})) Σ\|x a\|`, worst measured 0.70 of it), Lanczos3 forward also within a derived bound of the `image` crate's `resize` where their filters coincide (upsampling, `align_corners = false`). **Lanczos3 backward has no Flex reference** (Flex panics): it is the adjoint of the forward map, checked against `Aᵀg` with `A` read off Flex on unit impulses. BF16 widens, computes in F32 and rounds once (silicon-only gate: the narrowing pack is PACR `0x105`). Refused by name: F16, an empty input plane, and a resize staging more than 2^24 host addresses (`taps x N x C x outputs`; a separable two-pass is the follow-up, roadmap Track 2). Equal to Flex's shapes for an empty batch or output. Card 0 (step155, 15 tests, and Burn's 22 tests, in the smoke tier: 814 of 814, log `1791733008`) | -- (conformance round 2) |
| `deform_conv2d`, `ctc_loss`, `rfft`/`irfft` | | not planned until a model needs them |

## `IntTensorOps`, `BoolTensorOps`, `QTensorOps`

| Methods | Device | Item |
|---|:-:|---|
| storage on the device | x `I32`, `Bool` (any store); other int dtypes host | D3 |
| `{int,bool}_{reshape, slice, swap_dims, transpose}` | `~` views, as `float_`'s. A reshape that changes the stored matrix is a view where the strides express it, else a native copy (`reshaped_strided`: stored words, never read), exactly Flex's including `i32::MIN`/`MAX`; card 0 (step150, found by `burn-tt-conformance`) | D3 |
| `bool_argwhere` (and `nonzero`) | x device stable argsort of the negated mask and `I32` division/remainder by the shape's radices; the only host read is the 4-byte count (the result's shape); masks up to the device sort's 1,024 elements, a longer one refused by this op's name; exactly Flex's; card 0 (step153, found by `burn-tt-conformance`) | D4 |
| zero-extent tensors (`empty`, `zeros`, `cat` with an empty input) | `~` built on the host (a device buffer cannot have a zero dimension), skipped by `cat`; bit-exact against Flex, card 0 (step152, found by `burn-tt-conformance`) | D4 |
| `int_{add,sub,mul}{,_scalar}`, comparisons, `bitwise_*` | x I32 full-width wrapping ALU, signed comparisons, masked shifts; both-card validated | S5 |
| `int_{div,remainder}{,_scalar}`, `int_mean_dim` | `~` checked native I32, simulator/card-0 validated (`1791232718`) | S5 |
| `int_neg` | Burn composition over native wrapping ALU | S5 |
| `int_abs` | x wrapping (`i32::MIN` stays `i32::MIN`, as Flex); card 0 (step126, run `1791560877`) | S5 (T1) |
| `int_clamp*`, `int_sign`, `int_max_abs*` | x Burn defaults over native mask fill/abs, no downloads (step126) | S5 (T1) |
| `int_into_float` | x to F32 (SFPU, exact) | S4 (10.2d) |
| `bool_into_float`, `bool_into_int` | x native exact 0/1, F32/I32 output only | S6 |
| `int_cast` | `~` I32→I32 native; other widths fail by name (`[-]`: the device stores I32 only) (step126) | S6 (T1) |
| `int_sum*`, `int_max*`, `int_prod*`, `int_min*` | x I32 reductions (step80) | R1 |
| `int_matmul` | x exact modulo 2^32 composition (expand, `int_mul`, `int_sum_dim`), size budget 2^22 elements, never an f32 matmul; card 0 (step128) | S5 (T2) |
| `int_cumsum`, `int_cumprod`, `int_cummin`, `int_cummax`, `int_argmax`, `int_argmin` | x wrapping/signed resident scans and first-extreme indices (step131-132, card 0) | R1 |
| `bool_and`, `bool_or`, `bool_xor`, `bool_not`, `bool_equal`, `bool_equal_elem` | x (SFPU, exact) | D3 |
| `bool_any{,_dim}`, `bool_all{,_dim}` | x raw 0/1 OR/AND reductions on all axes; both-card validated | R1 |
| `int_mask_where`, `int_mask_fill`, `bool_mask_where`, `bool_mask_fill` | x raw-bit selects and fills (16777217 and `i32::MIN` exact); card 0 (step126) | S5 (T1) |
| `{int,bool}_{permute, flip, unfold}` | x dtype-generic views and copies; card 0 (step125, run `1791560886`) | D3 (T1) |
| indexing (`*_gather`, `*_select`, `*_cat`, `*_slice*`, `*_scatter*`) | `~` resident arbitrary-axis gather/select, stepped slices/assignment/cat/repeat, I32 wrapping and Boolean OR updates; ND gather/scatter are native (the `*_gather_nd`/`*_scatter_nd` row above, step127; the component `float_gather`/`float_select`/`float_cat` rows above stay `~`) | D4 |
| `QTensorOps` | explicit unsupported | Deferred; D2 is backend tensor storage, not portable quantization |
