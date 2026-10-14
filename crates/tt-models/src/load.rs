//! Loading a pretrained safetensors file into a Burn module, strictly.
//!
//! `burn-store`'s own `load_from` is lenient: a tensor the module has and the file lacks keeps its
//! random initialisation, and a shape mismatch is recorded and skipped (`Applier::apply_tensor`
//! returns `None` and the parameter stays as it was). For a pretrained model that is the worst
//! failure there is, because the network still runs and produces plausible nonsense. So
//! [`load_safetensors`] turns every one of those outcomes into an error and names the tensors:
//!
//! * a parameter the file does not provide (`missing`),
//! * a shape or dtype mismatch, or an unreadable tensor (`errors`),
//! * a tensor in the file the model does not consume (`unused`), unless the caller's `ignore` says
//!   it is expected (PyTorch's `num_batches_tracked` counters, a tied `lm_head`).
//!
//! The dtype is the file's: `burn-store` builds each parameter with `Tensor::from_data(data,
//! (device, snapshot.dtype))`, so a BF16 checkpoint becomes a BF16 model. Nothing here casts.

use std::fmt;
use std::path::{Path, PathBuf};

use burn::module::Module;
use burn::tensor::backend::Backend;
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};

/// Where `cargo xtask fetch-weights --model <model>` puts a model's weights.
pub fn weights_path(model: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../vendor/weights")
        .join(model)
        .join("model.safetensors")
}

/// How a checkpoint's names and layouts map onto the module.
pub struct Layout<'a> {
    /// `(regex, replacement)` applied to the file's tensor names, in order.
    pub remaps: &'a [(&'a str, &'a str)],
    /// Convert PyTorch layouts: `Linear` weights `[out, in]` to Burn's `[in, out]`, and the
    /// normalisation layers' `weight`/`bias` to `gamma`/`beta`.
    pub pytorch: bool,
    /// File tensors the module deliberately does not consume.
    pub ignore: &'a dyn Fn(&str) -> bool,
}

/// What a successful load did.
#[derive(Debug)]
pub struct Loaded {
    /// Parameters the file provided.
    pub applied: usize,
    /// File tensors left unconsumed on purpose (`Layout::ignore`).
    pub ignored: Vec<String>,
}

/// Why a load was refused.
#[derive(Debug)]
pub enum LoadError {
    /// The weights are not there; `cargo xtask fetch-weights` puts them in place.
    NotFetched(PathBuf),
    /// `burn-store` could not read or apply the file.
    Store(String),
    /// The file and the module disagree; each entry names the tensor and the disagreement.
    Mismatch(Vec<String>),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::NotFetched(p) => write!(
                f,
                "{} does not exist: run `cargo xtask fetch-weights` first",
                p.display()
            ),
            LoadError::Store(e) => write!(f, "burn-store: {e}"),
            LoadError::Mismatch(m) => {
                write!(
                    f,
                    "the checkpoint does not match the model ({} problems):",
                    m.len()
                )?;
                for line in m.iter().take(24) {
                    write!(f, "\n  {line}")?;
                }
                if m.len() > 24 {
                    write!(f, "\n  ... and {} more", m.len() - 24)?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for LoadError {}

/// Load `path` into `model`, refusing anything short of an exact match (see the module docs).
pub fn load_safetensors<B: Backend, M: Module<B>>(
    model: &mut M,
    path: &Path,
    layout: &Layout,
) -> Result<Loaded, LoadError> {
    if !path.exists() {
        return Err(LoadError::NotFetched(path.to_path_buf()));
    }
    let mut store = SafetensorsStore::from_file(path);
    if layout.pytorch {
        store = store.with_from_adapter(PyTorchToBurnAdapter);
    }
    for (from, to) in layout.remaps {
        store = store.with_key_remapping(*from, *to);
    }
    // Leniency is switched off by checking the result below, not by `allow_partial(false)`,
    // which aborts at the first missing name and so reports one problem at a time.
    store = store.allow_partial(true);

    let result = model
        .load_from(&mut store)
        .map_err(|e| LoadError::Store(e.to_string()))?;

    let mut problems = Vec::new();
    for e in &result.errors {
        problems.push(e.to_string());
    }
    for (name, why) in &result.missing {
        problems.push(format!("'{name}' is not in the file ({why})"));
    }
    // `burn-store` lists a norm layer's `weight` and `bias` as unused even though the adapter
    // found them under the module's `gamma` and `beta` and applied them (checked in
    // `a_norm_layers_scale_and_shift_load_although_burn_store_lists_them_unused`). A file name
    // counts as consumed when its alias was applied; anything else unused is still reported.
    let applied: std::collections::HashSet<&str> =
        result.applied.iter().map(String::as_str).collect();
    let consumed_by_alias = |name: &str| {
        layout.pytorch
            && [(".weight", ".gamma"), (".bias", ".beta")]
                .iter()
                .any(|(file, module)| {
                    name.strip_suffix(file)
                        .is_some_and(|stem| applied.contains(format!("{stem}{module}").as_str()))
                })
    };
    let (ignored, unexpected): (Vec<String>, Vec<String>) = result
        .unused
        .iter()
        .filter(|name| !consumed_by_alias(name))
        .cloned()
        .partition(|name| (layout.ignore)(name));
    for name in unexpected {
        problems.push(format!(
            "'{name}' is in the file and the model has no such parameter"
        ));
    }
    if problems.is_empty() {
        Ok(Loaded {
            applied: result.applied.len(),
            ignored,
        })
    } else {
        Err(LoadError::Mismatch(problems))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::nn::{BatchNorm, BatchNormConfig, Linear, LinearConfig};
    use burn::tensor::{Tensor, TensorData};
    use burn_flex::{Flex, FlexDevice};
    use burn_store::BurnToPyTorchAdapter;

    #[derive(Module, Debug)]
    struct Small<B: Backend> {
        fc: Linear<B>,
        bn: BatchNorm<B>,
    }

    #[derive(Module, Debug)]
    struct Bigger<B: Backend> {
        fc: Linear<B>,
        bn: BatchNorm<B>,
        extra: Linear<B>,
    }

    fn small() -> Small<Flex> {
        Small {
            fc: LinearConfig::new(5, 3).init(&FlexDevice),
            bn: BatchNormConfig::new(3).init(&FlexDevice),
        }
    }

    fn scratch(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "tt-models-{}-{name}.safetensors",
            std::process::id()
        ))
    }

    /// Save `model` in the layout a PyTorch checkpoint has.
    fn save<M: ModuleSnapshot<Flex>>(model: &M, path: &Path) {
        let mut store = SafetensorsStore::from_file(path)
            .with_to_adapter(BurnToPyTorchAdapter)
            .overwrite(true);
        model.save_into(&mut store).unwrap();
    }

    fn bits<const D: usize>(t: Tensor<Flex, D>) -> Vec<u32> {
        t.into_data()
            .to_vec::<f32>()
            .unwrap()
            .into_iter()
            .map(f32::to_bits)
            .collect()
    }

    const NONE: &dyn Fn(&str) -> bool = &|_| false;

    fn torch<'a>() -> Layout<'a> {
        Layout {
            remaps: &[],
            pytorch: true,
            ignore: NONE,
        }
    }

    #[test]
    fn a_checkpoint_loads_bit_for_bit() {
        let source = small();
        let path = scratch("roundtrip");
        save(&source, &path);

        let mut target = small();
        assert_ne!(
            bits(source.fc.weight.val()),
            bits(target.fc.weight.val()),
            "control: two initialisations must differ, or the test proves nothing"
        );
        let loaded = load_safetensors(&mut target, &path, &torch()).unwrap();
        assert!(loaded.applied >= 4, "applied {}", loaded.applied);
        assert_eq!(bits(source.fc.weight.val()), bits(target.fc.weight.val()));
        assert_eq!(
            bits(source.fc.bias.as_ref().unwrap().val()),
            bits(target.fc.bias.as_ref().unwrap().val())
        );
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_parameter_the_file_lacks_is_an_error_not_a_random_initialisation() {
        let path = scratch("missing");
        save(&small(), &path);
        let mut bigger = Bigger::<Flex> {
            fc: LinearConfig::new(5, 3).init(&FlexDevice),
            bn: BatchNormConfig::new(3).init(&FlexDevice),
            extra: LinearConfig::new(3, 2).init(&FlexDevice),
        };
        let err = load_safetensors(&mut bigger, &path, &torch()).unwrap_err();
        let text = err.to_string();
        assert!(matches!(err, LoadError::Mismatch(_)), "{text}");
        assert!(text.contains("extra"), "{text}");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_tensor_the_model_does_not_consume_is_an_error_unless_ignored() {
        let path = scratch("unused");
        save(
            &Bigger::<Flex> {
                fc: LinearConfig::new(5, 3).init(&FlexDevice),
                bn: BatchNormConfig::new(3).init(&FlexDevice),
                extra: LinearConfig::new(3, 2).init(&FlexDevice),
            },
            &path,
        );
        let err = load_safetensors(&mut small(), &path, &torch()).unwrap_err();
        assert!(err.to_string().contains("extra"), "{err}");

        let ignore_extra: &dyn Fn(&str) -> bool = &|n| n.starts_with("extra.");
        let loaded = load_safetensors(
            &mut small(),
            &path,
            &Layout {
                remaps: &[],
                pytorch: true,
                ignore: ignore_extra,
            },
        )
        .unwrap();
        assert_eq!(loaded.ignored.len(), 2, "{:?}", loaded.ignored);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_shape_mismatch_is_an_error() {
        let path = scratch("shape");
        save(&small(), &path);
        let mut wrong = Small::<Flex> {
            fc: LinearConfig::new(5, 4).init(&FlexDevice),
            bn: BatchNormConfig::new(4).init(&FlexDevice),
        };
        let err = load_safetensors(&mut wrong, &path, &torch()).unwrap_err();
        assert!(err.to_string().contains("expected"), "{err}");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn missing_weights_say_how_to_fetch_them() {
        let err = load_safetensors(
            &mut small(),
            Path::new("/nonexistent/m.safetensors"),
            &torch(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("fetch-weights"), "{err}");
    }

    #[test]
    fn a_norm_layers_scale_and_shift_load_although_burn_store_lists_them_unused() {
        // Gamma and beta start at 1 and 0 in both models, so a load that skipped them would
        // still match. Moving the source's off those values is what makes the check able to fail.
        let mut source = small();
        source.bn.gamma = source.bn.gamma.map(|t| t.add_scalar(0.5));
        source.bn.beta = source.bn.beta.map(|t| t.add_scalar(-0.25));
        let path = scratch("norm");
        save(&source, &path);

        let mut target = small();
        load_safetensors(&mut target, &path, &torch()).unwrap();
        assert_eq!(bits(source.bn.gamma.val()), bits(target.bn.gamma.val()));
        assert_eq!(bits(source.bn.beta.val()), bits(target.bn.beta.val()));
        assert_eq!(
            target.bn.gamma.val().into_data().to_vec::<f32>().unwrap(),
            vec![1.5; 3]
        );

        // The alias rule must not hide a real extra: `bn` itself is consumed, `stray` is not.
        let stray = Bigger::<Flex> {
            fc: LinearConfig::new(5, 3).init(&FlexDevice),
            bn: BatchNormConfig::new(3).init(&FlexDevice),
            extra: LinearConfig::new(3, 2).init(&FlexDevice),
        };
        save(&stray, &path);
        let err = load_safetensors(&mut small(), &path, &torch()).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("extra.weight"), "{text}");
        assert!(!text.contains("bn."), "{text}");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn data_round_trips_through_tensordata() {
        // The control for `bits`: a known tensor reads back as the bits it was built from.
        let t = Tensor::<Flex, 1>::from_data(TensorData::new(vec![1.0f32, -0.0], [2]), &FlexDevice);
        assert_eq!(bits(t), vec![1.0f32.to_bits(), (-0.0f32).to_bits()]);
    }
}
