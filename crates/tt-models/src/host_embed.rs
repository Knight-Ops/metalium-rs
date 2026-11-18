//! An embedding table kept in host memory instead of GDDR: a step uploads only its tokens' rows.
//!
//! A large vocabulary makes the table one of a model's largest tensors (Qwen3.5's 248,320 rows of
//! 5,120 are 2.5 GB in BF16), while a step reads a single row of it. With an untied head nothing
//! else on the card reads the table, so it can stay in the checkpoint: the file is mapped, and
//! [`HostEmbedding::rows`] widens the rows a step or a chunk needs to F32, exactly, as the
//! device's own lookup of a BF16 table widened to an F32 stream does. A traced step then takes
//! those rows as its input (`FixedShapeDecoder::host_embedding`).

use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use rayon::prelude::*;
use safetensors::{Dtype, SafeTensors};

use crate::load::LoadError;

/// A `[vocab, hidden]` table in a mapped checkpoint file.
pub struct HostEmbedding {
    map: Mmap,
    /// Where the table's data starts in `map`, and the element format.
    offset: usize,
    dtype: Dtype,
    vocab: usize,
    hidden: usize,
}

impl std::fmt::Debug for HostEmbedding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostEmbedding")
            .field("dtype", &self.dtype)
            .field("vocab", &self.vocab)
            .field("hidden", &self.hidden)
            .finish()
    }
}

fn error(path: &Path, what: impl std::fmt::Display) -> LoadError {
    LoadError::Store(format!("{}: {what}", path.display()))
}

impl HostEmbedding {
    /// The first tensor `is_table` names (BF16 or F32, `[vocab, hidden]`) in whichever of `paths`
    /// holds it.
    pub fn from_shards(
        paths: &[PathBuf],
        is_table: &dyn Fn(&str) -> bool,
    ) -> Result<Self, LoadError> {
        for path in paths {
            let file = File::open(path).map_err(|e| error(path, e))?;
            // SAFETY: the checkpoint is read only and is not written while a model uses it.
            let map = unsafe { Mmap::map(&file) }.map_err(|e| error(path, e))?;
            let (offset, dtype, shape) = {
                let st = SafeTensors::deserialize(&map).map_err(|e| error(path, e))?;
                let Some(name) = st.names().into_iter().find(|n| is_table(n)) else {
                    continue;
                };
                let view = st.tensor(name).map_err(|e| error(path, e))?;
                let data = view.data();
                let offset = data.as_ptr() as usize - map.as_ptr() as usize;
                (offset, view.dtype(), view.shape().to_vec())
            };
            let [vocab, hidden] = shape[..] else {
                return Err(error(path, format!("the table is {shape:?}")));
            };
            if !matches!(dtype, Dtype::BF16 | Dtype::F32) {
                return Err(error(
                    path,
                    format!("the table is {dtype:?}, not BF16 or F32"),
                ));
            }
            return Ok(Self {
                map,
                offset,
                dtype,
                vocab,
                hidden,
            });
        }
        Err(LoadError::Store(
            "no shard holds the embedding table".into(),
        ))
    }

    pub fn vocab(&self) -> usize {
        self.vocab
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// The rows of `tokens`, `tokens.len() * hidden` values, each row widened to F32 exactly.
    /// A token past the vocabulary is refused by name.
    pub fn rows(&self, tokens: &[u32]) -> Vec<f32> {
        let h = self.hidden;
        let mut out = vec![0f32; tokens.len() * h];
        out.par_chunks_mut(h)
            .zip(tokens.par_iter())
            .for_each(|(row, &t)| self.row_into(t as usize, row));
        out
    }

    fn row_into(&self, token: usize, row: &mut [f32]) {
        assert!(
            token < self.vocab,
            "token {token} past a vocabulary of {}",
            self.vocab
        );
        let h = self.hidden;
        match self.dtype {
            Dtype::BF16 => {
                let at = self.offset + token * h * 2;
                let bytes = &self.map[at..at + h * 2];
                for (y, b) in row.iter_mut().zip(bytes.chunks_exact(2)) {
                    *y = f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16);
                }
            }
            Dtype::F32 => {
                let at = self.offset + token * h * 4;
                let bytes = &self.map[at..at + h * 4];
                for (y, b) in row.iter_mut().zip(bytes.chunks_exact(4)) {
                    *y = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                }
            }
            _ => unreachable!("checked at load"),
        }
    }
}

/// Write `values` (`[vocab, hidden]`, F32) as the tensor `name` of a safetensors file: a table
/// for tests and for synthetic models, which have no checkpoint.
#[doc(hidden)]
pub fn write_table(
    path: &Path,
    name: &str,
    values: &[f32],
    vocab: usize,
    hidden: usize,
) -> Result<(), LoadError> {
    assert_eq!(values.len(), vocab * hidden);
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let view = safetensors::tensor::TensorView::new(Dtype::F32, vec![vocab, hidden], &bytes)
        .map_err(|e| error(path, e))?;
    safetensors::serialize_to_file([(name, view)], None, path).map_err(|e| error(path, e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::TensorView;

    /// A BF16 table written to a file and read back row by row: the rows are the values widened,
    /// in the order the tokens ask for them, a token repeated read twice.
    #[test]
    fn rows_are_the_tables_values_widened() {
        let (vocab, hidden) = (5, 3);
        let values: Vec<f32> = (0..vocab * hidden).map(|i| i as f32 * 0.5 - 3.0).collect();
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let view = TensorView::new(Dtype::BF16, vec![vocab, hidden], &bytes).unwrap();
        let dir = std::env::temp_dir().join(format!("host-embed-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.safetensors");
        safetensors::serialize_to_file([("embed.weight", view)], None, &path).unwrap();

        let table =
            HostEmbedding::from_shards(std::slice::from_ref(&path), &|n| n == "embed.weight")
                .unwrap();
        assert_eq!((table.vocab(), table.hidden()), (vocab, hidden));
        let got = table.rows(&[4, 0, 4, 2]);
        let want: Vec<f32> = [4, 0, 4, 2]
            .iter()
            .flat_map(|&t| values[t * hidden..(t + 1) * hidden].to_vec())
            .collect();
        assert_eq!(got, want);
        assert!(HostEmbedding::from_shards(&[path], &|n| n == "missing").is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
