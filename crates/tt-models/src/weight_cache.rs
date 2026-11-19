//! Quantized weights as the bytes they occupy in GDDR.
//!
//! The first load packs each matrix on the card and writes each channel's extent here: the
//! column layout when that is how the weight was stored, the interleaved layout otherwise,
//! including each slot's alignment tail. Every later start allocates that same layout and
//! writes the bytes back. The header checksum covers the checkpoint, the precision plan, the
//! file version and the layout stamp (channel count, column mode, architecture). Each record
//! carries its own checksum (hashed on all cores), checked before its bytes reach the card;
//! the trailer counts the records and hashes their checksums, so a write that stopped early
//! is refused.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

/// File format. Stays 1 until the format is merged; until then a changed format means deleting old caches.
pub const VERSION: u32 = 1;

const MAGIC: &[u8; 8] = b"TTWCLAY1";
const TRAILER: &[u8; 8] = b"TTWCEND1";

/// What the header checksum is taken over.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Identity {
    /// Stable description of the checkpoint (paths, sizes, modification times).
    pub model: String,
    /// The precision plan, one rule a line.
    pub plan: String,
    /// Channel count, column mode and the architecture the images were packed for.
    pub layout: String,
}

impl Identity {
    /// The bytes the header checksum covers, version first.
    pub fn bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(VERSION.to_le_bytes());
        out.extend(self.model.as_bytes());
        out.push(0);
        out.extend(self.plan.as_bytes());
        out.push(0);
        out.extend(self.layout.as_bytes());
        out
    }

    pub fn checksum(&self) -> u64 {
        fnv1a64(&self.bytes())
    }
}

/// One record. An [`Record::Image`] is one weight's GDDR extents. [`Record::Values`] is a
/// parameter the card does not store as BFP (a norm, the convolution, a derived scale), as
/// its logical floats.
#[derive(Clone, Debug, PartialEq)]
pub enum Record {
    Image {
        name: String,
        /// 8, 4 or 2.
        format: u8,
        /// 0 interleaved, 1 by column.
        layout: u8,
        rows: u32,
        cols: u32,
        slot_bytes: u32,
        channels: Vec<Vec<u8>>,
    },
    Values {
        name: String,
        /// 0 F32, 1 BF16, 2 F16. The values are the F32 the dtype widens to.
        dtype: u8,
        dims: Vec<u32>,
        values: Vec<f32>,
    },
}

impl Record {
    pub fn name(&self) -> &str {
        match self {
            Self::Image { name, .. } | Self::Values { name, .. } => name,
        }
    }

    fn check(&self) -> Result<(), String> {
        match self {
            Self::Image {
                name,
                format,
                layout,
                rows,
                cols,
                slot_bytes,
                channels,
            } => {
                if !matches!(format, 2 | 4 | 8) {
                    return Err(format!("{name}: format {format} is not BFP2/4/8"));
                }
                if !matches!(layout, 0 | 1) {
                    return Err(format!("{name}: layout {layout}"));
                }
                if *rows == 0 || *cols == 0 || *slot_bytes == 0 || slot_bytes % 64 != 0 {
                    return Err(format!("{name}: empty or unaligned image"));
                }
                if channels.is_empty()
                    || channels.iter().any(|c| c.len() % *slot_bytes as usize != 0)
                {
                    return Err(format!("{name}: a channel is not a whole number of slots"));
                }
                let n = channels[0].len();
                if channels.iter().any(|c| c.len() != n) {
                    return Err(format!("{name}: channels differ in length"));
                }
                Ok(())
            }
            Self::Values {
                name,
                dtype,
                dims,
                values,
            } => {
                if !matches!(dtype, 0..=2) {
                    return Err(format!("{name}: dtype {dtype}"));
                }
                if dims.is_empty() || dims.contains(&0) {
                    return Err(format!("{name}: empty shape"));
                }
                let n: usize = dims.iter().map(|d| *d as usize).product();
                if values.len() != n {
                    return Err(format!("{name}: {} values for {n}", values.len()));
                }
                Ok(())
            }
        }
    }
}

/// Reads a finished cache whose header names `identity`, one record at a time.
pub struct Reader {
    file: BufReader<File>,
    payload: u64,
    records: u64,
    total: u64,
    read: u64,
    done: bool,
}

impl Reader {
    /// `Ok(None)` when the file is absent, names something else, or has another version.
    /// A record whose checksum does not match, or a torn or missing trailer, is an error from
    /// [`Reader::next_record`].
    pub fn open(path: &Path, identity: &Identity) -> Result<Option<Self>, String> {
        if !path.exists() {
            return Ok(None);
        }
        let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let total = file.metadata().map(|m| m.len()).unwrap_or(0);
        let mut file = BufReader::with_capacity(1 << 20, file);
        let mut magic = [0u8; 8];
        if file.read_exact(&mut magic).is_err() || &magic != MAGIC {
            return Ok(None);
        }
        let version = read_u32(&mut file).unwrap_or(0);
        if version != VERSION {
            return Ok(None);
        }
        let id_len = read_u32(&mut file)? as usize;
        if id_len > 1 << 24 {
            return Ok(None);
        }
        let mut id_bytes = vec![0u8; id_len];
        file.read_exact(&mut id_bytes)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        let checksum = read_u64(&mut file)?;
        if checksum != identity.checksum() || id_bytes != identity.bytes() {
            return Ok(None);
        }
        Ok(Some(Self {
            file,
            payload: fnv1a64(&[]),
            records: 0,
            total,
            read: 20 + id_len as u64,
            done: false,
        }))
    }

    /// The file's length and the bytes read so far, for progress.
    pub fn progress(&self) -> (u64, u64) {
        (self.read, self.total)
    }

    /// The next record, or `Ok(None)` after a trailer that matches. A torn file is an error.
    pub fn next_record(&mut self) -> Result<Option<Record>, String> {
        if self.done {
            return Ok(None);
        }
        let mut mark = [0u8; 8];
        if let Err(e) = self.file.read_exact(&mut mark) {
            return Err(if e.kind() == std::io::ErrorKind::UnexpectedEof {
                "weight cache: no trailer".into()
            } else {
                e.to_string()
            });
        }
        self.read += 8;
        if &mark == TRAILER {
            let records = read_u64(&mut self.file)?;
            let got = read_u64(&mut self.file)?;
            if records != self.records || got != self.payload {
                return Err("weight cache: trailer does not match".into());
            }
            let mut extra = [0u8; 1];
            if self.file.read(&mut extra).unwrap_or(0) != 0 {
                return Err("weight cache: bytes after the trailer".into());
            }
            self.read += 16;
            self.done = true;
            return Ok(None);
        }
        let len = u64::from_le_bytes(mark);
        if len < 8 || len > self.total.saturating_sub(self.read) {
            return Err("weight cache: a record longer than the file".into());
        }
        let mut body = vec![0u8; len as usize];
        self.file.read_exact(&mut body).map_err(|e| e.to_string())?;
        let want = read_u64(&mut self.file)?;
        self.read += len + 8;
        let got = parallel_hash(&body);
        if got != want {
            return Err(format!(
                "weight cache: record {} does not hash",
                self.records
            ));
        }
        self.payload = fnv1a64_continue(self.payload, &got.to_le_bytes());
        self.records += 1;
        let record = parse(&body)?;
        record.check()?;
        Ok(Some(record))
    }
}

/// One record's body: kind and name length, name, then the kind's fields.
fn parse(body: &[u8]) -> Result<Record, String> {
    let mut c = Cursor { b: body, at: 0 };
    let name_len = c.u32()? as usize;
    let kind = c.u8()?;
    if c.take(3)? != [0, 0, 0] {
        return Err("weight cache: bad record header".into());
    }
    let name = String::from_utf8(c.take(name_len)?.to_vec()).map_err(|e| e.to_string())?;
    let record = match kind {
        1 => {
            let format = c.u8()?;
            let layout = c.u8()?;
            c.take(2)?;
            let rows = c.u32()?;
            let cols = c.u32()?;
            let slot_bytes = c.u32()?;
            let nchan = c.u32()? as usize;
            if nchan > 1024 {
                return Err(format!("{name}: {nchan} channels"));
            }
            let mut channels = Vec::with_capacity(nchan);
            for _ in 0..nchan {
                let len = c.u64()? as usize;
                channels.push(c.take(len)?.to_vec());
            }
            Record::Image {
                name,
                format,
                layout,
                rows,
                cols,
                slot_bytes,
                channels,
            }
        }
        2 => {
            let dtype = c.u8()?;
            let rank = c.u8()? as usize;
            c.take(2)?;
            let mut dims = Vec::with_capacity(rank);
            for _ in 0..rank {
                dims.push(c.u32()?);
            }
            let n = c.u32()? as usize;
            let values = c
                .take(n * 4)?
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                .collect();
            Record::Values {
                name,
                dtype,
                dims,
                values,
            }
        }
        _ => return Err(format!("weight cache: kind {kind}")),
    };
    if c.at != body.len() {
        return Err(format!("{}: bytes after the record", record.name()));
    }
    Ok(record)
}

struct Cursor<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self
            .at
            .checked_add(n)
            .filter(|e| *e <= self.b.len())
            .ok_or("weight cache: a record ends early")?;
        let out = &self.b[self.at..end];
        self.at = end;
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
}

/// Writes a cache, trailer last. Dropping it without [`Writer::finish`] leaves no trailer.
pub struct Writer {
    file: BufWriter<File>,
    payload: u64,
    records: u64,
    written: u64,
    finished: bool,
}

impl Writer {
    pub fn create(path: &Path, identity: &Identity) -> Result<Self, String> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
        let file = File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut file = BufWriter::with_capacity(1 << 20, file);
        file.write_all(MAGIC).map_err(io)?;
        file.write_all(&VERSION.to_le_bytes()).map_err(io)?;
        let id = identity.bytes();
        file.write_all(&(id.len() as u32).to_le_bytes())
            .map_err(io)?;
        file.write_all(&id).map_err(io)?;
        file.write_all(&identity.checksum().to_le_bytes())
            .map_err(io)?;
        Ok(Self {
            file,
            payload: fnv1a64(&[]),
            records: 0,
            written: 20 + id.len() as u64,
            finished: false,
        })
    }

    pub fn push(&mut self, record: &Record) -> Result<(), String> {
        record.check()?;
        let mut body = Vec::new();
        let name = record.name().as_bytes();
        body.extend((name.len() as u32).to_le_bytes());
        match record {
            Record::Image {
                format,
                layout,
                rows,
                cols,
                slot_bytes,
                channels,
                ..
            } => {
                body.reserve(32 + name.len() + channels.iter().map(|c| c.len() + 8).sum::<usize>());
                body.push(1);
                body.extend([0, 0, 0]);
                body.extend(name);
                body.push(*format);
                body.push(*layout);
                body.extend([0, 0]);
                body.extend(rows.to_le_bytes());
                body.extend(cols.to_le_bytes());
                body.extend(slot_bytes.to_le_bytes());
                body.extend((channels.len() as u32).to_le_bytes());
                for channel in channels {
                    body.extend((channel.len() as u64).to_le_bytes());
                    body.extend(channel);
                }
            }
            Record::Values {
                dtype,
                dims,
                values,
                ..
            } => {
                body.push(2);
                body.extend([0, 0, 0]);
                body.extend(name);
                body.push(*dtype);
                body.push(dims.len() as u8);
                body.extend([0, 0]);
                for dim in dims {
                    body.extend(dim.to_le_bytes());
                }
                body.extend((values.len() as u32).to_le_bytes());
                for value in values {
                    body.extend(value.to_le_bytes());
                }
            }
        }
        let hash = parallel_hash(&body);
        self.file
            .write_all(&(body.len() as u64).to_le_bytes())
            .map_err(io)?;
        self.file.write_all(&body).map_err(io)?;
        self.file.write_all(&hash.to_le_bytes()).map_err(io)?;
        self.payload = fnv1a64_continue(self.payload, &hash.to_le_bytes());
        self.records += 1;
        self.written += body.len() as u64 + 16;
        Ok(())
    }

    /// Bytes written so far.
    pub fn written(&self) -> u64 {
        self.written
    }

    pub fn finish(mut self) -> Result<(), String> {
        self.file.write_all(TRAILER).map_err(io)?;
        self.file
            .write_all(&self.records.to_le_bytes())
            .map_err(io)?;
        self.file
            .write_all(&self.payload.to_le_bytes())
            .map_err(io)?;
        self.file.flush().map_err(io)?;
        self.finished = true;
        Ok(())
    }
}

impl Drop for Writer {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.file.flush();
        }
    }
}

fn io(e: std::io::Error) -> String {
    e.to_string()
}

fn read_u32(file: &mut impl Read) -> Result<u32, String> {
    let mut b = [0u8; 4];
    file.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(file: &mut impl Read) -> Result<u64, String> {
    let mut b = [0u8; 8];
    file.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(u64::from_le_bytes(b))
}

/// Bytes a [`parallel_hash`] chunk covers.
const HASH_CHUNK: usize = 1 << 20;

/// FNV-1a 64 over each 1 MiB chunk on all cores, then FNV-1a over the chunk hashes and the
/// length. FNV runs at about a byte a cycle, so one thread over a 24 GB cache would take
/// most of the load.
fn parallel_hash(bytes: &[u8]) -> u64 {
    use rayon::prelude::*;
    let parts: Vec<u64> = bytes.par_chunks(HASH_CHUNK).map(fnv1a64_words).collect();
    let mut h = fnv1a64(&(bytes.len() as u64).to_le_bytes());
    for p in parts {
        h = fnv1a64_continue(h, &p.to_le_bytes());
    }
    h
}

/// FNV-1a over 8-byte words (the tail byte-wise): 8× fewer multiplies than byte-wise.
fn fnv1a64_words(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut words = bytes.chunks_exact(8);
    for w in &mut words {
        hash ^= u64::from_le_bytes(w.try_into().unwrap());
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    fnv1a64_continue(hash, words.remainder())
}

/// FNV-1a 64. Stable across Rust versions, unlike `std::hash`.
fn fnv1a64(bytes: &[u8]) -> u64 {
    fnv1a64_continue(0xcbf2_9ce4_8422_2325, bytes)
}

fn fnv1a64_continue(mut hash: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Checkpoint identity from the shard files a load will read: path, length and modification time.
pub fn model_identity(paths: &[PathBuf]) -> Result<String, String> {
    let mut lines = Vec::new();
    for path in paths {
        let meta = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let modified = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        lines.push(format!("{} {} {}", path.display(), meta.len(), modified));
    }
    Ok(lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(plan: &str) -> Identity {
        Identity {
            model: "model.safetensors 4 1".into(),
            plan: plan.into(),
            layout: "channels=8 columns=1 layers=1".into(),
        }
    }

    fn image(name: &str) -> Record {
        Record::Image {
            name: name.into(),
            format: 8,
            layout: 1,
            rows: 32,
            cols: 64,
            slot_bytes: 64,
            channels: vec![[1, 2, 3, 4].repeat(16), [9, 8, 7, 6].repeat(16)],
        }
    }

    fn values(name: &str) -> Record {
        Record::Values {
            name: name.into(),
            dtype: 0,
            dims: vec![2],
            values: vec![1.0, -2.0],
        }
    }

    #[test]
    fn a_finished_cache_reads_back_and_a_different_plan_does_not() {
        let dir = std::env::temp_dir().join(format!("ttwc-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("weights.ttwc");
        let identity = id("mlp gate_proj Bfp4");
        let a = image("layers.0.mlp.down_proj.weight");
        let b = values("layers.0.input_layernorm.weight");
        let mut w = Writer::create(&path, &identity).unwrap();
        w.push(&a).unwrap();
        w.push(&b).unwrap();
        w.finish().unwrap();
        let mut reader = Reader::open(&path, &identity).unwrap().unwrap();
        assert_eq!(reader.next_record().unwrap().unwrap(), a);
        assert_eq!(reader.next_record().unwrap().unwrap(), b);
        assert!(reader.next_record().unwrap().is_none());
        assert!(Reader::open(&path, &id("other")).unwrap().is_none());
        let mut other = id("mlp gate_proj Bfp4");
        other.layout = "channels=7 columns=1 layers=1".into();
        assert!(Reader::open(&path, &other).unwrap().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_unfinished_file_is_refused() {
        let dir = std::env::temp_dir().join(format!("ttwc-open-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("weights.ttwc");
        let identity = id("plan");
        let mut w = Writer::create(&path, &identity).unwrap();
        w.push(&values("w")).unwrap();
        drop(w);
        let mut reader = Reader::open(&path, &identity).unwrap().unwrap();
        assert!(reader.next_record().unwrap().is_some());
        assert!(reader.next_record().is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_flipped_byte_is_found_before_the_record_is_used() {
        let dir = std::env::temp_dir().join(format!("ttwc-flip-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("weights.ttwc");
        let identity = id("plan");
        let mut w = Writer::create(&path, &identity).unwrap();
        w.push(&image("a")).unwrap();
        w.finish().unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        let n = bytes.len();
        bytes[n - 40] ^= 1;
        std::fs::write(&path, &bytes).unwrap();
        let mut reader = Reader::open(&path, &identity).unwrap().unwrap();
        assert!(reader.next_record().is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_parallel_hash_sees_every_chunk() {
        let a = vec![7u8; 3 * HASH_CHUNK + 5];
        let mut b = a.clone();
        b[2 * HASH_CHUNK + 1] = 8;
        assert_ne!(parallel_hash(&a), parallel_hash(&b));
        assert_ne!(parallel_hash(&a), parallel_hash(&a[..a.len() - 1]));
        assert_eq!(parallel_hash(&a), parallel_hash(&a.clone()));
    }

    #[test]
    fn the_checksum_changes_with_the_model_the_plan_and_the_version() {
        let a = id("Bfp8");
        let b = Identity {
            model: "other 1 2".into(),
            plan: a.plan.clone(),
            layout: a.layout.clone(),
        };
        let c = Identity {
            model: a.model.clone(),
            plan: "Bfp4".into(),
            layout: a.layout.clone(),
        };
        assert_ne!(a.checksum(), b.checksum());
        assert_ne!(a.checksum(), c.checksum());
        assert_eq!(a.checksum(), id("Bfp8").checksum());
        assert!(a.bytes().starts_with(&VERSION.to_le_bytes()));
    }
}
