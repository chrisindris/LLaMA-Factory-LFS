# Spec: PyTorch / LLaMA-Factory dataloader for Spatial-SSRL H5 images

**Audience:** an LLM (or engineer) implementing image loading for Deep Learning training with **LLaMA-Factory** (this repo’s SFT-coldstart tree under `training/SFT-coldstart/`, a LLaMA-Factory fork).  
**Goal:** replace filesystem JPEG trees with the packed dataset at  
`/scratch/indrisch/Spatial-SSRL_images_h5`  
without changing model code, templates, or the text side of the SFT JSON.

Related packing docs: [`docs/images_details.md`](images_details.md).  
Packing pipeline: `scripts/format_SpatialSSRL_multinode_wrapper.sh` → `scripts/format_SpatialSSRL_multinode.sh` → `scripts/format_SpatialSSRL_multinode.py`.

---

## 1. Provenance and paths

| Role | Path |
|------|------|
| **Packed images root (default)** | `/scratch/indrisch/Spatial-SSRL_images_h5` |
| **Source (default HF snapshot)** | `$HF_HOME/datasets--internlm--Spatial-SSRL-81k/snapshots/54b82086060a5612f95588b4979446da2282bcd9` |
| **How packed** | Multinode pack: white-pad JPEGs per logical folder into one HDF5 each; merge JSON shards into one global index |

Env overrides used by the wrapper:

- `SPATIALSSRL_INPUT_DIR` — input snapshot root  
- `SPATIALSSRL_OUTPUT_DIR` — output H5 root (default above)  
- `SPATIALSSRL_SHARDING_MODE` — `folder` (default) or `file`

**Sharding mode that produced the current tree:** `folder` → exactly **one H5 + one JSON shard per logical folder**, plus a merged global index.

Do **not** assume JPEG folders still exist under the H5 root. The H5 root contains only:

```text
/scratch/indrisch/Spatial-SSRL_images_h5/
├── coldstart_SFT.h5
├── coldstart_SFT.json
├── crop.h5
├── crop.json
├── depth.h5
├── depth.json
├── flip.h5
├── flip.json
├── position.h5
├── position.json
├── shuffle.h5
├── shuffle.json
└── spatial_ssrl_images_index.json   # USE THIS as the canonical index
```

Approximate on-disk size: ~80 GB total. Example H5 sizes: crop ~19 G, depth ~21 G, position ~27 G, shuffle ~8.8 G, coldstart_SFT ~3.5 G, flip ~2.4 G.

---

## 2. Logical folders and counts

Six logical folders (names used as H5 basenames and as the first path segment of index keys):

| Logical name | Source relative to snapshot | Index key prefix | Approx. N (this build) |
|--------------|-----------------------------|------------------|------------------------|
| `crop` | `images/crop` | `crop/` | 101000 |
| `depth` | `images/depth` | `depth/` | 20620 |
| `flip` | `images/flip` | `flip/` | 4005 |
| `position` | `images/position` | `position/` | 20200 |
| `shuffle` | `images/shuffle` | `shuffle/` | 16028 |
| `coldstart_SFT` | `coldstart_SFT_images` | `coldstart_SFT/` | 3597 |

**Total indexed images:** 165450 (sum of the six folders).

Per-folder max canvas sizes **from this build’s index** (max of stored `width`/`height`; H5 axis sizes match folder max after packing):

| Folder | max width | max height |
|--------|-----------|------------|
| `crop` | 640 | 640 |
| `depth` | 959 | 738 |
| `flip` | 640 | 640 |
| `position` | 1185 | 885 |
| `shuffle` | 640 | 640 |
| `coldstart_SFT` | 1185 | 885 |

**Do not hardcode these maxes for decoding.** Always crop using the per-image `width`/`height` from the JSON index. Maxes only describe the H5 canvas size.

---

## 3. HDF5 layout (mandatory)

Created by `pack_folder()` in `scripts/format_SpatialSSRL_multinode.py`.

### 3.1 Dataset

- **File name:** `{folder}.h5` in folder mode (e.g. `crop.h5`).  
  In multi-node **file** sharding: `node_{i}_{folder}.h5` (not the current default tree).
- **Dataset name:** `images` (only data array needed for training).
- **Shape:** `(N, max_width, max_height, 3)`
- **Dtype:** `uint8`
- **Compression:** gzip (level 4 by default), **chunks** `(1, max_width, max_height, 3)` (one image per chunk — good for random access).

### 3.2 Axis convention (critical — easy to get wrong)

```text
axis 0  →  image index  (h5index)
axis 1  →  width   (W)   ← NOT height
axis 2  →  height  (H)
axis 3  →  channel (RGB)
```

So each slice is **`(W, H, 3)` (WHC)**, **not** the usual PIL/numpy `(H, W, 3)` (HWC).

Packing stores:

1. Load JPEG with PIL → numpy HWC  
2. Transpose to WHC  
3. Allocate white canvas `np.full((max_w, max_h, 3), 255, uint8)`  
4. Paste real pixels into **top-left**: `canvas[:w, :h, :] = arr[:w, :h, :]`

### 3.3 Attributes (optional but useful)

On the `images` dataset (and mirrored on the file):

| Attr | Meaning |
|------|---------|
| `max_width`, `max_height` | Canvas size |
| `folder` | Logical folder name |
| `n_images` | N after successful writes |
| `layout` | `"N,max_width,max_height,3"` |
| `pad_value` | `255` |

### 3.4 Decode algorithm (required)

```python
# entry from spatial_ssrl_images_index.json
# entry = {"width": W, "height": H, "h5file": "crop.h5", "h5index": i}

canvas = h5["images"][entry["h5index"]]          # (max_w, max_h, 3), uint8
image_whc = canvas[: entry["width"], : entry["height"], :]  # (W, H, 3)
image_hwc = np.transpose(image_whc, (1, 0, 2))   # (H, W, 3) for PIL / torch vision
pil = Image.fromarray(image_hwc, mode="RGB")
```

**Rules:**

1. **Always crop** with index `width`/`height`. Training on the full white-padded canvas is wrong (distorts aspect ratio and wastes pixels under `image_max_pixels`).
2. **Always transpose** WHC → HWC before `Image.fromarray` or before any code that assumes HWC.
3. Values are already RGB uint8; no JPEG re-encode required for training.

---

## 4. JSON index schema (mandatory)

### 4.1 Canonical global index

**File:** `/scratch/indrisch/Spatial-SSRL_images_h5/spatial_ssrl_images_index.json`

Structure: one flat `dict`:

```json
{
  "coldstart_SFT/img_0.jpg": {
    "width": 1185,
    "height": 807,
    "h5file": "coldstart_SFT.h5",
    "h5index": 0
  },
  "crop/blackened_image_29801.jpg": {
    "width": 500,
    "height": 375,
    "h5file": "crop.h5",
    "h5index": 0
  },
  "depth/image_1.jpg": {
    "width": 954,
    "height": 723,
    "h5file": "depth.h5",
    "h5index": 0
  }
}
```

| Field | Type | Meaning |
|-------|------|---------|
| **key** | string | `{logical_folder}/{basename}.jpg` — folder-qualified because basenames collide across folders (especially depth vs position). |
| `width` | int | Original pixel width (≤ folder max_w) |
| `height` | int | Original pixel height (≤ folder max_h) |
| `h5file` | string | **Basename only** of the H5 (resolve against the dataset root) |
| `h5index` | int | Row along axis 0 of `images` |

Per-folder shards (`crop.json`, …) use the **same key/value schema** and are superseded by the global merge for loaders. Prefer the global file; ignore shards unless debugging a pack.

Keys starting with `_` are reserved/skipped by the merger; do not use them as image keys.

### 4.2 Path remapping from annotation files (critical)

Training annotations still use **original filesystem-relative paths**, not H5 index keys.

| Source of truth | Example path in annotations | Index key to look up |
|-----------------|----------------------------|----------------------|
| SFT coldstart JSON (`SFT-coldstart.json`) | `coldstart_SFT_images/img_0.jpg` | `coldstart_SFT/img_0.jpg` |
| GRPO / parquet-style (folder-qualified under `images/`) | `crop/….jpg`, `depth/….jpg`, … | same as key if already `crop/…` etc. |
| Snapshot layout | `images/crop/….jpg` | `crop/….jpg` |
| Snapshot coldstart dir | `coldstart_SFT_images/img_0.jpg` | `coldstart_SFT/img_0.jpg` |

**Normalize every media path before index lookup:**

```text
1. Strip leading media_dir / absolute prefix if present.
2. Replace backslashes with '/'.
3. Map:
     coldstart_SFT_images/  →  coldstart_SFT/
     images/crop/           →  crop/
     images/depth/          →  depth/
     images/flip/           →  flip/
     images/position/       →  position/
     images/shuffle/        →  shuffle/
4. If path already starts with one of:
     crop/ depth/ flip/ position/ shuffle/ coldstart_SFT/
   keep as-is (after basename folder check).
5. Lookup key in spatial_ssrl_images_index.json.
6. On miss → raise a clear error (do not silently fall back to Image.open of a missing JPEG unless explicitly configured).
```

`media_dir` in LLaMA-Factory currently joins `media_dir` + relative path and expects a real file (`converter.DatasetConverter._find_medias`). With H5, **file existence checks against JPEG trees will fail**. The loader must either:

- resolve to **PIL `Image` objects** (or bytes) **before** `_regularize_images` opens paths, or  
- patch path resolution so “paths” never hit `os.path.isfile` / `Image.open(path)` on missing JPEGs.

---

## 5. How LLaMA-Factory consumes images today

Relevant code (SFT tree):

| Component | Path | Behavior |
|-----------|------|----------|
| Dataset registry | `training/SFT-coldstart/data/dataset_info.json` | e.g. `"coldstart"` → `SFT-coldstart.json`, columns `instruction`/`input`/`output`/`images` |
| Train YAML | e.g. `sft_scripts/Qwen2.5-VL-3B.yaml` | `dataset: coldstart`, `media_dir: ./data`, `template: qwen2_vl` |
| Path join | `data/converter.py` → `_find_medias` | `os.path.join(media_dir, path)` if file exists; else keeps original path + warning |
| Decode | `data/mm_plugin.py` → `_regularize_images` | Accepts `str` (path) \| `bytes` \| `dict{bytes,path}` \| **`PIL.Image`** → RGB preprocess / resize |

`ImageInput` type:

```python
ImageInput = Union[str, bytes, EncodedImage, BinaryIO, ImageObject]
```

**Implication for the H5 loader:** the cleanest integration is to put **`PIL.Image.Image` instances** (or in-memory JPEG/PNG `bytes`) into the example’s image field **after** alignment, so `_regularize_images` never calls `Image.open` on a filesystem path.

Coldstart annotation shape (Alpaca-style, already registered):

```json
{
  "instruction": "... text with <image> ...",
  "input": "",
  "output": "...",
  "images": ["coldstart_SFT_images/img_0.jpg"]
}
```

`dataset_info.json` entry pattern:

```json
"coldstart": {
  "file_name": "SFT-coldstart.json",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "images": "images"
  }
}
```

Text / chat conversion stays unchanged. Only the **media resolution** path changes.

---

## 6. Required deliverable: what to implement

Implement a reusable image backend plus a LLaMA-Factory-friendly integration. Prefer **minimal invasive patches** to LLaMA-Factory.

### 6.1 Core class: `SpatialSSRLH5ImageStore` (required)

Responsibilities:

1. Load `spatial_ssrl_images_index.json` once.  
2. Resolve `h5file` basenames against `root` (`/scratch/indrisch/Spatial-SSRL_images_h5`).  
3. Keep a cache of open `h5py.File(..., "r")` handles (lazy open per file).  
4. `normalize_key(path: str) -> str` implementing §4.2.  
5. `get_pil(path_or_key: str) -> PIL.Image.Image` implementing §3.4.  
6. `get_numpy_hwc(path_or_key: str) -> np.ndarray` optional.  
7. Thread/process safety notes (see §7).

Suggested constructor:

```python
SpatialSSRLH5ImageStore(
    root: str | Path = "/scratch/indrisch/Spatial-SSRL_images_h5",
    index_name: str = "spatial_ssrl_images_index.json",
)
```

Suggested public API:

```python
def normalize_key(self, path: str) -> str: ...
def contains(self, path: str) -> bool: ...
def get_entry(self, path: str) -> dict: ...  # width, height, h5file, h5index
def get_pil(self, path: str) -> Image.Image: ...
def get_batch_pil(self, paths: list[str]) -> list[Image.Image]: ...
def close(self) -> None: ...
```

Dependencies: `h5py`, `numpy`, `Pillow` (already used by packing and training).

### 6.2 LLaMA-Factory integration options (pick one; A preferred)

#### Option A — Resolve to PIL in a thin wrapper around dataset conversion (recommended)

1. After HF `datasets` loads `SFT-coldstart.json` (or any multimodal JSON), map the `images` column (or `_images` after `align_dataset`) from path strings → `PIL.Image` via the store.  
2. Ensure `_find_medias` does **not** require `os.path.isfile` for these entries (either skip join when value is already `Image`, or replace `_find_medias` so non-str media pass through).  
3. `_regularize_images` already accepts PIL — no mm_plugin change needed if images are PIL by then.

Hook points (in priority order):

1. Subclass / wrap `align_dataset` or the Alpaca converter’s `_images` assignment.  
2. Or map the column with `dataset.map(fn, num_proc=1)` **before** training, with `fn` that opens H5 → PIL.  
   - Caution: PIL objects and multiprocessing `num_proc>1` / dataset cache can be painful; prefer `num_proc=1` or resolve **lazily in the collator / processor**, not in a cached map.

**Better variant of A (lazy):** keep string keys in the HF dataset; patch only `MMPluginMixin._regularize_images` (or a small helper it calls) so that when `isinstance(image, str)` and the path is not a real file, call `store.get_pil(image)`.

Pseudo:

```python
if isinstance(image, str):
    if os.path.isfile(image):
        image = Image.open(image)
    else:
        image = h5_store.get_pil(image)  # normalize_key inside
```

This preserves LLaMA-Factory’s caching of tokenized text while loading pixels at train time (desired).

#### Option B — Custom `torch.utils.data.Dataset`

Build a standalone `Dataset` that:

- Loads the SFT JSON list  
- `__getitem__` returns the same dict structure LLaMA-Factory’s collator expects after processing  

Only use this if bypassing LLaMA-Factory’s `get_dataset` entirely. **Not preferred** for SFT-coldstart YAML workflows, but fine for experiments.

#### Option C — Materialize JPEG cache

Extract all H5 images back to a directory tree matching annotation paths. Works with stock LLaMA-Factory but **defeats** the H5 design (file-count / inode goals). Do not do this unless forced.

### 6.3 `dataset_info` / YAML (if adding a new dataset name)

If registering a dedicated H5-backed entry (optional; can keep `coldstart` and only change loading):

```yaml
### dataset
dataset: coldstart
# Point media_dir at annotation JSON parent; images come from H5 store, not media_dir files.
media_dir: /path/to/dir/containing/SFT-coldstart.json
# Or keep media_dir unused for images when using Option A lazy resolve.
```

Place `SFT-coldstart.json` (text annotations) somewhere LLaMA-Factory can load via `dataset_dir` / `file_name`. Images are **not** read from `media_dir` when using the H5 store.

---

## 7. PyTorch DataLoader / multiprocessing constraints

`h5py.File` handles are **not reliably fork-safe** across DataLoader workers.

**Requirements:**

1. Do **not** open H5 files in the parent and share handles into forked workers.  
2. Use one of:
   - `worker_init_fn` that constructs a **per-worker** `SpatialSSRLH5ImageStore` (global in worker process), or  
   - lazy open on first use **inside** the worker with handles stored on `threading.local()` / process-global singleton keyed by `pid`.  
3. Prefer `persistent_workers=True` when `num_workers > 0` so handles stay warm.  
4. `num_workers=0` is always correct (simpler debugging).  
5. Chunking is per-image → random `h5index` access is fine; sequential access is slightly better but not required.  
6. For multi-GPU DDP, each rank’s workers open their own read-only H5 handles (HDF5 read-only multi-process is OK if each process has its own handle).

Example worker pattern:

```python
_STORE = None

def worker_init_fn(_worker_id: int) -> None:
    global _STORE
    _STORE = SpatialSSRLH5ImageStore("/scratch/indrisch/Spatial-SSRL_images_h5")

def get_store() -> SpatialSSRLH5ImageStore:
    global _STORE
    if _STORE is None:
        _STORE = SpatialSSRLH5ImageStore("/scratch/indrisch/Spatial-SSRL_images_h5")
    return _STORE
```

LLaMA-Factory’s `dataloader_num_workers` (YAML) maps to HF Trainer’s DataLoader — the patch in `_regularize_images` / collator must use the per-process store singleton.

---

## 8. Reference implementation sketch (normative behavior)

The following is the **behavior** the implementation must match (language may differ).

```python
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image

# Prefix rewrites: longest-prefix first
_PREFIX_MAP: list[tuple[str, str]] = [
    ("coldstart_SFT_images/", "coldstart_SFT/"),
    ("images/crop/", "crop/"),
    ("images/depth/", "depth/"),
    ("images/flip/", "flip/"),
    ("images/position/", "position/"),
    ("images/shuffle/", "shuffle/"),
]

_KNOWN_ROOTS = (
    "crop/",
    "depth/",
    "flip/",
    "position/",
    "shuffle/",
    "coldstart_SFT/",
)


class SpatialSSRLH5ImageStore:
    def __init__(
        self,
        root: str | os.PathLike = "/scratch/indrisch/Spatial-SSRL_images_h5",
        index_name: str = "spatial_ssrl_images_index.json",
    ) -> None:
        self.root = Path(root).resolve()
        index_path = self.root / index_name
        if not index_path.is_file():
            raise FileNotFoundError(f"Missing global index: {index_path}")
        import json

        with index_path.open("r", encoding="utf-8") as f:
            self.index: dict[str, dict[str, Any]] = json.load(f)
        self._files: dict[str, h5py.File] = {}

    def normalize_key(self, path: str) -> str:
        p = path.replace("\\", "/").lstrip("./")
        # Drop absolute root if someone joined media_dir + relpath incorrectly
        # Keep only the trailing logical suffix when possible.
        for src, dst in _PREFIX_MAP:
            if p.endswith(src) or f"/{src}" in f"/{p}":
                # take substring from first occurrence of src
                idx = p.find(src)
                if idx >= 0:
                    p = dst + p[idx + len(src) :]
                    break
            if p.startswith(src):
                p = dst + p[len(src) :]
                break
        # Already logical?
        if not p.startswith(_KNOWN_ROOTS):
            # try basename-only coldstart: img_123.jpg → coldstart_SFT/img_123.jpg if unique
            # Prefer failing closed: require folder-qualified keys for multi-folder corpora.
            pass
        return p

    def get_entry(self, path: str) -> dict[str, Any]:
        key = self.normalize_key(path)
        try:
            return self.index[key]
        except KeyError as e:
            raise KeyError(
                f"Image key not in H5 index after normalize: {path!r} -> {key!r}"
            ) from e

    def _h5(self, h5file: str) -> h5py.File:
        if h5file not in self._files:
            path = self.root / h5file
            if not path.is_file():
                raise FileNotFoundError(path)
            self._files[h5file] = h5py.File(path, "r")
        return self._files[h5file]

    def get_pil(self, path: str) -> Image.Image:
        entry = self.get_entry(path)
        h5 = self._h5(entry["h5file"])
        canvas = h5["images"][entry["h5index"]]  # (max_w, max_h, 3)
        w, h = int(entry["width"]), int(entry["height"])
        whc = np.asarray(canvas[:w, :h, :], dtype=np.uint8)
        hwc = np.transpose(whc, (1, 0, 2))
        return Image.fromarray(hwc, mode="RGB")

    def close(self) -> None:
        for f in self._files.values():
            f.close()
        self._files.clear()
```

### 8.1 Patch sketch for LLaMA-Factory `mm_plugin._regularize_images`

```python
# Inside the loop over images, replace bare Image.open(str):
if isinstance(image, str):
    if os.path.isfile(image):
        image = Image.open(image)
    else:
        image = get_store().get_pil(image)
```

Also update `_find_medias` so missing files under `media_dir` do not only warn and keep a useless relative path — either leave the relative path for the H5 store, or resolve early. Leaving the **annotation-relative** string (e.g. `coldstart_SFT_images/img_0.jpg`) is correct for `normalize_key`.

---

## 9. Correctness checklist (implementer must satisfy)

- [ ] Uses **global** index `spatial_ssrl_images_index.json`, not only one folder shard.  
- [ ] Resolves `h5file` as `root / basename`.  
- [ ] Crops with per-image `width`/`height` (no white border in model input).  
- [ ] Transposes WHC → HWC before PIL.  
- [ ] Maps `coldstart_SFT_images/…` → `coldstart_SFT/…`.  
- [ ] Maps `images/{crop,depth,flip,position,shuffle}/…` → `{folder}/…`.  
- [ ] Does not rely on JPEG trees existing on disk for training.  
- [ ] Produces `PIL.Image` RGB (or path/bytes that open to the same pixels as original JPEGs).  
- [ ] Safe under `dataloader_num_workers > 0` (per-process H5 handles).  
- [ ] Key miss raises; does not train on blank/white full canvas silently.  
- [ ] Works for coldstart N≈3597 and scales to full 165k multi-folder GRPO-style paths if those annotations are used later.

### 9.1 Smoke tests (required)

```python
store = SpatialSSRLH5ImageStore("/scratch/indrisch/Spatial-SSRL_images_h5")

# Coldstart path as in SFT-coldstart.json
im = store.get_pil("coldstart_SFT_images/img_0.jpg")
assert im.mode == "RGB"
assert im.size == (store.get_entry("coldstart_SFT_images/img_0.jpg")["width"],
                   store.get_entry("coldstart_SFT_images/img_0.jpg")["height"])
# PIL size is (W, H)

# Folder-qualified index key
im2 = store.get_pil("depth/image_1.jpg")
assert im2.size[0] == store.get_entry("depth/image_1.jpg")["width"]

# Collision safety: same basename different folders must differ if both exist
# (depth vs position share patterns like image_10.jpg)
a = store.get_pil("depth/image_10.jpg")
b = store.get_pil("position/image_10.jpg")
assert a.size != b.size or np.any(np.asarray(a) != np.asarray(b))
```

Optional: if original JPEGs still exist on the snapshot, assert `np.allclose` / exact equality of HWC arrays between H5 decode and `Image.open(jpeg)`.

---

## 10. Out of scope / non-goals

- Re-packing or rewriting H5 files.  
- Changing GRPO/EasyR1 parquet schema (same image store can be reused by swapping `process_image` in `training/GRPO/verl/utils/dataset.py` similarly).  
- Training hyperparameters, LoRA config, or templates.  
- Writing a new HF `datasets` script hub package (local store + small LLaMA-Factory patch is enough).

---

## 11. Quick reference card

```text
ROOT   = /scratch/indrisch/Spatial-SSRL_images_h5
INDEX  = ROOT/spatial_ssrl_images_index.json
H5     = ROOT/{crop,depth,flip,position,shuffle,coldstart_SFT}.h5
DSET   = h5["images"]  # (N, max_w, max_h, 3) uint8, WHC per slice, pad=255
KEY    = "{folder}/{basename}.jpg"
 ann   coldstart_SFT_images/img_0.jpg  →  coldstart_SFT/img_0.jpg
DECODE = crop [:w,:h,:] → transpose (1,0,2) → PIL RGB
LFAC   = feed PIL (or patch _regularize_images); keep SFT JSON text as-is
MP     = one H5 store per DataLoader worker process
```

---

## 12. Source of truth in this repo

| Topic | Location |
|-------|----------|
| Pack algorithm, WHC layout, index fields | `scripts/format_SpatialSSRL_multinode.py` |
| SLURM multinode launch + merge step | `scripts/format_SpatialSSRL_multinode_wrapper.sh`, `scripts/format_SpatialSSRL_multinode.sh` |
| Human notes on folders / retrieval | `docs/images_details.md` |
| SFT annotations + `dataset_info` | HF snapshot `SFT-coldstart.json`; `training/SFT-coldstart/data/dataset_info.json` |
| Image open path in LLaMA-Factory | `training/SFT-coldstart/src/llamafactory/data/mm_plugin.py` |
| media_dir join | `training/SFT-coldstart/src/llamafactory/data/converter.py` |

When packing code and this spec disagree, **prefer the packing script and the on-disk index** at `/scratch/indrisch/Spatial-SSRL_images_h5`.
