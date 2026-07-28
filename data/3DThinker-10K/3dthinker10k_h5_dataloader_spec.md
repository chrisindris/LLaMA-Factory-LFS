# 3DThinker-10k H5 dataloader — LLM implementation spec

**Audience:** an LLM (or engineer) implementing a PyTorch dataloader for training with LLaMA-Factory or a similar multimodal SFT stack.

**Goal:** load training examples whose **images** come from the packed H5 store and whose **text** comes from the existing annotations — without requiring the loose `other_all_image_resize/` image tree at train time.

**Out of scope:** re-packing H5; VGGT / Stage-1 3D latent supervision; Stage-2 RL.

---

## 1. Paths and artifacts

| Role | Default path |
|------|----------------|
| Packed image dir | `/scratch/indrisch/3DThinker10K_images_h5` |
| H5 | `/scratch/indrisch/3DThinker10K_images_h5/3dthinker10k_images.h5` |
| Index JSON | `/scratch/indrisch/3DThinker10K_images_h5/3dthinker10k_images_index.json` |
| Annotations | `data_3DThinker-10K/data_output3d_begin_10k_resized.jsonl` (repo-relative) |

Packing code of record (do **not** use Spatial-SSRL packers for this layout):

- `scripts/format_3DThinker10K_multinode_wrapper.sh`
- `scripts/format_3DThinker10K_multinode.sh`
- `scripts/format_3DThinker10K_multinode.py`

Background on original dataset + use cases: `docs/dataset_jankin123--3DThinker-10K.md`.  
Pack layout details: `docs/jankin123_3DThinker10K_images_details.md`.

---

## 2. On-disk contracts (packed images)

### 2.1 H5

- File: `3dthinker10k_images.h5`
- Dataset name: `images`
- Shape: `(N, 480, 640, 3)`, `dtype=uint8`
- Layout: **WHC** — axis 0 = image index, axis 1 = **width** (480), axis 2 = **height** (640), axis 3 = RGB
- No padding (all source images are already 480×640)
- Compression: gzip (chunked per image: `(1, 480, 640, 3)`)
- Expected `N`: **2785**

### 2.2 Index JSON

- File: `3dthinker10k_images_index.json`
- Type: object / dict
- Keys: relative paths starting with `other_all_image_resize/…` (e.g. `other_all_image_resize/among/<scene>/back_164.png`)
- Values: plain **integer** H5 row index (not an object)

### 2.3 Contrast with Spatial-SSRL (do not mix schemas)

| | 3DThinker-10k | Spatial-SSRL |
|--|---------------|--------------|
| H5 files | **1** | **6** |
| Index value | plain `int` | `{width, height, h5file, h5index}` |
| Padding | none | white pad to per-folder max |
| Key style | full path under `other_all_image_resize/` | folder-qualified basename |

### 2.4 Low-level retrieval (must transpose for PIL)

```python
import json
import h5py
import numpy as np
from PIL import Image

with open("3dthinker10k_images_index.json") as f:
    D = json.load(f)

key = "other_all_image_resize/among/3a582cdbad3d207460b4ffea8f185b2ca8dd6d2a29274db83390a0c9a2a11c54/back_164.png"
idx = D[key]
with h5py.File("3dthinker10k_images.h5", "r") as h5:
    image_whc = h5["images"][idx]  # (480, 640, 3) uint8, axes (W, H, C)

image_hwc = np.transpose(image_whc, (1, 0, 2))  # (640, 480, 3) for HWC libs
pil = Image.fromarray(image_hwc)                 # PIL size (width, height) = (480, 640)
assert pil.size == (480, 640)
assert pil.mode == "RGB"
```

**Rule:** always convert WHC → HWC before `Image.fromarray` or any HWC-expecting preprocessor.

---

## 3. Annotation contract

### 3.1 File format pitfall

`data_output3d_begin_10k_resized.jsonl` is **not** strict one-JSON-object-per-line. It is a concatenation of **pretty-printed multi-line** JSON objects. `json.loads(line)` on each line will fail.

**Required parse pattern:**

```python
import json
from pathlib import Path

def load_annotations(path: str | Path) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    idx = 0
    n = len(text)
    examples: list[dict] = []
    while idx < n:
        while idx < n and text[idx].isspace():
            idx += 1
        if idx >= n:
            break
        obj, end = decoder.raw_decode(text, idx)
        examples.append(obj)
        idx = end
    return examples
```

Expected count: **10_000** examples.

Optional preprocess (not required for the Dataset): rewrite once to true JSONL (`json.dumps(obj) + "\n"` per example) for faster reloads.

### 3.2 Fields per example

| Field | Type | Role |
|-------|------|------|
| `mindcube_input` | `str` | System / task framing; contains one `<image>` placeholder per input image |
| `text_input` | `str` | User question (often MCQ) |
| `image_input` | `list[str]` | Ordered image paths (see remapping) |
| `text_output` | `str` | **CoT** target (`<output_3D>`, `<think>…</think>`, `<answer>…</answer>`) |
| `answer` | `str` | Short answer (non-CoT target option) |
| `mindcube_output` | `str` | Short tagged answer, e.g. `<answer>…</answer>` |
| `idx` | `int` | Example id |

### 3.3 Images per example (distribution)

| `#image_input` | `#examples` |
|----------------|-------------|
| 2 | 1668 |
| 3 | 916 |
| 4 | 7416 |

Invariant (all 10k): `mindcube_input.count("<image>") == len(image_input)`.

### 3.4 SFT field mapping (intended use)

CoT SFT (default, matches dataset notes):

| Training role | Source field |
|---------------|--------------|
| System | `mindcube_input` (keep `<image>` tokens) |
| User | `text_input` |
| Images | resolved from `image_input` via H5 → `list[PIL.Image]` |
| Assistant target | `text_output` |

Non-CoT alternatives: target = `answer` or `mindcube_output`.

---

## 4. Path remapping (mandatory)

Annotation paths look like:

```text
data/other_all_image_resize/among/ball_440/back_115.jpg
```

Index keys look like:

```text
other_all_image_resize/among/ball_440/back_115.jpg
```

**Normalize before every lookup:**

```python
def to_index_key(annotation_path: str) -> str:
    p = annotation_path.strip().lstrip("./")
    if p.startswith("data/"):
        p = p[len("data/") :]
    return p
```

Verified: after stripping `data/`, **all** annotation image refs (35_748) hit the index (0 missing). Unique images = 2785 = index size.

If a key is missing after normalize → fail fast with the original and normalized path.

---

## 5. Recommended Dataset API

Implement a `torch.utils.data.Dataset` with approximately this interface.

```python
from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from PIL import Image


class Thinker10KH5Dataset(torch.utils.data.Dataset):
    """3DThinker-10k SFT examples with images from packed H5."""

    def __init__(
        self,
        annotations_path: str | Path,
        h5_path: str | Path,
        index_path: str | Path,
        *,
        target: Literal["cot", "answer", "mindcube_output"] = "cot",
        max_samples: int | None = None,
        validate_paths: bool = True,
    ) -> None:
        ...

    def __len__(self) -> int: ...

    def __getitem__(self, i: int) -> dict:
        """
        Returns a LLaMA-Factory-friendly dict:

        {
          "system": str,          # mindcube_input
          "prompt": str,          # text_input
          "response": str,        # text_output | answer | mindcube_output
          "images": list[Image.Image],  # RGB, size (480, 640), order = image_input
          "idx": int,
        }
        """
        ...
```

### 5.1 Target mode

| `target` | `response` field |
|----------|------------------|
| `"cot"` (default) | `text_output` |
| `"answer"` | `answer` |
| `"mindcube_output"` | `mindcube_output` |

### 5.2 Init responsibilities

1. Parse annotations (Section 3.1); optionally truncate with `max_samples`.
2. `json.load` the full index into memory (small).
3. Open H5 only as needed for validation / workers (Section 6) — do **not** require sharing one handle across processes.
4. If `validate_paths`: for every `image_input` entry, `to_index_key` must exist in the index; assert placeholder count matches image count.
5. Assert `len(index) == h5["images"].shape[0]` and spatial shape is `(480, 640, 3)`.

### 5.3 `__getitem__` image steps (per path, in order)

1. `key = to_index_key(path)`
2. `row = index[key]`
3. Read `h5["images"][row]` → WHC `(480, 640, 3)`
4. `hwc = np.transpose(arr, (1, 0, 2))`
5. `Image.fromarray(hwc)` → append to list

Preserve order of `image_input`. Do not drop or reorder frames.

---

## 6. H5 I/O and DataLoader workers

### 6.1 Rules

- Do **not** open one `h5py.File` in the parent and share it across `num_workers > 0` without a per-process strategy (fork + HDF5 is a common source of hangs/crashes).
- Load the **index JSON once** in `__init__` (read-only dict is fine to share after fork).
- Prefer **lazy open per process**:

```python
import h5py

class Thinker10KH5Dataset(torch.utils.data.Dataset):
    def __init__(self, ..., h5_path: str | Path, ...):
        self.h5_path = str(h5_path)
        self._h5 = None  # process-local

    def _get_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, i):
        h5 = self._get_h5()
        ...
```

After fork, each worker gets `_h5 is None` again (or use `worker_init_fn` to open explicitly). Closing on `__del__` is optional.

### 6.2 Performance notes (optional, not blocking)

- Chunks are one image per chunk → random row access is reasonable.
- Gzip slows random reads vs uncompressed; a future memmap/cache is an optimization, not required for correctness.

### 6.3 Collate

Default PyTorch collation will fail on lists of PIL images. For a standalone `DataLoader` smoke test, use `batch_size=1` or a custom collate that returns `list[dict]` / keeps `images` as nested lists. LLaMA-Factory has its own collator — do not reimplement that unless writing a fully standalone trainer.

---

## 7. LLaMA-Factory integration

Repo surfaces:

- Config example: `SFT/train_sft.yaml` (`dataset: mindcube`, `template: qwen2_vl`)
- Vendored package: `SFT/env/src/llamafactory/`
- Cluster install also referenced as `/scratch/indrisch/LLaMA-Factory/` with `data/dataset_info.json`

### 7.1 Why return PIL, not filesystem paths

`mm_plugin._regularize_images` accepts:

- path strings
- `bytes` / `BytesIO`
- `PIL.Image.Image`
- dicts with `bytes` / `path`

`converter.DatasetConverter._find_medias` only rewrites **string** paths under `media_dir`. **PIL objects pass through** unchanged.

Returning PIL from H5 means training does **not** need `other_all_image_resize/` on disk and does not depend on `media_dir` for these images.

### 7.2 Preferred integration: `dataset_info.json` + script dataset

Register something like:

```json
"3dthinker10k_h5_cot": {
  "script_url": "3dthinker10k_h5",
  "formatting": "alpaca",
  "columns": {
    "system": "system",
    "prompt": "prompt",
    "response": "response",
    "images": "images"
  }
}
```

Implement `script_url` as a HuggingFace `datasets` loading script (or the LLaMA-Factory `script` load path) that yields rows with those column names and `images` as `list[PIL.Image]`.

Wire training yaml:

```yaml
dataset: 3dthinker10k_h5_cot
template: qwen2_vl   # or project-standard VL template
# media_dir not required when images are already PIL
```

### 7.3 Alternative: pure PyTorch

Use `Thinker10KH5Dataset` directly outside LLaMA-Factory. Same return schema; pair with the project’s own collate/processor.

### 7.4 Not preferred

Materializing thousands of JPEGs back to disk only to feed path-based LLaMA-Factory file mode. If file mode is unavoidable, document it as a fallback, not the default.

---

## 8. Implementation checklist (for the coding agent)

Execute in order:

1. [ ] Implement `load_annotations` with `JSONDecoder.raw_decode`.
2. [ ] Implement `to_index_key` and path preflight against the index.
3. [ ] Validate H5 shape `(N, 480, 640, 3)`, `N == len(index) == 2785`.
4. [ ] Implement WHC→HWC→PIL image fetch.
5. [ ] Implement `Thinker10KH5Dataset` with `target` modes.
6. [ ] Smoke `DataLoader` with `num_workers=0` and `num_workers=2` (batch size 1 or custom collate).
7. [ ] (Optional) HF/LLaMA-Factory script + `dataset_info.json` entry + yaml.
8. [ ] (Optional) small smoke script under `scripts/` or test under `tests/`.

Suggested module path (implementer may choose): e.g. `SFT/src/thinker10k_h5_dataset.py` or `preprocessing/thinker10k_h5_dataset.py`.

Dependencies: `torch`, `h5py`, `numpy`, `Pillow`. (Same stack as packing scripts.)

---

## 9. Acceptance tests

The implementation is done only when all of the following pass.

| ID | Check | Criterion |
|----|--------|-----------|
| A1 | Index / H5 size | `N = h5["images"].shape[0] == len(index) == 2785`; `max(index.values()) == N - 1` |
| A2 | Spatial shape | `h5["images"].shape[1:] == (480, 640, 3)` |
| A3 | Path coverage | For every annotation `image_input` path, `to_index_key(p) in index` |
| A4 | Placeholders | For every example, `mindcube_input.count("<image>") == len(image_input)` |
| A5 | PIL geometry | Each loaded image: `mode == "RGB"`, `size == (480, 640)` |
| A6 | Multi-image | Draw examples with 2, 3, and 4 images; `len(images)` matches; order preserved |
| A7 | CoT target | With `target="cot"`, `response` non-empty and contains answer markup (e.g. `<answer>`) for sampled rows |
| A8 | Workers | `DataLoader(ds, batch_size=1, num_workers=2)` iterates ≥ 8 samples without H5 errors |
| A9 | No loose tree | Run using only annotations + H5 dir (no `other_all_image_resize/` filesystem tree) |

### 9.1 Minimal smoke snippet

```python
from torch.utils.data import DataLoader
# from <module> import Thinker10KH5Dataset

ds = Thinker10KH5Dataset(
    annotations_path="data_3DThinker-10K/data_output3d_begin_10k_resized.jsonl",
    h5_path="/scratch/indrisch/3DThinker10K_images_h5/3dthinker10k_images.h5",
    index_path="/scratch/indrisch/3DThinker10K_images_h5/3dthinker10k_images_index.json",
    target="cot",
    validate_paths=True,
)
assert len(ds) == 10_000
ex = ds[0]
assert set(ex) >= {"system", "prompt", "response", "images", "idx"}
assert all(im.size == (480, 640) for im in ex["images"])
assert ex["system"].count("<image>") == len(ex["images"])

loader = DataLoader(ds, batch_size=1, num_workers=2, shuffle=False)
for i, batch in enumerate(loader):
    if i >= 7:
        break
print("smoke OK")
```

---

## 10. Defaults / knobs the implementer should support

| Knob | Default |
|------|---------|
| Packed dir | `/scratch/indrisch/3DThinker10K_images_h5` |
| H5 name | `3dthinker10k_images.h5` |
| Index name | `3dthinker10k_images_index.json` |
| Annotations | `data_3DThinker-10K/data_output3d_begin_10k_resized.jsonl` |
| `target` | `cot` |
| Image output | PIL RGB HWC-equivalent, size (480, 640) |

Environment overrides (optional but useful): e.g. `THINKER10K_H5_DIR`, `THINKER10K_ANNOTATIONS`.

---

## 11. Reference map

| Doc / code | Why |
|------------|-----|
| `docs/dataset_jankin123--3DThinker-10K.md` | CoT SFT field mapping and use cases |
| `docs/jankin123_3DThinker10K_images_details.md` | H5/JSON pack layout |
| `scripts/format_3DThinker10K_multinode.py` | Ground-truth writer (WHC, keys, shapes) |
| `docs/SpatialSSRL_images_details.md` | Different layout — do not copy index schema |
| `SFT/train_sft.yaml` | Existing LLaMA-Factory train entry |
| `SFT/env/src/llamafactory/data/converter.py` | `_find_medias` path rewriting |
| `SFT/env/src/llamafactory/data/mm_plugin.py` | `_regularize_images` accepts PIL |
| `SFT/env/src/llamafactory/data/parser.py` | `dataset_info.json` / `script_url` |

---

## 12. Success criterion for this spec

An implementer (human or LLM) given **this file** plus the artifact paths above can ship a working H5-backed Dataset that:

1. Joins 10k CoT annotations to 2785 packed images,
2. Never needs the loose image tree,
3. Survives multi-worker DataLoader reads,
4. Emits LLaMA-Factory-compatible `system` / `prompt` / `response` / `images` (PIL) fields.
