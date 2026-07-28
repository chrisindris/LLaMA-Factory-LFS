# Details of the 6 image folders

The `images` directory, a child directory of `/scratch/indrisch/huggingface/hub/datasets--internlm--Spatial-SSRL-81k/snapshots/54b82086060a5612f95588b4979446da2282bcd9/`, contains five subdirs, with details below.

```csv
subdir,file_count,filename_regex,max_image_width,max_image_height
crop,101000,(blackened_image_\d{5}_\d{5}|cropped_img_\d{5}_\d{1})\.jpg,$MAX_IMAGE_WIDTH_CROP,$MAX_IMAGE_HEIGHT_CROP
depth,20620,image_\d+\.jpg,$MAX_IMAGE_WIDTH_DEPTH,$MAX_IMAGE_HEIGHT_DEPTH
flip,4005,image_\d+_\d{1}\.jpg,$MAX_IMAGE_WIDTH_FLIP,$MAX_IMAGE_HEIGHT_FLIP
position,20200,image_\d+\.jpg,$MAX_IMAGE_WIDTH_POSITION,$MAX_IMAGE_HEIGHT_POSITION
shuffle,16028,image_\d+_\d{1}\.jpg,$MAX_IMAGE_WIDTH_SHUFFLE,$MAX_IMAGE_HEIGHT_SHUFFLE
```

The `coldstart_SFT_images` directory, a child directory of `/scratch/indrisch/huggingface/hub/datasets--internlm--Spatial-SSRL-81k/snapshots/54b82086060a5612f95588b4979446da2282bcd9/`, contains 3597 images, all named like `img_\d+\.jpg`. The maximum image width and height are given as $MAX_IMAGE_WIDTH_COLDSTART and $MAX_IMAGE_HEIGHT_COLDSTART.

The actual values determined by scripts are as follows:
```csv
subdir,max_image_width,max_image_height
crop,640,640
depth,959,738
flip,640,640
position,999,885
shuffle,640,640
coldstart_SFT,998,885
```

# JSON mapping of each file to their width and height

The following files, for each of the 5 `images` folders and the `coldstart_SFT_images` folder, are .json files which contain key:value pairs where the key is the filename and the value is the dimension of the image (by construction, no height is greater than the max_image_height and no width is greater than the max_image_width for that image).

```csv
subdir,json
crop,crop_file_info.json
depth,depth_file_info.json
flip,flip_file_info.json
position,position_file_info.json
shuffle,shuffle_file_info.json
coldstart_SFT,coldstart_SFT_file_info.json
```

# Desired Output

Instead of having 6 folders of images, we want to reduce the total file count by producing 6 h5 files and 1 json for indexing.
- Each .h5 file is a stack of all of the images in the folder. Implemented layout (see `scripts/format_SpatialSSRL_multinode.py`):
  - dataset name: `images`
  - shape: `(N, MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT, 3)`, `dtype=uint8`
  - axis 0 is the image index; each slice is a white-padded canvas of shape `(MAX_W, MAX_H, 3)`
- For most images, width/height are smaller than the folder max; pad with white (value **255**) so every slice is `(MAX_W, MAX_H, 3)`. Real pixels live in the **top-left** `[:width, :height, :]` region.
- The json file is a map from keys to `{width, height, h5file, h5index}`.
  - Keys are **folder-qualified** basenames (e.g. `coldstart_SFT/img_10.jpg`, `depth/image_10.jpg`) because basenames collide across folders (notably depth vs position).
  - `h5file` is the H5 basename (e.g. `coldstart_SFT.h5`); `h5index` is the row along axis 0.

## Retrieval example

```python
import json
import h5py
import numpy as np

with open("spatial_ssrl_images_index.json") as f:
    D = json.load(f)

key = "coldstart_SFT/img_10.jpg"
entry = D[key]
with h5py.File(entry["h5file"], "r") as h5:
    canvas = h5["images"][entry["h5index"]]  # (max_w, max_h, 3)
image = canvas[: entry["width"], : entry["height"], :]  # original RGB, axes (W, H, 3)
# optional: to HWC for typical image libs
image_hwc = np.transpose(image, (1, 0, 2))
```

## How to build the packed dataset

```bash
# SLURM multinode (4 nodes default)
sbatch scripts/format_SpatialSSRL_multinode_wrapper.sh

# Or single-node dry run on a subset
python scripts/format_SpatialSSRL_multinode.py \
  --input-dataset-dir /path/to/snapshot \
  --combined-dataset /path/to/out \
  --folders flip coldstart_SFT
python scripts/format_SpatialSSRL_multinode.py \
  --combined-dataset /path/to/out --merge-only
```

Defaults (wrapper): input = HF snapshot under `$HF_HOME/datasets--internlm--Spatial-SSRL-81k/snapshots/54b82086060a5612f95588b4979446da2282bcd9`, output = `/scratch/indrisch/Spatial-SSRL_images_h5`. Override with `SPATIALSSRL_INPUT_DIR` / `SPATIALSSRL_OUTPUT_DIR`.

## Loading for training (LLaMA-Factory)

For an LLM-oriented specification of a PyTorch / LLaMA-Factory dataloader over the packed H5 dataset (index schema, WHC layout, path remapping, multiprocessing, integration points), see [`docs/h5_dataloader_spec.md`](h5_dataloader_spec.md).

