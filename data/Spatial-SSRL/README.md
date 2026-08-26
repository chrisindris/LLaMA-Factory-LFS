# Spatial-SSRL in LLaMA-Factory

## Registered dataset

| Name | Description |
|------|-------------|
| `SpatialSSRL_coldstart` | Coldstart SFT JSON (3597 rows) from the Spatial-SSRL-81k HF snapshot |

Annotations (`file_name` in `dataset_info.json`; `${HF_HUB_CACHE}` is expanded at load time):

```text
${HF_HUB_CACHE}/datasets--internlm--Spatial-SSRL-81k/snapshots/54b82086060a5612f95588b4979446da2282bcd9/SFT-coldstart.with_question_id.json
```

## Packed images (H5)

| Env | Default |
|-----|---------|
| `SPATIALSSRL_H5_DIR` | `/scratch/indrisch/Spatial-SSRL_images_h5` |

Index: `spatial_ssrl_images_index.json`  
H5 files: `coldstart_SFT.h5`, `crop.h5`, `depth.h5`, `flip.h5`, `position.h5`, `shuffle.h5`

Annotation paths like `coldstart_SFT_images/img_0.jpg` are normalized to index keys (`coldstart_SFT/img_0.jpg`) and decoded with crop + WHC→HWC transpose. No loose JPEG tree is required at train time.

## Train example

```bash
llamafactory-cli train examples/train_lora/qwen2_5vl_lora_sft_SpatialSSRL_coldstart.yaml
```

## Spec

See [`h5_dataloader_spec.md`](h5_dataloader_spec.md) for packing layout and decode rules.
