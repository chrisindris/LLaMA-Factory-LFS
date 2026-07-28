# 3DThinker-10k in LLaMA-Factory

## Registered dataset

| Name | Description |
|------|-------------|
| `3DThinker10k` | CoT SFT (10k rows): system=`mindcube_input`, user=`text_input`, target=`text_output` |

Prepared annotations:

```text
data/3DThinker-10K/out/3dthinker10k_cot.jsonl
```

Source (multi-line JSON objects, not strict JSONL):

```text
/scratch/indrisch/huggingface/hub/datasets--jankin123--3DThinker-10K/snapshots/2b16e1e73cf985e5d46b84cc90c13956bc7205f2/data_output3d_begin_10k_resized.jsonl
```

Regenerate prepared JSONL:

```bash
python data/3DThinker-10K/prepare_3dthinker10k.py
# optional: --annotations PATH --h5-dir PATH --output PATH --max-samples N
```

## Packed images (H5)

| Env | Default |
|-----|---------|
| `THINKER10K_H5_DIR` | `/scratch/indrisch/3DThinker10K_images_h5` |

- H5: `3dthinker10k_images.h5` — shape `(2785, 480, 640, 3)` WHC uint8  
- Index: `3dthinker10k_images_index.json` — keys `other_all_image_resize/...`, values row indices  

Annotation paths `data/other_all_image_resize/...` strip the `data/` prefix for index lookup. No loose `other_all_image_resize/` tree is required at train time.

## Field mapping (CoT)

| Role | Source field |
|------|----------------|
| System (with `<image>` tags) | `mindcube_input` |
| User question | `text_input` |
| Images | `image_input` → H5 |
| Assistant target | `text_output` |

Image placeholders live in the system string; this fork expands them in `SupervisedDatasetProcessor` so Qwen2-VL vision tokens still match `len(images)`.

## Train example

Upstream reference: `/scratch/indrisch/3DThinker/SFT/train_sft.yaml` (may be a slightly older LLaMA-Factory). Adapted config:

```bash
llamafactory-cli train examples/train_lora/qwen2_5vl_lora_sft_3DThinker10k.yaml
```

## Specs

- [`3dthinker10k_h5_dataloader_spec.md`](3dthinker10k_h5_dataloader_spec.md)
- [`dataset_jankin123--3DThinker-10K.md`](dataset_jankin123--3DThinker-10K.md)
