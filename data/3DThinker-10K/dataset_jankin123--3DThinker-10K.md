# jankin123/3DThinker-10k

Contains the following, which can be produced also by the Step 1 (Data Generation) in [MLL-Lab--MindCube](https://github.com/zhangquanchen/3DThinker/blob/main/docs/dataset_MLL-Lab--MindCube.md):

-- "data_output3d_begin_10k_resized.jsonl"; this is already produced in [jankin123/3DThinker-10k](https://huggingface.co/datasets/jankin123/3DThinker-10K/tree/main) --> has full CoT traces, inputs, images and outputs.
-- "data/resized_images"; this is found in "other_all_image_resize.zip" of [jankin123/3DThinker-10k](https://huggingface.co/datasets/jankin123/3DThinker-10K/tree/main)

-----

# Based on the above, we can do the following:

1. We can convert to h5/json ("data_output3d_begin_10k_resized.jsonl" and "data/resized_images") — pack layout: [`jankin123_3DThinker10K_images_details.md`](jankin123_3DThinker10K_images_details.md); packed dir default `/scratch/indrisch/3DThinker10K_images_h5`.
2. We can run it in LLaMA-Factory using their settings (though we might want to adjust to the version of LLaMA-Factory we use for everything else) --> note that this would be a CoT SFT since "data_output3d_begin_10k_resized.jsonl" contains chain-of-thought:
Inputs: "mindcube_input" (system prompt), "text_input" (question), "image_input" (image context)
Target: ["text_output" (if doing CoT) or "answer" (if doing non-CoT)]

For implementing a PyTorch dataloader over the **compressed** H5/JSON images + these annotations (LLaMA-Factory-oriented LLM build spec), see [`3dthinker10k_h5_dataloader_spec.md`](3dthinker10k_h5_dataloader_spec.md).