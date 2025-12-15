# from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, Qwen2_5_VLProcessor, Qwen2VLImageProcessor
# from huggingface_hub import snapshot_download
# import sys, os
# from src.llamafactory.data.formatter import EmptyFormatter, FunctionFormatter, StringFormatter, ToolFormatter
# from src.llamafactory.data.template import register_template
# from src.llamafactory.data.mm_plugin import get_mm_plugin

# model_path = "Video-R1/Qwen2.5-VL-7B-COT-SFT"
# huggingface_hub = "/scratch/indrisch/huggingface/hub/"
# tokenizer = AutoTokenizer.from_pretrained(os.path.join(huggingface_hub, "models--Video-R1--Qwen2.5-VL-7B-COT-SFT/snapshots/f71f0f1e22c015007fccd080eef87824fe292a10/"), local_files_only=True, cache_dir=huggingface_hub) # type(tokenizer) is transformers.models.qwen2.tokenization_qwen2_fast.Qwen2TokenizerFast
# processor = AutoProcessor.from_pretrained(os.path.join(huggingface_hub, "models--Video-R1--Qwen2.5-VL-7B-COT-SFT/snapshots/f71f0f1e22c015007fccd080eef87824fe292a10/"), local_files_only=True, cache_dir=huggingface_hub) # type(processor) is transformers.models.qwen2_5_vl.processing_qwen2_5_vl.Qwen2_5_VLProcessor

# tokenizer_qwen2_vl = AutoTokenizer.from_pretrained(os.path.join(huggingface_hub, "models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5/"), local_files_only=True, cache_dir=huggingface_hub) # type(tokenizer) is transformers.models.qwen2.tokenization_qwen2_fast.Qwen2TokenizerFast
# processor_qwen2_vl = AutoProcessor.from_pretrained(os.path.join(huggingface_hub, "models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5/"), local_files_only=True, cache_dir=huggingface_hub) # type(processor) is transformers.models.qwen2_5_vl.processing_qwen2_5_vl.Qwen2_5_VLProcessor


# messages = [
#     {"role": "user", "content": r"{{content}}"},
#     {"role": "assistant", "content": r"{{content}}"},
#     {"role": "system", "content": r"{{content}}"},
#     {"role": "tool", "content": r"{{content}}"}
# ]

# text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# print("========== Template (Qwen2.5-VL-7B-COT-SFT) ==========")
# print(text)

# text_qwen2_vl = tokenizer_qwen2_vl.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# print("========== Template (Qwen2.5-VL-7B-Instruct) ==========")
# print(text_qwen2_vl)

# # copied from qwen template
# register_template(
#     name="qwen2_vl",
#     format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
#     format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
#     format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
#     format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen"),
#     format_observation=StringFormatter(
#         slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
#     ),
#     format_tools=ToolFormatter(tool_format="qwen"),
#     default_system="You are a helpful assistant.",
#     stop_words=["<|im_end|>"],
#     replace_eos=True,
#     mm_plugin=get_mm_plugin(name="qwen2_vl", image_token="<|image_pad|>", video_token="<|video_pad|>"),
# )

# # Video-R1
# register_template(
#     name="videor1sft",
#     format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
#     format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
#     format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
#     format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen"),
#     format_observation=StringFormatter(
#         slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
#     ),
#     format_tools=ToolFormatter(tool_format="qwen"),
#     default_system="You are a helpful assistant.",
#     stop_words=["<|im_end|>"],
#     replace_eos=True,
#     mm_plugin=get_mm_plugin(name="videor1sft", image_token="<|image_pad|>", video_token="<|video_pad|>"),
# )