#!/usr/bin/env python3
"""
Script to upload LoRA model checkpoints to HuggingFace Hub.

This script uploads the trained LoRA adapter model from the saves directory
to HuggingFace Hub. It can upload either the final model or a specific checkpoint.
"""

import os
import json
import re
import argparse
from pathlib import Path
from huggingface_hub import HfApi, upload_folder, create_repo
from huggingface_hub.utils import HfHubHTTPError


def upload_lora_checkpoint(
    checkpoint_path: str,
    repo_id: str,
    token: str = None,
    private: bool = False,
    commit_message: str = "Upload LoRA adapter checkpoint"
):
    """
    Upload a LoRA checkpoint to HuggingFace Hub.
    
    Args:
        checkpoint_path: Path to the checkpoint directory
        repo_id: HuggingFace repository ID (e.g., "username/model-name")
        token: HuggingFace token (if None, will try to use cached token)
        private: Whether to create a private repository
        commit_message: Commit message for the upload
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    if not checkpoint_path.is_dir():
        raise ValueError(f"Checkpoint path must be a directory: {checkpoint_path}")
    
    # Check for required LoRA files
    adapter_model = checkpoint_path / "adapter_model.safetensors"
    adapter_config = checkpoint_path / "adapter_config.json"
    
    if not adapter_model.exists() and not (checkpoint_path / "adapter_model.bin").exists():
        raise ValueError(
            f"No adapter model file found in {checkpoint_path}. "
            "Expected adapter_model.safetensors or adapter_model.bin"
        )
    
    if not adapter_config.exists():
        raise ValueError(f"No adapter_config.json found in {checkpoint_path}")
    
    # Read base model from adapter_config.json
    with open(adapter_config, 'r') as f:
        adapter_config_data = json.load(f)
    base_model = adapter_config_data.get('base_model_name_or_path', '')
    
    if not base_model:
        raise ValueError("base_model_name_or_path not found in adapter_config.json")
    
    # Fix README.md if it exists and has empty base_model
    readme_path = checkpoint_path / "README.md"
    if readme_path.exists():
        with open(readme_path, 'r', encoding='utf-8') as f:
            readme_content = f.read()
        
        # Fix empty base_model in YAML frontmatter
        # Match: base_model: '' or base_model: "" or base_model: (empty)
        # Pattern matches only empty base_model values
        pattern = r"base_model:\s*(''|\"\"|)\s*\n"
        if re.search(pattern, readme_content):
            readme_content = re.sub(
                pattern,
                f'base_model: {base_model}\n',
                readme_content
            )
            # Write back the fixed README.md
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            print(f"Fixed README.md: set base_model to {base_model}")
    
    print(f"Uploading checkpoint from: {checkpoint_path}")
    print(f"Repository: {repo_id}")
    print(f"Private: {private}")
    print(f"Base model: {base_model}")
    
    # Initialize HuggingFace API
    api = HfApi(token=token)
    
    # Create repository if it doesn't exist
    try:
        create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            repo_type="model",
            exist_ok=True
        )
        print(f"Repository {repo_id} created or already exists")
    except HfHubHTTPError as e:
        if "already exists" not in str(e).lower():
            raise
        print(f"Repository {repo_id} already exists")
    
    # Files to upload (exclude DeepSpeed checkpoint files and other unnecessary files)
    exclude_patterns = [
        "checkpoint-*/**",  # Intermediate checkpoint directories (only upload final model)
        "global_step*/**",  # DeepSpeed checkpoint directories
        "zero_pp_rank_*",  # DeepSpeed model states
        "bf16_zero_pp_rank_*",  # DeepSpeed optimizer states
        "rng_state_*.pth",  # Random state files
        "scheduler.pt",  # Scheduler state
        "training_args.bin",  # Training arguments (not needed for inference)
        "trainer_state.json",  # Trainer state (not needed for inference)
        "zero_to_fp32.py",  # DeepSpeed conversion script
        "latest",  # Symlink to latest checkpoint
        "*.png",  # Training loss plots
        "trainer_log.jsonl",  # Training logs
        "train_results.json",  # Training results
        "all_results.json",  # All results
        "eval_results.json",  # Eval results
    ]
    
    # Upload the folder
    print("\nUploading files...")
    upload_folder(
        folder_path=str(checkpoint_path),
        repo_id=repo_id,
        token=token,
        commit_message=commit_message,
        ignore_patterns=exclude_patterns,
        repo_type="model"
    )
    
    print(f"\n✓ Successfully uploaded checkpoint to https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload LoRA model checkpoint to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload final model
  python upload_model_checkpoint.py \\
    --checkpoint saves/qwen2_5vl-7b/lora/sft/SQA3Devery24_traineval \\
    --repo-id username/qwen2_5vl-7b-lora-sqa3d
  
  # Upload specific checkpoint
  python upload_model_checkpoint.py \\
    --checkpoint saves/qwen2_5vl-7b/lora/sft/SQA3Devery24_traineval/checkpoint-465 \\
    --repo-id username/qwen2_5vl-7b-lora-sqa3d-checkpoint-465
  
  # Upload as private repository
  python upload_model_checkpoint.py \\
    --checkpoint saves/qwen2_5vl-7b/lora/sft/SQA3Devery24_traineval \\
    --repo-id username/qwen2_5vl-7b-lora-sqa3d \\
    --private
        """
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint directory to upload"
    )
    
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repository ID (e.g., 'username/model-name')"
    )
    
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace token (if not provided, will use cached token from huggingface-cli login)"
    )
    
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository"
    )
    
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload LoRA adapter checkpoint",
        help="Commit message for the upload"
    )
    
    args = parser.parse_args()
    
    # Get token from environment if not provided
    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    
    # Resolve checkpoint path relative to script location or use absolute path
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        # Try relative to script directory
        script_dir = Path(__file__).parent
        base_dir = script_dir.parent.parent  # Go up to LLaMA-Factory root
        checkpoint_path = base_dir / args.checkpoint
    
    upload_lora_checkpoint(
        checkpoint_path=str(checkpoint_path),
        repo_id=args.repo_id,
        token=token,
        private=args.private,
        commit_message=args.commit_message
    )


if __name__ == "__main__":
    main()

