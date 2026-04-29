#!/usr/bin/env python3
"""
Script to upload merged model checkpoints to HuggingFace Hub.

This script uploads the merged full model from the specified directory
to HuggingFace Hub.
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, upload_folder, create_repo
from huggingface_hub.utils import HfHubHTTPError


def upload_merged_checkpoint(
    checkpoint_path: str,
    repo_id: str,
    token: str = None,
    private: bool = False,
    commit_message: str = "Upload merged model checkpoint"
):
    """
    Upload a merged model checkpoint to HuggingFace Hub.
    
    Args:
        checkpoint_path: Path to the merged model directory
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
    
    # Check for required full model files (e.g. config.json)
    config_file = checkpoint_path / "config.json"
    
    if not config_file.exists():
        raise ValueError(
            f"No config.json found in {checkpoint_path}. "
            "Are you sure this is a merged model directory?"
        )
    
    print(f"Uploading merged checkpoint from: {checkpoint_path}")
    print(f"Repository: {repo_id}")
    print(f"Private: {private}")
    
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
    
    # Files to upload (exclude intermediate files)
    exclude_patterns = [
        "checkpoint-*/**",
        "global_step*/**",
        "zero_pp_rank_*",
        "bf16_zero_pp_rank_*",
        "rng_state_*.pth",
        "scheduler.pt",
        "training_args.bin",
        "trainer_state.json",
        "zero_to_fp32.py",
        "latest",
        "*.png",
        "trainer_log.jsonl",
        "train_results.json",
        "all_results.json",
        "eval_results.json",
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
    
    print(f"\n✓ Successfully uploaded merged model to https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload merged model checkpoint to HuggingFace Hub"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the merged checkpoint directory to upload"
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
        default="Upload merged model checkpoint",
        help="Commit message for the upload"
    )
    
    args = parser.parse_args()
    
    # Get token from environment if not provided
    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    
    # Resolve checkpoint path relative to script location or use absolute path
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        script_dir = Path(__file__).parent
        base_dir = script_dir.parent.parent
        checkpoint_path = base_dir / args.checkpoint
    
    upload_merged_checkpoint(
        checkpoint_path=str(checkpoint_path),
        repo_id=args.repo_id,
        token=token,
        private=args.private,
        commit_message=args.commit_message
    )

if __name__ == "__main__":
    main()
