#!/usr/bin/env python3
"""
Convert DeepSpeed ZeRO Stage 3 checkpoints to work with different world sizes.

This script converts DeepSpeed checkpoints saved with one world size (number of GPUs)
to work with a different world size, preserving optimizer states and training state.

Usage:
    python scripts/convert_deepspeed_checkpoint_world_size.py \
        --checkpoint_path saves/model/checkpoint-465 \
        --original_world_size 4 \
        --target_world_sizes 1 8 \
        --output_dir saves/model/checkpoint-465_converted
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


def load_optimizer_states(checkpoint_dir: Path, original_world_size: int) -> Tuple[Dict, Dict]:
    """
    Load and gather optimizer states from all partitions.
    
    In ZeRO Stage 3, each rank only stores optimizer states for parameters it owns.
    We need to merge all ranks' optimizer states to get the complete state.
    
    Args:
        checkpoint_dir: Path to the checkpoint directory containing global_step* subdirectory
        original_world_size: Original number of GPUs used for training
        
    Returns:
        Tuple of (gathered_optim_states, metadata_dict) where metadata_dict contains
        top-level metadata from the optimizer state files (like dp_world_size)
    """
    # Find the global_step directory
    global_step_dirs = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("global_step")]
    if not global_step_dirs:
        raise ValueError(f"No global_step directory found in {checkpoint_dir}")
    
    global_step_dir = global_step_dirs[0]
    print(f"Loading optimizer states from {global_step_dir}")
    
    # Load optimizer states from all ranks
    # In ZeRO Stage 3, each rank has optimizer states for different parameters
    # We need to merge them all
    gathered_optim_states = {}
    metadata = {}
    
    for rank in range(original_world_size):
        optim_file = global_step_dir / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
        if not optim_file.exists():
            # Try fp16 format
            optim_file = global_step_dir / f"fp16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
        
        if not optim_file.exists():
            raise FileNotFoundError(f"Optimizer state file not found for rank {rank}: {optim_file}")
        
        print(f"  Loading optimizer states from rank {rank}...")
        rank_optim_states_raw = torch.load(optim_file, map_location="cpu", weights_only=False)
        
        # Extract metadata from the first rank BEFORE unwrapping (all ranks should have the same metadata)
        if rank == 0 and isinstance(rank_optim_states_raw, dict):
            # Common metadata keys that DeepSpeed stores
            metadata_keys = ['dp_world_size', 'world_size', 'data_parallel_size', 
                           'mp_world_size', 'pp_world_size', 'param_groups',
                           'base_optimizer_state']
            for key in metadata_keys:
                if key in rank_optim_states_raw:
                    metadata[key] = rank_optim_states_raw[key]
                    print(f"    Found metadata key: {key} = {metadata[key]}")
        
        # Debug: inspect the structure
        if rank == 0:
            print(f"    Debug: Type of rank_optim_states_raw: {type(rank_optim_states_raw)}")
            if isinstance(rank_optim_states_raw, dict):
                print(f"    Debug: Number of keys: {len(rank_optim_states_raw)}")
                print(f"    Debug: Top-level keys: {list(rank_optim_states_raw.keys())[:10]}")
        
        # DeepSpeed optimizer states structure can vary:
        # Option 1: {param_id: {optimizer_key: tensor, ...}, ...}
        # Option 2: {param_id: tensor, ...}  (simpler structure)
        # Option 3: Nested structure with different organization
        # Option 4: Top-level metadata keys like 'optimizer_state_dict', 'param_groups', etc.
        rank_optim_states = rank_optim_states_raw
        if isinstance(rank_optim_states, dict):
            # Extract the actual optimizer state dict (skip metadata keys)
            # Check if this is a wrapped checkpoint with metadata
            if 'optimizer_state_dict' in rank_optim_states:
                # Unwrap the optimizer state dict
                rank_optim_states = rank_optim_states['optimizer_state_dict']
                print(f"    Found wrapped checkpoint, unwrapping optimizer_state_dict")
            
            # Skip metadata keys when processing parameter states
            metadata_keys_to_skip = {'dp_world_size', 'world_size', 'data_parallel_size',
                                   'mp_world_size', 'pp_world_size', 'param_groups',
                                   'base_optimizer_state', 'optimizer_state_dict'}
            
            for param_id, optim_state in rank_optim_states.items():
                # Skip metadata keys
                if param_id in metadata_keys_to_skip:
                    continue
                
                # Skip non-dict, non-tensor values (metadata, strings, etc.)
                if not isinstance(optim_state, (dict, torch.Tensor)):
                    if rank == 0:  # Only print for first rank to avoid spam
                        print(f"    Skipping non-optimizer-state entry: {param_id} (type: {type(optim_state)})")
                    continue
                
                # Handle different structures
                if isinstance(optim_state, dict):
                    # Standard structure: {optimizer_key: tensor, ...}
                    if param_id in gathered_optim_states:
                        # If we see overlap, it means the parameter is split across ranks
                        print(f"    Warning: Parameter {param_id} found in multiple ranks, merging...")
                        for key, value in optim_state.items():
                            if isinstance(value, torch.Tensor):
                                if key in gathered_optim_states[param_id]:
                                    # Concatenate along first dimension
                                    gathered_optim_states[param_id][key] = torch.cat(
                                        [gathered_optim_states[param_id][key], value], dim=0
                                    )
                                else:
                                    gathered_optim_states[param_id][key] = value
                            else:
                                # Non-tensor values - use the first one (should be same)
                                if key not in gathered_optim_states[param_id]:
                                    gathered_optim_states[param_id][key] = value
                    else:
                        # New parameter - just copy it
                        gathered_optim_states[param_id] = {}
                        for key, value in optim_state.items():
                            if isinstance(value, torch.Tensor):
                                gathered_optim_states[param_id][key] = value.clone()
                            else:
                                gathered_optim_states[param_id][key] = value
                elif isinstance(optim_state, torch.Tensor):
                    # Simple structure: param_id -> tensor directly
                    if param_id in gathered_optim_states:
                        # Concatenate tensors
                        if isinstance(gathered_optim_states[param_id], torch.Tensor):
                            gathered_optim_states[param_id] = torch.cat(
                                [gathered_optim_states[param_id], optim_state], dim=0
                            )
                        else:
                            # Convert dict to tensor or handle mismatch
                            print(f"    Warning: Parameter {param_id} has mismatched types, keeping tensor")
                            gathered_optim_states[param_id] = optim_state.clone()
                    else:
                        gathered_optim_states[param_id] = optim_state.clone()
                else:
                    # Other types (string, int, etc.) - just copy
                    if param_id not in gathered_optim_states:
                        gathered_optim_states[param_id] = optim_state
                    # If it exists, keep the first one
        else:
            raise ValueError(f"Unexpected optimizer state structure: {type(rank_optim_states)}")
    
    print(f"  Gathered optimizer states for {len(gathered_optim_states)} parameters")
    return gathered_optim_states, metadata


def repartition_optimizer_states(
    gathered_optim_states: Dict,
    metadata: Dict,
    target_world_size: int
) -> List[Dict]:
    """
    Repartition optimizer states for the target world size.
    
    In ZeRO Stage 3, parameters are distributed across ranks. We need to
    redistribute the optimizer states accordingly. We'll use a simple round-robin
    distribution based on parameter ID.
    
    Args:
        gathered_optim_states: Gathered optimizer states from all original ranks
        metadata: Metadata dictionary from original checkpoint
        target_world_size: Target number of GPUs
        
    Returns:
        List of optimizer state dictionaries, one per target rank, each including updated metadata
    """
    repartitioned = [{} for _ in range(target_world_size)]
    
    # Update metadata with new world size
    updated_metadata = metadata.copy()
    if 'dp_world_size' in updated_metadata:
        updated_metadata['dp_world_size'] = target_world_size
    if 'world_size' in updated_metadata:
        updated_metadata['world_size'] = target_world_size
    if 'data_parallel_size' in updated_metadata:
        updated_metadata['data_parallel_size'] = target_world_size
    
    # Sort parameter IDs for consistent distribution
    param_ids = sorted(gathered_optim_states.keys())
    
    for idx, param_id in enumerate(param_ids):
        # Distribute parameters round-robin across ranks
        target_rank = idx % target_world_size
        
        # Copy the entire optimizer state for this parameter to the target rank
        repartitioned[target_rank][param_id] = {}
        for key, value in gathered_optim_states[param_id].items():
            if isinstance(value, torch.Tensor):
                repartitioned[target_rank][param_id][key] = value.clone()
            else:
                repartitioned[target_rank][param_id][key] = value
    
    # Add updated metadata to each rank's optimizer state
    for rank in range(target_world_size):
        # Merge metadata into each rank's state dict
        for key, value in updated_metadata.items():
            repartitioned[rank][key] = value
    
    return repartitioned


def load_model_states(checkpoint_dir: Path, original_world_size: int) -> Dict:
    """
    Load model states from all partitions.
    
    In ZeRO Stage 3 with stage3_gather_16bit_weights_on_model_save=true,
    model weights are gathered and saved on rank 0. However, we still check
    all ranks to handle cases where gathering might not have occurred or
    where parameters are still partitioned.
    
    Args:
        checkpoint_dir: Path to the checkpoint directory
        original_world_size: Original number of GPUs
        
    Returns:
        Gathered model states
    """
    global_step_dirs = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("global_step")]
    if not global_step_dirs:
        raise ValueError(f"No global_step directory found in {checkpoint_dir}")
    
    global_step_dir = global_step_dirs[0]
    
    # Load model states from all ranks
    # In ZeRO Stage 3, each rank may have model states for different parameters
    # or all parameters may be on rank 0 if gathering was enabled
    gathered_model_states = {}
    
    for rank in range(original_world_size):
        model_file = global_step_dir / f"zero_pp_rank_{rank}_mp_rank_00_model_states.pt"
        if not model_file.exists():
            raise FileNotFoundError(f"Model state file not found for rank {rank}: {model_file}")
        
        print(f"  Loading model states from rank {rank}...")
        rank_model_states = torch.load(model_file, map_location="cpu", weights_only=False)
        
        # Merge model states
        if isinstance(rank_model_states, dict):
            for key, value in rank_model_states.items():
                if key not in gathered_model_states:
                    # New parameter - copy it
                    if isinstance(value, torch.Tensor):
                        gathered_model_states[key] = value.clone()
                    else:
                        gathered_model_states[key] = value
                else:
                    # Parameter exists - might be split across ranks
                    # This can happen if gathering was not enabled
                    if isinstance(value, torch.Tensor) and isinstance(gathered_model_states[key], torch.Tensor):
                        # Check if shapes match - if so, they're duplicates (from gathering)
                        if value.shape == gathered_model_states[key].shape:
                            # Same shape - likely duplicate from gathering, keep the first one
                            pass
                        else:
                            # Different shapes - concatenate along first dimension
                            gathered_model_states[key] = torch.cat([gathered_model_states[key], value], dim=0)
                    elif isinstance(value, torch.Tensor):
                        # Replace non-tensor with tensor
                        gathered_model_states[key] = value.clone()
                    # If both are non-tensors, keep the first one
    
    return gathered_model_states


def repartition_model_states(
    gathered_model_states: Dict,
    target_world_size: int
) -> List[Dict]:
    """
    Repartition model states for the target world size.
    
    In ZeRO Stage 3, model parameters are distributed across ranks.
    We'll use a simple round-robin distribution based on parameter name.
    
    Args:
        gathered_model_states: Gathered model states
        target_world_size: Target number of GPUs
        
    Returns:
        List of model state dictionaries, one per target rank
    """
    repartitioned = [{} for _ in range(target_world_size)]
    
    # Sort parameter names for consistent distribution
    param_names = sorted(gathered_model_states.keys())
    
    for idx, param_name in enumerate(param_names):
        # Distribute parameters round-robin across ranks
        target_rank = idx % target_world_size
        
        value = gathered_model_states[param_name]
        if isinstance(value, torch.Tensor):
            repartitioned[target_rank][param_name] = value.clone()
        else:
            repartitioned[target_rank][param_name] = value
    
    return repartitioned


def copy_metadata_files(source_dir: Path, target_dir: Path):
    """
    Copy metadata files that don't depend on world size.
    
    Note: rng_state_*.pth files are rank-specific and are not copied.
    They will be regenerated by DeepSpeed when training resumes.
    """
    metadata_files = [
        "trainer_state.json",
        "scheduler.pt",
        "training_args.bin",
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "vocab.json",
        "added_tokens.json",
        "merges.txt",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "chat_template.jinja",
        "README.md",
    ]
    
    for filename in metadata_files:
        source_file = source_dir / filename
        if source_file.exists():
            target_file = target_dir / filename
            shutil.copy2(source_file, target_file)
            print(f"  Copied {filename}")


def create_latest_file(checkpoint_dir: Path, global_step_name: str):
    """
    Create or update the 'latest' file that DeepSpeed uses to track the latest checkpoint.
    
    The 'latest' file is a simple text file containing the name of the latest
    checkpoint directory (e.g., 'global_step465').
    
    Args:
        checkpoint_dir: Path to the checkpoint directory
        global_step_name: Name of the global_step directory (e.g., 'global_step465')
    """
    latest_file = checkpoint_dir / "latest"
    with open(latest_file, "w") as f:
        f.write(global_step_name)
    print(f"  Created 'latest' file pointing to {global_step_name}")


def convert_checkpoint(
    checkpoint_path: str,
    original_world_size: int,
    target_world_size: int,
    output_dir: str,
):
    """
    Convert a DeepSpeed checkpoint to work with a different world size.
    
    Args:
        checkpoint_path: Path to the source checkpoint directory
        original_world_size: Original number of GPUs used
        target_world_size: Target number of GPUs
        output_dir: Output directory for the converted checkpoint
    """
    checkpoint_dir = Path(checkpoint_path)
    if not checkpoint_dir.exists():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint_path}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Converting checkpoint from {original_world_size} GPUs to {target_world_size} GPUs")
    print(f"Source: {checkpoint_path}")
    print(f"Target: {output_dir}")
    print(f"{'='*60}\n")
    
    # Find global_step directory
    global_step_dirs = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("global_step")]
    if not global_step_dirs:
        raise ValueError(f"No global_step directory found in {checkpoint_path}")
    
    global_step_name = global_step_dirs[0].name
    target_global_step_dir = output_path / global_step_name
    target_global_step_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and repartition optimizer states
    print("Step 1: Loading optimizer states...")
    gathered_optim_states, metadata = load_optimizer_states(checkpoint_dir, original_world_size)
    
    print(f"Step 2: Repartitioning optimizer states for {target_world_size} GPUs...")
    repartitioned_optim_states = repartition_optimizer_states(gathered_optim_states, metadata, target_world_size)
    
    # Determine optimizer state file prefix (bf16 or fp16)
    optim_prefix = "bf16"
    test_file = checkpoint_dir / global_step_name / f"bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt"
    if not test_file.exists():
        optim_prefix = "fp16"
    
    # Save repartitioned optimizer states
    print(f"Step 3: Saving repartitioned optimizer states...")
    for rank in range(target_world_size):
        optim_file = target_global_step_dir / f"{optim_prefix}_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
        torch.save(repartitioned_optim_states[rank], optim_file)
        print(f"  Saved optimizer states for rank {rank}")
    
    # Load and repartition model states
    print(f"\nStep 4: Loading model states...")
    gathered_model_states = load_model_states(checkpoint_dir, original_world_size)
    
    print(f"Step 5: Repartitioning model states for {target_world_size} GPUs...")
    repartitioned_model_states = repartition_model_states(gathered_model_states, target_world_size)
    
    # Save repartitioned model states
    print(f"Step 6: Saving repartitioned model states...")
    for rank in range(target_world_size):
        model_file = target_global_step_dir / f"zero_pp_rank_{rank}_mp_rank_00_model_states.pt"
        torch.save(repartitioned_model_states[rank], model_file)
        print(f"  Saved model states for rank {rank}")
    
    # Copy metadata files
    print(f"\nStep 7: Copying metadata files...")
    copy_metadata_files(checkpoint_dir, output_path)
    
    # Create the 'latest' file that DeepSpeed expects
    print(f"\nStep 8: Creating 'latest' file...")
    create_latest_file(output_path, global_step_name)
    
    print(f"\n{'='*60}")
    print(f"✓ Successfully converted checkpoint to {target_world_size} GPUs")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert DeepSpeed ZeRO Stage 3 checkpoints to different world sizes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert checkpoint from 4 GPUs to 1 GPU
  python scripts/convert_deepspeed_checkpoint_world_size.py \\
    --checkpoint_path saves/model/checkpoint-465 \\
    --original_world_size 4 \\
    --target_world_sizes 1 \\
    --output_dir saves/model/checkpoint-465_1gpu
  
  # Convert checkpoint from 4 GPUs to both 1 and 8 GPUs
  python scripts/convert_deepspeed_checkpoint_world_size.py \\
    --checkpoint_path saves/model/checkpoint-465 \\
    --original_world_size 4 \\
    --target_world_sizes 1 8 \\
    --output_dir saves/model/checkpoint-465_converted
        """
    )
    
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the source checkpoint directory"
    )
    
    parser.add_argument(
        "--original_world_size",
        type=int,
        required=True,
        help="Original number of GPUs used for training"
    )
    
    parser.add_argument(
        "--target_world_sizes",
        type=int,
        nargs="+",
        required=True,
        help="Target world size(s) to convert to (e.g., 1 8)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Base output directory. If not specified, will create subdirectories for each target world size"
    )
    
    args = parser.parse_args()
    
    # Convert for each target world size
    for target_world_size in args.target_world_sizes:
        if args.output_dir:
            if len(args.target_world_sizes) > 1:
                # Multiple targets - create subdirectory
                output_dir = os.path.join(args.output_dir, f"world_size_{target_world_size}")
            else:
                output_dir = args.output_dir
        else:
            # Default: create subdirectory based on checkpoint path
            checkpoint_name = os.path.basename(os.path.normpath(args.checkpoint_path))
            output_dir = os.path.join(
                os.path.dirname(args.checkpoint_path),
                f"{checkpoint_name}_world_size_{target_world_size}"
            )
        
        convert_checkpoint(
            checkpoint_path=args.checkpoint_path,
            original_world_size=args.original_world_size,
            target_world_size=target_world_size,
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()

