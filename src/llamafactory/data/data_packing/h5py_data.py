#!/usr/bin/env python3
"""
Script to convert ScanNet scene folders to h5py format.

Each scene folder (sceneXXXX_YY) contains a 'color' folder with images.
This script processes these images and saves them as binary data in an h5py file,
along with a JSON mapping file that maps original image numbers to indices.
"""

import argparse
import fnmatch
import json
import os
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

import h5py
import numpy as np


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert ScanNet scene folders to h5py format, or retrieve images from h5py files"
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/scratch/indrisch/data/ScanNet/scans/",
        help="Input directory containing scene folders (default: /scratch/indrisch/data/ScanNet/scans/)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory where processed h5py files will be saved (required for packing mode)"
    )
    
    parser.add_argument(
        "--get",
        action="store_true",
        help="Retrieval mode: retrieve a specific image from an h5py file"
    )
    
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="The path to the image to retrieve. This can be passed instead of --output_dir && --scene_name && --image_name."
    )
    
    parser.add_argument(
        "--scene_name",
        type=str,
        default=None,
        help="Scene name for retrieval mode (e.g., 'scene0000_00')"
    )
    
    parser.add_argument(
        "--image_name",
        type=str,
        default=None,
        help="Image name for retrieval mode (e.g., '4.jpg')"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output file path for retrieved image (required for retrieval mode)"
    )
    
    parser.add_argument(
        "--scene_pattern",
        type=str,
        default="scene*_*",
        help="Pattern to match scene folders. Options: 'scene*_*' (default, all scenes), "
             "'scene0001_00' (specific scene), 'scene0001_00,scene0002_00' (multiple scenes), "
             "or glob patterns like 'scene0001_*' or 'scene*_00' (default: scene*_*)"
    )
    
    parser.add_argument(
        "--color_folder",
        type=str,
        default="color",
        help="Name of the folder containing color images (default: color)"
    )
    
    parser.add_argument(
        "--image_ext",
        type=str,
        default=".jpg",
        help="Image file extension (default: .jpg)"
    )
    
    parser.add_argument(
        "--h5py_filename",
        type=str,
        default="images.hdf5",
        help="Filename for the saved h5py file (default: images.hdf5)"
    )
    
    parser.add_argument(
        "--json_filename",
        type=str,
        default="image_mapping.json",
        help="Filename for the JSON mapping file (default: image_mapping.json)"
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for parallel processing (default: 4)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def process_scene(scene_path, output_dir, color_folder, image_ext, h5py_filename, json_filename, verbose=False):
    """
    Process a single scene folder and create an h5py file with binary image data.
    
    Args:
        scene_path: Path to the scene folder (e.g., scene0000_00)
        output_dir: Directory where the h5py file will be saved
        color_folder: Name of the folder containing color images
        image_ext: Image file extension (e.g., .jpg)
        h5py_filename: Name of the output h5py file
        json_filename: Name of the JSON mapping file
        verbose: Whether to print verbose output
    
    Returns:
        True if successful, False otherwise
    """
    scene_name = scene_path.name
    
    # Check if output files already exist
    output_path = output_dir / scene_name
    h5py_file = output_path / h5py_filename
    json_file = output_path / json_filename
    if h5py_file.exists() and json_file.exists():
        if verbose:
            print(f"Skipping {scene_name}: h5py and JSON files already exist")
        return True
    
    color_dir = scene_path / color_folder
    
    if not color_dir.exists():
        if verbose:
            print(f"Warning: Color folder not found for {scene_name}, skipping...")
        return False
    
    # Find all image files and extract their numbers
    image_files = list(color_dir.glob(f"*{image_ext}"))
    if not image_files:
        if verbose:
            print(f"Warning: No images found in {scene_name}, skipping...")
        return False
    
    # Extract image numbers from filenames
    image_numbers = []
    for img_file in image_files:
        # Extract number from filename (e.g., "40.jpg" -> 40)
        match = re.search(r'(\d+)' + re.escape(image_ext) + r'$', img_file.name)
        if match:
            image_numbers.append(int(match.group(1)))
    
    if not image_numbers:
        if verbose:
            print(f"Warning: Could not extract image numbers from {scene_name}, skipping...")
        return False
    
    # Sort image numbers for consistent ordering
    image_numbers = sorted(image_numbers)
    
    # Load images as binary data and create mapping
    loaded_images = []
    image_number_to_idx = {}
    
    for img_num in image_numbers:
        img_path = color_dir / f"{img_num}{image_ext}"
        try:
            # Read image as binary data
            with open(img_path, 'rb') as fin:
                binary_data = fin.read()
            
            loaded_images.append(binary_data)
            image_number_to_idx[str(img_num)] = len(loaded_images) - 1
            
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to load image {img_num} in {scene_name}: {e}")
            continue
    
    if not loaded_images:
        if verbose:
            print(f"Warning: No images successfully loaded for {scene_name}, skipping...")
        return False
    
    num_loaded = len(loaded_images)
    
    # Create output directory
    output_path = output_dir / scene_name
    output_path.mkdir(parents=True, exist_ok=True)
    h5py_file_path = output_path / h5py_filename
    json_file_path = output_path / json_filename
    
    # Save images to h5py file
    with h5py.File(h5py_file_path, 'w') as f:
        # Create variable-length dataset for binary data
        dt = h5py.special_dtype(vlen=np.dtype('uint8'))
        dset = f.create_dataset('binary_data', (num_loaded,), dtype=dt)
        
        # Save each image's binary data
        for idx, binary_data in enumerate(loaded_images):
            dset[idx] = np.frombuffer(binary_data, dtype='uint8')
    
    # Save JSON mapping file
    # Format: {image_number: index_in_h5py}
    # Example: {"5": 0, "10": 1, "15": 2}
    with open(json_file_path, 'w') as f:
        json.dump(image_number_to_idx, f, indent=2)
    
    if verbose:
        print(f"Processed {scene_name}: {num_loaded} images saved to {h5py_filename}")
    
    return True


def _process_scene_wrapper(args):
    """
    Wrapper function for process_scene to enable parallel processing.
    
    Args:
        args: Tuple of (scene_path, output_dir, color_folder, image_ext, h5py_filename, json_filename, verbose)
    
    Returns:
        Tuple of (scene_name, success, error_message)
    """
    scene_path, output_dir, color_folder, image_ext, h5py_filename, json_filename, verbose = args
    scene_name = scene_path.name if isinstance(scene_path, Path) else Path(scene_path).name
    
    try:
        # Convert string paths to Path objects if needed
        scene_path = Path(scene_path) if isinstance(scene_path, str) else scene_path
        output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
        
        success = process_scene(scene_path, output_dir, color_folder, image_ext, 
                               h5py_filename, json_filename, verbose)
        return (scene_name, success, None)
    except Exception as e:
        return (scene_name, False, str(e))


def retrieve_image(image_path=None, output_dir=None, scene_name=None, image_name=None, output_path=None, 
                   h5py_filename="images.hdf5", json_filename="image_mapping.json", 
                   verbose=False):
    """
    Retrieve a specific image from an h5py file and save it to a file path or return as bytes.
    
    Args:
        image_path: Path to the image to retrieve. This can be passed instead of --output_dir && --scene_name && --image_name. Note that this path may not actually have a file; hence, this code simulates having a file there.
        output_dir: Directory containing the h5py files (where scenes are stored)
        scene_name: Name of the scene (e.g., "scene0000_00")
        image_name: Name of the image file (e.g., "4.jpg")
        output_path: Path where the retrieved image will be saved. If None, returns image bytes instead.
        h5py_filename: Name of the h5py file (default: images.hdf5)
        json_filename: Name of the JSON mapping file (default: image_mapping.json)
        verbose: Whether to print verbose output
    
    Returns:
        If output_path is provided: True if successful, False otherwise
        If output_path is None: bytes object containing the image data
    
    Raises:
        FileNotFoundError: If scene directory, h5py file, or JSON file does not exist
        ValueError: If image number is not found in mapping or index is out of range
        Exception: If there are errors loading JSON or reading from h5py file
    """
    if image_path:
        # Extract output_dir, scene_name, and image_name from image_path
        r = re.search(r"(.*)(scene\d+_\d+).*color/(.*jpg)", image_path)
        if r:
            output_dir = Path(r.group(1)) # this is the path up to "scans"
            scene_name = r.group(2) # the scene name 
            image_name = r.group(3) # the image name
        else:
            raise ValueError(f"Invalid image path: {image_path}")
    
    
    output_dir_path = Path(output_dir)
    scene_path = output_dir_path / scene_name
    
    # Check if scene directory exists
    if not scene_path.exists():
        error_msg = f"Scene directory does not exist: {scene_path}"
        if output_path is None:
            raise FileNotFoundError(error_msg)
        print(f"Error: {error_msg}")
        return False
    
    # Check if h5py and JSON files exist
    h5py_file = scene_path / h5py_filename
    json_file = scene_path / json_filename
    
    if not h5py_file.exists():
        error_msg = f"H5py file does not exist: {h5py_file}"
        if output_path is None:
            raise FileNotFoundError(error_msg)
        print(f"Error: {error_msg}")
        return False
    
    if not json_file.exists():
        error_msg = f"JSON mapping file does not exist: {json_file}"
        if output_path is None:
            raise FileNotFoundError(error_msg)
        print(f"Error: {error_msg}")
        return False
    
    # Extract image number from image name (e.g., "4.jpg" -> "4")
    image_ext = Path(image_name).suffix
    image_number_str = Path(image_name).stem
    
    # Load JSON mapping
    try:
        with open(json_file, 'r') as f:
            image_number_to_idx = json.load(f)
    except Exception as e:
        error_msg = f"Failed to load JSON mapping file: {e}"
        if output_path is None:
            raise Exception(error_msg) from e
        print(f"Error: {error_msg}")
        return False
    
    # Check if image number exists in mapping
    if image_number_str not in image_number_to_idx:
        error_msg = f"Image number '{image_number_str}' not found in mapping for scene {scene_name}"
        if verbose:
            error_msg += f". Available image numbers: {sorted(image_number_to_idx.keys())}"
        if output_path is None:
            raise ValueError(error_msg)
        print(f"Error: {error_msg}")
        return False
    
    # Get the index in the h5py dataset
    idx = image_number_to_idx[image_number_str]
    
    if verbose:
        print(f"Retrieving image {image_name} (index {idx}) from scene {scene_name}")
    
    # Load binary data from h5py file
    try:
        with h5py.File(h5py_file, 'r') as f:
            dset = f['binary_data']
            if idx >= len(dset):
                error_msg = f"Index {idx} is out of range (dataset has {len(dset)} images)"
                if output_path is None:
                    raise ValueError(error_msg)
                print(f"Error: {error_msg}")
                return False
            
            # Get binary data
            binary_data = dset[idx]
            # Convert numpy array back to bytes
            if isinstance(binary_data, np.ndarray):
                image_bytes = binary_data.tobytes()
            else:
                image_bytes = bytes(binary_data)
            
    except Exception as e:
        error_msg = f"Failed to read from h5py file: {e}"
        if output_path is None:
            raise Exception(error_msg) from e
        print(f"Error: {error_msg}")
        return False
    
    # If output_path is None, return the image bytes
    if output_path is None:
        if verbose:
            print(f"Successfully retrieved image {image_name} from scene {scene_name}")
        return image_bytes
    
    # Save image to output path
    output_file_path = Path(output_path)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(output_file_path, 'wb') as f:
            f.write(image_bytes)
        
        if verbose:
            print(f"Successfully saved image to {output_path}")
        
        return True
    except Exception as e:
        print(f"Error: Failed to save image to {output_path}: {e}")
        return False


def match_scene_pattern(scene_name, pattern):
    """
    Check if a scene name matches the given pattern.
    
    Args:
        scene_name: Name of the scene folder (e.g., "scene0001_00")
        pattern: Pattern to match. Can be:
            - "scene*_*" (default) - matches all scenes
            - "scene0001_00" - matches specific scene
            - "scene0001_00,scene0002_00" - matches multiple specific scenes (comma-separated)
            - "scene0001_*" - matches all scenes starting with scene0001_
            - "scene*_00" - matches all scenes ending with _00
            - Any glob pattern supported by fnmatch
    
    Returns:
        True if scene matches pattern, False otherwise
    """
    # Default pattern matches all scenes
    if pattern == "scene*_*" or pattern == "*":
        return True
    
    # Check if pattern contains commas (multiple specific scenes)
    if ',' in pattern:
        scene_list = [s.strip() for s in pattern.split(',')]
        return scene_name in scene_list
    
    # Use fnmatch for glob patterns or exact match
    return fnmatch.fnmatch(scene_name, pattern)


def build_dataset_copy(input_dir, output_dir, scene_pattern="scene*_*", color_folder="color", 
                       image_ext=".jpg", h5py_filename="images.hdf5", json_filename="image_mapping.json",
                       num_workers=4, verbose=False):
    """
    Build a copy of the dataset with images packed as binary data in h5py format.
    
    Args:
        input_dir: Input directory containing scene folders
        output_dir: Output directory where h5py files will be saved
        scene_pattern: Pattern to match scene folders
        color_folder: Name of the folder containing color images
        image_ext: Image file extension
        h5py_filename: Name of the output h5py file
        json_filename: Name of the JSON mapping file
        num_workers: Number of worker processes for parallel processing
        verbose: Whether to print verbose output
    
    Returns:
        Number of successfully processed scenes
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Find all scene folders
    all_folders = [p for p in input_path.iterdir() if p.is_dir()]
    
    # Filter scenes based on pattern
    scene_folders = []
    for folder in all_folders:
        scene_name = folder.name
        # Only consider folders that look like scenes (start with "scene" and match scene pattern format)
        # unless pattern explicitly specifies something else
        if scene_pattern == "scene*_*" or scene_pattern == "*":
            # Default: only process scene folders matching sceneXXXX_YY format
            if scene_name.startswith("scene") and re.match(r'scene\d+_\d+', scene_name):
                scene_folders.append(folder)
        else:
            # Custom pattern: check if it matches
            if match_scene_pattern(scene_name, scene_pattern):
                scene_folders.append(folder)
    
    scene_folders = sorted(scene_folders)
    
    if not scene_folders:
        if verbose:
            print(f"No scene folders found in {input_dir}")
        return 0
    
    # Filter out scenes that already have the output files
    scenes_to_process = []
    skipped_count = 0
    for scene_path in scene_folders:
        scene_name = scene_path.name
        output_scene_path = output_path / scene_name
        h5py_file = output_scene_path / h5py_filename
        json_file = output_scene_path / json_filename
        if h5py_file.exists() and json_file.exists():
            skipped_count += 1
            if verbose:
                print(f"Skipping {scene_name}: h5py and JSON files already exist")
        else:
            scenes_to_process.append(scene_path)
    
    if verbose:
        print(f"Found {len(scene_folders)} scene folders total")
        if skipped_count > 0:
            print(f"Skipping {skipped_count} scenes that already have h5py files")
        print(f"Processing {len(scenes_to_process)} scenes")
        if num_workers > 1:
            print(f"Using {num_workers} worker processes for parallel processing")
    
    if not scenes_to_process:
        if verbose:
            print("All scenes already processed. Nothing to do.")
        return len(scene_folders)  # Return total count as all were "successful" (already done)
    
    # Process scenes sequentially or in parallel
    success_count = skipped_count  # Count skipped scenes as successful
    
    if num_workers <= 1:
        # Sequential processing
        for scene_path in scenes_to_process:
            try:
                if process_scene(scene_path, output_path, color_folder, image_ext, 
                               h5py_filename, json_filename, verbose):
                    success_count += 1
            except Exception as e:
                if verbose:
                    print(f"Error processing {scene_path.name}: {e}")
                continue
    else:
        # Parallel processing
        # Prepare arguments for each scene
        scene_args = [
            (scene_path, output_path, color_folder, image_ext, h5py_filename, json_filename, verbose)
            for scene_path in scenes_to_process
        ]
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_scene = {
                executor.submit(_process_scene_wrapper, args): args[0]
                for args in scene_args
            }
            
            # Process completed tasks as they finish
            for future in as_completed(future_to_scene):
                scene_path = future_to_scene[future]
                try:
                    scene_name, success, error = future.result()
                    if success:
                        success_count += 1
                    elif verbose and error:
                        print(f"Error processing {scene_name}: {error}")
                except Exception as e:
                    if verbose:
                        print(f"Error processing {scene_path.name}: {e}")
    
    if verbose:
        print(f"Successfully processed {success_count}/{len(scene_folders)} scenes")
        if skipped_count > 0:
            print(f"  ({skipped_count} were already processed and skipped)")
    
    return success_count


def main():
    """Main function."""
    args = parse_args()
    
    if args.get:
        # Retrieval mode
        if (not args.image_path) or (not args.output_dir and not args.scene_name and not args.image_name):
            raise ValueError("Please provide either --image_path or --output_dir && --scene_name && --image_name to specify which image to retrieve")
        if not args.output_path:
            warnings.warn("By not providing --output_path, the image will be returned as a bytes object.")
        
        if args.verbose:
            print(args)
        
        # Retrieve the image
        success = retrieve_image(
            image_path=args.image_path if args.image_path else None,
            output_dir=args.output_dir if args.output_dir else None,
            scene_name=args.scene_name if args.scene_name else None,
            image_name=args.image_name if args.image_name else None,
            output_path=args.output_path if args.output_path else None,
            h5py_filename=args.h5py_filename,
            json_filename=args.json_filename,
            verbose=args.verbose
        )
        
        if success:
            print(f"Successfully retrieved {args.image_name if args.image_name else args.image_path} from {args.scene_name if args.scene_name else args.output_dir} to {args.output_path if args.output_path else 'bytes object'}")
        else:
            print(f"Failed to retrieve {args.image_name if args.image_name else args.image_path} from {args.scene_name if args.scene_name else args.output_dir}")
            exit(1)
    
    else:
        # Packing mode (default)
        if not args.output_dir:
            raise ValueError("--output_dir is required in packing mode")
        
        # Validate input directory
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        # Create output directory if it doesn't exist
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.verbose:
            print(f"Packing mode:")
            print(f"  Input directory: {input_dir}")
            print(f"  Output directory: {output_dir}")
            print(f"  Scene pattern: {args.scene_pattern}")
            print(f"  Color folder: {args.color_folder}")
            print(f"  Image extension: {args.image_ext}")
            print(f"  H5py filename: {args.h5py_filename}")
            print(f"  JSON filename: {args.json_filename}")
            print(f"  Number of workers: {args.num_workers}")
        
        # Build the dataset copy
        success_count = build_dataset_copy(
            input_dir=input_dir,
            output_dir=output_dir,
            scene_pattern=args.scene_pattern,
            color_folder=args.color_folder,
            image_ext=args.image_ext,
            h5py_filename=args.h5py_filename,
            json_filename=args.json_filename,
            num_workers=args.num_workers,
            verbose=args.verbose
        )
        
        print(f"Processing complete. Successfully processed {success_count} scenes.")


if __name__ == "__main__":
    main()
