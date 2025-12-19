#!/usr/bin/env python3
"""
Script to modify specific keys in a YAML file and write to a new file.
"""

import argparse
import sys
import yaml
from pathlib import Path
from typing import Any, Dict, List


def set_nested_value(data: Dict[str, Any], key_path: str, value: Any) -> None:
    """
    Set a value in a nested dictionary using a dot-separated key path.
    
    Args:
        data: The dictionary to modify
        key_path: Dot-separated path to the key (e.g., "foo.bar.baz")
        value: The value to set
    """
    keys = key_path.split('.')
    current = data
    
    # Navigate to the parent of the target key
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        elif not isinstance(current[key], dict):
            # If the intermediate key exists but is not a dict, convert it
            current[key] = {}
        current = current[key]
    
    # Set the final value
    final_key = keys[-1]
    current[final_key] = value


def modify_yaml(input_path: str, output_path: str, keys: List[str], values: List[str]) -> None:
    """
    Read a YAML file, modify specified keys, and write to output file.
    
    Args:
        input_path: Path to input YAML file
        output_path: Path to output YAML file
        keys: List of keys to modify (supports dot notation for nested keys)
        values: List of new values (in order matching keys)
    """
    if len(keys) != len(values):
        raise ValueError(f"Number of keys ({len(keys)}) must match number of values ({len(values)})")
    
    # Read input YAML file
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    
    # Modify specified keys
    for key, value in zip(keys, values):
        # Try to parse value as appropriate type (int, float, bool, or keep as string)
        parsed_value = parse_value(value)
        
        if '.' in key:
            # Handle nested keys
            set_nested_value(data, key, parsed_value)
        else:
            # Handle top-level keys
            data[key] = parsed_value
    
    # Write output YAML file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def parse_value(value: str) -> Any:
    """
    Parse a string value to appropriate Python type.
    
    Args:
        value: String value to parse
        
    Returns:
        Parsed value (int, float, bool, None, or str)
    """
    # Try boolean
    if value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    
    # Try None/null
    if value.lower() in ('null', 'none'):
        return None
    
    # Try integer
    try:
        return int(value)
    except ValueError:
        pass
    
    # Try float
    try:
        return float(value)
    except ValueError:
        pass
    
    # Return as string
    return value


def main():
    parser = argparse.ArgumentParser(
        description='Modify specific keys in a YAML file and write to a new file.'
    )
    parser.add_argument(
        '--input_yaml',
        type=str,
        required=True,
        help='Path to input YAML file'
    )
    parser.add_argument(
        '--output_yaml',
        type=str,
        required=True,
        help='Path to output YAML file'
    )
    parser.add_argument(
        '--keys',
        nargs='+',
        required=True,
        help='List of keys to modify (supports dot notation for nested keys)'
    )
    parser.add_argument(
        '--values',
        nargs='+',
        required=True,
        help='List of new values for the keys (in order)'
    )
    
    args = parser.parse_args()
    
    try:
        modify_yaml(args.input_yaml, args.output_yaml, args.keys, args.values)
        print(f"Successfully modified YAML file: {args.output_yaml}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    import sys
    main()

