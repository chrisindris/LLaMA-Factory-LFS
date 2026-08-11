#!/usr/bin/env python3

""" Utilities to create a copy YAML file automatically for runs.
"""

import argparse
import typing
import sys
import os
import ruamel.yaml

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge a base model snapshot and one LoRA checkpoint into a full model "
            "using LLaMA-Factory export logic."
        )
    )
    parser.add_argument(
        "--yaml-template-path",
        required=True,
        help="Path to base YAML template.",
    )
    parser.add_argument(
        "--yaml-output-path",
        required=True,
        help="Path to output YAML file.",
    )
    return parser.parse_known_args()

def correct_comment_after_tag(s):
    # if a previous line ends in a tag and this line has enough spaces
    # at the start, append the end of the line to the previous one
    res = []
    prev_line = -1 # -1 if previous line didn't end in tag, else length of previous line
    for line in s.splitlines():
        linesplit = line.split()
        if linesplit and linesplit[-1].startswith('!'):
            prev_line = len(line)
        else:
            if prev_line > 0:
                if line.lstrip().startswith('#') and line.find('#') > prev_line:
                    res[-1] += line[prev_line:]
                    prev_line = -1
                    continue
            prev_line = -1
        res.append(line)
    return '\n'.join(res)+'\n'

def infer_data_type(value: str):
    """Infer data type of the given string value."""
    if value.lower() in ['none', 'null']:
        return None
    elif value.lower() == "true":
        return True
    elif value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value

def parse_key_value_pairs(modifications):
    extra_args = {}
    iterator = iter(modifications)
    for item in iterator:
        if item.startswith("--"):
            key = item.lstrip("-")
            try:
                extra_args[key] = infer_data_type(next(iterator))
            except StopIteration:
                extra_args[key] = True
    return extra_args     

def modify_yaml(data, modifications):
    """Apply CLI key/value overrides onto a loaded YAML mapping.

    Special-case null handling for resume fields (fresh start / epoch 0):
    - resume_from_checkpoint=None  -> keep key as YAML null
    - adapter_name_or_path=None    -> omit key entirely (LoRA from scratch)
    """
    for k, v in modifications.items():
        if v is None and k == "adapter_name_or_path":
            data.pop(k, None)
        elif v is None and k == "resume_from_checkpoint":
            data[k] = None
        else:
            data[k] = v
    return data

if __name__ == "__main__":
    args, modifications = parse_args()
    modifications = parse_key_value_pairs(modifications)
    with open(args.yaml_template_path, "r") as f:
        yaml = ruamel.yaml.YAML()
        yaml.width = 4096
        yaml.preserve_quotes = True
        # Emit explicit `null` (matches non-resume train YAMLs) instead of empty values.
        yaml.representer.add_representer(
            type(None),
            lambda rep, _data: rep.represent_scalar("tag:yaml.org,2002:null", "null"),
        )
        data = yaml.load(f)
    data = modify_yaml(data, modifications)
    with open(args.yaml_output_path, "w") as f:
        yaml.dump(data, f, transform=correct_comment_after_tag)

    