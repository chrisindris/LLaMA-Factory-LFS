# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate a portable dataset_info.json whose file_name values are repo-relative.

LLaMA-Factory joins ``dataset_dir`` with each ``file_name``. Moving the registry
from ``data/`` to ``data/annotations/`` therefore requires rewriting every
``file_name``:

* Absolute paths become ``<dataset_name>/<basename>`` and are reached through a
  symlink created next to the generated registry, so no large file is copied.
* Already-relative paths are re-anchored with ``..`` so they keep resolving.

Usage:
    python scripts/make_portable_dataset_info.py
    python scripts/make_portable_dataset_info.py --source A --dest B --no-symlinks
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple


def rewrite_registry(
    registry: Dict[str, Any], source_dir: str, dest_dir: str
) -> Tuple[Dict[str, Any], List[Tuple[str, str]]]:
    """Rewrite every ``file_name`` so it resolves against ``dest_dir``.

    Args:
        registry: Parsed contents of a dataset_info.json file.
        source_dir: Directory the original relative file names resolve against.
        dest_dir: Directory the rewritten file names must resolve against.

    Returns:
        A tuple of the new registry and a list of ``(link_relpath, target)``
        pairs describing the symlinks needed for absolute entries.
    """
    new_registry: Dict[str, Any] = {}
    links: List[Tuple[str, str]] = []

    for name, attrs in registry.items():
        if not isinstance(attrs, dict) or "file_name" not in attrs:
            new_registry[name] = attrs
            continue

        new_attrs = dict(attrs)
        file_name = attrs["file_name"]

        if os.path.isabs(file_name):
            link_relpath = "{}/{}".format(name, os.path.basename(file_name))
            new_attrs["file_name"] = link_relpath
            links.append((link_relpath, file_name))
        else:
            absolute = os.path.join(source_dir, file_name)
            new_attrs["file_name"] = os.path.relpath(absolute, dest_dir).replace(os.sep, "/")

        new_registry[name] = new_attrs

    return new_registry, links


def _create_symlink(link_path: str, target: str) -> None:
    """Create or refresh a single symlink, never clobbering a real file."""
    os.makedirs(os.path.dirname(link_path), exist_ok=True)

    if os.path.islink(link_path):
        if os.path.realpath(link_path) == os.path.realpath(target):
            return

        os.unlink(link_path)
    elif os.path.exists(link_path):
        raise RuntimeError("refusing to replace non-symlink: {}".format(link_path))

    os.symlink(target, link_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point. Returns a process exit code."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=os.path.join(repo_root, "data", "dataset_info.json"))
    parser.add_argument("--dest", default=os.path.join(repo_root, "data", "annotations", "dataset_info.json"))
    parser.add_argument("--no-symlinks", action="store_true", help="rewrite paths without creating symlinks")
    args = parser.parse_args(argv)

    with open(args.source, encoding="utf-8") as f:
        registry = json.load(f)

    source_dir = os.path.dirname(os.path.abspath(args.source))
    dest_dir = os.path.dirname(os.path.abspath(args.dest))
    new_registry, links = rewrite_registry(registry, source_dir, dest_dir)

    exit_code = 0
    for link_relpath, target in links:
        if not os.path.exists(target):
            print("missing annotation source for {}: {}".format(link_relpath, target))
            exit_code = 1
            continue

        if not args.no_symlinks:
            _create_symlink(os.path.join(dest_dir, link_relpath), target)

    os.makedirs(dest_dir, exist_ok=True)
    with open(args.dest, "w", encoding="utf-8") as f:
        json.dump(new_registry, f, indent=2)
        f.write("\n")

    print("wrote {} ({} entries, {} symlinked)".format(args.dest, len(new_registry), len(links)))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
