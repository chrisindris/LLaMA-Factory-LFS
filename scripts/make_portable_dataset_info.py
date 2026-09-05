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
  Entries may name a *directory* (the repo's registry records seven such, with a
  trailing slash); the trailing slash is preserved in the rewritten value while
  the symlink itself is created at the slash-free path.
* Already-relative paths are re-anchored with ``..`` so they keep resolving.

Usage:
    python scripts/make_portable_dataset_info.py
    python scripts/make_portable_dataset_info.py --source A --dest B --no-symlinks
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from typing import Any


def rewrite_registry(
    registry: dict[str, Any], source_dir: str, dest_dir: str, overrides: dict[str, str] | None = None
) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    """Rewrite every ``file_name`` so it resolves against ``dest_dir``.

    Args:
        registry: Parsed contents of a dataset_info.json file.
        source_dir: Directory the original relative file names resolve against.
        dest_dir: Directory the rewritten file names must resolve against.
        overrides: Optional ``{dataset_name: absolute_path}`` replacing the
            recorded source before the link target is derived. This is how a
            second user redirects an entry whose registry path belongs to
            someone else, without editing the registry.

    Returns:
        A tuple of the new registry and a list of ``(link_relpath, target)``
        pairs describing the symlinks needed for absolute entries. Each
        ``link_relpath`` is slash-free even when the rewritten ``file_name``
        keeps its trailing slash.
    """
    overrides = overrides or {}
    new_registry: dict[str, Any] = {}
    links: list[tuple[str, str]] = []

    for name, attrs in registry.items():
        if not isinstance(attrs, dict) or "file_name" not in attrs:
            # Copy so the returned registry never shares mutable state with the caller's.
            new_registry[name] = dict(attrs) if isinstance(attrs, dict) else attrs
            continue

        new_attrs = dict(attrs)
        file_name = overrides.get(name, attrs["file_name"])

        if not isinstance(file_name, str):
            new_registry[name] = new_attrs
            continue

        if os.path.isabs(file_name):
            # A trailing slash marks a directory entry. basename() would return
            # "" for it, which would place the symlink at "<name>/" -- a path that
            # cannot be created once "<name>" exists as the link's parent.
            is_dir = file_name.endswith("/")
            target = file_name.rstrip("/")
            link_relpath = f"{name}/{os.path.basename(target)}"
            new_attrs["file_name"] = f"{link_relpath}/" if is_dir else link_relpath
            links.append((link_relpath, target))
        else:
            absolute = os.path.join(source_dir, file_name)
            new_attrs["file_name"] = os.path.relpath(absolute, dest_dir).replace(os.sep, "/")

        new_registry[name] = new_attrs

    return new_registry, links


def _create_symlink(link_path: str, target: str) -> None:
    """Create or refresh a single symlink, never clobbering a real file.

    Raises:
        RuntimeError: If ``link_path`` exists and is not a symlink, so that
            staging can never destroy real data.
    """
    os.makedirs(os.path.dirname(link_path), exist_ok=True)

    if os.path.islink(link_path):
        if os.path.realpath(link_path) == os.path.realpath(target):
            return

        os.unlink(link_path)
    elif os.path.exists(link_path):
        raise RuntimeError(f"refusing to replace non-symlink: {link_path}")

    os.symlink(target, link_path)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=os.path.join(repo_root, "data", "dataset_info.json"))
    parser.add_argument("--dest", default=os.path.join(repo_root, "data", "annotations", "dataset_info.json"))
    parser.add_argument("--no-symlinks", action="store_true", help="rewrite paths without creating symlinks")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="replace one dataset's recorded source path; repeatable",
    )
    parser.add_argument(
        "--require",
        default="",
        metavar="NAME[,NAME...]",
        help="datasets whose sources must exist; a missing source elsewhere only warns",
    )
    args = parser.parse_args(argv)

    overrides: dict[str, str] = {}
    for item in args.override:
        name, sep, path = item.partition("=")
        if not sep or not name or not path:
            parser.error(f"--override needs NAME=PATH, got: {item}")

        overrides[name] = path

    required = {n for n in args.require.split(",") if n}

    with open(args.source, encoding="utf-8") as f:
        registry = json.load(f)

    # A typo in --require would otherwise silently downgrade a required dataset
    # to "warn only", which is the failure mode this flag exists to prevent.
    unknown = sorted(required - set(registry))
    if unknown:
        parser.error(f"--require names datasets absent from {args.source}: {', '.join(unknown)}")

    source_dir = os.path.dirname(os.path.abspath(args.source))
    dest_dir = os.path.dirname(os.path.abspath(args.dest))
    new_registry, links = rewrite_registry(registry, source_dir, dest_dir, overrides)

    exit_code = 0
    in_place = 0
    missing_required: list[str] = []
    skipped = 0
    for link_relpath, target in links:
        dataset = link_relpath.split("/", 1)[0]
        # Every entry is rewritten, but only the datasets this job actually loads may
        # fail the run. Seven of the nine absolute entries belong to unrelated ablations;
        # failing on those would make the exit code useless as a gate.
        is_required = dataset in required

        if not os.path.exists(target):
            if is_required:
                missing_required.append(dataset)
                print(f"missing source for {link_relpath}: {target}")
                exit_code = 1
            else:
                # Summarised below rather than printed per entry: on a machine that
                # only has this job's data, seven such lines every run would bury the
                # two that matter.
                skipped += 1

            continue

        if args.no_symlinks:
            continue

        try:
            # Counts a link that was already correct too, so the number describes the
            # resulting tree rather than how much churn this particular run caused.
            _create_symlink(os.path.join(dest_dir, link_relpath), target)
            in_place += 1
        except (OSError, RuntimeError) as err:
            print(f"cannot link {link_relpath}: {err}")
            if is_required:
                exit_code = 1

    os.makedirs(dest_dir, exist_ok=True)
    # Publish atomically and only on success: a registry whose links are dangling
    # would otherwise satisfy preflight's existence check and the job would die
    # after the allocation starts, on a node that cannot fetch the missing data.
    if skipped:
        print(f"skipped {skipped} entries that are absent and not required")

    if exit_code == 0:
        tmp_dest = f"{args.dest}.tmp"
        with open(tmp_dest, "w", encoding="utf-8") as f:
            json.dump(new_registry, f, indent=2)
            f.write("\n")

        os.replace(tmp_dest, args.dest)
        print(f"wrote {args.dest} ({len(new_registry)} entries, {in_place} links in place)")
    else:
        print(
            f"not writing {args.dest}: required datasets unavailable "
            f"({', '.join(sorted(set(missing_required)))}). "
            f"Set the matching PORTABLE_SRC_*_ANNOTATION in scripts/site.env."
        )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
