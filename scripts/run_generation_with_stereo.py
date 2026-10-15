#!/usr/bin/env python3
"""Run the standard generation CLI with the reviewed stereo writer extension."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

SERIALIZATION_PATCH_SHA256 = "6962088d83c98ecd8a3ba26d776ebf4b18130a055aea8bf6c208656326db6b35"


def project_root():
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "src/kinase_gap/generate.py").is_file() and (
                candidate / "scripts/run_generation_shard.py").is_file():
            return candidate
    raise FileNotFoundError("Place this entrypoint and its sibling extension inside the project archive")


def _load(path, name):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def main():
    root = project_root()
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(1, str(root / "vendor/DiffLinker"))
    extension_path = Path(__file__).resolve().with_name("serialization_patch.py")
    if hashlib.sha256(extension_path.read_bytes()).hexdigest() != SERIALIZATION_PATCH_SHA256:
        raise ValueError("Stereo writer extension checksum mismatch")
    extension = _load(extension_path, "kinase_public_stereo_extension")
    extension.install()
    native = _load(root / "scripts/run_generation_shard.py", "kinase_native_generation_entrypoint")
    # argparse receives the original argv: all modes/options retain native meaning.
    return native.main()


if __name__ == "__main__":
    raise SystemExit(main())
