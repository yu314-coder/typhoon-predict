"""Export a verified training checkpoint as inference-only Trackformer 1.2 weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


EXPECTED_SHA256 = "f194a23d3f91ea76ad776dfad942fabd669eeae8b3fd665815463095367e9ee0"
EXPECTED_ARCH = "v173-moving-core-multiscale-attention"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    if sha256(args.checkpoint) != EXPECTED_SHA256:
        raise ValueError("Checkpoint SHA-256 does not match the selected release source")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (payload.get("version"), payload.get("architecture"), payload.get("epoch")) != (
        "1.2.73", EXPECTED_ARCH, 4
    ):
        raise ValueError("Checkpoint identity does not match the selected release source")
    if len(payload.get("history", [])) != 4:
        raise ValueError("Expected four completed epochs in checkpoint history")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights = args.output_dir / "weights.pt"
    torch.save(payload["model"], weights)
    source = args.output_dir
    manifest = {
        "public_version": "1.2",
        "release_status": "research_candidate_not_operational",
        "source_checkpoint_version": payload["version"],
        "source_checkpoint_epoch": payload["epoch"],
        "architecture": payload["architecture"],
        "source_checkpoint_sha256": EXPECTED_SHA256,
        "inference_weights_sha256": sha256(weights),
        "dataset_sha256": payload["dataset_sha256"],
        "source_module_sha256": {name: sha256(source / name) for name in (
            "model.py", "baseline_model.py", "v165_base.py")},
        "data_contract": payload["data_contract"],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: manifest[key] for key in (
        "public_version", "architecture", "inference_weights_sha256")}, indent=2))


if __name__ == "__main__":
    main()
