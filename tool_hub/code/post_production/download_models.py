from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path("/data1/AgentG/hhb/CameraMaster")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download post-production model repos into tool_hub/models")
    parser.add_argument(
        "--hf-endpoint",
        type=str,
        default="https://huggingface.co",
        help="HF endpoint (e.g., https://huggingface.co)",
    )
    parser.add_argument("--only", choices=["all", "bgm", "dubbing"], default="all")
    args = parser.parse_args()

    os.environ["HF_ENDPOINT"] = args.hf_endpoint

    targets = []
    if args.only in {"all", "bgm"}:
        targets.append(
            (
                "ACE-Step/Ace-Step1.5",
                ROOT / "tool_hub/models/bgm/Ace-Step1.5",
            )
        )
    if args.only in {"all", "dubbing"}:
        targets.append(
            (
                "k2-fsa/OmniVoice",
                ROOT / "tool_hub/models/dubbing/OmniVoice",
            )
        )

    for repo_id, local_dir in targets:
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {repo_id} -> {local_dir}")
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"Done: {path}")


if __name__ == "__main__":
    main()

