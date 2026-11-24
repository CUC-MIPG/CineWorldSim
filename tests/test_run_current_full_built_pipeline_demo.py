from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def test_current_full_built_pipeline_demo_dry_run(tmp_path: Path) -> None:
    output_dir = tmp_path / "current_full_demo"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_current_full_built_pipeline_demo.py",
            "--output-dir",
            str(output_dir),
            "--max-steps",
            "1",
            "--backend-mode",
            "all_banana",
            "--path-mode",
            "llm",
            "--no-strict-online",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    root_manifest = output_dir / "pipeline_manifest.json"
    backend_result = output_dir / "all_banana" / "pipeline_result.json"
    path_root = output_dir / "all_banana" / "path_orchestration"

    assert root_manifest.exists()
    assert backend_result.exists()
    assert (path_root / "narrative_nodes.json").exists()
    assert (path_root / "path_candidates.json").exists()
    assert (path_root / "path_llm_direct" / "narrative_path_plan.json").exists()

    manifest = json.loads(root_manifest.read_text(encoding="utf-8"))
    assert manifest["status"] == "done"
    assert manifest["runs"][0]["candidate_count"] >= 1
