from __future__ import annotations

import argparse
import base64
import json
import logging
import mimetypes
import os
import re
from pathlib import Path
import sys
from typing import Any

import requests
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.logging_utils import configure_logging
from agents.schemas import Character, Shot, Storyboard
from agents.http_utils import openai_client_kwargs, requests_get

logger = logging.getLogger(__name__)


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        key = k.strip()
        if not key or key in os.environ:
            continue
        val = v.strip().strip('"').strip("'")
        os.environ[key] = val


def _normalize_base_url(url: str) -> str:
    text = (url or "").strip()
    if not text:
        return text
    if not text.startswith(("http://", "https://")):
        text = f"https://{text}"
    return text.rstrip("/")


def _encode_image_to_data_uri(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError(f"Unsupported image type: {path}")
    raw = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{raw}"


def _to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


def _extract_image_data(content: Any) -> tuple[str | None, str | None]:
    text = _to_text(content)
    m = re.search(r"data:image/(png|jpeg|jpg|webp);base64,([A-Za-z0-9+/=]+)", text)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def _extract_image_url(content: Any) -> str | None:
    text = _to_text(content)
    urls = re.findall(r"https?://[^\s)\]\">]+", text, re.IGNORECASE)
    for u in urls:
        low = u.lower()
        if any(k in low for k in [".png", ".jpg", ".jpeg", ".webp", "/response_images/"]):
            return u
    return None


def _match_characters_for_shot(shot: Shot, characters: dict[str, Character]) -> list[Character]:
    if not characters:
        return []
    by_name: list[Character] = []
    seen: set[str] = set()
    for n in shot.character_names:
        key = str(n or "").strip()
        if not key:
            continue
        for cname, c in characters.items():
            if cname == key and cname not in seen:
                by_name.append(c)
                seen.add(cname)
    if by_name:
        return by_name
    return list(characters.values())


def _collect_reference_images(
    *,
    shot: Shot,
    scene_images: list[str],
    characters: dict[str, Character],
    prev_keyframe: str | None,
    max_images: int,
) -> list[Path]:
    refs: list[Path] = []
    if prev_keyframe:
        p = Path(prev_keyframe)
        if p.exists():
            refs.append(p)
    for s in scene_images:
        p = Path(s)
        if p.exists():
            refs.append(p)
    for c in _match_characters_for_shot(shot, characters):
        if c.reference_image:
            p = Path(c.reference_image)
            if p.exists():
                refs.append(p)
        for s in c.reference_images:
            p = Path(s)
            if p.exists():
                refs.append(p)
    uniq: list[Path] = []
    seen: set[str] = set()
    for p in refs:
        rp = str(p.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(p)
        if len(uniq) >= max_images:
            break
    return uniq


def _build_shot_prompt(shot: Shot, scene_plot: str, has_prev: bool) -> str:
    continuity = (
        "Keep strict visual continuity with the previous keyframe in character identity, costume, lighting, lens, and color palette."
        if has_prev
        else "This is the first shot of the scene; establish the scene look and characters clearly."
    )
    return (
        "You are generating a cinematic storyboard keyframe image.\n"
        f"Shot ID: {shot.shot_id}\n"
        f"Scene ID: {shot.scene_id}\n"
        f"Shot sequence: {shot.sequence}\n"
        f"Shot description: {shot.auto_prompt()}\n"
        f"Scene story: {scene_plot}\n"
        f"Characters in shot: {', '.join(shot.character_names) if shot.character_names else 'not explicitly specified'}\n"
        f"{continuity}\n"
        "Output exactly one high-quality frame image."
    )


def _request_nanobanana_image(
    *,
    client: OpenAI,
    model_name: str,
    prompt: str,
    ref_images: list[Path],
    timeout_sec: int,
    output_path: Path,
) -> str:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for p in ref_images:
        content.append({"type": "image_url", "image_url": {"url": _encode_image_to_data_uri(p)}})

    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content}],
        timeout=timeout_sec,
    )
    if not resp.choices:
        raise RuntimeError("No choices in nano_banana response")
    raw = resp.choices[0].message.content

    ext, b64 = _extract_image_data(raw)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if b64:
        suffix = f".{ext or 'png'}"
        final_path = output_path if output_path.suffix else output_path.with_suffix(suffix)
        final_path.write_bytes(base64.b64decode(b64))
        return str(final_path)

    url = _extract_image_url(raw)
    if url:
        r = requests_get(url, timeout=timeout_sec)
        r.raise_for_status()
        final_path = output_path if output_path.suffix else output_path.with_suffix(".png")
        final_path.write_bytes(r.content)
        return str(final_path)

    raise RuntimeError(f"nano_banana response does not contain image data: {str(raw)[:240]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recursive keyframe generation with nano_banana: prev-shot keyframe + current shot text + scene refs + character refs."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("/data1/AgentG/hhb/CameraMaster/outputs/five_agents_e2e_nb_magref_omni_ace_real"),
    )
    parser.add_argument("--storyboard-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--api-key", type=str, default="")
    parser.add_argument("--base-url", type=str, default="")
    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument("--timeout-sec", type=int, default=480)
    parser.add_argument("--max-scenes", type=int, default=0, help="0 means all scenes.")
    parser.add_argument(
        "--scene-ids",
        type=str,
        default="",
        help="Optional comma-separated scene ids to process, e.g. SC002,SC003",
    )
    parser.add_argument("--max-shots-per-scene", type=int, default=0)
    parser.add_argument("--max-reference-images", type=int, default=8)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If keyframe file exists, skip regeneration.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)

    run_dir = args.run_dir.resolve()
    storyboard_json = args.storyboard_json.resolve() if args.storyboard_json else (run_dir / "storyboard.json")
    output_dir = args.output_dir.resolve() if args.output_dir else (run_dir / "keyframes_recursive_nanobanana")
    output_dir.mkdir(parents=True, exist_ok=True)
    _load_env_file(ROOT / ".env")

    if not storyboard_json.exists():
        raise SystemExit(f"storyboard-json not found: {storyboard_json}")

    api_key = (args.api_key or os.getenv("NANO_BANANA_API_KEY", "")).strip()
    base_url = (args.base_url or os.getenv("NANO_BANANA_BASE_URL", "https://api.xi-ai.cn/v1")).strip()
    model_name = (args.model_name or os.getenv("NANO_BANANA_MODEL_NAME", "gemini-2.5-flash-image-preview")).strip()
    if not api_key:
        raise SystemExit("NANO_BANANA_API_KEY is required (or pass --api-key).")

    client = OpenAI(
        api_key=api_key,
        base_url=_normalize_base_url(base_url),
        **openai_client_kwargs(),
    )
    storyboard = Storyboard.from_dict(json.loads(storyboard_json.read_text(encoding="utf-8")))
    ordered_shots = storyboard.ordered_shots_scene_first(unique_by_scene_sequence=True)
    shot_map: dict[str, list[Shot]] = {}
    for sh in ordered_shots:
        shot_map.setdefault(sh.scene_id, []).append(sh)

    scene_ids = sorted(storyboard.scenes.keys())
    if args.scene_ids.strip():
        requested = {x.strip() for x in args.scene_ids.split(",") if x.strip()}
        scene_ids = [sid for sid in scene_ids if sid in requested]
    if args.max_scenes and args.max_scenes > 0:
        scene_ids = scene_ids[: args.max_scenes]

    report_rows: list[dict[str, Any]] = []
    success = 0
    failed = 0
    for scene_id in scene_ids:
        scene = storyboard.scenes[scene_id]
        shots = shot_map.get(scene_id, [])
        if args.max_shots_per_scene and args.max_shots_per_scene > 0:
            shots = shots[: args.max_shots_per_scene]
        prev_keyframe: str | None = None
        logger.info("keyframe scene status=start scene_id=%s shots=%d", scene_id, len(shots))
        for idx, shot in enumerate(shots, start=1):
            out_path = output_dir / scene_id / f"keyframe_{shot.shot_id}.png"
            prev_for_current = prev_keyframe if idx > 1 else None
            refs = _collect_reference_images(
                shot=shot,
                scene_images=scene.reference_images,
                characters=storyboard.screenplay.characters,
                prev_keyframe=prev_for_current,
                max_images=max(1, args.max_reference_images),
            )
            prompt = _build_shot_prompt(shot, scene.plot_description, has_prev=bool(prev_for_current))
            if args.resume and out_path.exists() and out_path.stat().st_size > 0:
                generated = str(out_path)
                ok = True
                mode = "resume_skip"
            else:
                try:
                    generated = _request_nanobanana_image(
                        client=client,
                        model_name=model_name,
                        prompt=prompt,
                        ref_images=refs,
                        timeout_sec=args.timeout_sec,
                        output_path=out_path,
                    )
                    ok = True
                    mode = "generated"
                except Exception as exc:
                    generated = ""
                    ok = False
                    mode = f"failed:{type(exc).__name__}:{exc}"

            if ok:
                success += 1
                prev_keyframe = generated
                logger.info(
                    "keyframe shot status=done scene_id=%s shot_id=%s refs=%d output=%s",
                    scene_id,
                    shot.shot_id,
                    len(refs),
                    generated,
                )
            else:
                failed += 1
                logger.warning(
                    "keyframe shot status=failed scene_id=%s shot_id=%s refs=%d reason=%s",
                    scene_id,
                    shot.shot_id,
                    len(refs),
                    mode,
                )

            report_rows.append(
                {
                    "scene_id": scene_id,
                    "shot_id": shot.shot_id,
                    "sequence": int(shot.sequence),
                    "is_first_shot_in_scene": idx == 1,
                    "previous_keyframe_used": bool(prev_for_current),
                    "reference_images": [str(p) for p in refs],
                    "output_keyframe": generated,
                    "ok": ok,
                    "status": mode,
                }
            )
        logger.info("keyframe scene status=done scene_id=%s", scene_id)

    report = {
        "run_dir": str(run_dir),
        "storyboard_json": str(storyboard_json),
        "output_dir": str(output_dir),
        "model_name": model_name,
        "scene_count": len(scene_ids),
        "total_tasks": len(report_rows),
        "success": success,
        "failed": failed,
        "rows": report_rows,
    }
    report_path = output_dir / "keyframe_recursive_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=== Recursive Nano Banana Keyframe Generation Completed ===")
    print(f"Output dir: {output_dir}")
    print(f"Report: {report_path}")
    print(f"Success: {success}, Failed: {failed}")


if __name__ == "__main__":
    main()
