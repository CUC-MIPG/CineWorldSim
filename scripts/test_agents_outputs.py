from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.client import resolve_llm_env
from agents.director_agent import DirectorAgent
from agents.movie_producer_agent import MovieProducerAgent
from agents.post_production_agent import PostProductionAgent
from agents.schemas import Storyboard
from agents.shot_agent import ShotAgent
from agents.story_supervisor_agent import StorySupervisorAgent
from agents.video_agent import VideoAgent


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _has_api_key(scope: str) -> bool:
    key, _, _ = resolve_llm_env(scope)
    return bool(key)


def _make_result(agent_name: str) -> dict[str, Any]:
    return {
        "agent": agent_name,
        "status": "unknown",
        "started_at": _now(),
        "ended_at": None,
        "checks": [],
        "artifacts": {},
        "error": None,
    }


def _finish(result: dict[str, Any], status: str) -> dict[str, Any]:
    result["status"] = status
    result["ended_at"] = _now()
    return result


def _check(result: dict[str, Any], name: str, ok: bool, detail: str = "") -> None:
    result["checks"].append(
        {
            "name": name,
            "ok": bool(ok),
            "detail": detail,
        }
    )


def _skip_for_key(result: dict[str, Any], scope: str, strict_keys: bool) -> dict[str, Any] | None:
    if _has_api_key(scope):
        return None
    msg = f"{scope}_API_KEY (or OPENAI_API_KEY fallback) is missing"
    if strict_keys:
        raise RuntimeError(msg)
    result["error"] = msg
    return _finish(result, "skipped")


def test_director(
    out_dir: Path,
    prompt_library: Path,
    storyline: str,
    strict_keys: bool,
) -> tuple[dict[str, Any], Storyboard | None]:
    result = _make_result("DirectorAgent")
    try:
        skipped = _skip_for_key(result, "DIRECTOR", strict_keys)
        if skipped is not None:
            return skipped, None

        agent = DirectorAgent(
            prompt_library=prompt_library,
            strict_online=False,
            retries=1,
        )
        storyboard = agent.build_storyboard(storyline)
        storyboard = agent.ensure_character_images(
            storyboard=storyboard,
            image_dir=out_dir / "director" / "characters",
            image_policy="never",
        )
        storyboard = agent.ensure_scene_images(
            storyboard=storyboard,
            image_dir=out_dir / "director" / "images",
            image_policy="never",
        )

        sb_path = out_dir / "director" / "director_storyboard.json"
        storyboard.save_json(sb_path)
        result["artifacts"]["storyboard_json"] = str(sb_path)

        _check(result, "title_not_empty", bool(storyboard.title), storyboard.title)
        _check(result, "scenes_exist", len(storyboard.scenes) > 0, f"scenes={len(storyboard.scenes)}")
        _check(
            result,
            "characters_dict_exists",
            isinstance(storyboard.screenplay.characters, dict),
            f"characters={len(storyboard.screenplay.characters)}",
        )

        failed = [c for c in result["checks"] if not c["ok"]]
        return _finish(result, "passed" if not failed else "failed"), storyboard
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        return _finish(result, "failed"), None


def test_shot(
    out_dir: Path,
    prompt_library: Path,
    storyboard: Storyboard | None,
    max_shots_per_scene: int,
    strict_keys: bool,
) -> tuple[dict[str, Any], Storyboard | None]:
    result = _make_result("ShotAgent")
    try:
        if storyboard is None:
            raise RuntimeError("Missing storyboard from DirectorAgent")

        skipped = _skip_for_key(result, "SHOT", strict_keys)
        if skipped is not None:
            return skipped, storyboard

        agent = ShotAgent(
            prompt_library=prompt_library,
            strict_online=False,
            retries=1,
        )
        storyboard = agent.loop_storyboard_generation(
            storyboard=storyboard,
            max_shots_per_scene=max_shots_per_scene,
        )

        sb_path = out_dir / "shot" / "shot_storyboard.json"
        sb_path.parent.mkdir(parents=True, exist_ok=True)
        storyboard.save_json(sb_path)
        result["artifacts"]["storyboard_json"] = str(sb_path)

        _check(result, "shots_exist", len(storyboard.shots) > 0, f"shots={len(storyboard.shots)}")
        sequences = [s.sequence for s in storyboard.shots.values()]
        _check(result, "shot_sequence_positive", all(x > 0 for x in sequences), f"sequence_count={len(sequences)}")

        failed = [c for c in result["checks"] if not c["ok"]]
        return _finish(result, "passed" if not failed else "failed"), storyboard
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        return _finish(result, "failed"), storyboard


def test_story_supervisor(
    out_dir: Path,
    storyboard: Storyboard | None,
    strict_keys: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    result = _make_result("StorySupervisorAgent")
    try:
        if storyboard is None:
            raise RuntimeError("Missing storyboard from previous stage")

        skipped = _skip_for_key(result, "SUPERVISOR", strict_keys)
        if skipped is not None:
            return skipped, None

        agent = StorySupervisorAgent(strict_online=False, retries=1)
        sup_dir = out_dir / "story_supervisor"
        plan = agent.generate_story_plan(storyboard=storyboard, output_dir=sup_dir)

        plan_path = sup_dir / "story_supervisor_plan.json"
        result["artifacts"]["plan_json"] = str(plan_path)

        _check(
            result,
            "scene_count_match",
            int(plan.get("scene_count", -1)) == len(storyboard.scenes),
            f"plan={plan.get('scene_count')} storyboard={len(storyboard.scenes)}",
        )
        _check(
            result,
            "scene_plans_exist",
            isinstance(plan.get("scene_plans"), list) and len(plan.get("scene_plans", [])) > 0,
            f"scene_plans={len(plan.get('scene_plans', [])) if isinstance(plan.get('scene_plans'), list) else -1}",
        )
        scene_plans = plan.get("scene_plans", []) if isinstance(plan.get("scene_plans"), list) else []
        shot_plans_count_match = True
        shot_plans_duration_positive = True
        shot_plans_total_match_scene = True
        for sp in scene_plans:
            if not isinstance(sp, dict):
                shot_plans_count_match = False
                shot_plans_duration_positive = False
                shot_plans_total_match_scene = False
                continue
            scene_id = str(sp.get("scene_id", ""))
            expected_count = len(storyboard.get_scene_shots(scene_id))
            shot_plans = sp.get("shot_plans", [])
            if not isinstance(shot_plans, list):
                shot_plans_count_match = False
                shot_plans_duration_positive = False
                shot_plans_total_match_scene = False
                continue
            if len(shot_plans) != expected_count:
                shot_plans_count_match = False

            durations: list[float] = []
            for row in shot_plans:
                if not isinstance(row, dict):
                    shot_plans_duration_positive = False
                    continue
                dur = row.get("duration_sec")
                if not isinstance(dur, (int, float)) or dur <= 0:
                    shot_plans_duration_positive = False
                    continue
                durations.append(float(dur))

            est = sp.get("estimated_scene_duration_sec")
            if isinstance(est, (int, float)) and est > 0 and durations:
                if abs(sum(durations) - float(est)) > 0.6:
                    shot_plans_total_match_scene = False

        _check(result, "shot_plans_count_match_scene_shots", shot_plans_count_match, "per-scene shot_plans count")
        _check(result, "shot_plans_duration_positive", shot_plans_duration_positive, "all shot durations > 0")
        _check(
            result,
            "shot_plans_duration_sum_match_scene",
            shot_plans_total_match_scene,
            "sum(shot_plans.duration_sec) ~= estimated_scene_duration_sec",
        )

        failed = [c for c in result["checks"] if not c["ok"]]
        return _finish(result, "passed" if not failed else "failed"), plan
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        return _finish(result, "failed"), None


def test_video(
    out_dir: Path,
    storyboard: Storyboard | None,
    strict_keys: bool,
) -> tuple[dict[str, Any], Storyboard | None]:
    result = _make_result("VideoAgent")
    try:
        if storyboard is None:
            raise RuntimeError("Missing storyboard from previous stage")

        skipped = _skip_for_key(result, "VIDEO", strict_keys)
        if skipped is not None:
            return skipped, storyboard

        agent = VideoAgent(strict_online=False, retries=1)
        video_dir = out_dir / "video"
        storyboard = agent.process_shots(
            storyboard=storyboard,
            output_dir=video_dir,
            dry_run=True,
            save_task_json=True,
            video_mode="reference",
        )
        sb_path = video_dir / "video_storyboard.json"
        storyboard.save_json(sb_path)

        task_json = video_dir / "video_info.json"
        tasks = _load_json(task_json) if task_json.exists() else []
        result["artifacts"]["storyboard_json"] = str(sb_path)
        result["artifacts"]["task_json"] = str(task_json)

        _check(result, "task_json_exists", task_json.exists(), str(task_json))
        _check(result, "task_count_positive", isinstance(tasks, list) and len(tasks) > 0, f"tasks={len(tasks)}")
        video_paths = [s.video_path for s in storyboard.shots.values() if s.video_path]
        _check(result, "video_paths_assigned", len(video_paths) > 0, f"assigned={len(video_paths)}")

        failed = [c for c in result["checks"] if not c["ok"]]
        return _finish(result, "passed" if not failed else "failed"), storyboard
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        return _finish(result, "failed"), storyboard


def test_post_production(
    out_dir: Path,
    storyboard: Storyboard | None,
    supervision_plan: dict[str, Any] | None,
) -> dict[str, Any]:
    result = _make_result("PostProductionAgent")
    try:
        if storyboard is None:
            raise RuntimeError("Missing storyboard from previous stage")

        agent = PostProductionAgent()
        post_dir = out_dir / "post_production"
        summary = agent.process(
            storyboard=storyboard,
            output_dir=post_dir,
            final_video_path="",
            supervision_plan=supervision_plan,
        )
        info_json = post_dir / "post_production_info.json"
        result["artifacts"]["post_production_info_json"] = str(info_json)

        _check(result, "info_json_exists", info_json.exists(), str(info_json))
        _check(
            result,
            "scene_outputs_positive",
            int(summary.get("scene_count", 0)) > 0,
            f"scene_count={summary.get('scene_count')}",
        )
        master_todo = (
            ((summary.get("master_edit") or {}).get("todo_file", ""))
            if isinstance(summary, dict)
            else ""
        )
        _check(result, "master_todo_exists", bool(master_todo), str(master_todo))

        failed = [c for c in result["checks"] if not c["ok"]]
        return _finish(result, "passed" if not failed else "failed")
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        return _finish(result, "failed")


def test_movie_producer(out_dir: Path, question: str, strict_keys: bool) -> dict[str, Any]:
    result = _make_result("MovieProducerAgent")
    try:
        skipped = _skip_for_key(result, "RAG", strict_keys)
        if skipped is not None:
            return skipped

        agent = MovieProducerAgent(root=ROOT)
        mounted = agent.list_knowledge_bases()
        query_result = agent.query_scene_design(
            question=question,
            k_per_book=1,
            top_k_total=8,
        )

        out_json = out_dir / "movie_producer" / "movie_producer_result.json"
        agent.save_query_result(query_result, out_json)
        result["artifacts"]["query_result_json"] = str(out_json)

        _check(result, "mounted_books_positive", len(mounted) > 0, f"mounted={len(mounted)}")
        _check(
            result,
            "retrieved_docs_positive",
            len(query_result.get("retrieved_docs", [])) > 0,
            f"retrieved={len(query_result.get('retrieved_docs', []))}",
        )
        _check(result, "answer_not_empty", bool(query_result.get("answer", "").strip()), "answer text")

        failed = [c for c in result["checks"] if not c["ok"]]
        return _finish(result, "passed" if not failed else "failed")
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        return _finish(result, "failed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test output contracts for each agent and write a unified report."
    )
    parser.add_argument(
        "--prompt-library",
        type=Path,
        default=ROOT / "prompt_library" / "default",
    )
    parser.add_argument(
        "--storyline",
        type=str,
        default="A photographer with memory loss follows fragmented clues in a rainy city and uncovers a hidden conspiracy.",
    )
    parser.add_argument(
        "--movie-producer-question",
        type=str,
        default="如何把叙事学原则转换为可执行的场景与镜头设计策略？",
    )
    parser.add_argument(
        "--max-shots-per-scene",
        type=int,
        default=2,
        help="Used by ShotAgent test to reduce runtime.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/agent_output_tests/latest"),
    )
    parser.add_argument(
        "--strict-keys",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail test instead of skipping when an agent API key is missing.",
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "started_at": _now(),
        "output_dir": str(out_dir),
        "prompt_library": str(args.prompt_library),
        "strict_keys": bool(args.strict_keys),
        "results": [],
    }

    director_res, storyboard = test_director(
        out_dir=out_dir,
        prompt_library=args.prompt_library,
        storyline=args.storyline,
        strict_keys=bool(args.strict_keys),
    )
    report["results"].append(director_res)

    shot_res, storyboard = test_shot(
        out_dir=out_dir,
        prompt_library=args.prompt_library,
        storyboard=storyboard,
        max_shots_per_scene=int(args.max_shots_per_scene),
        strict_keys=bool(args.strict_keys),
    )
    report["results"].append(shot_res)

    supervisor_res, supervision_plan = test_story_supervisor(
        out_dir=out_dir,
        storyboard=storyboard,
        strict_keys=bool(args.strict_keys),
    )
    report["results"].append(supervisor_res)

    video_res, storyboard = test_video(
        out_dir=out_dir,
        storyboard=storyboard,
        strict_keys=bool(args.strict_keys),
    )
    report["results"].append(video_res)

    post_res = test_post_production(
        out_dir=out_dir,
        storyboard=storyboard,
        supervision_plan=supervision_plan,
    )
    report["results"].append(post_res)

    mp_res = test_movie_producer(
        out_dir=out_dir,
        question=args.movie_producer_question,
        strict_keys=bool(args.strict_keys),
    )
    report["results"].append(mp_res)

    passed = [r for r in report["results"] if r.get("status") == "passed"]
    failed = [r for r in report["results"] if r.get("status") == "failed"]
    skipped = [r for r in report["results"] if r.get("status") == "skipped"]

    report["ended_at"] = _now()
    report["summary"] = {
        "total": len(report["results"]),
        "passed": len(passed),
        "failed": len(failed),
        "skipped": len(skipped),
    }

    report_path = out_dir / "agent_tests_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=== Agent Output Tests Completed ===")
    print(f"Output dir: {out_dir}")
    print(f"Report: {report_path}")
    print(
        "Summary: "
        f"total={report['summary']['total']}, "
        f"passed={report['summary']['passed']}, "
        f"failed={report['summary']['failed']}, "
        f"skipped={report['summary']['skipped']}"
    )
    for r in report["results"]:
        print(f"- {r['agent']}: {r['status']}")


if __name__ == "__main__":
    main()
