#!/usr/bin/env python3
"""Run the repository evaluator for the five requested correction experiments.

Does not alter evaluator code, datasets, or repository splits. Unhealthy held-out
real episodes are documented in an output-local split. Optional Numba compilation
preserves the original DTW recurrence and validates paths against its Python code.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

TASKS = {
    "cube": ("stack_redcube_v1", "stack_redcube_v1_1"),
    "hanging": ("hanging", "hanging_1"),
    "bowl": ("bowl", "bowl_1"),
    "cola": ("cola", "cola_1012"),
    "drawer": ("drawer", "drawer_1012"),
}
VARIANTS = ["finger_center_hys085__none", "finger_center_hys085__position_rotation"]
METRICS = ["rho_pos", "rho_rot", "rho_se3", "D_pos_mm", "D_rot_deg"]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def compile_dtw():
    import numpy as np
    from numba import njit
    import evaluation.align as align
    original = align._dtw_path
    source = inspect.getsource(original)
    # np.argmin accepts a tuple in NumPy, but Numba requires an array.
    compiled_source = source.replace("np.argmin(candidates)", "np.argmin(np.asarray(candidates))")
    scope = dict(original.__globals__)
    exec(compile(compiled_source, "<validated_dtw>", "exec"), scope)
    compiled = njit(scope["_dtw_path"])
    rng = np.random.default_rng(20261012)
    cases = 0
    for n, m in [(2, 2), (5, 9), (31, 27), (200, 200)]:
        for cost in [rng.random((n, m)), np.zeros((n, m)), np.ones((n, m))]:
            for window in [None, max(abs(n-m), 3)]:
                expected = original(cost, window)
                actual = compiled(cost, window)
                assert all(np.array_equal(a, b) for a, b in zip(expected, actual))
                cases += 1
    align._dtw_path = compiled
    return {"compiled": True, "exact_path_and_cost_cases": cases,
            "change": "np.argmin(tuple) -> np.argmin(np.asarray(tuple)); no fastmath",
            "source": source}


def run_task(args):
    import yaml
    from evaluation import run_eval as ev
    from evaluation.loaders import load_real_fk_json, load_real_manifest
    from evaluation.schemas import sha256_file

    name = args.task
    task, default_experiment = TASKS[name]
    experiment = args.experiment or default_experiment
    out = args.output / name
    out.mkdir(parents=True, exist_ok=True)
    dtw = compile_dtw()
    write_json(out / "dtw_validation.json", dtw)
    if args.skip_dispersion:
        # eta_disp is independent of the five requested metrics and otherwise
        # requires O(N_ego^2) extra alignments. Keep the repository aggregation.
        def omitted_dispersion(ego_cache, active_sides, config, noise_floors):
            return {side: {"available": False, "value": None,
                    "reason": "Not requested; skipped with --skip-dispersion"}
                    for side in active_sides}
        ev._compute_eta_dispersion = omitted_dispersion
    manifest = args.root / "outputs/experiments" / experiment / "manifest.json"
    real_manifest = args.root / "outputs/arx_real_flange" / (task + "_arx") / "manifest.json"
    config_path = args.root / "cfg/evaluation" / (task + "_arx.yaml")
    split_path = args.root / "cfg/evaluation/splits" / (task + ".json")
    config = ev._merge(ev.DEFAULT_CONFIG, yaml.safe_load(config_path.read_text()))
    split = json.loads(split_path.read_text())
    experiment_data = json.loads(manifest.read_text())
    real_data, outputs = load_real_manifest(real_manifest)
    ev._validate_split(split, outputs, config, experiment_data, real_data["_fingerprint"])
    # Check BOTH fitted artifacts, including the separate rotation artifact.
    artifact_audit = []
    for key in ["artifact", "rotation_artifact"]:
        p = Path(experiment_data["variants"][VARIANTS[1]]["correction_config"][key])
        ids = ev._verify_no_leakage(json.loads(p.read_text()), [str(i) for i in split["eval"]], str(p))
        artifact_audit.append({"kind": key, "path": str(p), "sha256": sha256_file(p), "calibration_ids": ids})
    exclusions, healthy = [], []
    for eid in split["eval"]:
        traj = load_real_fk_json(outputs[str(eid)], episode_id=str(eid))
        bad = []
        for side in config["active_sides"]:
            reason, health = ev._health_reason(traj, side, config)
            if reason:
                bad.append({"side": side, "reason": reason, "health": health})
        if bad:
            exclusions.append({"real_episode_id": str(eid), "failures": bad})
        else:
            healthy.append(eid)
    assert len(healthy) >= 2
    split["eval"] = healthy
    if exclusions:
        split["health_exclusions"] = exclusions
        split["source_split"] = str(split_path)
        effective_split = out / "real_split_healthy.json"
        write_json(effective_split, split)
    else:
        effective_split = split_path
    write_json(out / "preflight.json", {
        "task": name, "experiment": experiment, "source_split": str(split_path),
        "source_split_sha256": sha256_file(split_path), "effective_split": str(effective_split),
        "real_eval_ids": healthy, "real_health_exclusions": exclusions,
        "skip_unrequested_eta_disp": args.skip_dispersion,
        "correction_artifacts": artifact_audit,
        "evaluator_source_sha256": {str(p.relative_to(args.root)): sha256_file(p)
            for p in sorted((args.root / "evaluation").rglob("*.py"))},
    })
    print(name, "start", len(healthy), "held-out real episodes", flush=True)
    result = ev.run(argparse.Namespace(
        experiment_manifest=str(manifest), real_manifest=str(real_manifest),
        real_split=str(effective_split), eval_config=str(config_path),
        out=str(out / "evaluation"), variants=VARIANTS, episodes=None,
        sides=None, segment=None, pairing=None, limit=None, dry_run=False,
        resume=True, force=False))
    comparison = json.loads((out / "evaluation/comparison.json").read_text())
    assert comparison["comparable"], f"{name}: evaluated cohorts differ across variants"
    for s in result["summaries"]:
        for side in config["active_sides"]:
            assert s["metrics"]["per_side"][side]["n_ego_episodes"] > 0
    print(name, "complete", flush=True)


def collect(args):
    rows, per_side_rows, counts = [], [], {}
    for name in TASKS:
        root = args.output / name
        preflight = json.loads((root / "preflight.json").read_text())
        counts[name] = {"real_eval": len(preflight["real_eval_ids"]), "ego_per_side": {}}
        for label, variant in zip(["before", "after"], VARIANTS):
            s = json.loads((root / "evaluation" / variant / "summary.json").read_text())
            row = {"task": name, "variant": label, **{k: s["metrics"]["bimanual_active_side_mean"][k] for k in METRICS}}
            rows.append(row)
            for side, block in s["metrics"]["per_side"].items():
                per_side_rows.append({"task": name, "variant": label, "side": side,
                    "n_ego": block["n_ego_episodes"], "n_real": counts[name]["real_eval"],
                    **{k: block["aggregate"][k]["mean"] for k in METRICS}})
                counts[name]["ego_per_side"].setdefault(label, {})[side] = block["n_ego_episodes"]
    for filename, values in [("metrics.csv", rows), ("metrics_per_arm.csv", per_side_rows)]:
        with (args.output / filename).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(values[0]))
            w.writeheader(); w.writerows(values)
    write_json(args.output / "metrics.json", {"task_metrics": rows, "per_arm": per_side_rows, "counts": counts})
    print(json.dumps({"metrics": rows, "counts": counts}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/mnt/data/xule/TATE"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=list(TASKS))
    parser.add_argument("--experiment",
                        help="Override the experiment directory for a single --task run")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--skip-dispersion", action="store_true",
                        help="Skip only the independent, unrequested eta_disp metric")
    args = parser.parse_args()
    args.root = args.root.resolve(); args.output = args.output.resolve()
    if args.experiment and not args.task:
        parser.error("--experiment requires --task")
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.root))
    if args.task:
        run_task(args)
        return
    def worker(name):
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        with (args.output / f"{name}.log").open("w") as log:
            command = [sys.executable, str(Path(__file__).resolve()), "--root", str(args.root),
                       "--output", str(args.output), "--task", name]
            if args.skip_dispersion:
                command.append("--skip-dispersion")
            p = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        print(name, "exit", p.returncode, flush=True)
        return p.returncode
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        codes = list(pool.map(worker, TASKS))
    if any(codes):
        raise RuntimeError("One or more tasks failed; inspect task logs")
    collect(args)


if __name__ == "__main__":
    main()
