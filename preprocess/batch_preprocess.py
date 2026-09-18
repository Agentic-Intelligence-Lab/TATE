"""Batch-process LeRobot ego episodes into reusable WiLoR and EEF variants."""

from __future__ import annotations

import argparse
from pathlib import Path

from preprocess.batch.config import load_experiment_config
from preprocess.batch.dataset import dataset_fingerprint, discover_episodes
from preprocess.batch.runner import BatchRunner, STAGE_ORDER as PIPELINE_STAGE_ORDER


STAGE_ORDER = (*PIPELINE_STAGE_ORDER[:-1], "visualize", PIPELINE_STAGE_ORDER[-1])


def parse_ids(value: str | None) -> set[int] | None:
    if not value:
        return None
    result: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token or "-" in token:
            separator = ":" if ":" in token else "-"
            start_text, end_text = token.split(separator, 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"invalid episode range: {token}")
            result.update(range(start, end + 1))
        else:
            result.add(int(token))
    return result


def parse_names(value: str | None) -> set[str] | None:
    return None if not value else {token.strip() for token in value.split(",") if token.strip()}


def parse_stages(value: str) -> set[str]:
    if value == "all":
        return set(STAGE_ORDER)
    stages = parse_names(value) or set()
    unknown = stages.difference(STAGE_ORDER)
    if unknown:
        raise ValueError(f"unknown stages {sorted(unknown)}; choose from {list(STAGE_ORDER)}")
    return stages


def print_plan(config, episodes, runs, dataset_fp, stages) -> None:
    print("TATE batch preprocess plan")
    print(f"  experiment : {config['experiment_id']}")
    print(f"  source     : {config['source']['ego_dataset']}")
    print(f"  dataset id : {dataset_fp[:16]}")
    print(f"  video key  : {config['source']['video_key']}")
    print(
        f"  arm mode   : {config['trajectory']['arm_mode']} "
        f"({', '.join(config['trajectory']['active_sides'])})"
    )
    trim = config["trajectory"]["trim"]
    print(
        f"  trim       : start={trim['start_seconds']:.3f}s "
        f"end={trim['end_seconds']:.3f}s"
    )
    print(f"  episodes   : {len(episodes)}")
    print(f"  stages     : {', '.join(stage for stage in STAGE_ORDER if stage in stages)}")
    print(f"  variants   : {', '.join(run['id'] for run in runs)}")
    for episode in episodes:
        print(
            f"    episode_{episode.episode_index:06d}: "
            f"frames={episode.length}->{episode.effective_length} "
            f"crop={episode.crop_start_frames}/{episode.crop_end_frames} "
            f"video={episode.video_path} start={episode.processing_start_frame}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Batch experiment YAML")
    parser.add_argument(
        "--stages",
        default="all",
        help="all or comma-separated wilor,eef,correct,retarget,visualize,package",
    )
    parser.add_argument("--episodes", default=None, help="IDs/ranges, e.g. 0,2,5:9")
    parser.add_argument("--variants", default=None, help="Comma-separated run IDs")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force-stage", default=None, help="Comma-separated stages whose matching outputs are regenerated"
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--override-experiment",
        "--override",
        dest="override_experiment",
        action="store_true",
        help=(
            "replace an existing experiment only when its configuration or source "
            "dataset fingerprint differs; shared WiLoR caches are preserved"
        ),
    )
    parser.add_argument("--visualization-axis-length", type=float, default=0.06)
    parser.add_argument("--visualization-ratio-plot-max", type=float, default=1.5)
    parser.add_argument("--visualization-max-frames", type=int, default=None)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    stages = parse_stages(args.stages)
    force_stages = parse_stages(args.force_stage) if args.force_stage else set()
    selected_ids = parse_ids(args.episodes)
    selected_variants = parse_names(args.variants)
    all_run_ids = {run["id"] for run in config["runs"]}
    if selected_variants:
        unknown = selected_variants.difference(all_run_ids)
        if unknown:
            raise ValueError(f"unknown variants: {sorted(unknown)}")
    runs = [
        run for run in config["runs"]
        if selected_variants is None or run["id"] in selected_variants
    ]

    all_episodes = discover_episodes(
        config["source"]["ego_dataset"], config["source"]["video_key"]
    )
    source_fp = dataset_fingerprint(config["source"]["ego_dataset"], all_episodes)
    trim = config["trajectory"]["trim"]
    all_episodes = [
        episode.with_trim_seconds(trim["start_seconds"], trim["end_seconds"])
        for episode in all_episodes
    ]
    episodes = [
        episode for episode in all_episodes
        if selected_ids is None or episode.episode_index in selected_ids
    ]
    if selected_ids is not None:
        missing = selected_ids.difference(episode.episode_index for episode in episodes)
        if missing:
            raise ValueError(f"requested source episodes do not exist: {sorted(missing)}")
    if args.limit > 0:
        episodes = episodes[: args.limit]
    if not episodes:
        raise ValueError("episode selection is empty")

    print_plan(config, episodes, runs, source_fp, stages)
    if args.dry_run:
        return

    runner = BatchRunner(
        config=config,
        episodes=episodes,
        source_dataset_fingerprint=source_fp,
        source_episode_count=len(all_episodes),
        selected_run_ids=selected_variants,
        force_stages=force_stages,
        fail_fast=args.fail_fast,
        override_experiment=args.override_experiment,
    )
    runner.run_episode_stages(stages)
    if "visualize" in stages:
        from preprocess.batch.visualization import visualize_hand2gripper_variants

        visualize_hand2gripper_variants(
            runner,
            axis_length=args.visualization_axis_length,
            ratio_plot_max=args.visualization_ratio_plot_max,
            max_frames=args.visualization_max_frames,
        )
    if "package" in stages and config["lerobot"]["enabled"]:
        from preprocess.batch.lerobot import package_variants

        package_variants(runner, force="package" in force_stages)
    runner.write_summary_tsv()
    print(f"Done. Manifest: {runner.experiment_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
