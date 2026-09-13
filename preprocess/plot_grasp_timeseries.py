"""Plot windowed grasp-signal changes and binary states per episode.

The script reads the EEF artifacts produced by ``batch_preprocess.py``.  It can
process one experiment or several experiments in one invocation; by default it
targets the stack-cola and stack-cube correction-ablation experiments.  The
plotted change signal is ``median(x[t:t+w]) - median(x[t-w:t])``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENTS = (
    REPO_ROOT / "outputs/experiments/stack_cola_correction_ablation_v1",
    REPO_ROOT / "outputs/experiments/stack_cube_correction_ablation_v1",
)
SIDES = (("left", "hand_l", "tab:blue"), ("right", "hand_r", "tab:orange"))


def _episode_number(path: Path) -> int:
    try:
        return int(path.parent.name.removeprefix("episode_"))
    except ValueError as error:
        raise ValueError(f"Cannot parse episode number from {path}") from error


def _parse_episode_spec(spec: str | None) -> set[int] | None:
    """Parse comma-separated episode numbers/ranges, e.g. ``0,3,8-11``."""
    if spec is None or spec.strip().lower() in {"", "all"}:
        return None
    selected: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" not in token:
            selected.add(int(token))
            continue
        first_text, last_text = token.split("-", 1)
        first, last = int(first_text), int(last_text)
        if last < first:
            raise ValueError(f"Invalid descending episode range: {token!r}")
        selected.update(range(first, last + 1))
    return selected


def _discover_variant(experiment: Path, requested: str | None) -> str:
    eef_root = experiment / "artifacts/eef"
    if requested:
        if not (eef_root / requested).is_dir():
            raise FileNotFoundError(
                f"Hand2gripper variant {requested!r} is absent under {eef_root}"
            )
        return requested
    variants = sorted(path.name for path in eef_root.iterdir() if path.is_dir())
    if len(variants) != 1:
        raise ValueError(
            f"Expected one hand2gripper variant under {eef_root}, found {variants}; "
            "select one with --variant"
        )
    return variants[0]


def _discover_eef_files(
    experiment: Path, variant: str, selected: set[int] | None
) -> list[Path]:
    files = list((experiment / "artifacts/eef" / variant).glob("episode_*/eef_raw.json"))
    files.sort(key=_episode_number)
    if selected is not None:
        files = [path for path in files if _episode_number(path) in selected]
        found = {_episode_number(path) for path in files}
        missing = sorted(selected - found)
        if missing:
            print(f"[warning] {experiment.name}: missing episodes {missing}")
    return files


def _time_seconds(frames: list[dict[str, Any]], fps: float) -> np.ndarray:
    # EEF schema v2 stores ``ts`` in nanoseconds.  Fall back to frame/fps for
    # older artifacts and normalize the first sample to t=0.
    timestamps = np.asarray([frame.get("ts", np.nan) for frame in frames], dtype=float)
    if timestamps.size and np.all(np.isfinite(timestamps)):
        timestamps = (timestamps - timestamps[0]) * 1e-9
        if len(timestamps) < 2 or timestamps[-1] > 0.0:
            return timestamps
    return np.arange(len(frames), dtype=float) / fps


def _side_series(
    frames: Iterable[dict[str, Any]], hand_key: str
) -> tuple[np.ndarray, np.ndarray]:
    ratio: list[float] = []
    state: list[float] = []
    for frame in frames:
        hand = frame.get(hand_key)
        if hand is None:
            ratio.append(np.nan)
            state.append(np.nan)
            continue
        raw_ratio = hand.get("grasp_ratio")
        raw_state = hand.get("grasp_state")
        ratio.append(np.nan if raw_ratio is None else float(raw_ratio))
        state.append(np.nan if raw_state is None else float(raw_state))
    return np.asarray(ratio), np.asarray(state)


def _windowed_median_change(signal: np.ndarray, window_frames: int) -> np.ndarray:
    """Return median(future window) - median(past window) at each frame."""
    change = np.full(len(signal), np.nan, dtype=float)
    for index in range(window_frames, len(signal) - window_frames + 1):
        past = signal[index - window_frames : index]
        future = signal[index : index + window_frames]
        past = past[np.isfinite(past)]
        future = future[np.isfinite(future)]
        if len(past) and len(future):
            change[index] = float(np.median(future) - np.median(past))
    return change


def _shade_missing(ax: plt.Axes, time_s: np.ndarray, valid: np.ndarray) -> None:
    if len(time_s) == 0 or np.all(valid):
        return
    ax.fill_between(
        time_s,
        0,
        1,
        where=~valid,
        transform=ax.get_xaxis_transform(),
        color="0.75",
        alpha=0.35,
        linewidth=0,
        label="missing hand detection",
    )


def plot_episode(
    eef_path: Path, output: Path, *, dpi: int, window_frames: int
) -> None:
    payload = json.loads(eef_path.read_text(encoding="utf-8"))
    frames = payload.get("frames") or []
    if not frames:
        raise ValueError(f"No frames in {eef_path}")

    fps = float(payload.get("fps") or 30.0)
    time_s = _time_seconds(frames, fps)
    hysteresis = (payload.get("hand2gripper") or {}).get("grasp_hysteresis") or {}
    signal = hysteresis.get(
        "signal", "thumb-index tip distance / wrist-middle-MCP palm size"
    )

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(14, 9),
        sharex=True,
        gridspec_kw={"height_ratios": (3.0, 1.15, 3.0, 1.15), "hspace": 0.10},
    )
    episode = _episode_number(eef_path)
    fig.suptitle(
        f"{eef_path.parents[4].name} · episode {episode:06d} · grasp change diagnostics",
        fontsize=14,
    )

    for side_index, (side, hand_key, color) in enumerate(SIDES):
        change_ax, state_ax = axes[2 * side_index : 2 * side_index + 2]
        ratio, state = _side_series(frames, hand_key)
        change = _windowed_median_change(ratio, window_frames)
        valid = np.isfinite(ratio) & np.isfinite(state)
        valid_states = state[valid]
        transitions = int(np.count_nonzero(np.diff(valid_states)))
        coverage = 100.0 * float(np.count_nonzero(valid)) / len(valid)

        change_ax.plot(
            time_s,
            change,
            color=color,
            linewidth=1.15,
            label="windowed median change",
        )
        change_ax.axhline(
            0.0,
            color="0.25",
            linestyle="--",
            linewidth=1.0,
            label="zero change",
        )
        _shade_missing(change_ax, time_s, valid)
        change_ax.set_ylabel(f"{side.title()} change")
        change_ax.grid(True, alpha=0.25)
        change_ax.legend(loc="upper right", ncol=3, fontsize=8)
        change_ax.text(
            0.01,
            0.96,
            f"coverage {coverage:.1f}%  |  transitions {transitions}",
            transform=change_ax.transAxes,
            va="top",
            fontsize=8,
            color="0.25",
        )
        if not np.any(valid):
            change_ax.text(
                0.5,
                0.5,
                "NO VALID HAND DETECTIONS",
                transform=change_ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
                weight="bold",
                color="0.35",
            )

        state_ax.step(
            time_s,
            state,
            where="post",
            color=color,
            linewidth=1.35,
            label="final binary state",
        )
        state_ax.fill_between(
            time_s,
            0,
            np.nan_to_num(state, nan=0.0),
            where=np.isfinite(state),
            step="post",
            color=color,
            alpha=0.18,
        )
        _shade_missing(state_ax, time_s, valid)
        state_ax.set_yticks((0, 1), labels=("OPEN", "CLOSED"))
        state_ax.set_ylim(-0.16, 1.16)
        state_ax.set_ylabel("decision")
        state_ax.grid(True, axis="x", alpha=0.25)

    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_xlim(float(time_s[0]), float(time_s[-1]))
    fig.text(
        0.5,
        0.015,
        "Change: median(x[t:t+w]) - median(x[t-w:t])  |  "
        f"w={window_frames} frames ({window_frames / fps:.2f} s)  |  x: {signal}",
        ha="center",
        fontsize=9,
        color="0.3",
    )
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.10, top=0.90)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiments",
        nargs="*",
        type=Path,
        help="Experiment roots (default: stack cola and stack cube correction ablations)",
    )
    parser.add_argument(
        "--episodes",
        help="Comma-separated episode IDs/ranges, e.g. 0,3,8-11 (default: all)",
    )
    parser.add_argument(
        "--variant",
        help="Hand2gripper artifact variant (auto-detected when there is only one)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Common output root; default is "
            "<experiment>/artifacts/grasp_change_timeseries/<variant>"
        ),
    )
    parser.add_argument(
        "--window-frames",
        type=int,
        default=15,
        help="Past/future median window length in frames (default: 15)",
    )
    parser.add_argument("--format", choices=("png", "pdf"), default="png")
    parser.add_argument("--dpi", type=int, default=160)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    if args.window_frames <= 0:
        raise ValueError("--window-frames must be positive")
    experiments = args.experiments or list(DEFAULT_EXPERIMENTS)
    selected = _parse_episode_spec(args.episodes)
    total = 0
    for raw_experiment in experiments:
        experiment = raw_experiment.expanduser().resolve()
        if not experiment.is_dir():
            raise FileNotFoundError(f"Experiment directory does not exist: {experiment}")
        variant = _discover_variant(experiment, args.variant)
        eef_files = _discover_eef_files(experiment, variant, selected)
        if not eef_files:
            print(f"[warning] {experiment.name}: no matching EEF artifacts")
            continue
        output_dir = (
            args.output_root.expanduser().resolve() / experiment.name / variant
            if args.output_root
            else experiment / "artifacts/grasp_change_timeseries" / variant
        )
        for eef_path in eef_files:
            episode = _episode_number(eef_path)
            output = output_dir / f"episode_{episode:06d}.{args.format}"
            plot_episode(
                eef_path,
                output,
                dpi=args.dpi,
                window_frames=args.window_frames,
            )
            print(f"[output] {output}")
            total += 1
    print(f"Generated {total} grasp diagnostic figure(s).")


if __name__ == "__main__":
    main()
