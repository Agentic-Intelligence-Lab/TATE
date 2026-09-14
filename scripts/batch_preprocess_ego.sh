#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-/home/ymq/miniconda3/envs/lifego/bin/python}"
CFG_PATH="${REPO_ROOT}/cfg/preprocess/base/Preprocess.yaml"
TASK="fold the paper boxes."
SKIP_EXISTING=0
CONTINUE_ON_ERROR=0
MAX_FILES=0
POSITIONAL=()

usage() {
    printf 'Usage: %s [options] INPUT_VIDEO_DIR OUTPUT_DIR\n' "$0"
    printf '\nOptions:\n'
    printf '  --python PATH             Python interpreter (default: %s)\n' "$PYTHON_BIN"
    printf '  --cfg-path PATH          Preprocess config path\n'
    printf '  --task TEXT              Task text (default: %s)\n' "$TASK"
    printf '  --skip-existing          Skip episodes with both expected outputs\n'
    printf '  --continue-on-error      Continue after a failed episode\n'
    printf '  --max-files N            Process at most N videos; 0 means all\n'
    printf '  -h, --help               Show this help\n'
}

while (($# > 0)); do
    case "$1" in
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --cfg-path)
            CFG_PATH="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --skip-existing)
            SKIP_EXISTING=1
            shift
            ;;
        --continue-on-error)
            CONTINUE_ON_ERROR=1
            shift
            ;;
        --max-files)
            MAX_FILES="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            while (($# > 0)); do
                POSITIONAL+=("$1")
                shift
            done
            ;;
        -*)
            printf 'Unknown option: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

if ((${#POSITIONAL[@]} != 2)); then
    usage >&2
    exit 2
fi

INPUT_DIR="$(realpath "${POSITIONAL[0]}")"
OUTPUT_DIR="$(realpath -m "${POSITIONAL[1]}")"

if [[ ! -d "$INPUT_DIR" ]]; then
    printf 'Input directory does not exist: %s\n' "$INPUT_DIR" >&2
    exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    printf 'Python interpreter is not executable: %s\n' "$PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -f "$CFG_PATH" ]]; then
    printf 'Preprocess config does not exist: %s\n' "$CFG_PATH" >&2
    exit 1
fi
if ! [[ "$MAX_FILES" =~ ^[0-9]+$ ]]; then
    printf '--max-files must be a non-negative integer\n' >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"

mapfile -t VIDEO_PATHS < <(find "$INPUT_DIR" -type f \( -iname '*.mp4' -o -iname '*.mov' -o -iname '*.avi' -o -iname '*.mkv' \) | sort)
if ((${#VIDEO_PATHS[@]} == 0)); then
    printf 'No videos found under %s\n' "$INPUT_DIR" >&2
    exit 1
fi

processed=0
failed=0
for video_path in "${VIDEO_PATHS[@]}"; do
    if ((MAX_FILES > 0 && processed >= MAX_FILES)); then
        break
    fi

    relative_path="${video_path#"$INPUT_DIR"/}"
    episode_rel="${relative_path%.*}"
    episode_dir="${OUTPUT_DIR}/${episode_rel}"
    eef_path="${episode_dir}/preprocess/eef.json"
    vis_path="${episode_dir}/preprocess/hand_keypoints_eef_vis.mp4"

    if ((SKIP_EXISTING == 1)) && [[ -f "$eef_path" && -f "$vis_path" ]]; then
        printf '[skip] %s\n' "$relative_path"
        processed=$((processed + 1))
        continue
    fi

    mkdir -p "$episode_dir"
    printf '[run] %s\n' "$relative_path"
    if "$PYTHON_BIN" -m preprocess.Preprocess \
        --mps_path "$episode_dir" \
        --video_path "$video_path" \
        --cfg_path "$CFG_PATH" \
        --task "$TASK" \
        --no-gif; then
        processed=$((processed + 1))
    else
        failed=$((failed + 1))
        printf '[failed] %s\n' "$relative_path" >&2
        if ((CONTINUE_ON_ERROR == 0)); then
            exit 1
        fi
    fi
done

printf 'Processed: %d, failed: %d, output: %s\n' "$processed" "$failed" "$OUTPUT_DIR"
exit 0
