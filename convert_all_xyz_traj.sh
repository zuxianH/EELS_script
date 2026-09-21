#!/usr/bin/env bash

# Convert every XYZ/extended-XYZ trajectory in a directory to an ASE .traj file.

set -u

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
converter="$script_dir/convert_xyz_traj.py"

jobs=1
recursive=false
overwrite=false
dry_run=false
directory=""

usage() {
    cat <<'EOF'
Usage: convert_all_xyz_traj.sh [OPTIONS] [DIRECTORY]

Convert all *.xyz and *.extxyz files in DIRECTORY. If DIRECTORY is omitted,
the current directory is used. Output files are placed beside their inputs.

Options:
  -j, --jobs N        Run N conversions in parallel (default: 1)
  -r, --recursive     Also scan subdirectories
      --overwrite     Replace existing .traj files
      --dry-run       Show what would be converted without writing files
  -h, --help          Show this help

Examples:
  cd /path/to/trajectory/folder
  /path/to/convert_all_xyz_traj.sh

  /path/to/convert_all_xyz_traj.sh -j 4 /path/to/trajectory/folder
EOF
}

while (($#)); do
    case "$1" in
        -j|--jobs)
            if (($# < 2)); then
                echo "error: $1 requires a positive integer" >&2
                exit 2
            fi
            jobs=$2
            shift 2
            ;;
        -r|--recursive)
            recursive=true
            shift
            ;;
        --overwrite)
            overwrite=true
            shift
            ;;
        --dry-run)
            dry_run=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            if (($# > 1)); then
                echo "error: only one directory may be specified" >&2
                exit 2
            fi
            if (($# == 1)); then
                directory=$1
            fi
            break
            ;;
        -*)
            echo "error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            if [[ -n $directory ]]; then
                echo "error: only one directory may be specified" >&2
                exit 2
            fi
            directory=$1
            shift
            ;;
    esac
done

if [[ ! $jobs =~ ^[1-9][0-9]*$ ]]; then
    echo "error: --jobs must be a positive integer" >&2
    exit 2
fi

if [[ -z $directory ]]; then
    directory=.
fi
if [[ ! -d $directory ]]; then
    echo "error: directory does not exist: $directory" >&2
    exit 1
fi
if [[ ! -f $converter ]]; then
    echo "error: converter not found: $converter" >&2
    exit 1
fi

directory=$(cd -- "$directory" && pwd -P)
find_args=("$directory")
if [[ $recursive == false ]]; then
    find_args+=(-maxdepth 1)
fi
find_args+=(-type f \( -name '*.xyz' -o -name '*.extxyz' \) -print0)

inputs=()
while IFS= read -r -d '' input; do
    inputs+=("$input")
done < <(find "${find_args[@]}" | sort -z)

if ((${#inputs[@]} == 0)); then
    echo "No .xyz or .extxyz files found in $directory"
    exit 0
fi

pending=()
skipped=0
for input in "${inputs[@]}"; do
    output="${input%.*}.traj"
    if [[ -e $output && $overwrite == false ]]; then
        echo "Skipping existing output: $output"
        ((skipped += 1))
    else
        pending+=("$input")
    fi
done

echo "Found ${#inputs[@]} input(s): ${#pending[@]} to convert, $skipped skipped."

if [[ $dry_run == true ]]; then
    for input in "${pending[@]}"; do
        echo "Would convert: $input -> ${input%.*}.traj"
    done
    exit 0
fi

if ((${#pending[@]} == 0)); then
    exit 0
fi

run_conversion() {
    local input=$1
    local command=(python3 "$converter" "$input")
    if [[ $overwrite == true ]]; then
        command+=(--overwrite)
    fi
    echo "Converting: $input"
    "${command[@]}"
}

failures=0
pids=()
names=()

wait_for_batch() {
    local index
    for index in "${!pids[@]}"; do
        if ! wait "${pids[$index]}"; then
            echo "FAILED: ${names[$index]}" >&2
            ((failures += 1))
        fi
    done
    pids=()
    names=()
}

for input in "${pending[@]}"; do
    run_conversion "$input" &
    pids+=("$!")
    names+=("$input")
    if ((${#pids[@]} >= jobs)); then
        wait_for_batch
    fi
done
if ((${#pids[@]})); then
    wait_for_batch
fi

if ((failures)); then
    echo "Finished with $failures failed conversion(s)." >&2
    exit 1
fi

echo "Finished: converted ${#pending[@]} file(s) successfully."
