#!/bin/bash
#
# 批量处理 InSAR 数据集：
#   1) process_offset   : 生成/处理 offset
#   2) process_dataset  : lltenude -> grd -> 归档
#

set -euo pipefail

# ============================================================
# Configuration
# ============================================================

BASE_DIR="$HOME/myanmar"
OUTPUT_DIR="$HOME/py_inv/input_data/azi_data"

MAT_SCRIPT="$BASE_DIR/mat_lltenude.py"
ORIGIN_SCRIPT="$BASE_DIR/origin_lltenu.sh"
Collect_SCRIPT="$BASE_DIR/collect_sub.sh"
Merge_SCRIPT="$BASE_DIR/merge_sub.sh"

# 需要归档的文件模式 / 固定文件
MOVE_GLOBS=("look_*.grd")
MOVE_FILES=("los_clean_detrend.grd" "dem_low.grd")

# 数据集列表: "名称|样本目录"
DATASETS=(
    "D106|$BASE_DIR/D106/no_esd/sample_D106"
    "D33|$BASE_DIR/D33/newframe/sample_D33"
    "A143|$BASE_DIR/A143/sample_A143"
    "A70|$BASE_DIR/A70/sample_A70"
)

BASESETS=(
    "D106|$BASE_DIR/D106/no_esd"
    "D33|$BASE_DIR/D33/newframe"
    "A143|$BASE_DIR/A143"
    "A70|$BASE_DIR/A70"
)

declare -A SUBS=(
    [D106]="F1,F2"
    [D33]="F2,F3"
    [A143]="F2,F3"
    [A70]="F1,F2"
)


# ============================================================
# Helpers
# ============================================================

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
die() { log "ERROR: $*" >&2; exit 1; }

require_file() {
    [[ -f "$1" ]] || die "Required file not found: $1"
}

# ============================================================
# Process 1: Offset
# ============================================================

process_offset() {
    local name="$1"
    local sample_dir="$2"
    local sub="${SUBS[$name]}"

    echo
    echo "------------------------------------------------------------"
    log "[Offset] $name"
    echo "  sample_dir: $sample_dir"
    echo "  sub: $sub"
    echo "------------------------------------------------------------"

    [[ -d "$sample_dir" ]] || die "Sample directory not found: $sample_dir"

    (
        cd "$sample_dir" || die "Cannot cd to $sample_dir"
	pwd
	OFFSET_SCRIPT="$sample_dir/make_offset.sh"
	require_file $OFFSET_SCRIPT
        "$OFFSET_SCRIPT" filter 0 
	"$Collect_SCRIPT"  aa  "$sub"
	cd "collected_offsets"
	"$Merge_SCRIPT" "$sub" azi_offset_mask.grd
    )

    log "[Offset] Finished: $name"
}

# ============================================================
# Process 2: lltenude -> grd -> 归档
# ============================================================

process_dataset() {
    local name="$1"
    local sample_dir="$2"
    local output_dir="$OUTPUT_DIR/$name"

    echo
    echo "============================================================"
    log "Processing: $name"
    echo "  sample_dir: $sample_dir"
    echo "  output_dir: $output_dir"
    echo "============================================================"

    [[ -d "$sample_dir" ]] || die "Sample directory not found: $sample_dir"

    mkdir -p "$output_dir"

    (
        cd "$sample_dir" || die "Cannot cd to $sample_dir"

        # ----------------------------------------------------
        # 1. lltenude -> mat  (按需启用)
        # ----------------------------------------------------
        # log "[1/4] Creating los_samp0.mat ..."
        # python "$MAT_SCRIPT" test.lltenude los_samp0.mat
        # cp los_samp0.mat "$output_dir/"

        # ----------------------------------------------------
        # 2. 生成 merged.grd
        # ----------------------------------------------------
        log "[2/4] Creating merged.grd ..."
        "$ORIGIN_SCRIPT" .

        # ----------------------------------------------------
        # 3. 归档输出文件
        # ----------------------------------------------------
        log "[3/4] Moving output files ..."

        shopt -s nullglob

        local pattern files=()
        for pattern in "${MOVE_GLOBS[@]}"; do
            files=( $pattern )
            if (( ${#files[@]} > 0 )); then
                mv -- "${files[@]}" "$output_dir/"
                log "  Moved ${#files[@]} file(s) matching '$pattern'"
            else
                log "  Warning: no file matching '$pattern'"
            fi
        done

        local f
        for f in "${MOVE_FILES[@]}"; do
            if [[ -f "$f" ]]; then
                mv -- "$f" "$output_dir/"
                log "  Moved: $f"
            else
                log "  Warning: $f not found"
            fi
        done

        shopt -u nullglob
    )

    log "[4/4] Finished: $name"
}

# ============================================================
# Main
# ============================================================

main() {
    #require_file "$OFFSET_SCRIPT"
    require_file "$MAT_SCRIPT"

    local entry name dir
#    for entry in "${DATASETS[@]}"; do
#        name="${entry%%|*}"
#        dir="${entry#*|}"
#
#        process_dataset "$name" "$dir"
#    done

    for entry in "${BASESETS[@]}"; do
        name="${entry%%|*}"
        dir="${entry#*|}"

        process_offset  "$name" "$dir" &
	pids+=("$!")
	names+=("$name")
    done
    local i
    local status=0

    for i in "${!pids[@]}"; do
        if wait "${pids[$i]}"; then
            log "SUCCESS: ${names[$i]}"
        else
            log "FAILED: ${names[$i]}"
            status=1
        fi
    done

    if (( status != 0 )); then
        die "One or more datasets failed."
    fi

    echo
    echo "============================================================"
    log "All datasets completed successfully."
    echo "============================================================"
}

main "$@"
