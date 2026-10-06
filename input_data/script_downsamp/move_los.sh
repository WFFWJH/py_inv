#!/bin/bash

set -e

# ============================================================
# Configuration
# ============================================================

BASE_DIR="$HOME/myanmar"
OUTPUT_DIR="$HOME/py_inv/input_data/data_v2"

MAT_SCRIPT="$BASE_DIR/mat_lltenude.py"
ORIGIN_SCRIPT="$BASE_DIR/origin_lltenu.sh"


# ============================================================
# Function: process one dataset
# ============================================================

process_dataset() {
    local name="$1"
    local sample_dir="$2"

    local output_dir="$OUTPUT_DIR/$name"

    echo
    echo "============================================================"
    echo "Processing: $name"
    echo "Sample directory: $sample_dir"
    echo "Output directory: $output_dir"
    echo "============================================================"

    # Create output directory if it does not exist
    mkdir -p "$output_dir"

    # Enter sample directory
    cd "$sample_dir"

    # --------------------------------------------------------
    # 1. Convert lltenude -> mat
    # --------------------------------------------------------
    echo "[1/4] Creating los_samp0.mat ..."

    python "$MAT_SCRIPT" test.lltenude los_samp0.mat

    cp los_samp0.mat "$output_dir/"


    # --------------------------------------------------------
    # 2. Generate merged.grd
    # --------------------------------------------------------
    echo "[2/4] Creating merged.grd ..."

    "$ORIGIN_SCRIPT"  .


    # --------------------------------------------------------
    # 3. Move look_*.grd
    # --------------------------------------------------------
    echo "[3/4] Moving look_*.grd los_clean_detrend.grd dem_low.grd ..."
    shopt -s nullglob

    look_files=(look_*.grd)
    
    if (( ${#look_files[@]} > 0 )); then
        mv "${look_files[@]}" "$output_dir/"
    else
        echo "Warning: no look_*.grd found."
    fi

    # Move los_clean_detrend.grd
    if [[ -f los_clean_detrend.grd ]]; then
        mv los_clean_detrend.grd "$output_dir/"
        echo "  Moved: los_clean_detrend.grd"
    else
        echo "  Warning: dem_low.grd not found."
    fi


    # Move dem_low.grd
    if [[ -f dem_low.grd ]]; then
        mv dem_low.grd "$output_dir/"
        echo "  Moved: dem_low.grd"
    else
        echo "  Warning: dem_low.grd not found."
    fi


    # --------------------------------------------------------
    # 4. Finished
    # --------------------------------------------------------
    echo "[4/4] Finished: $name"

    echo
}


# ============================================================
# Dataset list
# ============================================================

process_dataset \
    "D106" \
    "$BASE_DIR/D106/no_esd/sample_D106"

process_dataset \
    "D33" \
    "$BASE_DIR/D33/newframe/sample_D33"

process_dataset \
    "A143" \
    "$BASE_DIR/A143/sample_A143"

process_dataset \
    "A70" \
    "$BASE_DIR/A70/sample_A70"


echo
echo "============================================================"
echo "All datasets completed successfully."
echo "============================================================"
