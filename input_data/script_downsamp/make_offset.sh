#!/bin/bash
cleanup() {
    trap - SIGINT SIGTERM

    echo
    echo "[STOP] Terminating all child processes..."

    jobs -pr | xargs -r kill

    wait

    exit 1
}

trap cleanup SIGINT SIGTERM

find_best_divisor() {
    local x=$1
    local dmin=$2
    local dmax=$3

    for ((d=dmin; d<=dmax; d++)); do

        r=$((x%d))

        printf "%d %d\n" "$d" "$r"
    done |
    sort -k2,2n -k1,1nr |
    head -10 
#    while read -r d err; do
#        printf "d=%4d  err=%3d\n" "$d" "$err"
#    done
}

# =========================
# Usage
# =========================
if [[ $# -lt 2 || $# -gt 6 ]]; then
    echo "Usage: $0 [a|r] [0|1] [nx] [ny] [xsearch] [ysearch]"
    echo ""
    echo "  First argument:"
    echo "    a - run make_a_offset.csh"
    echo "    r - run make_r_offset.csh"
    echo ""
    echo "  Second argument:"
    echo "    0 - disable xcorr"
    echo "    1 - enable xcorr"
    echo ""
    echo "  Optional:"
    echo "    nx ny xsearch ysearch"
    echo "    default: 4096 16384 16 16"
    exit 1
fi

# =========================
# Parse arguments
# =========================
mode="$1"
xcorr_flag="$2"

nx="${3:-4096}"
ny="${4:-8192}"
xsearch="${5:-16}"
ysearch="${6:-16}"

# =========================
# Select program
# =========================
case "$mode" in
    a)
        program="make_a_offset.csh"
        ;;
    r)
        program="make_r_offset.csh"
        ;;
    *)
        echo "ERROR: Invalid mode '$mode'"
        echo "Use 'a' or 'r'"
        exit 1
        ;;
esac

# =========================
# Check xcorr flag
# =========================
if [[ "$xcorr_flag" != "0" && "$xcorr_flag" != "1" ]]; then
    echo "ERROR: Invalid xcorr flag '$xcorr_flag'"
    echo "Use 0 or 1"
    exit 1
fi

# =========================
# Folder list
# =========================
folders=(F1  F2 F3 F4 F5)

# =========================
# Main loop
# =========================
for folder in "${folders[@]}"; do

    # Check folder
    if [[ ! -d "$folder" ]]; then
        echo "[skip] Folder not found: $folder"
        continue
    fi

    if [[ ! -d "$folder/SLC" ]]; then
        echo "[skip] Subfolder not found: $folder/SLC"
        continue
    fi

    echo "===================================="
    echo "Processing folder: $folder"
    echo "===================================="

    # Find PRM files
    mapfile -t prm_files < <(
        find "$folder/SLC" -maxdepth 1 -type f -name "*.PRM" | sort
    )

    if [[ ${#prm_files[@]} -lt 2 ]]; then
        echo "[ERROR] Less than two PRM files in $folder/SLC"
        continue
    fi

    prm1=$(basename "${prm_files[0]}")
    prm2=$(basename "${prm_files[1]}")

    echo "PRM1 = $prm1"
    echo "PRM2 = $prm2"
echo $prm1
 num_valid_az=$(awk '/num_valid_az/ {print $3}' "$folder/SLC/$prm1")
 num_rng_bins=$(awk '/num_rng_bins/ {print $3}' "$folder/SLC/$prm1") 
# nx=$(awk '/num_rng_bins/ {print int($3/4)}' "$folder/SLC/$prm1") 
# ny=$(awk '/num_valid_az/ {print int($3/6)}' "$folder/SLC/$prm1")
echo $num_valid_az
 echo $num_rng_bins
find_best_divisor $(($num_rng_bins  - 6 * $xsearch)) 1500 2500 
 nx=$(find_best_divisor $(($num_rng_bins  - 6 * $xsearch)) 1500 2500 | awk 'NR==1{print ($1 - 3)}')
echo $nx

      (
    cd "$folder" || exit 1

    echo ""
    echo "Running:"

    cmd=( "$program"
          "$prm1"
          "$prm2"
          "$nx"
          "$ny"
          "$xsearch"
          "$ysearch"
          "$xcorr_flag" )

    printf '%q ' "${cmd[@]}"
    echo ""

    if "${cmd[@]}"; then
        echo "[OK] $program finished in $folder"
    else
        echo "[ERROR] $program failed in $folder"
    fi
) &
done
wait
echo ""
echo "All done."
