#!/bin/bash -x
#
#
set -e


if [ $# -lt 2 ]; then
    echo "Usage: $0 los_model.lltenude los_clean_detrend.grd"
    echo "step_begin = 1 means .llde already exit, do not downsamp"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

awk '{print $1,$2}' "$1" | gmt grdtrack -T -G"$2" | awk '{print $3}' > value1.tmp

awk '
NR==FNR {value1[NR]=$1; next}
{
    $7=value1[FNR]
    print
}
' value1.tmp "$1" > out.txt

python "$SCRIPT_DIR/mat_lltenude.py" "out.txt"  "los_samp1.mat"
