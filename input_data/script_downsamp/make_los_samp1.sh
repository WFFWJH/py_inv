#!/bin/bash -x
#
#
set -e


if [ $# -lt 2 ]; then
    echo "Usage: $0 file.grd fault_area step_begin"
    echo "step_begin = 1 means .llde already exit, do not downsamp"
    exit 1
fi

step_begin="${3:-0}"
if [ "$step_begin" != "0" ] && [ "$step_begin" != "1" ]; then
    echo "ERROR: step_begin must be 0 or 1"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
input="$1"
prefix="${input%.grd}"

if [ "$step_begin" -le 0 ]; then
	fault_txt="$2"
	
	buffer="${prefix}_buffer.txt"
	mask="${prefix}_mask.grd"
	mask1="${prefix}_mask1.grd"
	
	gmt spatial "$fault_txt" -Sb0.15 -fg > "$buffer"
	
	gmt grdmask "$buffer" -R"$input" -N1/NaN/NaN -G"$mask"
	
	gmt grdmath "$input" "$mask" MUL = "${prefix}_out.grd"
	gmt grdmath "los_clean_detrend.grd" "$mask" MUL = "los_clean_detrend_out.grd"
	
	#python sample.py "${prefix}_out.grd" 500 50 10 1 0 1
	
#	python "$SCRIPT_DIR/sample.py" "${prefix}_out.grd" 250 50 10 1 0 1 "los_clean_detrend_out.grd"
	python "$SCRIPT_DIR/sample.py" "${prefix}_out.grd" 500 50 10 1 0 1 "los_clean_detrend_out.grd"

	gmt grdmath "$mask" ISNAN 0 NAN = "$mask1"
	
	gmt grdmath "$input" "$mask1" MUL = "${prefix}_in.grd"
	gmt grdmath "los_clean_detrend.grd" "$mask1" MUL = "los_clean_detrend_in.grd" 
	
	python "$SCRIPT_DIR/sample.py" "${prefix}_in.grd" 1000 20 5 1 0 1 "los_clean_detrend_in.grd"

#	python "$SCRIPT_DIR/sample.py" "${prefix}_in.grd" 500 20 5 1 0 1 "los_clean_detrend_in.grd"

	#python sample.py "${prefix}_in.grd" 1000 20 5 1 0 1
	
	cat "${prefix}_in_py.llde" > "${prefix}.llde"
	cat "${prefix}_out_py.llde" >> "${prefix}.llde"
	
	"$SCRIPT_DIR/pplot.sh" "${prefix}.llde"
	"$SCRIPT_DIR/pplot.sh" "${prefix}_in_py.llde"
	"$SCRIPT_DIR/pplot.sh" "${prefix}_out_py.llde"
    rm $mask $mask1 $buffer ${prefix}_out.grd ${prefix}_in.grd ${prefix}_in_py.llde ${prefix}_out_py.llde

fi

if [ "$step_begin" -le 1 ]; then
    "$SCRIPT_DIR/lltenude.sh" "${1%.grd}.llde" "dem.grd"
    python "$SCRIPT_DIR/mat_lltenude.py" "${1%.grd}.lltenude" "${1%.grd}.mat"
    mv "${1%.grd}.mat" los_samp1.mat
#	awk '{print $1,$2}' "${1%.grd}.lltenude" | gmt grdtrack -t  -glos_clean_detrend.grd | awk '{print $3}' > value1.tmp
#
#	awk '
#	NR==FNR {value1[NR]=$1; next}
#	{
#	    $7=value1[FNR]
#	    print
#	}
#	' value1.tmp "${1%.grd}.lltenude" > out.txt
#	
#	python "$SCRIPT_DIR/mat_lltenude.py" "out.txt"  "los_samp1.mat"
fi

#gmt spatial fault1.txt -Sb0.15 -fg > buffer.txt
#gmt grdmask buffer.txt -R"$1" -N1/NaN/NaN -Gmask.grd
#gmt grdmath $1 mask.grd MUL =  "${1%.grd}_out.grd"
#
#python sample.py "${1%.grd}_out.grd" 500 50 10 1 0 1
#
#gmt grdmath mask.grd ISNAN 0 NAN = mask1.grd
#gmt grdmath $1 mask1.grd MUL =  "${1%.grd}_in.grd"
#
#python sample.py "${1%.grd}_in.grd" 1000 20 5 1 0 1
#
#cat "${1%.grd}_in_py.llde" > "${1%.grd}.llde" 
#cat "${1%.grd}_out_py.llde" >> "${1%.grd}.llde" 
#
#./pplot.sh "${1%.grd}.llde"
#./pplot.sh "${1%.grd}_in_py.llde"
#./pplot.sh "${1%.grd}_out_py.llde"

