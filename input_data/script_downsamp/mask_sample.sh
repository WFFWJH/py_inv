#!/bin/bash -x
#
#
set -e


if [ $# -lt 2 ]; then
    echo "Usage: $0 file_detrend.grd fault_area dir_of_dem step_begin"
    echo "step_begin = 1 means mask have done"
    echo "step_begin = 2 means .llde already exit, do not downsamp"
    exit 1
fi

search_dir="$3"
step_begin="${4:-0}"
if [ "$step_begin" != "0" ] && [ "$step_begin" != "1" ] && [ "$step_begin" != "2" ]; then
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

	gmt grdmath "$mask" ISNAN 0 NAN = "$mask1"
	
	gmt grdmath "$input" "$mask1" MUL = "${prefix}_in.grd"
fi
	
if [ "$step_begin" -le 1 ]; then
	#python sample.py "${prefix}_out.grd" 500 50 10 1 0 1
	
#	python "$SCRIPT_DIR/sample.py" "${prefix}_out.grd" 250 50 10 1 0 1
	python "$SCRIPT_DIR/sample.py" "${prefix}_out.grd" 500 500 10 1 0 1
	#python "$SCRIPT_DIR/sample.py" "${prefix}_out.grd" 500 50 10 1 0 1
#	gmt grdmath "$mask" ISNAN 0 NAN = "$mask1"
#	
#	gmt grdmath "$input" "$mask1" MUL = "${prefix}_in.grd"
	
	#python "$SCRIPT_DIR/sample.py" "${prefix}_in.grd" 1000 20 5 1 0 1
	python "$SCRIPT_DIR/sample.py" "${prefix}_in.grd" 500 300 5 1 0 1
	#python sample.py "${prefix}_in.grd" 1000 20 5 1 0 1
	
fi

if [ "$step_begin" -le 2 ]; then
 	cat "${prefix}_in_py.llde" "${prefix}_out_py.llde" > "${prefix}.llde"
	
	"$SCRIPT_DIR/pplot.sh" "${prefix}.llde"
	"$SCRIPT_DIR/pplot.sh" "${prefix}_in_py.llde"
	"$SCRIPT_DIR/pplot.sh" "${prefix}_out_py.llde"
    	rm $mask $mask1 $buffer ${prefix}_in_py.llde ${prefix}_out_py.llde
   "$SCRIPT_DIR/lltenude.sh" "${1%.grd}.llde" "$search_dir"
    python "$SCRIPT_DIR/mat_lltenude.py" "${1%.grd}.lltenude" "${1%.grd}.mat"
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

