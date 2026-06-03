#!/bin/bash -x
#
#
set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 file.grd"
    exit 1
fi

input="$1"
prefix="${input%.grd}"

buffer="${prefix}_buffer.txt"
mask="${prefix}_mask.grd"
mask1="${prefix}_mask1.grd"

gmt spatial fault1.txt -Sb0.15 -fg > "$buffer"

gmt grdmask "$buffer" -R"$input" -N1/NaN/NaN -G"$mask"

gmt grdmath "$input" "$mask" MUL = "${prefix}_out.grd"

#python sample.py "${prefix}_out.grd" 500 50 10 1 0 1
python sample.py "${prefix}_out.grd" 500 50 10 1 0 1

gmt grdmath "$mask" ISNAN 0 NAN = "$mask1"

gmt grdmath "$input" "$mask1" MUL = "${prefix}_in.grd"

python sample.py "${prefix}_in.grd" 1000 20 5 1 0 1
#python sample.py "${prefix}_in.grd" 1000 20 5 1 0 1

cat "${prefix}_in_py.llde" > "${prefix}.llde"
cat "${prefix}_out_py.llde" >> "${prefix}.llde"

./pplot.sh "${prefix}.llde"
./pplot.sh "${prefix}_in_py.llde"
./pplot.sh "${prefix}_out_py.llde"

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

