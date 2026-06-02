#!/bin/bash
#

if [ $# -lt 1 ]; then
    echo "Usage: $0 file"
    exit 1
fi

file="$1"
name=$(basename "${file%.*}")
ext="${file##*.}"

if [ "$ext" = "grd" ]; then

    echo "GRD plot"
	gmt begin $name png 
#		gmt subplot begin 1x2 -Fs10c/0 -JX15c "$region" -Sr -A -Baf 
		
#		gmt subplot set 0
		gmt basemap  --MAP_FRAME_TYPE=plain -R"$1" -JM15c -Baf
		gmt makecpt -Cjet -Do -T-3/3/0.1 -Z
#		gmt plot -Sc0.1 tmp1 -C
		gmt grdimage $1 -C

		gmt plot fault1.txt -W0.5p,black
	        gmt colorbar -D+h+e -S -Ba1f0.5+l"Azimuth displacement(m)"
	gmt end show


elif [ "$ext" = "llde" ]; then

    echo "LLDE plot"

    awk '{printf "%.6f %.6f %s\n", $1, $2, $3}' $1  > tmp1
    region=$(awk '
	function floor(x){return int(x)}
	function ceil(x){return (x==int(x))?x:int(x)+1}
	
	NR==1 {
	    xmin=xmax=$1
	    ymin=ymax=$2
	}
	{
	    if ($1<xmin) xmin=$1
	    if ($1>xmax) xmax=$1
	    if ($2<ymin) ymin=$2
	    if ($2>ymax) ymax=$2
	}
	END {
	    printf "-R%d/%d/%d/%d\n", floor(xmin), ceil(xmax), floor(ymin), ceil(ymax)
	}' tmp1)
	echo $region
	echo $name
	gmt begin $name png 
#		gmt subplot begin 1x2 -Fs10c/0 -JX15c "$region" -Sr -A -Baf 
		
#		gmt subplot set 0
		gmt basemap  --MAP_FRAME_TYPE=plain "$region" -JM15c -Baf
		gmt makecpt -Cjet -Do -T-3/3/0.1 -Z
		gmt plot -Sc0.1 tmp1 -C
	        gmt colorbar -D+h+e -S -Ba1f0.5+l"Azimuth displacement(m)"
	gmt end show

    
else

    echo "Unknown file type"

fi

