#!/bin/bash
#
#

    awk '{printf "%.6f %.6f %s\n", $1/100, $2/100, $3}' $1  > tmp1
    awk '{printf "%.6f %.6f %s\n", $1/100, $2/100, $3}' $2  > tmp2
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
	gmt begin data png 
		gmt subplot begin 1x2 -Fs10c/0 -JX15c "$region" -Sr -A -Baf 
		
		gmt subplot set 0
		gmt basemap  --MAP_FRAME_TYPE=plain
		gmt makecpt -Cjet -Do -T-3/3/0.1 -Z
		gmt plot -Sc0.1 tmp1 -C
	        gmt colorbar -D+h+e -S -Ba1f0.5+l"Azimuth displacement(cm)"


		gmt subplot set 1
		gmt basemap  --MAP_FRAME_TYPE=plain
		gmt makecpt -Cjet -Do -T-3/3/0.1 -Z
		gmt plot -Sc0.1 tmp2 -C
	        gmt colorbar -D+h+e -S -Ba1f0.5+l"Azimuth displacement(cm)"
	


	#	gmt subplot set 2
	#	gmt basemap  --MAP_FRAME_TYPE=plain
	#	gmt makecpt -Cjet -Do -T-100/100/1 -Z
	#	gmt plot -Sc0.1 "$residual" -C
	#        gmt colorbar -D+h+e -S -Ba50f10+l"Azimuth displacement(cm)"
			
		gmt subplot end
		
	gmt end show

