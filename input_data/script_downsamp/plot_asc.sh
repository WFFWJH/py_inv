#!/usr/bin/env -S bash -e
# GMT modern mode bash template
# Date:    2026-04-12T00:46:41
# User:    ffan
# Purpose: Purpose of this script
export GMT_SESSION_NAME=$$	# Set a unique session name
# 原始飞行方向角（可能为负）
#heading_raw=-167.68
heading_raw=-13

# 如果 heading 小于 0，则加 360，否则保留原值
if (( $(echo "$heading_raw < 0" | bc -l) )); then
    heading=$(echo "360 + $heading_raw" | bc -l)
else
    heading=$heading_raw
fi

# 计算 LOS 方向角（右视，+90°，结果仍保持在 0–360° 内）
los=$(echo "$heading + 90" | bc -l)
if (( $(echo "$los >= 360" | bc -l) )); then
    los=$(echo "$los - 360" | bc -l)
fi

echo "✅ 飞行方向角（Heading）: $heading°"
echo "✅ LOS 方向角: $los°"

array_lon=96.2
array_lat=17
#
track=A143
cd azi_best/$track
gmt begin figurename png
#	gmt basemap -JM12c -R94.5/97.5/18/24 -Baf
	gmt basemap -JM12c -R${track}_azi.grd -Baf
    # 让0到2000m海拔的颜色从白，灰再到黑色渐变，生成cpt
    gmt makecpt -Cwhite,gray,black -T0/2000 -Do -Z
    # 绘制灰度地形底图
    gmt grdimage dem.grd -C -I+d
    gmt makecpt -Cpolar -T-300/300/10 -Do -Z
    gmt grdimage ${track}_azi.grd -C  -Q
    gmt coast -Ia/1p,lightblue -S167/194/223 
    gmt clip -C
#    gmt coast -Ia/1p,167/194/223 
    # plot arraw LOS
    #
    echo "$array_lon $array_lat $heading 2.5" | gmt plot -SV0.5c+e+h0 -W1p,black -Gblack

    # LOS 方向箭头（蓝色）
    echo "$array_lon $array_lat $los 1.5" | gmt plot -SV0.5c+e+h0 -W1p,black -Gblack

    # 图例文字
	echo "$(awk "BEGIN{print $array_lon-0.2}") $(awk "BEGIN{print $array_lat+0.7}") Azimuth" | gmt text -F+f12p,Helvetica-Bold,black
 	echo "$(awk "BEGIN{print $array_lon+0.4}") $(awk "BEGIN{print $array_lat-0.1}") Range" | gmt text -F+f12p,Helvetica-Bold,black


    gmt colorbar -D+h+e -S -Ba100f50+l"Azimuth displacement(cm)"
gmt end show
