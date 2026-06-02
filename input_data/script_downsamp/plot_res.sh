#!/bin/bash
#
trap "echo 'User interrupt'; exit 1" SIGINT

# ==============function================
plot_dem(){
    # 让0到2000m海拔的颜色从白，灰再到黑色渐变，生成cpt
    gmt makecpt -Cwhite,gray,black -T0/2000 -Do -Z
    # 绘制灰度地形底图
    gmt grdimage "$1" -C -I+d
#    gmt coast -Ia/1p,lightblue -S167/194/223 
    # plot arraw LOS
    heading_raw="$2"
	array_lon="$3"
	array_lat="$4"	
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
	

    echo "$array_lon $array_lat $heading 1.5" | gmt plot -SV0.5c+e+h0 -W1.5p,black -Gblack

    # LOS 方向箭头（蓝色）
    echo "$array_lon $array_lat $los 1" | gmt plot -SV0.5c+e+h0 -W1.5p,red -Gred

    # 图例文字
    #echo "96.8 18.3  Flight Heading" | gmt text -F+f12p,Helvetica-Bold,black
    #echo "96.7 19.2 LOS Direction"  | gmt text -F+f12p,Helvetica-Bold,black

}


# =================== main =======================
#
tracks=( none A143 A70 D33 D106)
basepath="/home/ffan/inv/inversion/myanmar/input/azi_best/"
for i in {1..4}; do
    model="dataset${i}.forwards"
    residual="dataset${i}.residuals"
    origin="dataset${i}.origin"
    paste "$model" "$residual" | awk '{print $1,$2,$3-$6}' > "$origin"
    region=$(gmt info "$model" -I0.1)
    read xmin xmax ymin ymax zmin zmax <<< $(gmt info "$model" -C -I0.1)
	fullpath="$basepath""${tracks[i]}/dem.grd"
	if [[ ${tracks[i]} == A* ]]; then
	    heading_raw=-12.2
	    array_lon=$(awk "BEGIN{print $xmin+0.3}")
	    array_lat=$(awk "BEGIN{print $ymin+0.25}")
	else
	    heading_raw=-167.7
	    array_lon=$(awk "BEGIN{print $xmin+0.3}")
	    array_lat=$(awk "BEGIN{print $ymax-0.3}")
	fi
	
	gmt begin "data_${i}" png 
		gmt subplot begin 1x3 -Fs10c/0 -JM15c "$region" -Sr -A -Baf 
		
		gmt subplot set 0
		gmt basemap  --MAP_FRAME_TYPE=plain
		plot_dem $fullpath $heading_raw $array_lon $array_lat
		gmt makecpt -Cjet -Do -T-300/300/1 -Z
		gmt plot -Sc0.1 "$model" -C
	        gmt colorbar -D+h+e -S -Ba100f50+l"Azimuth displacement(cm)"


		gmt subplot set 1
		gmt basemap  --MAP_FRAME_TYPE=plain
		plot_dem $fullpath $heading_raw $array_lon $array_lat
		gmt makecpt -Cjet -Do -T-300/300/1 -Z
		gmt plot -Sc0.1 "$origin" -C
	        gmt colorbar -D+h+e -S -Ba100f50+l"Azimuth displacement(cm)"
	


		gmt subplot set 2
		gmt basemap  --MAP_FRAME_TYPE=plain
		plot_dem $fullpath $heading_raw $array_lon $array_lat
		gmt makecpt -Cjet -Do -T-100/100/1 -Z
		gmt plot -Sc0.1 "$residual" -C
	        gmt colorbar -D+h+e -S -Ba50f10+l"Azimuth displacement(cm)"
			
		gmt subplot end
		
	gmt end show
done


