#!/bin/bash

if [ $# -lt 4 ]; then
    echo "用法: $0 SRC_ROOT DEST_ROOT TYPES Process_level [ORBITS]"
    echo "例如: $0 /mnt/source /dst_dir 1 1"
    exit 1
fi

SRC="$1"
DST="$2"
TYPES_STR="$3"
FILE_TYPE="$4"

if [ $# -ge 5 ]; then
    ORBITS_STR="$5"
else
    ORBITS_STR="A143,A70,D33,D106"
fi

# 把逗号分隔转为空格数组
IFS=',' read -r -a TYPES <<< "$TYPES_STR"
IFS=',' read -r -a ORBITS <<< "$ORBITS_STR"

# 主循环
for orb in "${ORBITS[@]}"; do
#    for typ in "${TYPES[@]}"; do
#
#        SRC_DIR="${SRC}/${orb}/${typ}/${FILE_TYPE}"
#        DST_DIR="${DST}/${orb}/${typ}"
#
        SRC_DIR="${SRC}/${orb}"
#        DST_DIR="${DST}/${orb}"
        if [ ! -d "$SRC_DIR" ]; then
            echo "[跳过] 源目录不存在：$SRC_DIR"
            exit 1
        fi

#        if [ ! -d "$dst_dir" ]; then
#            echo "error $dst_dir not founD"
#            exit 1
#        fi
#
#	lower_orb="${orb,,}"
#	
#	files=("${SRC_DIR}/${lower_orb}azi_detrend.grd")
	files=("${SRC_DIR}"/los_clean_detrend.lltenude)
	
	if [ ! -f "${files[0]}" ]; then
	    echo "[空] 文件不存在：${files[0]}"
	    continue
	fi

 	echo "找到的文件："
	for f in "${files[@]}"; do
	    echo "  $f"
	done
	
#	files1=("${SRC_DIR}/${lower_orb}azi.mat")
#	
#	if [ ! -f "${files1[0]}" ]; then
#	    echo "[空] 文件不存在：${files[0]}"
#	    continue
#	fi
#
# 	echo "找到的文件："
#	for f in "${files1[@]}"; do
#	    echo "  $f"
#	done
	
#	 mv "${files[0]}" "${SRC_DIR}/los_clean_detrend.grd"
#	 mv "${files1[0]}" "${SRC_DIR}/los_samp0.mat" 

       # echo "[copy] 从 $SRC_DIR -> $DST_DIR (${#files[@]} 项)"
#	./script_downsamp/mask_sample.sh ${files[0]} fault1.txt	
	awk '{$7 /= 100}1' "${files[@]}" > "${files[@]}.tmp"
	python ./script_downsamp/mat_lltenude.py  "${files[@]}.tmp"  "${SRC_DIR}/los_samp0.mat"
#echo "	gmt grdedit -L "${files[@]}" -G""$DST_DIR"/los_clean.grd"  "
#	gmt grdedit -L "${files[@]}" -G""$DST_DIR"/los_clean.grd"  
#echo " gmt grdtrend  "$DST_DIR"/los_clean.grd -N10+r -D""$DST_DIR"/los_clean_detrend.grd" "
# gmt grdtrend  "$DST_DIR"/los_clean.grd -N10+r -D""$DST_DIR"/los_clean_detrend.grd" 

#	ln -s "${files[@]}" "$DST_DIR/"
#	rm "${files[@]}"


        if [ $? -ne 0 ]; then
            echo "[警告] cp 命令失败"
        fi

#    done
done

echo "完成。"
