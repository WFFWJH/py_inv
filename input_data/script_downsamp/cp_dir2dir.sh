#!/bin/bash

if [ $# -lt 4 ]; then
    echo "用法: $0 SRC_ROOT DEST_ROOT TYPES Process_level [ORBITS]"
    echo "例如: $0 /mnt/source 1 1 1"
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
        SRC_DIR="${SRC}/${orb}/"
    for typ in "${TYPES[@]}"; do
#
#        SRC_DIR="${SRC}/${orb}/${typ}/${FILE_TYPE}"
        DST_DIR="${DST}/${orb}/${typ}"
#
        if [ ! -d "$SRC_DIR" ]; then
            echo "[跳过] 源目录不存在：$SRC_DIR"
            exit 1
        fi

        if [ ! -d "$DST_DIR" ]; then
            echo "ERROR $DST_DIR NOT FOUND"
            exit 1
        fi

        files=("$SRC_DIR"/los_samp0.mat)

        # 判断是否为空（关键点！）
        if [ ! -e "${files[0]}" ]; then
            echo "[空] $SRC_DIR 中没有文件，跳过"
            continue
        fi

	echo "找到的文件："
	for f in "${files[@]}"; do
	    echo "  $f"
	done

#	python mat_lltenude.py "${files[@]}" "$DST_DIR/los_samp0.mat"	

        echo "[copy] 从 $SRC_DIR -> $DST_DIR (${#files[@]} 项)"
 	cp "${files[@]}" "$DST_DIR/"
#	rm "${files[@]}"


        if [ $? -ne 0 ]; then
            echo "[警告] cp 命令失败"
        fi

    done
done

echo "完成。"
