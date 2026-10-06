#!/usr/bin/bash
set -e

########################################
# 输入参数
########################################
#
# 用法：
#   ./script.sh F1,F2 azi_offset_mask.grd
#
# 第1个参数：F条带选择，例如 F1,F2 或 F1,F2,F3,F4,F5
# 第2个参数：suffix，例如 azi_offset_mask.grd
########################################

if [ $# -lt 2 ]; then
    echo "Usage:"
    echo "  $0 F1,F2 suffix"
    echo
    echo "Examples:"
    echo "  $0 F1,F2 azi_offset_mask.grd"
    echo "  $0 F2,F3 azi_offset_ll.grd"
    echo "  $0 F1,F2,F3,F4,F5 highpass_mask.grd"
    exit 1
fi

########################################
# F 条带选择
########################################

IFS=',' read -ra files <<< "$1"

########################################
# suffix
########################################

suffix="$2"

echo "========================================"
echo "F 条带：${files[*]}"
echo "suffix：$suffix"
echo "========================================"

########################################
# 获取脚本所在目录
########################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "脚本目录：$SCRIPT_DIR"

########################################
# 获取所有网格的最小/最大经纬度
########################################

xmin_list=()
xmax_list=()
ymin_list=()
ymax_list=()
xinc_list=()
yinc_list=()

for file in "${files[@]}"; do

    grd_file="${file}_${suffix}"

    if [ ! -f "$grd_file" ]; then
        echo "Warning: $grd_file 不存在，跳过"
        continue
    fi

    info=$(gmt grdinfo "$grd_file" -Cn)

    xmin_list+=("$(echo "$info" | awk '{print $1}')")
    xmax_list+=("$(echo "$info" | awk '{print $2}')")

    ymin_list+=("$(echo "$info" | awk '{print $3}')")
    ymax_list+=("$(echo "$info" | awk '{print $4}')")

    xinc_list+=("$(echo "$info" | awk '{print $7}')")
    yinc_list+=("$(echo "$info" | awk '{print $8}')")

done

########################################
# 检查是否找到输入文件
########################################

if [ ${#xmin_list[@]} -eq 0 ]; then
    echo "Error: 没有找到任何输入网格！"
    exit 1
fi

########################################
# 可选：手动指定统一范围
########################################

if [ $# -ge 8 ]; then

    echo "mode 1"

    x_min="$3"
    x_max="$4"
    y_min="$5"
    y_max="$6"
    x_inc="$7"
    y_inc="$8"

else

    echo "mode 2：自动计算统一范围"

    ########################################
    # 获取全局极值
    ########################################

    x_min=$(printf "%s\n" "${xmin_list[@]}" | sort -n | head -1)
    x_max=$(printf "%s\n" "${xmax_list[@]}" | sort -rn | head -1)

    y_min=$(printf "%s\n" "${ymin_list[@]}" | sort -n | head -1)
    y_max=$(printf "%s\n" "${ymax_list[@]}" | sort -rn | head -1)

    x_inc=$(printf "%s\n" "${xinc_list[@]}" | sort -n | head -1)
    y_inc=$(printf "%s\n" "${yinc_list[@]}" | sort -n | head -1)

fi

########################################
# 计算网格数
########################################

NX=$(awk -v xmin="$x_min" -v xmax="$x_max" -v dx="$x_inc" \
    'BEGIN {printf "%d", (xmax - xmin)/dx + 1}')

NY=$(awk -v ymin="$y_min" -v ymax="$y_max" -v dy="$y_inc" \
    'BEGIN {printf "%d", (ymax - ymin)/dy + 1}')

x_max_adj="$x_max"
y_max_adj="$y_max"

echo
echo "统一范围："
echo "经度范围：$x_min 到 $x_max_adj"
echo "纬度范围：$y_min 到 $y_max_adj"
echo "网格尺寸：NX=$NX, NY=$NY"
echo "inc：X=$x_inc, Y=$y_inc"
echo

########################################
# 原始数据绘图
########################################

gmt begin allplot png

    gmt basemap \
        -R${x_min}/${x_max_adj}/${y_min}/${y_max_adj} \
        -JM15c \
        -Baf

    for file in "${files[@]}"; do

        grd="${file}_${suffix}"

        [ -f "$grd" ] || continue

        gmt grdimage "$grd" -Cpolar -Q

    done

    gmt colorbar -DJMR -Baf -By+l"m"
    gmt coast -Wthinnest,black -Df

gmt end

########################################
# 重采样并裁剪
########################################

for i in "${!files[@]}"; do

    file="${files[$i]}"

    input="${file}_${suffix}"

    [ -f "$input" ] || continue

    local_xmin="${xmin_list[$i]}"
    local_xmax="${xmax_list[$i]}"

    local_ymin="${ymin_list[$i]}"
    local_ymax="${ymax_list[$i]}"

    tmp_grd="${file}_tmp.grd"

    ########################################
    # 第一次裁剪
    ########################################

    gmt grdcut "$input" \
        -G"$tmp_grd" \
        -N \
        -R${local_xmin}/${x_max_adj}/${local_ymin}/${y_max_adj}

    ########################################
    # 第二次裁剪到统一范围
    ########################################

    gmt grdcut "$tmp_grd" \
        -G"${file}_cut.grd" \
        -N \
        -R${x_min}/${x_max_adj}/${y_min}/${y_max_adj}

    echo "gmt grdcut $input -G$tmp_grd -N -R${local_xmin}/${x_max_adj}/${local_ymin}/${y_max_adj}"

    echo "gmt grdcut $tmp_grd -G${file}_cut.grd -N -R${x_min}/${x_max_adj}/${y_min}/${y_max_adj}"

    rm -f "$tmp_grd"

    ########################################
    # 重采样
    ########################################

    gmt grdsample "${file}_cut.grd" \
        -G"${file}_resampled.grd" \
        -I${x_inc}/${y_inc} \
        -n+c

done

########################################
# 合并网格
########################################

inputs=()

for file in "${files[@]}"; do

    input="${file}_resampled.grd"

    [ -f "$input" ] || continue

    inputs+=("$input")

done

########################################
# 检查输入数量
########################################

if [ ${#inputs[@]} -eq 0 ]; then
    echo "Error: 没有找到任何 resampled.grd 文件！"
    exit 1
fi

########################################
# grdmath AND 合并
########################################

grdmath_args=("${inputs[0]}")

for ((i=1; i<${#inputs[@]}; i++)); do
    grdmath_args+=("${inputs[$i]}" AND)
done

gmt grdmath "${grdmath_args[@]}" = merged.grd 

########################################
# grdblend
########################################

gmt grdblend "${inputs[@]}" \
    -Gblend.grd \
    -R${x_min}/${x_max_adj}/${y_min}/${y_max_adj} \
    -I${x_inc}/${y_inc} \

########################################
# resample_merge 绘图
########################################

gmt begin resample_merge png

    gmt basemap \
        -R${x_min}/${x_max_adj}/${y_min}/${y_max_adj} \
        -JM15c \
        -Baf

    for file in "${files[@]}"; do

        grd="${file}_resampled.grd"

        [ -f "$grd" ] || continue

        gmt grdimage "$grd" -Cpolar -Q

    done

    gmt colorbar -DJMR -Baf -By+l"m"
    gmt coast -Wthinnest,black -Df

gmt end 

########################################
# blend 绘图
########################################

gmt begin blend png

    gmt basemap \
        -R${x_min}/${x_max_adj}/${y_min}/${y_max_adj} \
        -JM15c \
        -Baf

    gmt grdimage blend.grd -Cpolar

    gmt colorbar -DJMR -Baf -By+l"m"
    gmt coast -Wthinnest,black -Df

gmt end

########################################
# merged 绘图
########################################

"$SCRIPT_DIR/pplot.sh" merged.grd
