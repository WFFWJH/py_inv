#!/usr/bin/env bash
set -e

########################################
# 参数检查
########################################

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage:"
    echo "  $0 [a|aa|mask|r] [F-selection]"
    echo
    echo "Modes:"
    echo "  a     - collect azi_offset_ll.grd"
    echo "  aa    - collect azi_offset_mask.grd"
    echo "  mask  - collect highpass_mask.grd"
    echo "  r     - collect rng_offset_ll.grd"
    echo
    echo "F-selection:"
    echo "  不指定 - 自动处理当前目录下所有 F*"
    echo "  F1,F2  - 只处理 F1 和 F2"
    echo "  F1,F3  - 只处理 F1 和 F3"
    exit 1
fi


########################################
# 模式
########################################

mode="$1"


########################################
# 根据 mode 设置 offset 文件
########################################

case "$mode" in

    a)
        offset_file="azi_offset_ll.grd"
        offset_dir="azi_offset"
        ;;

    aa)
        offset_file="azi_offset_mask.grd"
        offset_dir="azi_offset"
        ;;

    mask)
        offset_file="highpass_mask.grd"
        offset_dir="azi_offset"
        ;;

    r)
        offset_file="rng_offset_ll.grd"
        offset_dir="rng_offset"
        ;;

    *)
        echo "Invalid mode: $mode"
        echo "Use: a, aa, mask or r"
        exit 1
        ;;

esac


########################################
# 输出目录
########################################

output_dir="collected_offsets"

if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
fi


########################################
# F 子带选择
########################################

if [ $# -eq 2 ]; then

    # 用户指定，例如 F1,F2,F4
    IFS=',' read -ra folders <<< "$2"

else

    # 自动寻找当前目录下所有 F* 目录
    folders=()

    for dir in F*; do

        if [ -d "$dir" ]; then
            folders+=("$dir")
        fi

    done

fi


########################################
# 检查 F 子带
########################################

if [ ${#folders[@]} -eq 0 ]; then
    echo "Error: No F* directories found."
    exit 1
fi


########################################
# 显示处理信息
########################################

echo "========================================"
echo "Mode       : $mode"
echo "Offset dir : $offset_dir"
echo "Offset file: $offset_file"
echo "Output dir : $output_dir"
echo "F bands    : ${folders[*]}"
echo "========================================"
echo


########################################
# 遍历 F 子带
########################################

for folder in "${folders[@]}"; do

    echo "----------------------------------------"
    echo "Processing $folder"
    echo "----------------------------------------"

    ########################################
    # 检查 F 子带目录
    ########################################

    if [ ! -d "$folder" ]; then
        echo "Warning: $folder does not exist, skipping..."
        continue
    fi


    ########################################
    # 检查 intf
    ########################################

    intf_dir="$folder/intf"

    if [ ! -d "$intf_dir" ]; then
        echo "Warning: $intf_dir does not exist, skipping..."
        continue
    fi


    ########################################
    # 自动寻找 intf/*/offset_dir
    ########################################

    matches=()

    for candidate in "$intf_dir"/*/"$offset_dir"; do

        if [ -d "$candidate" ]; then
            matches+=("$candidate")
        fi

    done


    ########################################
    # 检查是否找到 offset_dir
    ########################################

    if [ ${#matches[@]} -eq 0 ]; then
        echo "Warning: No $offset_dir found under $intf_dir"
        continue
    fi


    ########################################
    # 正常情况下应该只有一个
    ########################################

    if [ ${#matches[@]} -gt 1 ]; then
        echo "Warning: Multiple $offset_dir directories found:"
        for match in "${matches[@]}"; do
            echo "    $match"
        done

        echo "Using the first one:"
        echo "    ${matches[0]}"
    fi


    ########################################
    # 使用第一个匹配目录
    ########################################

    full_offset_path="${matches[0]}"

    echo "Offset directory:"
    echo "    $full_offset_path"


    ########################################
    # 源文件
    ########################################

    source_file="$full_offset_path/$offset_file"

    if [ ! -f "$source_file" ]; then
        echo "Warning: File not found:"
        echo "    $source_file"
        echo "Skipping $folder..."
        continue
    fi


    ########################################
    # 目标文件
    ########################################

    dest_file="$output_dir/${folder}_${offset_file}"


    ########################################
    # 拷贝
    ########################################

    cp "$source_file" "$dest_file"

    echo "Copied:"
    echo "    $source_file"
    echo "        ↓"
    echo "    $dest_file"

done


########################################
# 完成
########################################

echo
echo "========================================"
echo "All processing completed."
echo "Output directory:"
echo "    $output_dir"
echo "========================================"
