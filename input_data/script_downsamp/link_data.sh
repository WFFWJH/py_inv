#!/bin/bash

# 检查参数
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <source_dir> <target_dir>"
    exit 1
fi

src="$1"
dst="$2"

# 转为绝对路径
src=$(realpath "$src")
dst=$(realpath "$dst")

# 检查源目录
if [ ! -d "$src" ]; then
    echo "Error: source directory does not exist: $src"
    exit 1
fi

# 检查目标目录（必须存在）
if [ ! -d "$dst" ]; then
    echo "Error: target directory does not exist: $dst"
    exit 1
fi

echo "Linking files from:"
echo "  $src"
echo "to:"
echo "  $dst"
echo "--------------------------"

# 遍历文件
for f in "$src"/*.lltenude; do
    if [ -f "$f" ]; then
        ln -sfn "$f" "$dst/"
        echo "Linked: $(basename "$f")"
    fi
done

echo "Done."
