#!/usr/bin/bash
set -e

echo "========================================"
echo "开始处理 merged.grd"
echo "========================================"

echo "[1/5] 复制 merged.grd ..."
echo "      cp ../collected_offset/merged.grd ."
cp ../collected_offset/merged.grd .

echo "      完成"
echo

echo "[2/5] 对 merged.grd 进行 mask/sample ..."
echo "      ./mask_sample_1.sh merged.grd"
./mask_sample_1.sh merged.grd

echo "      完成"
echo

echo "[3/5] 使用 fault 进行空间筛选 ..."
echo "      gmt select -Lfault+d0.005k -Il merged.llde > test.llde"
gmt select -Lfault+d0.005k -Il merged.llde > test.llde

echo "      完成"
echo

echo "[4/5] 检查 merged.llde 和 test.llde 的行数 ..."
echo "      wc -l merged.llde test.llde"
wc -l merged.llde test.llde

echo
echo "[5/5] 运行 lltenude ..."
echo "      ~/myanmar/lltenude.sh test.llde ."
~/myanmar/lltenude.sh test.llde .

echo
echo "========================================"
echo "全部处理完成！"
echo "========================================"
