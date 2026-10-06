#!/usr/bin/env bash
set -euo pipefail
usage() {
  echo "Usage: $0  <dem-dir>"
  echo "  Example: $0   /path contains merged.grd PRM LED dem.grd"
  exit 2
}

if [ $# -ne 1 ]; then
  usage
fi

#infile="$1"
demfile="$1"/dem.grd
losfile="$1"/merged.grd

search_dir="$1"

# 检查 to process 文件
if [ ! -f "$losfile" ]; then
  echo "ERROR: LOS file '$losfile' not found."
  exit 4
fi


# 检查 DEM 文件
if [ ! -f "$demfile" ]; then
  echo "ERROR: DEM file '$demfile' not found."
  exit 4
fi

# base name without last extension
#base="${infile%.*}"
basename=$( basename $losfile )

base="${basename%.*}"

# output filenames (same base, different suffix)
f_ll="${base}.ll"
f_llt="${base}.llt"
f_lld="${base}.lld"
f_lltenu="${base}.lltenu"
f_lltenud="${base}.lltenud"

echo "Input file:   $losfile"
echo "DEM file:     $demfile"
echo "Base name:    $base"
echo "Will create:  $f_ll, $f_llt, $f_lltenu, $f_lltenud"
echo

# 0) cut the pixels near fault
gmt grd2xyz $losfile -s > tmp
gmt select -Lfault+d0.005k -Il tmp > $f_lld
gmt xyz2grd $f_lld -R$losfile -Glos_clean_detrend.grd


# 1) 提取前两列到 base.ll （保留 Tab 为输出分隔符）
echo "[1/5] Extracting first two columns -> $f_ll"
#awk -v OFS=$'\t' '{print $1, $2}' "$infile" > "$f_ll"
awk '{print $1,$2}' $f_lld >  $f_ll
echo "  -> done ($f_ll)"

# 2) 用 GMT grdtrack 采样 DEM，输出 base.llt
echo "[2/5] Running: gmt grdtrack $f_ll -G$demfile > $f_llt"
gmt grdtrack "$f_ll" -G"$demfile" > "$f_llt"
echo "  -> done ($f_llt)"

# 3) 在目录查找 PRM 文件（大小写不敏感），取第一个
echo "[3/5] Searching for a .PRM file in current directory..."
#prmfile=$(find . -maxdepth 1 -type f -iname '*.prm' -print -quit || true)

prmfile=$(find "$search_dir" -maxdepth 1 -iname '*.prm' -print -quit || true)
if [ -z "$prmfile" ]; then
  echo "ERROR: No .PRM file found in current directory."
  exit 5
fi

#count_prm=$(find . -maxdepth 1 -type f -iname '*.prm' | wc -l)
count_prm=$(find "$search_dir" -maxdepth 1  -iname '*.prm' | wc -l)
if [ "$count_prm" -gt 1 ]; then
  echo "  Warning: multiple .PRM files found; using first: $prmfile"
else
  echo "  Found PRM: $prmfile"
fi

# 4) 运行 SAT_look：SAT_look $PRM < base.llt > base.lltenu
echo "[4/5] Running: SAT_look $prmfile < $f_llt > $f_lltenu"
SAT_look "$prmfile" < "$f_llt" > "$f_lltenu"
echo "  -> done ($f_lltenu)"
awk '{print $1,$2,$3}' $f_lltenu | gmt xyz2grd -R$losfile -Gdem_low.grd
awk '{print $1,$2,$4}' $f_lltenu | gmt xyz2grd -R$losfile -Glook_e.grd
awk '{print $1,$2,$5}' $f_lltenu | gmt xyz2grd -R$losfile -Glook_n.grd
awk '{print $1,$2,$6}' $f_lltenu | gmt xyz2grd -R$losfile -Glook_u.grd


# 5) paste base.lltenu 与 $f_lld 的最后1列 -> base.lltenude
echo "[5/5] Appending last cols of $f_lld to $f_lltenu -> $f_lltenud"
paste  <(tr -d '\r' < "$f_lltenu") <( tr -d '\r' < "$f_lld" |awk -v OFS=$'\t' '{print  $NF}') > "$f_lltenud"
echo "  -> done ($f_lltenud)"

echo
echo "All steps finished."
echo "Outputs:"
echo "rm  $f_ll"
rm "$f_ll"
echo "rm $f_llt"
#rm  "$f_llt"
echo "rm  $f_lltenu"
#rm "$f_lltenu"
echo "  $f_lltenud"

