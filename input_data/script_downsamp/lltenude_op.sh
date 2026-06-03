#!/usr/bin/env bash 
set -euo pipefail

usage() {
  echo "Usage: $0 <input-file> <dem-grd>"
  echo "  Example: $0 origin.llde/origin.grd dem.grd"
  exit 2
}

if [ $# -ne 2 ]; then
  usage
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
infile="$1"
demfile="$2"

# 检查输入文件
if [ ! -f "$infile" ]; then
  echo "ERROR: input file '$infile' not found."
  exit 3
fi

# 检查 DEM 文件
if [ ! -f "$demfile" ]; then
  echo "ERROR: DEM file '$demfile' not found."
  exit 4
fi

# base name without last extension
base="${infile%.*}"

# output filenames (same base, different suffix)
f_ll="${base}.ll"
f_llt="${base}.llt"
f_lle="${base}.lle"
f_lln="${base}.lln"
f_llu="${base}.llu"
f_lltenu="${base}.lltenu"
f_lltenude="${base}.lltenude"

echo "Input file:   $infile"
echo "DEM file:     $demfile"
echo "Base name:    $base"
echo "Will create:  $f_ll, $f_llt, $f_lltenu, $f_lltenude"
echo

ext="${infile##*.}"
if [ "$ext" = "llde" ]; then

	# 1) 提取前两列到 base.ll （保留 Tab 为输出分隔符）
	echo "[1/5] Extracting first two columns -> $f_ll"
	awk -v OFS=$'\t' '{print $1, $2}' "$infile" > "$f_ll"
	echo "  -> done ($f_ll)"
	# 2) 用 GMT grdtrack 采样 DEM，输出 base.llt
	echo "[2/5] Running: gmt grdtrack $f_ll -G$demfile > $f_llt"
	gmt grdtrack "$f_ll" -G"$demfile" > "$f_llt"
	echo "  -> done ($f_llt)"
elif  [ "$ext" =  "grd" ]; then
	echo "[1/5] Extracting grd's  first two columns -> $f_ll"
	gmt grd2xyz $infile -s | awk '{print $1,$2}'> $f_ll
	echo "  -> done ($f_ll)"
	echo "[2/5] Running: gmt grdtrack $f_ll -G$demfile > $f_llt"
	gmt grdtrack $f_ll -G"$demfile" > $f_llt
	echo "  -> done ($f_llt)"
else
	echo "Unkknoown file type"
fi

# 3) 在当前目录查找 PRM 文件（大小写不敏感），取第一个
echo "[3/5] Searching for a .PRM file in current directory..."
#prmfile=$(find . -maxdepth 1 -type f -iname '*.prm' -print -quit || true)

prmfile=$(find . -maxdepth 1 -iname '*.prm' -print -quit || true)
if [ -z "$prmfile" ]; then
  echo "ERROR: No .PRM file found in current directory."
  exit 5
fi

#count_prm=$(find . -maxdepth 1 -type f -iname '*.prm' | wc -l)
count_prm=$(find . -maxdepth 1  -iname '*.prm' | wc -l)
if [ "$count_prm" -gt 1 ]; then
  echo "  Warning: multiple .PRM files found; using first: $prmfile"
else
  echo "  Found PRM: $prmfile"
fi

# 4) 运行 SAT_look：SAT_look $PRM < base.llt > base.lltenu
echo "[4/5] Running: SAT_look $prmfile < $f_llt > $f_lltenu"
"$SCRIPT_DIR/SAT_look" "$prmfile" < "$f_llt" > "$f_lltenu"
echo "  -> done ($f_lltenu)"

# 5) paste base.lltenu 与 infile 的最后两列 -> base.lltenude
echo "[5/5] Appending last two cols of $infile to $f_lltenu -> $f_lltenude"
paste  <(tr -d '\r' < "$f_ll") <( tr -d '\r' < "$f_lltenu" |awk -v OFS=$'\t' '{print $4}') > "$f_lle"
paste  <(tr -d '\r' < "$f_ll") <( tr -d '\r' < "$f_lltenu" |awk -v OFS=$'\t' '{print $5}') > "$f_lln"
paste  <(tr -d '\r' < "$f_ll") <( tr -d '\r' < "$f_lltenu" |awk -v OFS=$'\t' '{print $6}') > "$f_llu"
#paste  <(tr -d '\r' < "$f_lltenu") <( tr -d '\r' < "$infile" |awk -v OFS=$'\t' '{print $(NF-1), $NF}') > "$f_lltenude"
echo "  -> done ($f_lle,$f_lln,$f_llu)"
echo " xyz2grd lle,lln,llu -> look_e.grd look_n.grd look_u.grd"
gmt xyz2grd $f_lle  -R"$infile"  -Glook_e.grd
gmt xyz2grd $f_lln  -R"$infile"  -Glook_n.grd
gmt xyz2grd $f_llu  -R"$infile"  -Glook_u.grd
#-I$(gmt grdinfo $infile  -Cn | awk '{print $7"/"$8}') 
gmt xyz2grd $f_llt  -R"$infile"  -Gdem_low.grd
gmt grdinfo $infile
gmt grdinfo dem_low.grd
echo
echo "All steps finished."
echo "Outputs:"
echo "rm  $f_ll"
rm "$f_ll"
echo "rm  $f_lltenu"
rm "$f_lltenu"
echo " rm $f_lle $f_lln $f_llu $f_llt"
rm $f_lle $f_lln $f_llu $f_llt

