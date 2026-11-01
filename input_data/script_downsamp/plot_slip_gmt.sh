#!/usr/bin/env bash
#
# 用 GMT 绘制有限断层模型 (三维 + 二维)
# 依赖 slip_to_gmt.py 生成的 slip_3dgrid.gmt / slip_2dgrid.gmt
#
#   bash plot_slip_gmt.sh
#
set -e
cd "$(dirname "$0")"

# ---------------- 参数 ----------------
# 三维: -R 经度/纬度/深度(km, 向下为负)
R3D=95.90/96.50/18.35/22.85/-20/0
PROJ3D=-JX6c/16c          # 水平方向: 经度 / 纬度
JZ=-JZ2c                  # 垂直方向: 深度
VIEW=-p100/20             # 方位角 / 仰角
CPT=white,lightblue,yellow,red
T3D=0/6.5/0.2             # CPT 范围 (滑移量, m)

# 二维展开剖面 (y 用负比例 -> 沿倾向距离向下增大)
R2D=0/478/0/20            # 沿走向 km / 沿倾向 km
PROJ2D=-JX20c/-2c

# 注意: gmt set 必须写在 gmt begin ... gmt end 之内,
# 否则会在当前目录落下一个 gmt.conf 并长期生效.

# ================= 三维有限断层模型 =================
gmt begin slip_gmt_3d png
  gmt set MAP_FRAME_TYPE plain
  gmt set MAP_GRID_PEN 0p,gray,-
  gmt set FONT_ANNOT_PRIMARY 8p FONT_LABEL 9p
  gmt set MAP_ANNOT_OFFSET_PRIMARY 2p MAP_FRAME_PEN thinner MAP_TICK_LENGTH 2p

  gmt basemap -R${R3D} ${PROJ3D} ${JZ} \
      -Bxa0.2fg+l"Longitude (deg)" -Bya0.5fg+l"Latitude (deg)" \
      -Bza5f1g5+l"Depth (km)" -BwSEnZ ${VIEW}

  # 滑移位错量 CPT: 最小 0, 最大 6.5 m
  gmt makecpt -C${CPT} -T${T3D} -Z

  # 逐个多边形填充 (-L 闭合多边形, -C 按段头 -Z 取色)
  gmt plot3d slip_3dgrid.gmt -C -L -W0p,gray ${VIEW}

  # 主震震中
#  echo "95.936 22.011 -10" | gmt plot3d -Sa0.45c -Gred -W0.5p,black ${VIEW}

  gmt colorbar -C -DjBR+w3c/0.3c+h+o1.5c/1.3c+ml -Ba1 -Bx+l"slip (m)"
gmt end 

# ================= 二维展开剖面 =================
gmt begin slip_gmt_2d png
  gmt set MAP_FRAME_TYPE plain
  gmt set FONT_ANNOT_PRIMARY 5p FONT_LABEL 6p
  gmt set MAP_ANNOT_OFFSET_PRIMARY 2p MAP_FRAME_PEN thinner MAP_TICK_LENGTH 2p

  gmt basemap -R${R2D} ${PROJ2D} \
      -Bxa50f25+l"Along-strike distance (km)" \
      -Bya5f2.5+l"Along-dip distance (km)" -BWS

  gmt makecpt -C${CPT} -T${T3D} -Z
  gmt plot slip_2dgrid.gmt -C -L -W0p,gray

  gmt colorbar -C -DJMR+w2c/0.2c+v+o0.2c/0c+ml   -Ba1 -Bx+l"slip (m)"
gmt end show

echo "done: slip_gmt_3d.png  slip_gmt_2d.png"
