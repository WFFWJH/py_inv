"""workflows — 编排层 (流程/命令行脚本, 非科学内核)。

模块
----
:mod:`workflows.inversion`     主反演流程 (downsample / invert1 / resamp / invert2)
:mod:`workflows.checkerboard`  棋盘格分辨率测试 (Okada 正演 -> 反演还原)
:mod:`workflows.dip_search`    断层倾角网格搜索
:mod:`workflows.make_rms`      平滑系数 λ 扫描 (L-curve)
:mod:`workflows.paragram`      COMSOL 对照脚本共享配置 (读取主工作流配置)
:mod:`workflows.paragram_checkboard` 棋盘格测试的独立参数与观测网格

科学内核一律放在 :mod:`geodetic` 包中; 本包只负责串联调用。
命令行入口见项目根目录 ``run_inversion.py``。
"""
