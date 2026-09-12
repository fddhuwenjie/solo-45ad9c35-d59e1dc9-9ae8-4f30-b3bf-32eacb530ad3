# -*- coding: utf-8 -*-
"""
吊装核算系统
============

刚体静力学受力求解 / 试吊重心反算 / 吊索-卸扣-吊梁利用率 /
调整量(吊点移位、索长垫片、配重) / SQLite 版本与审批流。

模块:
  linalg  -- 纯 Python 三维向量与线性方程组工具(无需 numpy)
  models  -- Pydantic v2 入参/出参模型
  physics -- 核心力学求解
  svg     -- 带尺寸标注的吊装示意图
  storage -- SQLite 版本库
  main    -- FastAPI 应用
"""

__version__ = "1.0.0"

# 计算摘要中的算法标识，批准版与复算 JSON 均记录，便于追溯
CALC_ENGINE_ID = f"liftcalc-core-{__version__}"
