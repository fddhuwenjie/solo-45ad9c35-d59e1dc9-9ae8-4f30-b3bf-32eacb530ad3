# -*- coding: utf-8 -*-
"""pytest 公共夹具。"""
import os
import sys

# 确保无需设置 PYTHONPATH 也能导入 app
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, ".pylibs", "lib", "python3.11", "site-packages")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import pytest  # noqa: E402

from app.models import LegIn, LiftInput, MeasurementIn, TrialIn  # noqa: E402

# 标准四点矩形吊点(单位 m)
PAD_RECT = [(-3.0, -2.0, 1.0), (-3.0, 2.0, 1.0),
            (3.0, -2.0, 1.0), (3.0, 2.0, 1.0)]


@pytest.fixture
def four_legs():
    return [LegIn(id=f"L{i+1}", pad=p, sling_capacity_kn=500.0,
                  shackle_capacity_kn=600.0, sling_ea_kn=50000.0)
            for i, p in enumerate(PAD_RECT)]


@pytest.fixture
def symmetric_input(four_legs):
    """对称四点吊, 重心在吊点形心正下方, 吊钩 (0,0,5)。"""
    return LiftInput(
        name="symmetric-4pt", component_weight_kn=1000.0,
        cog_theory=(0.0, 0.0, 1.0), legs=four_legs,
        hook_point=(0.0, 0.0, 5.0), lift_acceleration_mps2=0.1,
        min_dynamic_factor=1.1, hoist_capacity_kn=2000.0,
    )


@pytest.fixture
def vertical_four_legs():
    """4 根竖直索的等价几何(用极高吊钩使 sinβ≈1)。"""
    return [LegIn(id=f"L{i+1}", pad=p, sling_capacity_kn=500.0,
                  shackle_capacity_kn=600.0)
            for i, p in enumerate([(-1.0, -1.0, 0.0), (-1.0, 1.0, 0.0),
                                   (1.0, -1.0, 0.0), (1.0, 1.0, 0.0)])]


def make_vertical_input(vertical_four_legs, tensions, tension_is_static):
    """每索静载 100 kN 的竖直四点吊, φ=1.1。"""
    tr = TrialIn(id="TR1", measurements=[
        MeasurementIn(leg_id=f"L{i+1}", tension_kn=tensions[i],
                      tension_is_static=tension_is_static)
        for i in range(4)
    ])
    return LiftInput(
        name="vertical-static-equivalence", component_weight_kn=400.0,
        cog_theory=(0.0, 0.0, 0.0), legs=vertical_four_legs,
        hook_point=(0.0, 0.0, 1.0e6), lift_acceleration_mps2=0.981,
        min_dynamic_factor=1.1, trials=[tr], request_adjustment=False,
    )


@pytest.fixture
def vertical_dynamic_input(vertical_four_legs):
    """动态张力 110 kN/索(φ=1.1 已含)。"""
    return make_vertical_input(vertical_four_legs, [110.0] * 4, False)


@pytest.fixture
def vertical_static_input(vertical_four_legs):
    """静态张力 100 kN/索(系统须先乘 φ)。"""
    return make_vertical_input(vertical_four_legs, [100.0] * 4, True)
