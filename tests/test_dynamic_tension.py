# -*- coding: utf-8 -*-
"""动/静张力换算: 反算在动载方程中进行。

约定: 动态张力直接采用; 静态张力先乘 φ。φ=1.1 时,
同一 100 kN 静载, 无论输入 110 kN 动态张力还是 100 kN 静态张力,
反算静载总重都应为 400 kN(每索 100 kN)。
"""
from app.physics import inverse_cog_from_trials

PHI = 1.1
W_STATIC_TOTAL = 400.0
T_STATIC_PER_LEG = 100.0
T_DYN_PER_LEG = T_STATIC_PER_LEG * PHI


def test_static_tension_multiplied_by_phi(vertical_static_input):
    """静态 100 kN/索 -> 进入动载方程前必须先乘 φ。"""
    r = inverse_cog_from_trials(vertical_static_input)
    assert r["dynamic_factor"] == PHI
    assert r["total_weight_solved_static_kn"] == W_STATIC_TOTAL
    assert abs(r["total_weight_solved_dynamic_kn"] - W_STATIC_TOTAL * PHI) < 1e-6


def test_dynamic_tension_used_directly(vertical_dynamic_input):
    """动态 110 kN/索直接使用, 不重复乘 φ。"""
    r = inverse_cog_from_trials(vertical_dynamic_input)
    assert r["dynamic_factor"] == PHI
    assert r["total_weight_solved_static_kn"] == W_STATIC_TOTAL
    assert abs(r["total_weight_solved_dynamic_kn"] - W_STATIC_TOTAL * PHI) < 1e-6


def test_static_and_dynamic_inputs_equivalent(vertical_static_input,
                                              vertical_dynamic_input):
    """两种输入反算静载必须一致(这是本缺陷的核心验收)。"""
    rs = inverse_cog_from_trials(vertical_static_input)
    rd = inverse_cog_from_trials(vertical_dynamic_input)
    assert rs["total_weight_solved_static_kn"] == rd["total_weight_solved_static_kn"]
    assert abs(rs["total_weight_solved_static_kn"] - W_STATIC_TOTAL) < 1e-6
    assert abs(rs["total_weight_solved_dynamic_kn"]
               - rd["total_weight_solved_dynamic_kn"]) < 1e-6


def test_per_leg_implied_static_tension(vertical_static_input,
                                        vertical_dynamic_input):
    """逐索静载反推为 100 kN(= 竖向分量/φ)。"""
    for inp in (vertical_static_input, vertical_dynamic_input):
        r = inverse_cog_from_trials(inp)
        pt = r["per_trial"]["TR1"]
        # 竖直索: ΣT(动) = W_dyn; 除以索数再除 φ 得每索静载
        implied_static_per_leg = (
            pt["implied_total_weight_dynamic_kn"] / PHI / 4.0
        )
        assert abs(implied_static_per_leg - T_STATIC_PER_LEG) < 1e-6


def test_old_wrong_convention_would_fail(vertical_static_input):
    """防护: 旧错误约定(静态不乘 φ)会得到 363.6 kN, 断言不会发生。"""
    r = inverse_cog_from_trials(vertical_static_input)
    wrong_value = 4 * T_STATIC_PER_LEG / PHI  # 363.64: 旧 bug 的特征值
    assert abs(r["total_weight_solved_static_kn"] - wrong_value) > 1.0
