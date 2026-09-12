# -*- coding: utf-8 -*-
"""物理回归: 对称四点矩形吊的等张力/零倾斜/利用率。

这些数值是已复核的正确结果, 任何求解器或符号改动若破坏它们须显式说明。
"""
from app.physics import run_full_analysis


def test_symmetric_tilt_is_zero(symmetric_input):
    r = run_full_analysis(symmetric_input)
    att = r["predicted_attitude"]
    assert att["pitch_deg"] == 0.0
    assert att["roll_deg"] == 0.0
    assert att["tilt_deg"] == 0.0


def test_symmetric_equal_tensions(symmetric_input):
    r = run_full_analysis(symmetric_input)
    Td = [l["tension_dynamic_kn"] for l in r["legs"]]
    Ts = [l["tension_static_kn"] for l in r["legs"]]
    # 四索等张力
    assert max(Td) - min(Td) < 1e-9
    assert max(Ts) - min(Ts) < 1e-9
    # 静态: W/(4·sinβ), β=asin(4/5.657)=44.97° 名义 47.97° 见下;
    # 直接锁定复核值
    assert abs(Td[0] - 370.23) < 0.01
    assert abs(Ts[0] - 336.573) < 0.01
    # 动/静比例 = φ
    assert abs(Td[0] / Ts[0] - 1.1) < 1e-6


def test_symmetric_vertical_force_closure(symmetric_input):
    """Σ T_i · (索方向竖向分量) = φ W。"""
    r = run_full_analysis(symmetric_input)
    W_dyn = r["weights_kn"]["lower_assembly_dynamic"]
    total_vertical = sum(
        l["tension_dynamic_kn"] * l["vertical_share"] for l in r["legs"]
    )
    assert abs(total_vertical - W_dyn) / W_dyn < 1e-3
    assert abs(W_dyn - 1100.0) < 1e-9
    assert r["weights_kn"]["hook_dynamic"] == 1100.0


def test_symmetric_equal_shares(symmetric_input):
    r = run_full_analysis(symmetric_input)
    shares = r["distribution"]["vertical_load_shares"]
    assert set(shares) == {"L1", "L2", "L3", "L4"} or len(shares) == 4
    for v in shares.values():
        assert abs(v - 0.25) < 1e-9
    assert r["distribution"]["kind"] == "statically_indeterminate_elastic"
    assert r["distribution"]["equilibrium_residual_rel_rms"] < 1e-9


def test_symmetric_utilization_within_limits(symmetric_input):
    r = run_full_analysis(symmetric_input)
    # 容量 500/600, 动载 370.23 -> 0.7405 / 0.6171, 全部不超载
    for l in r["legs"]:
        assert abs(l["sling"]["utilization"] - 0.7405) < 1e-3
        assert abs(l["shackle"]["utilization"] - 0.6171) < 1e-3
        assert l["sling"]["safety_factor"] > 1.0
    assert r["utilization_summary"]["max_sling_utilization"] < 1.0
    assert r["utilization_summary"]["max_shackle_utilization"] < 1.0
    assert abs(r["hoist"]["utilization"] - 0.55) < 1e-9
    # 等张力工况不应有任何力学冲突(无实测重心的证据缺口除外)
    assert r["blocking_codes"] == []
