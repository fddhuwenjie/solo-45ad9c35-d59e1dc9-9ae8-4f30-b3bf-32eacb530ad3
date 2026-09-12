# -*- coding: utf-8 -*-
"""起重机工况校核: 载荷表插值、路径采样、支腿反力、违规定位与 API/版本流。

标准算例(已手工复核):
  部件 carrier 340 kN@(0,0) 不回转, super 260 kN@(-0.5,0) 回转,
  配重 400 kN@(-4.5,0) 回转, 吊钩滑轮组 10 kN, 吊钩动载 1100 kN。
  支腿 (±4,±4), 垫板 2x2 m, 总重 W = 2110 kN, 每腿基准 527.5 kN。
  Σ(wx) = 1110·r − 1930 -> R(x=+4) = 527.5 + Σ(wx)/16, R(x=−4) = 527.5 − Σ(wx)/16。
"""
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from app.crane import _chart_index, chart_capacity, check_crane_duty
from app.models import CraneSetupIn, LoadChartRowIn

HOOK_DYN = 1100.0  # 1000 kN 构件 × φ=1.1


def make_setup(**over):
    """全通过工况的标准起重机输入。"""
    data = {
        "crane_id": "CR1",
        "slewing_center": [0, 0],
        "boom_length_m": 20.0,
        "components": [
            {"id": "carrier", "weight_kn": 340.0, "cog": [0, 0, 1.2],
             "slews": False},
            {"id": "super", "weight_kn": 260.0, "cog": [-0.5, 0, 3.0]},
        ],
        "counterweight": {"weight_kn": 400.0, "cog": [-4.5, 0, 3.8]},
        "hook_block_weight_kn": 10.0,
        "outriggers": [
            {"id": "O1", "position": [4, 4], "mat_length_m": 2, "mat_width_m": 2},
            {"id": "O2", "position": [4, -4], "mat_length_m": 2, "mat_width_m": 2},
            {"id": "O3", "position": [-4, -4], "mat_length_m": 2, "mat_width_m": 2},
            {"id": "O4", "position": [-4, 4], "mat_length_m": 2, "mat_width_m": 2},
        ],
        "ground_bearing_limit_kpa": 300.0,
        "load_chart": [
            {"boom_length_m": 20, "zone": "360", "radius_m": 4,
             "capacity_kn": 1600},
            {"boom_length_m": 20, "zone": "360", "radius_m": 10,
             "capacity_kn": 1200},
            {"boom_length_m": 20, "zone": "side", "radius_m": 4,
             "capacity_kn": 1300},
            {"boom_length_m": 20, "zone": "side", "radius_m": 10,
             "capacity_kn": 1000},
        ],
        "path": [
            {"id": "P0", "position": [5, 0, 18]},
            {"id": "P1", "position": [8, 0, 18]},
        ],
        "path_step_m": 1.0,
    }
    data.update(over)
    return CraneSetupIn.model_validate(data)


def lift_payload(crane=None, **kw):
    p = {
        "name": "crane-duty", "component_weight_kn": 1000.0,
        "cog_theory": [0, 0, 1.0], "cog_actual_override": [0, 0, 1.0],
        "legs": [
            {"id": f"L{i+1}", "pad": list(p_),
             "sling_capacity_kn": 500.0, "shackle_capacity_kn": 600.0,
             "sling_ea_kn": 50000.0}
            for i, p_ in enumerate([(-3, -2, 1), (-3, 2, 1), (3, -2, 1), (3, 2, 1)])
        ],
        "hook_point": [0, 0, 5], "lift_acceleration_mps2": 0.1,
        "min_dynamic_factor": 1.1, "hoist_capacity_kn": 2000.0,
        "request_adjustment": False,
    }
    if crane is not None:
        p["crane"] = crane
    p.update(kw)
    return p


@pytest.fixture
def client_no_db(monkeypatch):
    monkeypatch.delenv("LIFTCALC_DB", raising=False)
    from app import main as main_mod
    main_mod._storage = None
    main_mod.DB_PATH = ""
    return TestClient(main_mod.app)


@pytest.fixture
def client_db(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    monkeypatch.setenv("LIFTCALC_DB", path)
    from app import main as main_mod
    main_mod.DB_PATH = path
    main_mod._storage = None
    yield TestClient(main_mod.app)
    main_mod.get_storage().close()
    os.unlink(path)


# ---------------------------------------------------------------- 载荷表
def test_chart_interpolation_within_adjacent_steps():
    idx = _chart_index([
        LoadChartRowIn(boom_length_m=20, zone="360", radius_m=4, capacity_kn=1600),
        LoadChartRowIn(boom_length_m=20, zone="360", radius_m=10, capacity_kn=1200),
    ])
    assert chart_capacity(idx, 20, "360", 4) == 1600.0
    assert chart_capacity(idx, 20, "360", 10) == 1200.0
    # 相邻档位间线性插值: 1600 - 400*(3/6)
    assert chart_capacity(idx, 20, "360", 7) == pytest.approx(1400.0)


def test_chart_never_extrapolates():
    idx = _chart_index([
        LoadChartRowIn(boom_length_m=20, zone="360", radius_m=4, capacity_kn=1600),
        LoadChartRowIn(boom_length_m=20, zone="360", radius_m=10, capacity_kn=1200),
    ])
    assert chart_capacity(idx, 20, "360", 3.999) is None   # 小于最小半径
    assert chart_capacity(idx, 20, "360", 10.001) is None  # 大于最大半径
    assert chart_capacity(idx, 20, "side", 7) is None      # 无该区段
    assert chart_capacity(idx, 25, "360", 7) is None       # 无该臂长


def test_chart_single_row_exact_match_only():
    idx = _chart_index([
        LoadChartRowIn(boom_length_m=20, zone="360", radius_m=6, capacity_kn=1000),
    ])
    assert chart_capacity(idx, 20, "360", 6) == 1000.0
    assert chart_capacity(idx, 20, "360", 6.1) is None


# ---------------------------------------------------------------- 路径采样
def test_path_sampling_by_step():
    r = check_crane_duty(make_setup(path_step_m=0.5), HOOK_DYN)
    radii = [s["radius_m"] for s in r["samples"]]
    assert radii == pytest.approx([5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0])
    assert r["samples"][0]["at_pose"] == "P0"
    assert r["samples"][-1]["at_pose"] == "P1"
    assert all(s["segment"] == "P0->P1" for s in r["samples"])
    assert r["path_summary"]["total_length_m"] == pytest.approx(3.0)


# ---------------------------------------------------------------- 全通过工况
def test_all_ok_duty_values():
    r = check_crane_duty(make_setup(), HOOK_DYN)
    assert r["status"] == "ok"
    assert r["conflicts"] == []
    assert r["first_violation"] is None
    s0, s8 = r["samples"][0], r["samples"][-1]
    # r=5: 毛能力 1600-400/6, 净额定再减吊钩滑轮组 10 kN
    assert s0["chart_gross_kn"] == pytest.approx(1533.333, abs=1e-3)
    assert s0["net_rated_kn"] == pytest.approx(1523.333, abs=1e-3)
    assert s0["load_utilization"] == pytest.approx(0.7221, abs=1e-3)
    # r=5: R(x=+4)=753.75 kN, p=188.44 kPa
    assert s0["outriggers"][0]["reaction_kn"] == pytest.approx(753.75)
    assert s0["outriggers"][0]["pressure_kpa"] == pytest.approx(188.44, abs=1e-2)
    # r=8: 利用率 1100/1323.33, R(x=-4)=93.125 kN
    assert s8["load_utilization"] == pytest.approx(0.8312, abs=1e-3)
    assert s8["outriggers"][2]["reaction_kn"] == pytest.approx(93.125)
    env = r["envelope"]
    assert env["max_load_utilization"] == pytest.approx(0.8312, abs=1e-3)
    assert env["max_ground_pressure_kpa"] == pytest.approx(240.47, abs=1e-2)
    assert env["min_outrigger_reaction_kn"] == pytest.approx(93.125)
    assert env["max_radius_m"] == pytest.approx(8.0)


def test_outrigger_equilibrium_closure():
    """每个采样点: ΣR = 总重 2110 kN, ΣR·x = Σ(wx) = 1110r − 1930。"""
    r = check_crane_duty(make_setup(), HOOK_DYN)
    for s in r["samples"]:
        rs = [o["reaction_kn"] for o in s["outriggers"]]
        assert sum(rs) == pytest.approx(2110.0, abs=1e-6)
        mx = sum(rr * x for rr, x in zip(rs, [4, 4, -4, -4]))
        assert mx == pytest.approx(1110.0 * s["radius_m"] - 1930.0, abs=1e-6)


def test_slew_rotation_moves_counterweight_side():
    """吊钩沿 +y 时回转 90°: 配重力臂转到 -y, 反力大的一侧换到 y=+4 支腿。"""
    setup = make_setup(path=[{"id": "P0", "position": [0, 5, 18]},
                             {"id": "P1", "position": [0, 8, 18]}])
    r = check_crane_duty(setup, HOOK_DYN)
    s8 = r["samples"][-1]
    assert s8["slew_angle_deg"] == pytest.approx(90.0)
    by_id = {o["id"]: o["reaction_kn"] for o in s8["outriggers"]}
    assert by_id["O1"] == pytest.approx(961.875)  # y=+4
    assert by_id["O4"] == pytest.approx(961.875)  # y=+4
    assert by_id["O2"] == pytest.approx(93.125)   # y=-4
    assert by_id["O3"] == pytest.approx(93.125)


# ---------------------------------------------------------------- 违规定位
def test_chart_overload_located():
    setup = make_setup(
        load_chart=[
            {"boom_length_m": 20, "zone": "360", "radius_m": 4,
             "capacity_kn": 1500},
            {"boom_length_m": 20, "zone": "360", "radius_m": 10,
             "capacity_kn": 1100},
        ],
        path=[{"id": "P0", "position": [5, 0, 18]},
              {"id": "P1", "position": [10, 0, 18]}],
    )
    r = check_crane_duty(setup, HOOK_DYN)
    codes = [c["code"] for c in r["conflicts"]]
    assert "CRANE_CHART_OVERLOAD" in codes
    # r=10 处净额定 1090 < 1100; 首个超载采样点 r=10
    fv = r["first_violation"]
    assert fv["kind"] == "CHART_OVERLOAD"
    assert fv["radius_m"] == pytest.approx(10.0)
    assert fv["interval"]["from"][0] == pytest.approx(9.0)


def test_outrigger_uplift_located():
    setup = make_setup(
        ground_bearing_limit_kpa=400.0,  # 提高地基限值, 隔离出拔起唯一违规
        load_chart=[
            {"boom_length_m": 20, "zone": "360", "radius_m": 4,
             "capacity_kn": 1600},
            {"boom_length_m": 20, "zone": "360", "radius_m": 12,
             "capacity_kn": 1200},
        ],
        path=[{"id": "P0", "position": [5, 0, 18]},
              {"id": "P1", "position": [12, 0, 18]}],
    )
    r = check_crane_duty(setup, HOOK_DYN)
    codes = [c["code"] for c in r["conflicts"]]
    assert codes == ["CRANE_OUTRIGGER_UPLIFT"]
    # 拔起零界 r = (527.5*16+1930)/1110 = 9.34 m; 采样点上首现于 r=10
    fv = r["first_violation"]
    assert fv["kind"] == "OUTRIGGER_UPLIFT"
    assert fv["radius_m"] == pytest.approx(10.0)
    assert r["envelope"]["min_outrigger_reaction_kn"] < 0


def test_ground_overpressure_located():
    setup = make_setup(ground_bearing_limit_kpa=200.0)
    r = check_crane_duty(setup, HOOK_DYN)
    codes = [c["code"] for c in r["conflicts"]]
    assert codes == ["CRANE_GROUND_OVERPRESSURE"]
    # r=6 处 p = 823.125/4 = 205.8 kPa 首超限
    fv = r["first_violation"]
    assert fv["kind"] == "GROUND_OVERPRESSURE"
    assert fv["radius_m"] == pytest.approx(6.0)


def test_out_of_chart_beyond_radius():
    setup = make_setup(
        path=[{"id": "P0", "position": [5, 0, 18]},
              {"id": "P1", "position": [10.5, 0, 18]}],
    )
    r = check_crane_duty(setup, 550.0)  # 轻载, 隔离出越表唯一违规
    codes = [c["code"] for c in r["conflicts"]]
    assert codes == ["CRANE_OUT_OF_CHART"]
    last = r["samples"][-1]
    assert last["radius_m"] == pytest.approx(10.5)
    assert last["chart_gross_kn"] is None
    assert last["net_rated_kn"] is None
    assert last["violations"] == ["OUT_OF_CHART"]


def test_zone_selection_and_unknown_zone():
    # 姿态指定 side 区段 -> 用 side 档(能力更低)
    setup = make_setup(path=[{"id": "P0", "position": [5, 0, 18], "zone": "side"},
                             {"id": "P1", "position": [8, 0, 18], "zone": "side"}])
    r = check_crane_duty(setup, HOOK_DYN)
    assert r["samples"][0]["zone"] == "side"
    assert r["samples"][0]["chart_gross_kn"] == pytest.approx(1250.0)
    # 未知区段 -> 越出载荷表
    setup2 = make_setup(path=[{"id": "P0", "position": [5, 0, 18], "zone": "front"},
                              {"id": "P1", "position": [8, 0, 18], "zone": "front"}])
    r2 = check_crane_duty(setup2, HOOK_DYN)
    assert [c["code"] for c in r2["conflicts"]] == ["CRANE_OUT_OF_CHART"]


def test_chart_summary_fingerprint_frozen():
    r = check_crane_duty(make_setup(), HOOK_DYN)
    cs = r["load_chart_summary"]
    assert cs["boom_length_m"] == 20.0
    assert cs["zones"] == ["360", "side"]
    assert cs["radius_range_m"] == [4.0, 10.0]
    assert cs["entries_total"] == 4
    assert cs["fingerprint"].startswith("sha256:")
    # 与行书写顺序无关
    rows = make_setup().load_chart
    idx_a = _chart_index(rows)
    idx_b = _chart_index(list(reversed(rows)))
    assert idx_a == idx_b


# ---------------------------------------------------------------- 配置隔离
def dual_config_chart():
    """同一臂长、同一 side 区段下的 A/B 双配置载荷表。"""
    return [
        {"boom_length_m": 20, "config": "A", "zone": "side", "radius_m": 4,
         "capacity_kn": 1600},
        {"boom_length_m": 20, "config": "A", "zone": "side", "radius_m": 10,
         "capacity_kn": 1200},
        {"boom_length_m": 20, "config": "B", "zone": "side", "radius_m": 4,
         "capacity_kn": 900},
        {"boom_length_m": 20, "config": "B", "zone": "side", "radius_m": 10,
         "capacity_kn": 700},
    ]


def test_chart_capacity_interpolates_within_same_config():
    idx = _chart_index([LoadChartRowIn(**r) for r in dual_config_chart()])
    # 同配置相邻档位内各自插值, 互不影响
    assert chart_capacity(idx, 20, "side", 7, config="A") == pytest.approx(1400.0)
    assert chart_capacity(idx, 20, "side", 7, config="B") == pytest.approx(800.0)


def test_no_cross_config_interpolation():
    """缺陷复现: A 只有 4 m 档、B 只有 10 m 档, 7 m 处不得跨配置插值出 1400 kN。"""
    idx = _chart_index([
        LoadChartRowIn(boom_length_m=20, config="A", zone="side", radius_m=4,
                       capacity_kn=1600),
        LoadChartRowIn(boom_length_m=20, config="B", zone="side", radius_m=10,
                       capacity_kn=1200),
    ])
    assert chart_capacity(idx, 20, "side", 7, config="A") is None
    assert chart_capacity(idx, 20, "side", 7, config="B") is None
    # 同配置单档精确匹配仍可用; 跨配置精确到对方半径也不可用
    assert chart_capacity(idx, 20, "side", 4, config="A") == 1600.0
    assert chart_capacity(idx, 20, "side", 10, config="B") == 1200.0
    assert chart_capacity(idx, 20, "side", 4, config="B") is None
    assert chart_capacity(idx, 20, "side", 10, config="A") is None


def test_duty_check_out_of_chart_when_config_lacks_adjacent_rows():
    """同配置无合法相邻档位 -> 判越出载荷表; 切到有档位的配置则正常。"""
    chart = [
        {"boom_length_m": 20, "config": "A", "zone": "side", "radius_m": 4,
         "capacity_kn": 1600},
        {"boom_length_m": 20, "config": "B", "zone": "side", "radius_m": 4,
         "capacity_kn": 1600},
        {"boom_length_m": 20, "config": "B", "zone": "side", "radius_m": 10,
         "capacity_kn": 1200},
    ]
    r_a = check_crane_duty(make_setup(config="A", default_zone="side",
                                      load_chart=chart), HOOK_DYN)
    assert [c["code"] for c in r_a["conflicts"]] == ["CRANE_OUT_OF_CHART"]
    assert all(s["chart_gross_kn"] is None for s in r_a["samples"])

    r_b = check_crane_duty(make_setup(config="B", default_zone="side",
                                      load_chart=chart), HOOK_DYN)
    assert r_b["status"] == "ok"
    assert r_b["samples"][0]["chart_gross_kn"] == pytest.approx(1533.333, abs=1e-3)


def test_api_dual_config_no_cross_interpolation(client_no_db):
    """API 回归: 双配置同区段, 当前配置无相邻档位的半径不得出现插值能力。"""
    crane = make_setup(
        config="A", default_zone="side",
        load_chart=[
            {"boom_length_m": 20, "config": "A", "zone": "side", "radius_m": 4,
             "capacity_kn": 1600},
            {"boom_length_m": 20, "config": "B", "zone": "side", "radius_m": 4,
             "capacity_kn": 1600},
            {"boom_length_m": 20, "config": "B", "zone": "side", "radius_m": 10,
             "capacity_kn": 1200},
        ],
    ).model_dump()
    r = client_no_db.post("/api/lifts/crane-check", json=lift_payload(crane))
    assert r.status_code == 200, r.text
    body = r.json()["crane"]
    assert body["config"] == "A"
    assert body["load_chart_summary"]["configs"] == ["A", "B"]
    assert [c["code"] for c in body["conflicts"]] == ["CRANE_OUT_OF_CHART"]
    gross_values = [s["chart_gross_kn"] for s in body["samples"]]
    assert all(g is None for g in gross_values)
    assert 1400.0 not in gross_values  # 旧缺陷在 7 m 处跨配置插值的特征值

    b = client_no_db.post("/api/lifts/analyze", json=lift_payload(crane)).json()
    assert b["approvable"] is False
    assert "CRANE_OUT_OF_CHART" in b["blocking_codes"]


def test_single_config_default_unchanged():
    """既有单配置输入(无 config 字段)默认 STD, 结果与修复前一致。"""
    setup = make_setup()
    assert setup.config == "STD"
    assert all(row.config == "STD" for row in setup.load_chart)
    r = check_crane_duty(setup, HOOK_DYN)
    assert r["status"] == "ok"
    assert r["config"] == "STD"
    assert r["load_chart_summary"]["configs"] == ["STD"]
    assert r["envelope"]["max_load_utilization"] == pytest.approx(0.8312, abs=1e-3)
    assert r["samples"][0]["chart_gross_kn"] == pytest.approx(1533.333, abs=1e-3)
    # 不带 config 实参的旧式查询等价于 STD 配置
    idx = _chart_index(setup.load_chart)
    assert chart_capacity(idx, 20, "360", 7) == pytest.approx(1400.0)


def test_chart_uniqueness_includes_config(client_no_db):
    # 同臂长同区段同半径、不同配置: 合法(双配置并存)
    ok = make_setup(load_chart=[
        {"boom_length_m": 20, "config": "A", "zone": "360", "radius_m": 4,
         "capacity_kn": 1600},
        {"boom_length_m": 20, "config": "B", "zone": "360", "radius_m": 4,
         "capacity_kn": 1500},
        {"boom_length_m": 20, "config": "A", "zone": "360", "radius_m": 10,
         "capacity_kn": 1200},
        {"boom_length_m": 20, "config": "B", "zone": "360", "radius_m": 10,
         "capacity_kn": 1100},
    ])
    assert len(ok.load_chart) == 4
    # 同臂长同配置同区段同半径: 重复 -> 422
    dup = make_setup().model_dump()
    dup["load_chart"] = [
        {"boom_length_m": 20, "config": "A", "zone": "360", "radius_m": 4,
         "capacity_kn": 1600},
        {"boom_length_m": 20, "config": "A", "zone": "360", "radius_m": 4,
         "capacity_kn": 1500},
    ]
    r = client_no_db.post("/api/lifts/analyze", json=lift_payload(dup))
    assert r.status_code == 422 and "载荷表存在重复" in r.text


def test_chart_fingerprint_covers_config():
    """配置标识进入载荷表指纹: 仅配置名不同, 指纹即不同(批准版可区分)。"""
    def fp(cfg):
        s = make_setup(load_chart=[
            {"boom_length_m": 20, "config": cfg, "zone": "360", "radius_m": 4,
             "capacity_kn": 1600}])
        return check_crane_duty(s, HOOK_DYN)["load_chart_summary"]["fingerprint"]
    assert fp("A") != fp("B")


# ---------------------------------------------------------------- API 集成
def test_analyze_includes_crane_and_approvable(client_no_db):
    r = client_no_db.post("/api/lifts/analyze",
                          json=lift_payload(make_setup().model_dump()))
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["crane"]["status"] == "ok"
    assert b["weights_kn"]["hook_dynamic"] == 1100.0
    assert b["crane"]["lifted_load_dynamic_kn"] == 1100.0
    # 已核实重心 + 无冲突 -> 可批准; 起重机包络进入利用率摘要
    assert b["approvable"] is True
    assert b["utilization_summary"]["crane_max_load_utilization"] == \
        pytest.approx(0.8312, abs=1e-3)


def test_analyze_crane_violation_blocks_approval(client_no_db):
    crane = make_setup(ground_bearing_limit_kpa=200.0).model_dump()
    r = client_no_db.post("/api/lifts/analyze", json=lift_payload(crane))
    b = r.json()
    assert b["approvable"] is False
    assert "CRANE_GROUND_OVERPRESSURE" in b["blocking_codes"]


def test_crane_check_endpoint_shares_samples_with_svg(client_no_db):
    payload = lift_payload(make_setup().model_dump())
    r = client_no_db.post("/api/lifts/crane-check", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["hook_load_dynamic_kn"] == 1100.0
    n = len(body["crane"]["samples"])
    assert n == 4  # 步长 1 m, 半径 5..8

    svg = client_no_db.post("/api/lifts/crane-station.svg", json=payload)
    assert svg.status_code == 200
    assert svg.headers["content-type"].startswith("image/svg+xml")
    text = svg.text
    assert text.startswith("<svg")
    assert "P0" in text and "P1" in text
    assert "回转中心" in text
    assert f"采样 {n} 点" in text  # JSON 与 SVG 共用逐姿态数据


def test_crane_endpoints_require_crane_field(client_no_db):
    r = client_no_db.post("/api/lifts/crane-check", json=lift_payload())
    assert r.status_code == 400
    r = client_no_db.post("/api/lifts/crane-station.svg", json=lift_payload())
    assert r.status_code == 400


def test_crane_input_validation(client_no_db):
    base = make_setup().model_dump()

    dup = make_setup().model_dump()
    dup["outriggers"][1]["id"] = "O1"
    r = client_no_db.post("/api/lifts/analyze", json=lift_payload(dup))
    assert r.status_code == 422 and "支腿 id 重复" in r.text

    col = make_setup().model_dump()
    col["outriggers"] = [
        {"id": "O1", "position": [0, 0], "mat_length_m": 2, "mat_width_m": 2},
        {"id": "O2", "position": [4, 0], "mat_length_m": 2, "mat_width_m": 2},
        {"id": "O3", "position": [8, 0], "mat_length_m": 2, "mat_width_m": 2},
    ]
    r = client_no_db.post("/api/lifts/analyze", json=lift_payload(col))
    assert r.status_code == 422 and "支腿布置共线" in r.text

    drow = make_setup().model_dump()
    drow["load_chart"].append(dict(drow["load_chart"][0]))
    r = client_no_db.post("/api/lifts/analyze", json=lift_payload(drow))
    assert r.status_code == 422 and "载荷表存在重复" in r.text

    one = make_setup().model_dump()
    one["path"] = one["path"][:1]
    r = client_no_db.post("/api/lifts/analyze", json=lift_payload(one))
    assert r.status_code == 422
    _ = base


# ---------------------------------------------------------------- 版本库集成
def test_revision_flow_freezes_chart_and_diffs_crane(client_db):
    # r1: 全通过工况, 可批准并锁定(载荷表摘要与路径随版本冻结)
    p1 = lift_payload(make_setup().model_dump())
    r1 = client_db.post("/api/plans/CR1/revisions", json=p1)
    assert r1.status_code == 201, r1.text
    rev1 = r1.json()
    assert rev1["approvable"] is True
    assert rev1["summary"]["crane"]["load_chart_summary"]["fingerprint"]
    assert rev1["summary"]["crane"]["path_summary"]["poses"] == 2

    ap = client_db.post("/api/plans/CR1/revisions/1/review",
                        json={"reviewer": "li", "action": "approve"})
    assert ap.status_code == 200 and ap.json()["status"] == "approved"

    # 工况变化必须从批准版派生: 路径延长到 r=12, 载荷表加 12 m 档
    crane2 = make_setup(
        load_chart=[
            {"boom_length_m": 20, "zone": "360", "radius_m": 4,
             "capacity_kn": 1600},
            {"boom_length_m": 20, "zone": "360", "radius_m": 10,
             "capacity_kn": 1200},
            {"boom_length_m": 20, "zone": "360", "radius_m": 12,
             "capacity_kn": 1200},
            {"boom_length_m": 20, "zone": "side", "radius_m": 4,
             "capacity_kn": 1300},
            {"boom_length_m": 20, "zone": "side", "radius_m": 10,
             "capacity_kn": 1000},
        ],
        path=[{"id": "P0", "position": [5, 0, 18]},
              {"id": "P1", "position": [12, 0, 18]}],
    ).model_dump()
    dr = client_db.post("/api/plans/CR1/derive",
                        json={"from_revision": 1, "change_note": "安装点半径加大",
                              "input": lift_payload(crane2)})
    assert dr.status_code == 201, dr.text
    rev2 = dr.json()
    assert rev2["derived_from"] == 1
    assert rev2["approvable"] is False  # 派生版支腿拔起

    # 版本差异包含起重机工况字段
    diff = client_db.get("/api/plans/CR1/diff/1/2").json()
    changed = [c["field"] for c in diff["summary_changes"]]
    assert "crane.envelope.min_outrigger_reaction_kn" in changed
    assert "crane.first_violation.kind" in changed
    assert "crane.load_chart_summary.fingerprint" in changed
    assert "CRANE_OUTRIGGER_UPLIFT" in diff["conflicts_added"]
    assert diff["approvable"] == {"from": True, "to": False}

    # 带病版本不能批准
    ap2 = client_db.post("/api/plans/CR1/revisions/2/review",
                         json={"reviewer": "li", "action": "approve"})
    assert ap2.status_code == 409
