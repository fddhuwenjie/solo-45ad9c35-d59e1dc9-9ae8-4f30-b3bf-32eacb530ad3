# -*- coding: utf-8 -*-
"""路径净空校核: 分离轴法、姿态插值、保守净空、证据缺口与 API/版本流。

标准几何(已手工复核):
  路径 P0(5,0,18) -> P1(9,0,18), 步长 1 m, 采样 x=5..9;
  包络 4x2x2 (半长 2/1/1), 吊钩至中心偏移 (0,0,-5) -> 中心 z=13;
  障碍物 OB1 x[8,9] y[-2,2] z[10,14]:
    x=5 净空 1.0, x=6 净空 0.0, x>=7 相交; 扫掠修正 0.5(步长 1 m 的一半),
    保守净空 x=5 -> 0.5(恰等于安全间距, 不违规), x=6 -> -0.5(首个碰撞点)。
"""
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from app.clearance import (
    geodesic_deg,
    rot_axes,
    rot_matrix,
    rot_vec,
    sat_clearance,
    unwrap_attitudes,
)
from app.crane import check_crane_duty
from test_crane_duty import HOOK_DYN, lift_payload, make_setup

# ---------------------------------------------------------------- 标准输入
OB1 = {"id": "OB1", "min_corner": [8, -2, 10], "max_corner": [9, 2, 14]}
EX1 = {"id": "EX1", "min_corner": [8, -2, 10], "max_corner": [9, 2, 14]}


def make_clearance_setup(**clr_over):
    """路径 x=5..9、带标准障碍物 OB1 的净空校核工况。"""
    clr = {
        "envelope": {"length_m": 4, "width_m": 2, "height_m": 2},
        "hook_to_center_offset": [0, 0, -5],
        "obstacles": [dict(OB1)],
        "exclusion_zones": [],
        "safety_margin_m": 0.5,
        "attitude_step_deg": 10.0,
    }
    clr.update(clr_over)
    return make_setup(
        path=[{"id": "P0", "position": [5, 0, 18]},
              {"id": "P1", "position": [9, 0, 18]}],
        clearance=clr,
    )


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


# ---------------------------------------------------------------- 分离轴法
def test_sat_axis_aligned_separated_touching_overlapping():
    axes = rot_axes(rot_matrix(0, 0, 0))
    half = (0.5, 0.5, 0.5)
    assert sat_clearance((0, 0, 0), axes, half, (2, -1, -1), (3, 1, 1)) == \
        pytest.approx(1.5)
    assert sat_clearance((0, 0, 0), axes, half, (0.5, -1, -1), (3, 1, 1)) == \
        pytest.approx(0.0)
    assert sat_clearance((0, 0, 0), axes, half, (0.4, -1, -1), (2, 1, 1)) == \
        pytest.approx(-0.1)


def test_sat_rotated_box_uses_cross_axes():
    """绕 z 转 45° 的单位盒(半长1): x 向半径 √2, 与 x>=1.5 的盒间距 1.5-√2。"""
    axes = rot_axes(rot_matrix(45, 0, 0))
    c = sat_clearance((0, 0, 0), axes, (1, 1, 1), (1.5, -5, -5), (3, 5, 5))
    assert c == pytest.approx(1.5 - 2 ** 0.5)


def test_rot_matrix_yaw90_maps_body_x_to_world_y():
    R = rot_matrix(90, 0, 0)
    assert rot_vec(R, (1, 0, 0)) == pytest.approx((0, 1, 0))
    assert rot_vec(R, (0, 0, -5)) == pytest.approx((0, 0, -5))
    assert geodesic_deg(rot_matrix(0, 0, 0), R) == pytest.approx(90.0)


def test_unwrap_attitudes_takes_shortest_branch():
    out = unwrap_attitudes([(350, 0, 0), (10, 0, 0)])
    assert out[1][0] == pytest.approx(370.0)  # 经 360° 而非倒退 340°


# ---------------------------------------------------------------- 姿态插值与采样
def test_rotation_drives_subdivision():
    """位置只需 4 段, 转角 90°/步长 10° 需 9 段 -> 10 个采样点, 姿态线性。"""
    setup = make_setup(
        path=[{"id": "P0", "position": [5, 0, 18], "yaw_deg": 0},
              {"id": "P1", "position": [9, 0, 18], "yaw_deg": 90}],
        clearance={
            "envelope": {"length_m": 4, "width_m": 2, "height_m": 2},
            "hook_to_center_offset": [0, 0, -5],
            "obstacles": [], "attitude_step_deg": 10.0,
        },
    )
    r = check_crane_duty(setup, HOOK_DYN)
    assert len(r["samples"]) == 10
    yaws = [s["attitude_deg"]["yaw"] for s in r["samples"]]
    assert yaws == pytest.approx([10 * k for k in range(10)], abs=1e-6)
    assert r["clearance"]["sampling"]["required_samples"] == 10


def test_attitude_unwrap_interpolates_through_360():
    setup = make_setup(
        path=[{"id": "P0", "position": [5, 0, 18], "yaw_deg": 350},
              {"id": "P1", "position": [9, 0, 18], "yaw_deg": 10}],
        clearance={
            "envelope": {"length_m": 4, "width_m": 2, "height_m": 2},
            "hook_to_center_offset": [0, 0, -5],
            "obstacles": [], "attitude_step_deg": 10.0,
        },
    )
    r = check_crane_duty(setup, HOOK_DYN)
    yaws = [s["attitude_deg"]["yaw"] for s in r["samples"]]
    # 位置需 4 段(主导), 姿态经 360° 展开插值, 显示回绕到 (-180,180]:
    # 350 -> 355 -> 0 -> 5 -> 10 即 -10 -> -5 -> 0 -> 5 -> 10
    assert yaws == pytest.approx([-10.0, -5.0, 0.0, 5.0, 10.0], abs=1e-6)


def test_hook_offset_rotates_with_attitude():
    """yaw=90° 时偏移 (1,0,-5) 转到世界 (0,1,-5): 包络中心在吊钩 +y 侧。"""
    setup = make_setup(
        path=[{"id": "P0", "position": [5, 0, 18], "yaw_deg": 0},
              {"id": "P1", "position": [9, 0, 18], "yaw_deg": 90}],
        clearance={
            "envelope": {"length_m": 4, "width_m": 2, "height_m": 2},
            "hook_to_center_offset": [1, 0, -5],
            "obstacles": [], "attitude_step_deg": 30.0,
        },
    )
    r = check_crane_duty(setup, HOOK_DYN)
    c0 = r["samples"][0]["clearance"]["envelope_center"]
    c1 = r["samples"][-1]["clearance"]["envelope_center"]
    assert c0 == pytest.approx([6.0, 0.0, 13.0])
    assert c1 == pytest.approx([9.0, 1.0, 13.0])


# ---------------------------------------------------------------- 碰撞/间距/侵入
def test_collision_located_and_conservative():
    r = check_crane_duty(make_clearance_setup(), HOOK_DYN)
    codes = [c["code"] for c in r["conflicts"]]
    assert "CRANE_COLLISION" in codes
    assert "CRANE_CLEARANCE_INSUFFICIENT" not in codes  # 碰撞优先, 不重复记间距
    clr = r["clearance"]
    assert clr["status"] == "violation"
    # 逐点数值: 净空 1.0/0.0/-1.0/-2.0/-2.0, 保守再减 0.5
    raw = [s["clearance"]["min_clearance_m"] for s in r["samples"]]
    cons = [s["clearance"]["conservative_clearance_m"] for s in r["samples"]]
    assert raw == pytest.approx([1.0, 0.0, -1.0, -2.0, -2.0])
    assert cons == pytest.approx([0.5, -0.5, -1.5, -2.5, -2.5])
    assert all(c <= r_ for c, r_ in zip(cons, raw))
    # 首个碰撞区间: x=5 -> x=6
    fv = clr["first_violation"]
    assert fv["kind"] == "COLLISION"
    assert fv["interval"]["from"][0] == pytest.approx(5.0)
    assert fv["interval"]["to"][0] == pytest.approx(6.0)
    assert fv["worst_obstacle_id"] == "OB1"
    assert clr["min_conservative_clearance_m"] == pytest.approx(-2.5)
    assert clr["worst_obstacle_id"] == "OB1"
    assert clr["samples_with_violations"] == 4
    # 起重机总首个违规即净空碰撞(同一样本点)
    assert r["first_violation"]["kind"] == "COLLISION"


def test_margin_insufficiency_without_collision():
    """障碍在包络可达范围之外 0.3 m: 不相交, 但保守净空低于安全间距 0.5。"""
    setup = make_setup(
        path=[{"id": "P0", "position": [5, 0, 18]},
              {"id": "P1", "position": [9, 0, 18]}],
        path_step_m=0.5,
        clearance={
            "envelope": {"length_m": 4, "width_m": 2, "height_m": 2},
            "hook_to_center_offset": [0, 0, -5],
            "obstacles": [{"id": "OB1", "min_corner": [11.3, -2, 10],
                           "max_corner": [12.3, 2, 14]}],
            "safety_margin_m": 0.5,
        },
    )
    r = check_crane_duty(setup, HOOK_DYN)
    codes = [c["code"] for c in r["conflicts"]]
    assert codes == ["CRANE_CLEARANCE_INSUFFICIENT"]
    fv = r["clearance"]["first_violation"]
    assert fv["kind"] == "CLEARANCE_INSUFFICIENT"
    # x=9 处净空 0.3, 保守 0.3-0.25=0.05 < 0.5
    assert fv["interval"]["to"][0] == pytest.approx(9.0)
    assert fv["conservative_clearance_m"] == pytest.approx(0.05)
    assert r["clearance"]["min_clearance_m"] == pytest.approx(0.3)


def test_exclusion_zone_zero_tolerance():
    """不可侵入区不适用安全间距: 区间外 0.5 m 不违规, 相交即侵入。"""
    setup = make_clearance_setup(obstacles=[], exclusion_zones=[dict(EX1)])
    r = check_crane_duty(setup, HOOK_DYN)
    codes = [c["code"] for c in r["conflicts"]]
    assert codes == ["CRANE_EXCLUSION_INTRUSION"]
    fv = r["clearance"]["first_violation"]
    assert fv["kind"] == "EXCLUSION_INTRUSION"
    assert fv["interval"]["to"][0] == pytest.approx(6.0)  # 保守净空 -0.5
    assert fv["worst_exclusion_id"] == "EX1"
    # x=5 保守净空 0.5 > 0: 零容忍但尚未侵入
    assert r["samples"][0]["clearance"]["violations"] == []


def test_clear_path_is_ok():
    far = {"id": "OB1", "min_corner": [20, -2, 10], "max_corner": [21, 2, 14]}
    r = check_crane_duty(make_clearance_setup(obstacles=[far]), HOOK_DYN)
    assert r["status"] == "ok"
    assert r["clearance"]["status"] == "ok"
    assert r["clearance"]["first_violation"] is None
    assert r["clearance"]["min_conservative_clearance_m"] == pytest.approx(8.5)
    assert r["evidence_gaps"] == []


# ---------------------------------------------------------------- 证据缺口
def test_degenerate_envelope_skips_check():
    r = check_crane_duty(
        make_clearance_setup(envelope={"length_m": 4, "width_m": 2, "height_m": 0}),
        HOOK_DYN)
    clr = r["clearance"]
    assert clr["status"] == "skipped"
    assert clr["min_clearance_m"] is None
    assert all("clearance" not in s for s in r["samples"])
    assert all("attitude_deg" in s for s in r["samples"])  # 姿态仍插值
    assert not any(c["code"].startswith("CRANE_COLLISION") for c in r["conflicts"])
    gaps = [g["code"] for g in r["evidence_gaps"]]
    assert "CLEARANCE_ENVELOPE_DEGENERATE" in gaps


def test_invalid_obstacle_geometry_listed_and_skipped():
    bad = {"id": "OB_BAD", "min_corner": [8, 2, 10], "max_corner": [9, 2, 14]}
    r = check_crane_duty(
        make_clearance_setup(obstacles=[dict(OB1), bad]), HOOK_DYN)
    gaps = {g["code"]: g for g in r["evidence_gaps"]}
    assert "CLEARANCE_OBSTACLE_INVALID" in gaps
    assert gaps["CLEARANCE_OBSTACLE_INVALID"]["evidence"]["invalid_obstacle_ids"] == \
        ["OB_BAD"]
    # 有效障碍仍照常校核
    assert "CRANE_COLLISION" in [c["code"] for c in r["conflicts"]]
    assert r["clearance"]["invalid_obstacle_ids"] == ["OB_BAD"]
    assert [o["id"] for o in r["clearance"]["obstacles"]] == ["OB1"]


def test_attitude_jump_gap():
    setup = make_setup(
        path=[{"id": "P0", "position": [5, 0, 18], "yaw_deg": 0},
              {"id": "P1", "position": [9, 0, 18], "yaw_deg": 170}],
        clearance={
            "envelope": {"length_m": 4, "width_m": 2, "height_m": 2},
            "hook_to_center_offset": [0, 0, -5],
            "obstacles": [], "attitude_jump_deg": 120.0,
        },
    )
    r = check_crane_duty(setup, HOOK_DYN)
    gaps = {g["code"]: g for g in r["evidence_gaps"]}
    assert "CLEARANCE_ATTITUDE_JUMP" in gaps
    seg = gaps["CLEARANCE_ATTITUDE_JUMP"]["evidence"]["segments"][0]
    assert seg["segment"] == "P0->P1"
    assert seg["rotation_deg"] == pytest.approx(170.0)
    # 转角 100° 不超阈值 -> 无缺口
    setup2 = make_setup(
        path=[{"id": "P0", "position": [5, 0, 18], "yaw_deg": 0},
              {"id": "P1", "position": [9, 0, 18], "yaw_deg": 100}],
        clearance={
            "envelope": {"length_m": 4, "width_m": 2, "height_m": 2},
            "hook_to_center_offset": [0, 0, -5],
            "obstacles": [], "attitude_jump_deg": 120.0,
        },
    )
    assert check_crane_duty(setup2, HOOK_DYN)["evidence_gaps"] == []


def test_sampling_cap_scales_down_and_flags():
    setup = make_setup(
        path=[{"id": "P0", "position": [5, 0, 18], "yaw_deg": 0},
              {"id": "P1", "position": [9, 0, 18], "yaw_deg": 90}],
        clearance={
            "envelope": {"length_m": 4, "width_m": 2, "height_m": 2},
            "hook_to_center_offset": [0, 0, -5],
            "obstacles": [], "attitude_step_deg": 10.0, "max_samples": 6,
        },
    )
    r = check_crane_duty(setup, HOOK_DYN)
    samp = r["clearance"]["sampling"]
    assert samp["required_samples"] == 10
    assert samp["used_samples"] <= 6
    assert samp["capped"] is True
    assert len(r["samples"]) == samp["used_samples"]
    # 仍覆盖全路径
    assert r["samples"][-1]["at_pose"] == "P1"
    assert r["samples"][-1]["position"][0] == pytest.approx(9.0)
    gaps = [g["code"] for g in r["evidence_gaps"]]
    assert "CLEARANCE_SAMPLING_CAP" in gaps


# ---------------------------------------------------------------- API 集成
def test_analyze_clearance_conflict_blocks_approvable(client_no_db):
    crane = make_clearance_setup().model_dump()
    r = client_no_db.post("/api/lifts/analyze", json=lift_payload(crane))
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["approvable"] is False
    assert "CRANE_COLLISION" in b["blocking_codes"]
    assert b["crane"]["clearance"]["status"] == "violation"
    assert b["utilization_summary"]["clearance_min_conservative_m"] == \
        pytest.approx(-2.5)


def test_analyze_clearance_ok_approvable(client_no_db):
    far = {"id": "OB1", "min_corner": [20, -2, 10], "max_corner": [21, 2, 14]}
    crane = make_clearance_setup(obstacles=[far]).model_dump()
    r = client_no_db.post("/api/lifts/analyze", json=lift_payload(crane))
    b = r.json()
    assert b["approvable"] is True
    assert b["crane"]["clearance"]["status"] == "ok"


def test_evidence_gaps_reach_analysis(client_no_db):
    crane = make_clearance_setup(
        envelope={"length_m": 4, "width_m": 2, "height_m": -1}).model_dump()
    r = client_no_db.post("/api/lifts/analyze", json=lift_payload(crane))
    b = r.json()
    assert "CLEARANCE_ENVELOPE_DEGENERATE" in \
        [g["code"] for g in b["all_evidence_gaps"]]


def test_crane_check_and_svg_share_attitude_and_collision(client_no_db):
    payload = lift_payload(make_clearance_setup().model_dump())
    r = client_no_db.post("/api/lifts/crane-check", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()["crane"]
    s0 = body["samples"][0]
    assert s0["attitude_deg"] == {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
    assert s0["clearance"]["min_clearance_m"] == pytest.approx(1.0)
    assert body["samples"][1]["clearance"]["violations"] == ["COLLISION"]

    svg = client_no_db.post("/api/lifts/crane-station.svg", json=payload)
    assert svg.status_code == 200
    text = svg.text
    assert "障碍 OB1" in text
    assert "<polygon" in text                      # 包络逐姿态投影
    assert "首个违规 COLLISION" in text            # 与 JSON 同一碰撞标记
    assert "最小保守净空 -2.5" in text
    assert "yaw=" in text                          # 姿态标注


def test_svg_marks_distinct_clearance_first(client_no_db):
    """净空首违与工况首违不同点时, 站位图分别标注。"""
    setup = make_setup(
        ground_bearing_limit_kpa=200.0,  # r=6 起地基超压
        path=[{"id": "P0", "position": [5, 0, 18]},
              {"id": "P1", "position": [9, 0, 18]}],
        clearance={
            "envelope": {"length_m": 4, "width_m": 2, "height_m": 2},
            "hook_to_center_offset": [0, 0, -5],
            "obstacles": [],
            # 侵入区 x[8.5,9.5]: 包络 x=7 起侵入(首现晚于 r=6 的地基超压)
            "exclusion_zones": [{"id": "EX1", "min_corner": [8.5, -2, 10],
                                 "max_corner": [9.5, 2, 14]}],
            "safety_margin_m": 0.5,
        },
    )
    r = client_no_db.post("/api/lifts/crane-station.svg",
                          json=lift_payload(setup.model_dump()))
    text = r.text
    assert "首个违规 GROUND_OVERPRESSURE" in text
    assert "首个净空违规 EXCLUSION_INTRUSION" in text
    assert "禁入 EX1" in text


def test_clearance_absent_keeps_previous_behavior(client_no_db):
    r = client_no_db.post("/api/lifts/crane-check",
                          json=lift_payload(make_setup().model_dump()))
    body = r.json()["crane"]
    assert body["clearance"] is None
    assert body["evidence_gaps"] == []
    assert all("attitude_deg" not in s for s in body["samples"])


def test_clearance_input_validation(client_no_db):
    crane = make_clearance_setup().model_dump()
    crane["clearance"]["obstacles"].append(
        {"id": "OB1", "min_corner": [0, 0, 0], "max_corner": [1, 1, 1]})
    r = client_no_db.post("/api/lifts/analyze", json=lift_payload(crane))
    assert r.status_code == 422 and "障碍物/不可侵入区 id 重复" in r.text


# ---------------------------------------------------------------- 版本库集成
def test_revision_flow_clearance_blocks_and_diffs(client_db):
    far = {"id": "OB1", "min_corner": [20, -2, 10], "max_corner": [21, 2, 14]}
    p1 = lift_payload(make_clearance_setup(obstacles=[far]).model_dump())
    r1 = client_db.post("/api/plans/CL1/revisions", json=p1)
    assert r1.status_code == 201, r1.text
    assert r1.json()["approvable"] is True
    ap = client_db.post("/api/plans/CL1/revisions/1/review",
                        json={"reviewer": "li", "action": "approve"})
    assert ap.status_code == 200

    # 派生版把障碍物移到路径上 -> 碰撞, 不可批准
    p2 = lift_payload(make_clearance_setup().model_dump())
    dr = client_db.post("/api/plans/CL1/derive",
                        json={"from_revision": 1, "change_note": "新增障碍物 OB1",
                              "input": p2})
    assert dr.status_code == 201, dr.text
    rev2 = dr.json()
    assert rev2["approvable"] is False
    assert rev2["summary"]["crane"]["clearance"]["first_violation"]["kind"] == \
        "COLLISION"

    diff = client_db.get("/api/plans/CL1/diff/1/2").json()
    changed = [c["field"] for c in diff["summary_changes"]]
    assert "crane.clearance.min_conservative_clearance_m" in changed
    assert "crane.clearance.first_violation.kind" in changed
    assert "CRANE_COLLISION" in diff["conflicts_added"]
    assert diff["approvable"] == {"from": True, "to": False}

    ap2 = client_db.post("/api/plans/CL1/revisions/2/review",
                         json={"reviewer": "li", "action": "approve"})
    assert ap2.status_code == 409
