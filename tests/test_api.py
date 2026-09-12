# -*- coding: utf-8 -*-
"""FastAPI 入口: 导入/启动、核算端点、参数校验、SQLite 审批与派生流。"""
import os
import tempfile

import pytest
from fastapi.testclient import TestClient


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


def symmetric_payload():
    return {
        "name": "symmetric-4pt", "component_weight_kn": 1000.0,
        "cog_theory": [0, 0, 1.0],
        "legs": [
            {"id": f"L{i+1}", "pad": list(p),
             "sling_capacity_kn": 500.0, "shackle_capacity_kn": 600.0,
             "sling_ea_kn": 50000.0}
            for i, p in enumerate([(-3, -2, 1), (-3, 2, 1), (3, -2, 1), (3, 2, 1)])
        ],
        "hook_point": [0, 0, 5], "lift_acceleration_mps2": 0.1,
        "min_dynamic_factor": 1.1, "hoist_capacity_kn": 2000.0,
    }


# ---------------------------------------------------------------- 基础
def test_app_importable_and_health(client_no_db):
    from app.main import app  # 导入即启动校验
    assert app.title.startswith("大型构件")
    r = client_no_db.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["storage"] == "disabled"
    assert body["version"]


def test_analyze_symmetric(client_no_db):
    r = client_no_db.post("/api/lifts/analyze", json=symmetric_payload())
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["engine"]
    assert b["predicted_attitude"]["tilt_deg"] == 0.0
    tensions = [l["tension_dynamic_kn"] for l in b["legs"]]
    assert max(tensions) - min(tensions) < 1e-6
    assert abs(tensions[0] - 370.23) < 0.01
    assert b["weights_kn"]["hook_dynamic"] == 1100.0
    assert b["distribution"]["kind"] == "statically_indeterminate_elastic"
    # 无试吊 -> 重心未核实, 系统不批准
    assert b["approvable"] is False
    assert "NO_VERIFIED_COG" in [g["code"] for g in b["all_evidence_gaps"]]


def test_validate_rejects_bad_input(client_no_db):
    bad = symmetric_payload()
    bad["component_weight_kn"] = -5
    r = client_no_db.post("/api/lifts/analyze", json=bad)
    assert r.status_code == 422
    assert "component_weight_kn" in r.text


def test_validate_rejects_duplicate_leg_id(client_no_db):
    bad = symmetric_payload()
    bad["legs"][1]["id"] = "L1"
    r = client_no_db.post("/api/lifts/analyze", json=bad)
    assert r.status_code == 422
    assert "吊索 id 重复" in r.text


def test_inverse_and_forward_endpoints(client_no_db):
    payload = symmetric_payload()
    r = client_no_db.post("/api/lifts/analyze-forward", json=payload)
    assert r.status_code == 200
    assert r.json()["predicted_attitude"]["tilt_deg"] == 0.0
    # 无试吊时反算明确返回数据不足
    r2 = client_no_db.post("/api/lifts/inverse-cog", json=payload)
    assert r2.status_code == 200
    assert r2.json()["status"] == "insufficient_data"
    assert r2.json()["evidence_gaps"][0]["code"] == "NO_TRIALS"


# ---------------------------------------------------------------- 动/静张力
def test_static_vs_dynamic_tension_api(client_no_db):
    def payload(t, static):
        return {
            "name": "v", "component_weight_kn": 400.0, "cog_theory": [0, 0, 0],
            "legs": [{"id": f"L{i+1}", "pad": list(p),
                      "sling_capacity_kn": 500.0, "shackle_capacity_kn": 600.0}
                     for i, p in enumerate([(-1, -1, 0), (-1, 1, 0),
                                            (1, -1, 0), (1, 1, 0)])],
            "hook_point": [0, 0, 1e6], "lift_acceleration_mps2": 0.981,
            "min_dynamic_factor": 1.1, "request_adjustment": False,
            "trials": [{"id": "TR1", "measurements": [
                {"leg_id": f"L{i+1}", "tension_kn": t,
                 "tension_is_static": static} for i in range(4)]}],
        }
    rs = client_no_db.post("/api/lifts/inverse-cog", json=payload(100.0, True)).json()
    rd = client_no_db.post("/api/lifts/inverse-cog", json=payload(110.0, False)).json()
    assert rs["total_weight_solved_static_kn"] == 400.0
    assert rd["total_weight_solved_static_kn"] == 400.0
    assert rs["total_weight_solved_dynamic_kn"] == rd["total_weight_solved_dynamic_kn"]


# ---------------------------------------------------------------- 版本库
def _approvable_payload():
    """构造一个 approvable=true 的输入: 双姿态试吊以识别重心高度。"""
    base = symmetric_payload()
    base["trials"] = [
        {"id": "TR1", "pitch_deg": 0, "roll_deg": 0, "measurements": [
            {"leg_id": f"L{i+1}", "tension_kn": t}
            for i, t in enumerate([370.23]*4)]},
        {"id": "TR2", "pitch_deg": 5, "roll_deg": 0, "measurements": [
            {"leg_id": f"L{i+1}", "tension_kn": t}
            for i, t in enumerate([300, 300, 445, 445])]},
    ]
    return base


def test_plan_endpoints_503_without_db(client_no_db):
    r = client_no_db.get("/api/plans")
    assert r.status_code == 503


def test_revision_save_review_lock_derive(client_db):
    # 理论重心、无试吊 -> approvable=False, 批准必须被拒
    r = client_db.post("/api/plans/P1/revisions", json=symmetric_payload())
    assert r.status_code == 201
    rev = r.json()
    assert rev["revision"] == 1 and rev["status"] == "draft"
    assert rev["approvable"] is False

    approve = client_db.post(
        "/api/plans/P1/revisions/1/review",
        json={"reviewer": "zhang", "action": "approve", "comment": "试一下"})
    assert approve.status_code == 409
    assert "不可批准" in approve.text

    # request_changes / reject 允许
    rc = client_db.post("/api/plans/P1/revisions/1/review",
                        json={"reviewer": "zhang", "action": "request_changes"})
    assert rc.status_code == 200 and rc.json()["status"] == "changes_requested"

    # 保存第二个 draft 版本(尚无批准版, 允许)
    r2 = client_db.post("/api/plans/P1/revisions", json=symmetric_payload())
    assert r2.status_code == 201 and r2.json()["revision"] == 2


def test_approved_revision_locks_and_derives(client_db):
    # 用 cog_actual_override 提供已核实重心 -> approvable=true
    payload = symmetric_payload()
    payload["cog_actual_override"] = [0, 0, 1.0]
    r = client_db.post("/api/plans/P2/revisions", json=payload)
    assert r.status_code == 201 and r.json()["approvable"] is True

    ap = client_db.post("/api/plans/P2/revisions/1/review",
                        json={"reviewer": "li", "action": "approve"})
    assert ap.status_code == 200 and ap.json()["status"] == "approved"

    # 锁定后不能直接再建版本, 必须派生
    blocked = client_db.post("/api/plans/P2/revisions", json=payload)
    assert blocked.status_code == 409 and "派生" in blocked.text

    # 从批准版派生 r2
    payload2 = symmetric_payload()
    payload2["cog_actual_override"] = [0.05, 0, 1.0]
    dr = client_db.post("/api/plans/P2/derive",
                        json={"from_revision": 1, "change_note": "重心微调",
                              "input": payload2})
    assert dr.status_code == 201, dr.text
    d = dr.json()
    assert d["revision"] == 2 and d["derived_from"] == 1

    # 从错误基版派生 -> 409
    bad = client_db.post("/api/plans/P2/derive",
                         json={"from_revision": 2, "input": payload2})
    assert bad.status_code == 409

    # diff
    diff = client_db.get("/api/plans/P2/diff/1/2")
    assert diff.status_code == 200
    dj = diff.json()
    changed_fields = [c["field"] for c in dj["summary_changes"]]
    assert "cog_used.cog" in changed_fields
    assert dj["status"] == {"from": "approved", "to": "draft"}


def test_list_and_get_revision(client_db):
    client_db.post("/api/plans/P3/revisions", json=symmetric_payload())
    lst = client_db.get("/api/plans").json()["plans"]
    assert any(p["plan_id"] == "P3" for p in lst)
    revs = client_db.get("/api/plans/P3/revisions").json()["revisions"]
    assert len(revs) == 1
    got = client_db.get("/api/plans/P3/revisions/1").json()
    assert got["input"]["name"] == "symmetric-4pt"
    assert got["summary"]["dynamic_factor"] == 1.1
    assert client_db.get("/api/plans/P3/revisions/9").status_code == 404


def test_manual_recheck_still_enforces_checks(client_no_db):
    payload = symmetric_payload()
    # 超载额定值 -> 即便人工提交也不能批准
    for leg in payload["legs"]:
        leg["sling_capacity_kn"] = 100.0
        leg["shackle_capacity_kn"] = 100.0
    payload["manual_adjustment"] = {
        "mode": "manual_mix", "comment": "尝试放短吊索",
        "shims": [{"leg_id": f"L{i+1}", "delta_length_m": 0.01} for i in range(4)],
    }
    r = client_no_db.post("/api/lifts/manual-recheck", json=payload)
    assert r.status_code == 200
    b = r.json()
    assert b["approvable"] is False
    assert "LEG_OVERLOAD" in b["blocking_codes"]
    assert b["manual_recheck"]["comment"] == "尝试放短吊索"


def test_manual_recheck_requires_adjustment(client_no_db):
    r = client_no_db.post("/api/lifts/manual-recheck", json=symmetric_payload())
    assert r.status_code == 400
