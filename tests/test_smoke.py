# -*- coding: utf-8 -*-
"""冒烟用例: 不依赖任何固定数据库文件, 也不写仓库目录。

覆盖:
  1. app / app.main 模块可导入;
  2. 未配置 LIFTCALC_DB 时 /health 正常(storage=disabled, 纯计算可用);
  3. 一次对称四索核算: 四索等张力、吊钩动载 1100 kN;
  4. 临时 SQLite(mkstemp) 版本保存与读回, 结束即清理。
"""
import os
import tempfile

from fastapi.testclient import TestClient


def _symmetric_payload():
    return {
        "name": "smoke-symmetric-4pt", "component_weight_kn": 1000.0,
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


def test_import_modules():
    import app  # noqa: F401
    from app.main import app as fastapi_app  # noqa: F401


def test_health_without_db(monkeypatch):
    monkeypatch.delenv("LIFTCALC_DB", raising=False)
    from app import main as main_mod
    monkeypatch.setattr(main_mod, "DB_PATH", "")
    monkeypatch.setattr(main_mod, "_storage", None)

    r = TestClient(main_mod.app).get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["storage"] == "disabled"
    assert body["version"]


def test_symmetric_four_sling_calc(monkeypatch):
    monkeypatch.delenv("LIFTCALC_DB", raising=False)
    from app import main as main_mod
    monkeypatch.setattr(main_mod, "DB_PATH", "")
    monkeypatch.setattr(main_mod, "_storage", None)

    r = TestClient(main_mod.app).post("/api/lifts/analyze",
                                      json=_symmetric_payload())
    assert r.status_code == 200, r.text
    b = r.json()
    # 对称几何: 构件水平(倾角 0), 四索动载张力相等
    assert b["predicted_attitude"]["tilt_deg"] == 0.0
    tensions = [leg["tension_dynamic_kn"] for leg in b["legs"]]
    assert max(tensions) - min(tensions) < 1e-6
    assert abs(tensions[0] - 370.23) < 0.01
    # φ = max(1+0.1/9.81, 1.1) = 1.1 -> 吊钩动载 1100 kN
    assert b["weights_kn"]["hook_dynamic"] == 1100.0


def test_revision_save_in_temp_sqlite(monkeypatch):
    fd, db_path = tempfile.mkstemp(prefix="liftcalc-smoke-", suffix=".db")
    os.close(fd)
    os.unlink(db_path)  # 让 sqlite 自行建库; 只需拿到一个无人使用的路径
    try:
        monkeypatch.setenv("LIFTCALC_DB", db_path)
        from app import main as main_mod
        monkeypatch.setattr(main_mod, "DB_PATH", db_path)
        monkeypatch.setattr(main_mod, "_storage", None)

        client = TestClient(main_mod.app)
        assert client.get("/health").json()["storage"] == "sqlite"

        r = client.post("/api/plans/SMOKE/revisions", json=_symmetric_payload())
        assert r.status_code == 201, r.text
        assert r.json()["revision"] == 1

        lst = client.get("/api/plans/SMOKE/revisions").json()["revisions"]
        assert len(lst) == 1 and lst[0]["status"] == "draft"

        main_mod.get_storage().close()
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)
