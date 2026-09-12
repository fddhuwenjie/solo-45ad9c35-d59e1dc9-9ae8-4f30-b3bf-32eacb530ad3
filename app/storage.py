# -*- coding: utf-8 -*-
"""
SQLite 版本库
==============

存储试吊与吊装方案版本: 输入(完整 JSON)、计算摘要、审批状态。
* 新版本状态为 draft; approve 仅在系统判定 approvable=true 时生效;
* 批准后输入与摘要冻结, plans.approved_revision 指向锁定版;
* 已有批准版时, 新版本必须从该批准版派生(保留来源链)。
"""
from __future__ import annotations

import datetime as _dt
import json
import sqlite3
from typing import Any, Dict, List, Optional


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# 版本对比时关注的摘要字段(点路径 -> 显示名)
_SUMMARY_PATHS = [
    ("approvable", "可批准"),
    ("dynamic_factor", "动载系数"),
    ("cog_used.source", "重心来源"),
    ("cog_used.cog", "采用重心[m]"),
    ("predicted_attitude.tilt_deg", "预测倾角[°]"),
    ("predicted_attitude.pitch_deg", "纵倾角[°]"),
    ("predicted_attitude.roll_deg", "横倾角[°]"),
    ("weights_kn.hook_dynamic", "吊钩动载[kN]"),
    ("utilization_summary.max_sling_utilization", "最大吊索利用率"),
    ("utilization_summary.max_shackle_utilization", "最大卸扣利用率"),
    ("utilization_summary.beam_stress_utilization", "吊梁应力利用率"),
    ("utilization_summary.hoist_utilization", "起升机利用率"),
    ("crane.envelope.max_load_utilization", "起重机载荷利用率"),
    ("crane.envelope.max_ground_pressure_kpa", "最大接地压力[kPa]"),
    ("crane.envelope.min_outrigger_reaction_kn", "最小支腿反力[kN]"),
    ("crane.first_violation.kind", "起重机首个违规"),
    ("crane.load_chart_summary.fingerprint", "载荷表指纹"),
]

_INPUT_SCALAR_PATHS = [
    ("name", "方案名称"),
    ("component_weight_kn", "构件重量[kN]"),
    ("cog_theory", "理论重心[m]"),
    ("lift_acceleration_mps2", "起升加速度[m/s²]"),
    ("min_dynamic_factor", "动载系数下限"),
    ("min_sling_angle_deg", "最小吊索角[°]"),
    ("crane.boom_length_m", "起重机臂长[m]"),
    ("crane.config", "起重机配置"),
    ("crane.hook_block_weight_kn", "吊钩滑轮组重量[kN]"),
    ("crane.ground_bearing_limit_kpa", "地基承压限值[kPa]"),
]


def _get_path(obj: Dict[str, Any], dotted: str) -> Any:
    cur: Any = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


class PlanStore:
    def __init__(self, db_path: str, check_same_thread: bool = True):
        self.conn = sqlite3.connect(db_path, check_same_thread=check_same_thread)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS plans (
                plan_id            TEXT PRIMARY KEY,
                name               TEXT NOT NULL,
                created_at         TEXT NOT NULL,
                approved_revision INTEGER
            );
            CREATE TABLE IF NOT EXISTS revisions (
                plan_id      TEXT NOT NULL,
                revision     INTEGER NOT NULL,
                input_json   TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                approvable   INTEGER NOT NULL,
                status       TEXT NOT NULL DEFAULT 'draft',
                derived_from INTEGER,
                change_note  TEXT NOT NULL DEFAULT '',
                created_at   TEXT NOT NULL,
                PRIMARY KEY (plan_id, revision)
            );
            CREATE TABLE IF NOT EXISTS review_events (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_id   TEXT NOT NULL,
                revision  INTEGER NOT NULL,
                action    TEXT NOT NULL,
                reviewer  TEXT NOT NULL,
                comment   TEXT NOT NULL DEFAULT '',
                at        TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    # ------------------------------------------------------------ 写入
    def save_revision(self, plan_id: str, inp, result: Dict[str, Any]) -> Dict[str, Any]:
        cur = self.conn.execute(
            "SELECT COALESCE(MAX(revision),0) AS m FROM revisions WHERE plan_id=?",
            (plan_id,),
        )
        next_rev = cur.fetchone()["m"] + 1
        approved = self._approved_revision(plan_id)
        if approved is not None:
            raise ValueError(
                f"方案 {plan_id} 已有批准版 r{approved}; 新版本必须使用 /derive 从它派生"
            )
        self._upsert_plan(plan_id, inp.name)
        self._insert_revision(plan_id, next_rev, inp, result, None, "")
        self.conn.commit()
        return self.get_revision(plan_id, next_rev)

    def derive_revision(self, plan_id: str, from_revision: Optional[int],
                        inp, result: Dict[str, Any], change_note: str) -> Dict[str, Any]:
        approved = self._approved_revision(plan_id)
        if approved is not None and from_revision != approved:
            raise ValueError(
                f"方案 {plan_id} 已锁定到批准版 r{approved}, "
                f"只能从 r{approved} 派生(请求的基版 r{from_revision})"
            )
        if from_revision is not None and self.get_revision(plan_id, from_revision) is None:
            raise ValueError(f"基版 r{from_revision} 不存在")
        cur = self.conn.execute(
            "SELECT COALESCE(MAX(revision),0) AS m FROM revisions WHERE plan_id=?",
            (plan_id,),
        )
        next_rev = cur.fetchone()["m"] + 1
        self._upsert_plan(plan_id, inp.name)
        self._insert_revision(plan_id, next_rev, inp, result, from_revision, change_note)
        self.conn.commit()
        return self.get_revision(plan_id, next_rev)

    def review(self, plan_id: str, revision: int, action: str,
               reviewer: str, comment: str) -> Dict[str, Any]:
        rec = self.get_revision(plan_id, revision)
        if rec is None:
            raise ValueError(f"版本 {plan_id} r{revision} 不存在")
        if action == "approve":
            if not rec["approvable"]:
                blockers = rec["summary"].get("blocking_codes", [])
                raise ValueError(
                    "系统判定该版本不可批准(存在冲突或证据缺口), 不能锁定: "
                    + (", ".join(blockers) if blockers else "未通过核算")
                )
            self.conn.execute(
                "UPDATE revisions SET status='approved' WHERE plan_id=? AND revision=?",
                (plan_id, revision),
            )
            self.conn.execute(
                "UPDATE plans SET approved_revision=? WHERE plan_id=?",
                (revision, plan_id),
            )
        elif action == "reject":
            self.conn.execute(
                "UPDATE revisions SET status='rejected' WHERE plan_id=? AND revision=?",
                (plan_id, revision),
            )
        else:
            self.conn.execute(
                "UPDATE revisions SET status='changes_requested' WHERE plan_id=? AND revision=?",
                (plan_id, revision),
            )
        self.conn.execute(
            "INSERT INTO review_events(plan_id,revision,action,reviewer,comment,at)"
            " VALUES(?,?,?,?,?,?)",
            (plan_id, revision, action, reviewer, comment, _now()),
        )
        self.conn.commit()
        return self.get_revision(plan_id, revision)

    # ------------------------------------------------------------ 读取
    def list_plans(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT p.plan_id, p.name, p.created_at, p.approved_revision,
                      COUNT(r.revision) AS n_revisions
               FROM plans p LEFT JOIN revisions r ON r.plan_id = p.plan_id
               GROUP BY p.plan_id ORDER BY p.created_at"""
        ).fetchall()
        return [dict(r) for r in rows]

    def list_revisions(self, plan_id: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT revision, approvable, status, derived_from, change_note, created_at
               FROM revisions WHERE plan_id=? ORDER BY revision""",
            (plan_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_revision(self, plan_id: str, revision: int) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            """SELECT * FROM revisions WHERE plan_id=? AND revision=?""",
            (plan_id, revision),
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["input"] = json.loads(d.pop("input_json"))
        d["summary"] = json.loads(d.pop("summary_json"))
        d["approvable"] = bool(d["approvable"])
        return d

    def review_history(self, plan_id: str, revision: int) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT action,reviewer,comment,at FROM review_events "
            "WHERE plan_id=? AND revision=? ORDER BY id",
            (plan_id, revision),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------ 对比
    def diff(self, plan_id: str, rev_a: int, rev_b: int) -> Dict[str, Any]:
        a, b = self.get_revision(plan_id, rev_a), self.get_revision(plan_id, rev_b)
        if a is None or b is None:
            raise ValueError(f"版本 r{rev_a} 或 r{rev_b} 不存在")
        ia, ib = a["input"], b["input"]
        sa, sb = a["summary"], b["summary"]

        input_changes = []
        for path, label in _INPUT_SCALAR_PATHS:
            va, vb = _get_path(ia, path), _get_path(ib, path)
            if va != vb:
                input_changes.append({"field": path, "label": label,
                                      "from": va, "to": vb})

        # 吊点/吊索的逐元素差异
        leg_changes = self._diff_legs(ia.get("legs", []), ib.get("legs", []))

        summary_changes = []
        for path, label in _SUMMARY_PATHS:
            va, vb = _get_path(sa, path), _get_path(sb, path)
            if va != vb:
                summary_changes.append({"field": path, "label": label,
                                        "from": va, "to": vb})

        ca = {c["code"] for c in sa.get("conflicts", [])}
        cb = {c["code"] for c in sb.get("conflicts", [])}
        return {
            "plan_id": plan_id,
            "from_revision": rev_a,
            "to_revision": rev_b,
            "status": {"from": a["status"], "to": b["status"]},
            "input_changes": input_changes,
            "leg_changes": leg_changes,
            "summary_changes": summary_changes,
            "conflicts_added": sorted(cb - ca),
            "conflicts_resolved": sorted(ca - cb),
            "approvable": {"from": a["approvable"], "to": b["approvable"]},
        }

    @staticmethod
    def _diff_legs(la_: List[Dict[str, Any]], lb_: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        by_a = {l["id"]: l for l in la_}
        by_b = {l["id"]: l for l in lb_}
        for lid in sorted(set(by_a) | set(by_b)):
            a, b = by_a.get(lid), by_b.get(lid)
            if a is None:
                out.append({"leg_id": lid, "change": "added"})
                continue
            if b is None:
                out.append({"leg_id": lid, "change": "removed"})
                continue
            fields = {}
            for key in ("pad", "length_m", "sling_capacity_kn", "shackle_capacity_kn",
                        "sling_count", "derating_factor"):
                if a.get(key) != b.get(key):
                    fields[key] = {"from": a.get(key), "to": b.get(key)}
            if fields:
                out.append({"leg_id": lid, "change": "modified", "fields": fields})
        return out

    # ------------------------------------------------------------ 内部
    def _upsert_plan(self, plan_id: str, name: str) -> None:
        self.conn.execute(
            "INSERT INTO plans(plan_id,name,created_at) VALUES(?,?,?) "
            "ON CONFLICT(plan_id) DO NOTHING",
            (plan_id, name, _now()),
        )

    def _insert_revision(self, plan_id, revision, inp, result, derived_from, note) -> None:
        self.conn.execute(
            "INSERT INTO revisions(plan_id,revision,input_json,summary_json,approvable,"
            "status,derived_from,change_note,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (plan_id, revision,
             json.dumps(inp.model_dump(), ensure_ascii=False, default=str),
             json.dumps(result, ensure_ascii=False, default=str),
             1 if result.get("approvable") else 0,
             "draft", derived_from, note, _now()),
        )

    def _approved_revision(self, plan_id: str) -> Optional[int]:
        row = self.conn.execute(
            "SELECT approved_revision FROM plans WHERE plan_id=?", (plan_id,)
        ).fetchone()
        return row["approved_revision"] if row else None

    def close(self) -> None:
        self.conn.close()
