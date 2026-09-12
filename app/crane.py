# -*- coding: utf-8 -*-
"""
起重机工况校核
==============

构件从试吊点回转到安装点的路径上, 按步长插值出相邻姿态, 逐姿态核算:
作业半径、载荷表净额定能力、支腿反力与垫板接地压力, 并定位首个
超载 / 支腿拔起 / 越出载荷表 / 地基超压区间。

约定
----
* 世界系: x 东、y 北、z 向上; 回转中心与吊钩路径均为世界坐标。
* 起重机参考系: 原点 = 回转中心, x = 吊臂正前方(回转角 0°, 世界方位角
  slew_reference_deg); 随转台回转的部件(slews=True)按吊钩方位角同步旋转。
* 载荷表按 (臂长, 回转区段) 分组, 仅在相邻半径档位间线性插值, 不外推。
* 支腿反力按刚性车体 + 等刚度支座(反力平面分布)求解, 复用
  physics.solve_force_system; 反力为负即支腿拔起(支座不能受拉)。
* 吊钩处的起升载荷复用吊装核算的吊钩动载(已含构件/索具/吊梁与动载系数)。
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Tuple

from .physics import ForceLane, solve_force_system

UPLIFT_TOL = 1.0e-6      # 支腿反力 < -tol 判定拔起 kN
OVERLOAD_TOL = 1.0       # 载荷利用率上限
RADIUS_TOL = 1.0e-6      # 半径档位匹配容差 m
BOOM_TOL = 1.0e-6        # 臂长匹配容差 m

# 违规类型与优先级(first_violation.kind 取列表中首个)
_VIOLATION_ORDER = ["OUT_OF_CHART", "CHART_OVERLOAD", "OUTRIGGER_UPLIFT",
                    "GROUND_OVERPRESSURE"]

_CONFLICT_CODE = {
    "OUT_OF_CHART": "CRANE_OUT_OF_CHART",
    "CHART_OVERLOAD": "CRANE_CHART_OVERLOAD",
    "OUTRIGGER_UPLIFT": "CRANE_OUTRIGGER_UPLIFT",
    "GROUND_OVERPRESSURE": "CRANE_GROUND_OVERPRESSURE",
}

_VIOLATION_MESSAGE = {
    "OUT_OF_CHART": "作业半径越出载荷表(该臂长/回转区段无对应档位, 不允许外推)",
    "CHART_OVERLOAD": "吊钩动载超过载荷表净额定能力",
    "OUTRIGGER_UPLIFT": "支腿反力为负, 支腿拔起(支座不能受拉)",
    "GROUND_OVERPRESSURE": "垫板接地压力超过地基承压限值",
}

_VIOLATION_EQUATIONS = {
    "OUT_OF_CHART": ["capacity(r): 仅同臂长同区段相邻半径档位线性插值, 禁止外推"],
    "CHART_OVERLOAD": ["P_hook_dynamic + W_hook_block <= Q_chart(radius, boom, zone)"],
    "OUTRIGGER_UPLIFT": ["R_i = W/n + Σ(wx)·x_i/Σx² + Σ(wy)·y_i/Σy² >= 0"],
    "GROUND_OVERPRESSURE": ["p_i = R_i / A_mat <= p_ground_allow"],
}


# ================================================================ 载荷表
def _chart_index(rows) -> Dict[Tuple[float, str], List[Tuple[float, float]]]:
    """(臂长, 区段) -> 按半径升序的 (半径, 能力) 档位。"""
    idx: Dict[Tuple[float, str], List[Tuple[float, float]]] = {}
    for r in rows:
        idx.setdefault((r.boom_length_m, r.zone), []).append((r.radius_m, r.capacity_kn))
    for k in idx:
        idx[k].sort()
    return idx


def chart_capacity(idx: Dict[Tuple[float, str], List[Tuple[float, float]]],
                   boom_length_m: float, zone: str, radius_m: float) -> Optional[float]:
    """
    查载荷表: 同一臂长与回转区段的相邻半径档位间线性插值。

    半径越出档位范围、或无该 (臂长, 区段) 组合时返回 None —— 不外推。
    单一档位仅在半径精确匹配时给出能力。
    """
    rows = None
    for (b, z), rr in idx.items():
        if abs(b - boom_length_m) <= BOOM_TOL and z == zone:
            rows = rr
            break
    if rows is None:
        return None
    if len(rows) == 1:
        return rows[0][1] if abs(rows[0][0] - radius_m) <= RADIUS_TOL else None
    if radius_m < rows[0][0] - RADIUS_TOL or radius_m > rows[-1][0] + RADIUS_TOL:
        return None
    for (r0, c0), (r1, c1) in zip(rows, rows[1:]):
        if r0 - RADIUS_TOL <= radius_m <= r1 + RADIUS_TOL:
            t = 0.0 if r1 <= r0 else (radius_m - r0) / (r1 - r0)
            t = max(0.0, min(1.0, t))
            return c0 + (c1 - c0) * t
    return None  # pragma: no cover


def _chart_fingerprint(rows) -> str:
    """载荷表内容指纹(批准版冻结用): 与行的书写顺序无关。"""
    data = sorted([r.boom_length_m, r.zone, r.radius_m, r.capacity_kn] for r in rows)
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


# ================================================================ 路径采样
def _sample_path(path, step_m: float) -> Tuple[List[Dict[str, Any]], float]:
    """
    相邻姿态间按步长线性插值(三维), 返回采样点与路径总长。
    每个采样点记录所属区段、命中的姿态 id 与回转区段(区段为离散标签,
    区间内取区段起点姿态的区段, 终点姿态用其自身区段)。
    """
    samples: List[Dict[str, Any]] = []
    total = 0.0
    for k, (a, b) in enumerate(zip(path, path[1:])):
        pa, pb = a.position, b.position
        dist = math.dist(pa, pb)
        total += dist
        n = max(1, math.ceil(dist / step_m - 1e-9))
        for j in range(0 if k == 0 else 1, n + 1):
            t = j / n
            pos = (pa[0] + (pb[0] - pa[0]) * t,
                   pa[1] + (pb[1] - pa[1]) * t,
                   pa[2] + (pb[2] - pa[2]) * t)
            at_pose = a.id if j == 0 else (b.id if j == n else None)
            zone = a.zone if j < n else b.zone
            samples.append({"segment": f"{a.id}->{b.id}", "position": pos,
                            "at_pose": at_pose, "zone": zone})
    return samples, total


# ================================================================ 支腿反力
def _rot2(x: float, y: float, ang_rad: float) -> Tuple[float, float]:
    c, s = math.cos(ang_rad), math.sin(ang_rad)
    return (x * c - y * s, x * s + y * c)


def _wrap180(deg: float) -> float:
    while deg > 180.0:
        deg -= 360.0
    while deg <= -180.0:
        deg += 360.0
    return deg


def _outrigger_reactions(setup, slew_rad: float, hook_rel: Tuple[float, float],
                         hook_load_kn: float) -> Tuple[List[float], float]:
    """
    支腿反力(相对回转中心平面内的刚性车体/等刚度支座解)。

    载荷: 起重机部件 + 配重(随转台回转者按吊钩方位角旋转) +
    吊钩滑轮组与吊钩动载(作用于吊钩平面位置)。
    返回 (各支腿反力 kN, 与 setup.outriggers 同序; 总重 kN)。
    """
    loads: List[Tuple[float, float, float]] = []  # (w_kn, x_rel, y_rel)
    for c in setup.components:
        x, y = c.cog[0], c.cog[1]
        if c.slews:
            x, y = _rot2(x, y, slew_rad)
        loads.append((c.weight_kn, x, y))
    if setup.counterweight is not None:
        x, y = setup.counterweight.cog[0], setup.counterweight.cog[1]
        if setup.counterweight.slews:
            x, y = _rot2(x, y, slew_rad)
        loads.append((setup.counterweight.weight_kn, x, y))
    # 吊钩滑轮组 + 吊起载荷, 作用于吊钩平面位置
    loads.append((setup.hook_block_weight_kn + hook_load_kn, hook_rel[0], hook_rel[1]))

    w_total = sum(w for w, _, _ in loads)
    # 对回转中心的矩(竖向载荷): r×(0,0,w) = (y·w, −x·w, 0)
    m_x = sum(w * y for w, _x, y in loads)
    m_y = -sum(w * x for w, x, _y in loads)
    lanes = [ForceLane(o.id, (o.position[0], o.position[1], 0.0), (0.0, 0.0, 1.0),
                       stiffness=1.0) for o in setup.outriggers]
    sol = solve_force_system(lanes, (0.0, 0.0, w_total), (m_x, m_y, 0.0),
                             label="outriggers")
    return sol.tensions, w_total


# ================================================================ 主校核
def check_crane_duty(setup, hook_load_dynamic_kn: float) -> Dict[str, Any]:
    """
    起重机工况校核主入口。

    hook_load_dynamic_kn: 复用吊装核算的吊钩动载(含动载系数的构件+索具+吊梁)。
    返回逐姿态数据(samples, JSON 与 SVG 站位图共用)、包络、首个违规区间与冲突。
    """
    idx = _chart_index(setup.load_chart)
    samples, total_len = _sample_path(setup.path, setup.path_step_m)
    cx, cy = setup.slewing_center
    limit = setup.ground_bearing_limit_kpa

    out_samples: List[Dict[str, Any]] = []
    first_of_kind: Dict[str, Dict[str, Any]] = {}
    first_violation: Optional[Dict[str, Any]] = None
    env = {"max_load_utilization": 0.0,
           "max_ground_pressure_kpa": 0.0,
           "min_outrigger_reaction_kn": math.inf,
           "max_radius_m": 0.0}

    for i, s in enumerate(samples):
        px, py, pz = s["position"]
        dx, dy = px - cx, py - cy
        radius = math.hypot(dx, dy)
        azimuth = math.degrees(math.atan2(dy, dx))
        slew = _wrap180(azimuth - setup.slew_reference_deg)
        zone = s["zone"] or setup.default_zone

        gross = chart_capacity(idx, setup.boom_length_m, zone, radius)
        net = gross - setup.hook_block_weight_kn if gross is not None else None
        util = (hook_load_dynamic_kn / net) if (net is not None and net > 0) else None

        reactions, _w = _outrigger_reactions(setup, math.radians(slew), (dx, dy),
                                             hook_load_dynamic_kn)
        rig = []
        max_p = 0.0
        min_r = math.inf
        for o, r_kn in zip(setup.outriggers, reactions):
            area = o.mat_length_m * o.mat_width_m
            p = r_kn / area
            max_p = max(max_p, p)
            min_r = min(min_r, r_kn)
            rig.append({
                "id": o.id,
                "reaction_kn": round(r_kn, 3),
                "mat_area_m2": round(area, 4),
                "pressure_kpa": round(p, 2),
                "pressure_utilization": round(p / limit, 4),
            })

        violations: List[str] = []
        if gross is None:
            violations.append("OUT_OF_CHART")
        elif util is not None and util > OVERLOAD_TOL:
            violations.append("CHART_OVERLOAD")
        if min_r < -UPLIFT_TOL:
            violations.append("OUTRIGGER_UPLIFT")
        if max_p > limit:
            violations.append("GROUND_OVERPRESSURE")
        violations.sort(key=_VIOLATION_ORDER.index)

        if util is not None:
            env["max_load_utilization"] = max(env["max_load_utilization"], util)
        env["max_ground_pressure_kpa"] = max(env["max_ground_pressure_kpa"], max_p)
        env["min_outrigger_reaction_kn"] = min(env["min_outrigger_reaction_kn"], min_r)
        env["max_radius_m"] = max(env["max_radius_m"], radius)

        entry = {
            "index": i,
            "segment": s["segment"],
            "at_pose": s["at_pose"],
            "position": [round(v, 4) for v in s["position"]],
            "radius_m": round(radius, 4),
            "slew_angle_deg": round(slew, 2),
            "zone": zone,
            "chart_gross_kn": round(gross, 3) if gross is not None else None,
            "net_rated_kn": round(net, 3) if net is not None else None,
            "load_utilization": round(util, 4) if util is not None else None,
            "outriggers": rig,
            "max_pressure_kpa": round(max_p, 2),
            "min_reaction_kn": round(min_r, 3),
            "violations": violations,
        }
        out_samples.append(entry)

        for kind in violations:
            if kind not in first_of_kind:
                first_of_kind[kind] = {
                    "sample_index": i, "segment": s["segment"],
                    "position": entry["position"], "radius_m": entry["radius_m"],
                    "zone": zone,
                }
        if violations and first_violation is None:
            prev = out_samples[i - 1]["position"] if i > 0 else entry["position"]
            first_violation = {
                "kind": violations[0],
                "kinds": violations,
                "sample_index": i,
                "segment": s["segment"],
                "interval": {"from": prev, "to": entry["position"]},
                "radius_m": entry["radius_m"],
                "zone": zone,
            }

    conflicts: List[Dict[str, Any]] = []
    for kind in _VIOLATION_ORDER:
        if kind not in first_of_kind:
            continue
        ev = first_of_kind[kind]
        conflicts.append({
            "code": _CONFLICT_CODE[kind],
            "message": f"回转路径上{_VIOLATION_MESSAGE[kind]}(首现于区段 "
                       f"{ev['segment']}, 半径 {ev['radius_m']} m)",
            "equations": _VIOLATION_EQUATIONS[kind],
            "evidence": ev,
        })

    # 载荷表与路径摘要(批准版随摘要冻结, 供版本差异与追溯)
    boom_rows = [(b, z, r, c) for (b, z), rr in idx.items() for (r, c) in rr
                 if abs(b - setup.boom_length_m) <= BOOM_TOL]
    chart_summary = {
        "boom_length_m": setup.boom_length_m,
        "zones": sorted({z for _b, z, _r, _c in boom_rows}),
        "radius_range_m": ([round(min(r for _b, _z, r, _c in boom_rows), 4),
                            round(max(r for _b, _z, r, _c in boom_rows), 4)]
                           if boom_rows else None),
        "entries_total": len(setup.load_chart),
        "fingerprint": _chart_fingerprint(setup.load_chart),
    }

    return {
        "status": "ok" if not conflicts else "violation",
        "crane_id": setup.crane_id,
        "boom_length_m": setup.boom_length_m,
        "hook_block_weight_kn": setup.hook_block_weight_kn,
        "lifted_load_dynamic_kn": round(hook_load_dynamic_kn, 3),
        "load_chart_summary": chart_summary,
        "path_summary": {
            "poses": len(setup.path),
            "samples": len(out_samples),
            "total_length_m": round(total_len, 3),
            "step_m": setup.path_step_m,
        },
        "setup": {
            "slewing_center": [cx, cy],
            "slew_reference_deg": setup.slew_reference_deg,
            "ground_bearing_limit_kpa": limit,
            "outriggers": [{
                "id": o.id,
                "position": [o.position[0], o.position[1]],
                "mat_length_m": o.mat_length_m,
                "mat_width_m": o.mat_width_m,
                "mat_angle_deg": o.mat_angle_deg,
            } for o in setup.outriggers],
        },
        "samples": out_samples,
        "envelope": {
            "max_load_utilization": round(env["max_load_utilization"], 4),
            "max_ground_pressure_kpa": round(env["max_ground_pressure_kpa"], 2),
            "min_outrigger_reaction_kn": round(env["min_outrigger_reaction_kn"], 3),
            "max_radius_m": round(env["max_radius_m"], 4),
        },
        "first_violation": first_violation,
        "conflicts": conflicts,
    }
