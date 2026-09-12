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
* 载荷表按 (臂长, 配置, 回转区段) 分组, 仅在同组相邻半径档位间线性插值,
  不跨配置、不外推。
* 支腿反力按刚性车体 + 等刚度支座(反力平面分布)求解, 复用
  physics.solve_force_system; 反力为负即支腿拔起(支座不能受拉)。
* 吊钩处的起升载荷复用吊装核算的吊钩动载(已含构件/索具/吊梁与动载系数)。
* 提供 crane.clearance 时启用路径净空校核: 姿态随路径同步插值
  (位置按 path_step_m、姿态按 attitude_step_deg 分别细分), 分离轴法
  逐点求构件定向包络与轴对齐障碍盒的保守净空, 几何实现见 clearance 模块。
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Tuple

from .clearance import (
    eval_sample as _clr_eval_sample,
    geodesic_deg,
    prepare_clearance,
    rot_matrix,
    sweep_bound,
    unwrap_attitudes,
)
from .physics import ForceLane, solve_force_system

UPLIFT_TOL = 1.0e-6      # 支腿反力 < -tol 判定拔起 kN
OVERLOAD_TOL = 1.0       # 载荷利用率上限
RADIUS_TOL = 1.0e-6      # 半径档位匹配容差 m
BOOM_TOL = 1.0e-6        # 臂长匹配容差 m

# 违规类型与优先级(first_violation.kind 取列表中首个)
_VIOLATION_ORDER = ["COLLISION", "EXCLUSION_INTRUSION", "OUT_OF_CHART",
                    "CHART_OVERLOAD", "OUTRIGGER_UPLIFT", "GROUND_OVERPRESSURE",
                    "CLEARANCE_INSUFFICIENT"]

_CONFLICT_CODE = {
    "COLLISION": "CRANE_COLLISION",
    "EXCLUSION_INTRUSION": "CRANE_EXCLUSION_INTRUSION",
    "OUT_OF_CHART": "CRANE_OUT_OF_CHART",
    "CHART_OVERLOAD": "CRANE_CHART_OVERLOAD",
    "OUTRIGGER_UPLIFT": "CRANE_OUTRIGGER_UPLIFT",
    "GROUND_OVERPRESSURE": "CRANE_GROUND_OVERPRESSURE",
    "CLEARANCE_INSUFFICIENT": "CRANE_CLEARANCE_INSUFFICIENT",
}

_VIOLATION_MESSAGE = {
    "COLLISION": "构件包络与障碍物相交(保守净空 < 0)",
    "EXCLUSION_INTRUSION": "构件包络侵入不可侵入区",
    "OUT_OF_CHART": "作业半径越出载荷表(该臂长/配置/回转区段无对应档位, 不跨配置、不外推)",
    "CHART_OVERLOAD": "吊钩动载超过载荷表净额定能力",
    "OUTRIGGER_UPLIFT": "支腿反力为负, 支腿拔起(支座不能受拉)",
    "GROUND_OVERPRESSURE": "垫板接地压力超过地基承压限值",
    "CLEARANCE_INSUFFICIENT": "构件包络保守净空低于安全间距",
}

_VIOLATION_EQUATIONS = {
    "COLLISION": ["gap(u) = |Δc·u| − Σh_a|a·u| − Σh_b|u| (15 条分离轴取 max)",
                  "clearance_cons = max_u gap(u) − sweep/2 < 0 ⇒ 包络与障碍相交"],
    "EXCLUSION_INTRUSION": ["clearance_cons(zone) < 0 ⇒ 侵入不可侵入区(零容忍)"],
    "OUT_OF_CHART": ["capacity(r): 仅同臂长同配置同区段相邻半径档位线性插值, 禁止跨配置与外推"],
    "CHART_OVERLOAD": ["P_hook_dynamic + W_hook_block <= Q_chart(radius, boom, config, zone)"],
    "OUTRIGGER_UPLIFT": ["R_i = W/n + Σ(wx)·x_i/Σx² + Σ(wy)·y_i/Σy² >= 0"],
    "GROUND_OVERPRESSURE": ["p_i = R_i / A_mat <= p_ground_allow"],
    "CLEARANCE_INSUFFICIENT": ["clearance_cons = max_u gap(u) − sweep/2 < safety_margin_m"],
}

_CLEARANCE_KINDS = ("COLLISION", "EXCLUSION_INTRUSION", "CLEARANCE_INSUFFICIENT")


# ================================================================ 载荷表
def _chart_index(rows) -> Dict[Tuple[float, str, str], List[Tuple[float, float]]]:
    """(臂长, 配置, 区段) -> 按半径升序的 (半径, 能力) 档位。"""
    idx: Dict[Tuple[float, str, str], List[Tuple[float, float]]] = {}
    for r in rows:
        idx.setdefault((r.boom_length_m, r.config, r.zone), []).append(
            (r.radius_m, r.capacity_kn))
    for k in idx:
        idx[k].sort()
    return idx


def chart_capacity(idx: Dict[Tuple[float, str, str], List[Tuple[float, float]]],
                   boom_length_m: float, zone: str, radius_m: float,
                   config: str = "STD") -> Optional[float]:
    """
    查载荷表: 仅在与当前工况相同臂长、配置和回转区段的相邻半径档位间线性插值。

    同配置没有合法相邻档位(半径越出该配置档位范围、该配置无此区段、
    或臂长/配置无记录)时返回 None —— 不跨配置合并档位, 不外推。
    单一档位仅在半径精确匹配时给出能力。
    config 缺省为 "STD", 与未声明配置的单配置载荷表兼容。
    """
    rows = None
    for (b, cfg, z), rr in idx.items():
        if abs(b - boom_length_m) <= BOOM_TOL and cfg == config and z == zone:
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
    data = sorted([r.boom_length_m, r.config, r.zone, r.radius_m, r.capacity_kn]
                  for r in rows)
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


# ================================================================ 路径采样
def _sample_path(path, step_m: float,
                 rot_step_deg: Optional[float] = None,
                 max_samples: Optional[int] = None,
                 jump_deg: Optional[float] = None
                 ) -> Tuple[List[Dict[str, Any]], float, Dict[str, Any]]:
    """
    相邻姿态间插值: 位置按 step_m 线性细分(三维); 提供 rot_step_deg 时
    姿态(yaw/pitch/roll, 逐轴展开后)同步线性插值, 并按测地转角
    rot_step_deg 进一步细分 —— 每段分段数取位置与姿态两者所需较大值。

    采样总量超过 max_samples 时各段等比放宽(每段至少 1 段), meta 标记
    capped; 相邻关键姿态测地转角超过 jump_deg 时记入 meta["jumps"]。
    每个采样点记录所属区段、命中的姿态 id 与回转区段(区段为离散标签,
    区间内取区段起点姿态的区段, 终点姿态用其自身区段)。
    """
    unwrapped: Optional[List[Tuple[float, float, float]]] = None
    if rot_step_deg is not None:
        unwrapped = unwrap_attitudes(
            [(p.yaw_deg, p.pitch_deg, p.roll_deg) for p in path])

    segs: List[Dict[str, Any]] = []
    total = 0.0
    required = 1
    jumps: List[Dict[str, Any]] = []
    for k, (a, b) in enumerate(zip(path, path[1:])):
        dist = math.dist(a.position, b.position)
        total += dist
        n = max(1, math.ceil(dist / step_m - 1e-9))
        rot = 0.0
        if unwrapped is not None:
            rot = geodesic_deg(rot_matrix(*unwrapped[k]),
                               rot_matrix(*unwrapped[k + 1]))
            n = max(n, math.ceil(rot / rot_step_deg - 1e-9))
            if jump_deg is not None and rot > jump_deg:
                jumps.append({"segment": f"{a.id}->{b.id}",
                              "from_pose": a.id, "to_pose": b.id,
                              "rotation_deg": round(rot, 2),
                              "threshold_deg": jump_deg})
        segs.append({"a": a, "b": b, "n_req": n, "n": n, "rot_deg": rot})
        required += n

    capped = False
    if max_samples is not None and required > max_samples:
        capped = True
        factor = max_samples / required
        used = 1
        for sg in segs:
            sg["n"] = max(1, int(sg["n_req"] * factor))
            used += sg["n"]
        while used > max_samples:  # 取整后仍超限, 从分段最多者继续削
            cand = max((sg for sg in segs if sg["n"] > 1),
                       key=lambda sg: sg["n"], default=None)
            if cand is None:
                break
            cand["n"] -= 1
            used -= 1

    samples: List[Dict[str, Any]] = []
    for k, sg in enumerate(segs):
        a, b, n = sg["a"], sg["b"], sg["n"]
        pa, pb = a.position, b.position
        for j in range(0 if k == 0 else 1, n + 1):
            t = j / n
            pos = (pa[0] + (pb[0] - pa[0]) * t,
                   pa[1] + (pb[1] - pa[1]) * t,
                   pa[2] + (pb[2] - pa[2]) * t)
            at_pose = a.id if j == 0 else (b.id if j == n else None)
            zone = a.zone if j < n else b.zone
            entry: Dict[str, Any] = {"segment": f"{a.id}->{b.id}", "position": pos,
                                     "at_pose": at_pose, "zone": zone}
            if unwrapped is not None:
                ua, ub = unwrapped[k], unwrapped[k + 1]
                entry["attitude"] = tuple(ua[i] + (ub[i] - ua[i]) * t
                                          for i in range(3))
            samples.append(entry)
    meta = {"required_samples": required, "used_samples": len(samples),
            "capped": capped, "jumps": jumps}
    return samples, total, meta


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
    提供 setup.clearance 时, 采样点同步携带插值姿态与净空结果(保守净空、
    碰撞/间距不足/侵入标记), 净空冲突与证据缺口一并汇总。
    """
    idx = _chart_index(setup.load_chart)
    clr = setup.clearance
    if clr is not None:
        samples, total_len, samp_meta = _sample_path(
            setup.path, setup.path_step_m,
            rot_step_deg=clr.attitude_step_deg,
            max_samples=clr.max_samples,
            jump_deg=clr.attitude_jump_deg)
        clr_ctx = prepare_clearance(clr)
    else:
        samples, total_len, samp_meta = _sample_path(setup.path,
                                                     setup.path_step_m)
        clr_ctx = None
    cx, cy = setup.slewing_center
    limit = setup.ground_bearing_limit_kpa

    # 净空: 相邻采样间包络表面点最大位移(扫掠界), 供保守净空折减
    sweeps: List[float] = [0.0] * len(samples)
    if clr_ctx is not None and not clr_ctx["degenerate"]:
        for i in range(1, len(samples)):
            d_hook = math.dist(samples[i - 1]["position"], samples[i]["position"])
            d_theta = geodesic_deg(rot_matrix(*samples[i - 1]["attitude"]),
                                   rot_matrix(*samples[i]["attitude"]))
            sweeps[i] = sweep_bound(d_hook, d_theta, clr_ctx["r_max"])

    out_samples: List[Dict[str, Any]] = []
    first_of_kind: Dict[str, Dict[str, Any]] = {}
    first_violation: Optional[Dict[str, Any]] = None
    clr_first: Optional[Dict[str, Any]] = None
    env = {"max_load_utilization": 0.0,
           "max_ground_pressure_kpa": 0.0,
           "min_outrigger_reaction_kn": math.inf,
           "max_radius_m": 0.0}
    clr_env = {"min_clearance_m": math.inf,
               "min_conservative_clearance_m": math.inf,
               "min_exclusion_conservative_m": math.inf,
               "worst_obstacle_id": None,
               "worst_exclusion_id": None,
               "samples_with_violations": 0}

    for i, s in enumerate(samples):
        px, py, pz = s["position"]
        dx, dy = px - cx, py - cy
        radius = math.hypot(dx, dy)
        azimuth = math.degrees(math.atan2(dy, dx))
        slew = _wrap180(azimuth - setup.slew_reference_deg)
        zone = s["zone"] or setup.default_zone

        gross = chart_capacity(idx, setup.boom_length_m, zone, radius,
                               config=setup.config)
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
        }

        # ---------- 路径净空(与工况共用同一采样点) ----------
        clr_entry = None
        if clr_ctx is not None:
            att = s["attitude"]
            entry["attitude_deg"] = {
                "yaw": round(_wrap180(att[0]), 2),
                "pitch": round(_wrap180(att[1]), 2),
                "roll": round(_wrap180(att[2]), 2),
            }
            if not clr_ctx["degenerate"]:
                ev = _clr_eval_sample(clr_ctx, s["position"], att)
                sweep = max(sweeps[i - 1] if i > 0 else 0.0,
                            sweeps[i + 1] if i + 1 < len(samples) else 0.0)
                margin = clr_ctx["safety_margin_m"]
                c_obs = ev["min_clearance_m"]
                c_obs_cons = c_obs - sweep / 2.0 if c_obs is not None else None
                c_exc = ev["exclusion_min_clearance_m"]
                c_exc_cons = c_exc - sweep / 2.0 if c_exc is not None else None
                cv: List[str] = []
                if c_obs_cons is not None:
                    if c_obs_cons < 0.0:
                        cv.append("COLLISION")
                    elif c_obs_cons < margin:
                        cv.append("CLEARANCE_INSUFFICIENT")
                if c_exc_cons is not None and c_exc_cons < 0.0:
                    cv.append("EXCLUSION_INTRUSION")
                clr_entry = {
                    "envelope_center": [round(v, 4) for v in ev["envelope_center"]],
                    "min_clearance_m": (round(c_obs, 4)
                                        if c_obs is not None else None),
                    "sweep_margin_m": round(sweep / 2.0, 4),
                    "conservative_clearance_m": (round(c_obs_cons, 4)
                                                 if c_obs_cons is not None else None),
                    "worst_obstacle_id": ev["worst_obstacle_id"],
                    "exclusion_min_clearance_m": (round(c_exc, 4)
                                                  if c_exc is not None else None),
                    "worst_exclusion_id": ev["worst_exclusion_id"],
                    "violations": cv,
                }
                entry["clearance"] = clr_entry
                violations.extend(cv)

                if c_obs is not None:
                    clr_env["min_clearance_m"] = min(clr_env["min_clearance_m"], c_obs)
                if c_obs_cons is not None and \
                        c_obs_cons <= clr_env["min_conservative_clearance_m"]:
                    clr_env["min_conservative_clearance_m"] = c_obs_cons
                    clr_env["worst_obstacle_id"] = ev["worst_obstacle_id"]
                if c_exc_cons is not None and \
                        c_exc_cons <= clr_env["min_exclusion_conservative_m"]:
                    clr_env["min_exclusion_conservative_m"] = c_exc_cons
                    clr_env["worst_exclusion_id"] = ev["worst_exclusion_id"]
                if cv:
                    clr_env["samples_with_violations"] += 1

        violations.sort(key=_VIOLATION_ORDER.index)
        entry["violations"] = violations

        if util is not None:
            env["max_load_utilization"] = max(env["max_load_utilization"], util)
        env["max_ground_pressure_kpa"] = max(env["max_ground_pressure_kpa"], max_p)
        env["min_outrigger_reaction_kn"] = min(env["min_outrigger_reaction_kn"], min_r)
        env["max_radius_m"] = max(env["max_radius_m"], radius)
        out_samples.append(entry)

        for kind in violations:
            if kind not in first_of_kind:
                first_of_kind[kind] = {
                    "sample_index": i, "segment": s["segment"],
                    "position": entry["position"], "radius_m": entry["radius_m"],
                    "zone": zone,
                }
        if violations:
            prev = out_samples[i - 1]["position"] if i > 0 else entry["position"]
            hit = {
                "kind": violations[0],
                "kinds": violations,
                "sample_index": i,
                "segment": s["segment"],
                "interval": {"from": prev, "to": entry["position"]},
                "radius_m": entry["radius_m"],
                "zone": zone,
            }
            if first_violation is None:
                first_violation = hit
            if clr_entry is not None and clr_entry["violations"] and clr_first is None:
                clr_first = dict(hit)
                clr_first["kind"] = clr_entry["violations"][0]
                clr_first["kinds"] = clr_entry["violations"]
                clr_first["conservative_clearance_m"] = \
                    clr_entry["conservative_clearance_m"]
                clr_first["worst_obstacle_id"] = clr_entry["worst_obstacle_id"]
                clr_first["worst_exclusion_id"] = clr_entry["worst_exclusion_id"]

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
    boom_rows = [(b, cfg, z, r, c) for (b, cfg, z), rr in idx.items() for (r, c) in rr
                 if abs(b - setup.boom_length_m) <= BOOM_TOL]
    chart_summary = {
        "boom_length_m": setup.boom_length_m,
        "config": setup.config,
        "configs": sorted({cfg for _b, cfg, _z, _r, _c in boom_rows}),
        "zones": sorted({z for _b, _cfg, z, _r, _c in boom_rows}),
        "radius_range_m": ([round(min(r for _b, _cfg, _z, r, _c in boom_rows), 4),
                            round(max(r for _b, _cfg, _z, r, _c in boom_rows), 4)]
                           if boom_rows else None),
        "entries_total": len(setup.load_chart),
        "fingerprint": _chart_fingerprint(setup.load_chart),
    }

    # 净空汇总与证据缺口(角度跳变/包络退化/障碍无效/采样上限)
    clr_summary = None
    evidence_gaps: List[Dict[str, Any]] = []
    if clr_ctx is not None:
        evidence_gaps.extend(clr_ctx["evidence_gaps"])
        if samp_meta["jumps"]:
            evidence_gaps.append({
                "code": "CLEARANCE_ATTITUDE_JUMP",
                "message": "相邻关键姿态转角超过阈值, 插值姿态可能不代表实际转向运动: "
                           + ", ".join(f"{j['segment']} {j['rotation_deg']}°"
                                       for j in samp_meta["jumps"]),
                "evidence": {"segments": samp_meta["jumps"]},
            })
        if samp_meta["capped"]:
            evidence_gaps.append({
                "code": "CLEARANCE_SAMPLING_CAP",
                "message": f"按步长需 {samp_meta['required_samples']} 个采样点, 超过上限 "
                           f"{clr.max_samples}, 已等比放宽至 {samp_meta['used_samples']} 点; "
                           "采样点之间的净空不再由步长保证",
                "evidence": {"required_samples": samp_meta["required_samples"],
                             "used_samples": samp_meta["used_samples"],
                             "max_samples": clr.max_samples},
            })
        n_clr_conf = sum(1 for c in conflicts
                         if c["code"] in ("CRANE_COLLISION",
                                          "CRANE_EXCLUSION_INTRUSION",
                                          "CRANE_CLEARANCE_INSUFFICIENT"))
        if clr_ctx["degenerate"]:
            clr_status = "skipped"
        elif n_clr_conf:
            clr_status = "violation"
        else:
            clr_status = "ok"
        inf = math.inf
        clr_summary = {
            "status": clr_status,
            "safety_margin_m": clr.safety_margin_m,
            "attitude_step_deg": clr.attitude_step_deg,
            "attitude_jump_deg": clr.attitude_jump_deg,
            "max_samples": clr.max_samples,
            "envelope": {
                "length_m": clr.envelope.length_m,
                "width_m": clr.envelope.width_m,
                "height_m": clr.envelope.height_m,
                "hook_to_center_offset": list(clr_ctx["offset"]),
            },
            "obstacles_total": len(clr.obstacles),
            "exclusion_zones_total": len(clr.exclusion_zones),
            "invalid_obstacle_ids": clr_ctx["invalid_obstacle_ids"],
            "invalid_exclusion_ids": clr_ctx["invalid_exclusion_ids"],
            "obstacles": [{"id": o["id"], "min": list(o["min"]),
                           "max": list(o["max"])} for o in clr_ctx["obstacles"]],
            "exclusion_zones": [{"id": z["id"], "min": list(z["min"]),
                                 "max": list(z["max"])}
                                for z in clr_ctx["exclusion_zones"]],
            "sampling": {
                "required_samples": samp_meta["required_samples"],
                "used_samples": samp_meta["used_samples"],
                "capped": samp_meta["capped"],
            },
            "min_clearance_m": (round(clr_env["min_clearance_m"], 4)
                                if clr_env["min_clearance_m"] < inf else None),
            "min_conservative_clearance_m": (
                round(clr_env["min_conservative_clearance_m"], 4)
                if clr_env["min_conservative_clearance_m"] < inf else None),
            "min_exclusion_conservative_m": (
                round(clr_env["min_exclusion_conservative_m"], 4)
                if clr_env["min_exclusion_conservative_m"] < inf else None),
            "worst_obstacle_id": clr_env["worst_obstacle_id"],
            "worst_exclusion_id": clr_env["worst_exclusion_id"],
            "samples_with_violations": clr_env["samples_with_violations"],
            "first_violation": clr_first,
        }

    return {
        "status": "ok" if not conflicts else "violation",
        "crane_id": setup.crane_id,
        "boom_length_m": setup.boom_length_m,
        "config": setup.config,
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
        "clearance": clr_summary,
        "first_violation": first_violation,
        "conflicts": conflicts,
        "evidence_gaps": evidence_gaps,
    }
