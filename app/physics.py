# -*- coding: utf-8 -*-
"""
核心力学求解
============

约定
----
* 本体坐标: x 纵向(梁轴), y 横向, z 竖直向上(水平姿态)。
* 力单位 kN, 长度 m, 角度内部用弧度(接口用度)。
* 刚体平衡的统一线性方程(本体坐标, 未知量为各索张力 T_i ≥ 0):

      Σ T_i d_i = F_ext                 (力平衡, 3 行)
      Σ r_i × T_i d_i = M_ext          (力矩平衡, 3 行)

  d_i 为第 i 根索对被吊体的拉力单位向量(指向吊钩), r_i 为作用点。
  写成  A T = b 后交给 linalg.solve_linear 自动判别
  静定 / 超定(最小二乘) / 静不定(最小范数, 柔度加权)。

* 试吊重心反算把未知量换成 [u_x,u_y,u_z, W](u = W·q, q 为总重心):
      Σ T_i d_i = W · n_b              (n_b = Rᵀ e_z, 世界竖直在本体中的方向)
      Σ r_i × T_i d_i + n_b × u = 0
  多次试吊(不同姿态)方程堆叠; 列秩不足即重心某分量不可观测。

* 动载系数 φ = max(1 + a/g, φ_min); 吊索与卸扣均按动载张力校核。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import linalg as la
from .linalg import Vec3, v_add, v_cross, v_dot, v_mul, v_norm, v_sub, v_unit

# 判定阈值
RESIDUAL_TOL = 1.0e-6          # 方程归一化残差容差
CONFLICT_RESIDUAL = 0.03       # 归一化残差 >3% 视为张力矛盾
WEIGHT_CHECK_TOL = 0.03        # 称重交叉核对 3%
SLACK_TOL = 1.0e-4             # 张力为负/零判定松弛
OVERLOAD_TOL = 1.0             # 利用率上限
MIN_TRIAL_POINTS = 3           # 有效张力测点最低数量


# ================================================================ 几何/姿态
def rot_matrix(pitch_rad: float, roll_rad: float) -> la.Matrix:
    """R = Rx(roll)·Ry(pitch)。世界向量 = R·本体向量(另加平移)。"""
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)
    Ry = [
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp],
    ]
    Rx = [
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr, cr],
    ]
    return la.mat_mul(Rx, Ry)


def rotate(R: la.Matrix, p: Vec3) -> Vec3:
    return tuple(la.mat_vec(R, list(p)))  # type: ignore[return-value]


def dynamic_factor(accel: float, g: float, phi_min: float) -> float:
    """动载系数: 起升加速度的惯性效应, 且不低于规范下限。"""
    return max(1.0 + accel / g, phi_min)


# ================================================================ 力系统
@dataclass
class ForceLane:
    """一根受拉构件(下吊索或上吊索)在力系统中的一行。"""

    id: str
    point: Vec3          # 对被隔离体的作用点(本体坐标)
    direction: Vec3      # 单位拉力方向(作用于被隔离体)
    stiffness: float     # 轴向刚度 kN/m, 静不定时按柔度加权


@dataclass
class ForceSolveResult:
    tensions: List[float]
    kind: str
    residual: List[float]
    residual_norm_rel: float
    nullspace_dim: int
    rank: int
    slack: List[str] = field(default_factory=list)


def solve_force_system(
    lanes: List[ForceLane],
    force_load: Vec3,
    moment_load: Vec3,
    label: str = "",
) -> ForceSolveResult:
    """
    解 Σ T_i d_i = force_load, Σ r_i×T_i d_i = moment_load。

    约定: d_i 为吊索对被隔离体的拉力方向(指向吊钩, 受拉为正)。
    调用方传入需要吊索承担的合力/合力矩(支撑侧), 例如抵抗重力时
    force_load = +W n_b, moment_load = q × W n_b。
    """
    n = len(lanes)
    A = la.mat_zeros(6, n)
    for j, ln in enumerate(lanes):
        d = ln.direction
        for i in range(3):
            A[i][j] = d[i]
        rx, ry, rz = ln.point
        A[3][j] = ry * d[2] - rz * d[1]
        A[4][j] = rz * d[0] - rx * d[2]
        A[5][j] = rx * d[1] - ry * d[0]
    b = [force_load[0], force_load[1], force_load[2],
         moment_load[0], moment_load[1], moment_load[2]]

    # 静不定时最小化弹性应变能 Σ T_i²/(2k_i)
    rank = la.matrix_rank(A)
    have_k = all(ln.stiffness > 0 for ln in lanes)
    if rank < n:
        C = la.mat_zeros(n, n)
        for i, ln in enumerate(lanes):
            k = ln.stiffness if (have_k and ln.stiffness > 0) else 1.0
            C[i][i] = 1.0 / math.sqrt(k)
        Aeq, beq = la.independent_system(A, b)
        if Aeq:
            x, _lam = la.solve_constrained_minnorm(Aeq, beq, C)
            kind = ("statically_indeterminate_elastic" if have_k
                    else "statically_indeterminate_equal_stiffness")
        else:
            x = [0.0] * n
            kind = "degenerate"
    else:
        sol = la.solve_linear(A, b)
        x, kind = sol["x"], sol["kind"]
        rank = sol["rank"]

    slack = [lanes[i].id for i in range(n) if x[i] < -SLACK_TOL]
    scale = la.rms(b) or 1.0
    res = la.residual(A, x, b)
    rel = la.rms(res) / scale
    return ForceSolveResult(
        tensions=x, kind=kind, residual=res, residual_norm_rel=rel,
        nullspace_dim=max(n - rank, 0), rank=rank, slack=slack,
    )


# ================================================================ 方案几何
@dataclass
class RigPart:
    weight_kn: float
    cog: Vec3
    name: str


@dataclass
class Geometry:
    pads: List[Vec3]
    leg_ids: List[str]
    lengths: List[float]
    lug_points: List[Vec3]          # 每根下吊索上端指向点(本体坐标)
    beam: Any = None
    hook: Optional[Vec3] = None
    tops: Optional[List[Any]] = None
    effective_top: Optional[Vec3] = None  # 多上索时取均值点(预测姿态参考)
    top_concurrent: bool = True


def build_geometry(inp, shims: Optional[Dict[str, float]] = None) -> Geometry:
    """
    从输入构造方案几何(水平姿态、本体坐标)。

    无吊梁: 所有下吊索指向同一吊钩点(显式给 hook_point, 否则取吊点形心
    正上方, 高度按平均索长或吊点展宽的 0.6 倍估计)。
    带吊梁: 吊耳位于 (lug_x, beam_y, beam_z); 上吊索汇交点取自 tops。
    shims: 索长增减 m(正放长/负收短), 仅改长度不改点。
    """
    shims = shims or {}
    beam = inp.beam
    pads = [tuple(lg.pad) for lg in inp.legs]  # type: ignore[misc]
    lengths = [lg.length_m for lg in inp.legs]

    if beam is not None:
        n_lugs = len(beam.lower_lugs_x)
        same_count = n_lugs == len(inp.legs)
        lug_points: List[Vec3] = []
        for i, lg in enumerate(inp.legs):
            li = lg.lug_index if lg.lug_index is not None else (i if same_count else 0)
            x = beam.lower_lugs_x[li]
            lug_points.append((x, beam.beam_y, beam.beam_z))
        tops = list(beam.tops)
        conv = [tuple(t.convergence) for t in tops]
        h_eff = (sum(c[0] for c in conv) / len(conv),
                 sum(c[1] for c in conv) / len(conv),
                 sum(c[2] for c in conv) / len(conv))
        spread = max(max(c[i] for c in conv) - min(c[i] for c in conv) for i in range(3))
        concurrent = spread < 1e-6
        hook = conv[0] if concurrent else None
        geo = Geometry(pads, [lg.id for lg in inp.legs], lengths, lug_points,
                       beam=beam, hook=hook, tops=tops,
                       effective_top=h_eff, top_concurrent=concurrent)
    else:
        if inp.hook_point is not None:
            hook = tuple(inp.hook_point)  # type: ignore[assignment]
        else:
            cx = sum(p[0] for p in pads) / len(pads)
            cy = sum(p[1] for p in pads) / len(pads)
            zmax = max(p[2] for p in pads)
            given = [L for L in lengths if L > 0]
            if given:
                h = sum(given) / len(given)
            else:
                span = max(
                    math.sqrt(max((p[0] - cx) ** 2 + (p[1] - cy) ** 2 for p in pads)),
                    1e-6,
                )
                h = max(span / 1.2, 1.0)  # 对应约 50° 吊索角
            hook = (cx, cy, zmax + h)
        lug_points = [hook for _ in pads]
        geo = Geometry(pads, [lg.id for lg in inp.legs], lengths, lug_points,
                       hook=hook, effective_top=hook, top_concurrent=True)

    # 公称长度缺省时由几何补齐(用于索角/刚度/调整量)
    for i, L in enumerate(geo.lengths):
        if L <= 0:
            geo.lengths[i] = v_norm(v_sub(geo.lug_points[i], geo.pads[i]))
    for lid, dl in shims.items():
        j = geo.leg_ids.index(lid)
        geo.lengths[j] += dl
    return geo


def nominal_body_directions(geo: Geometry) -> List[Vec3]:
    return [v_unit(v_sub(geo.lug_points[i], geo.pads[i])) for i in range(len(geo.pads))]


def tilted_pad_world(pad: Vec3, cog: Vec3, R: la.Matrix) -> Vec3:
    """
    倾斜后吊点的世界坐标(等价地: 水平姿态坐标系)。
    刚体绕重心转动 R, 重心在世界中的位置近似不动(始终在吊钩铅垂线上)。
    """
    return v_add(rotate(R, v_sub(pad, cog)), cog)


def tilted_body_directions(
    geo: Geometry, pitch_rad: float, roll_rad: float, cog: Vec3,
) -> Tuple[List[Vec3], Vec3]:
    """
    构件倾斜后下吊索的方向。

    模型: 上部连接点(吊钩/吊梁下耳)在世界系中位置不动; 构件为刚体绕重心
    转动 R。则
        d_w_i = û_i − p̂_i (上端点静止, 下端点随刚体转动),
        d_b_i = Rᵀ d_w_i (转回本体坐标供平衡方程使用)。
    水平姿态(R=I)时严格退化为名义几何; 索长变化忽略(柔性索, 静力仅取方向)。
    """
    R = rot_matrix(pitch_rad, roll_rad)
    dirs: List[Vec3] = []
    for i, pad in enumerate(geo.pads):
        pad_w = tilted_pad_world(pad, cog, R)
        d_w = v_unit(v_sub(geo.lug_points[i], pad_w))
        d_b = tuple(la.mat_vec(la.mat_T(R), list(d_w)))
        dirs.append(v_unit(d_b))
    n_b = tuple(R[k][2] for k in range(3))  # Rᵀ e3
    return dirs, n_b


def predicted_tilt(cog: Vec3, geo: Geometry) -> Tuple[float, float]:
    """
    刚体在单点(汇交)起吊下的预测平衡倾角(闭式)。

    条件: 相对上部汇交点的向量 r = cog − p0 经 R 后与世界铅垂线平行,
    即其两个水平分量为零。R = Rx(roll)·Ry(pitch):
        cp·r_x + sp·r_z = 0                 -> pitch = atan2(r_x, −r_z)
        cr·r_y − sr·z1  = 0,  z1=−sp r_x+cp r_z
                                            -> roll  = atan2(r_y, z1)
    多上索非汇交时 p0 取均值(工程近似, 返回结果中标记)。
    """
    p0 = geo.effective_top
    rx, ry, rz = v_sub(cog, p0)
    # 稳定(小角度)平衡支: R r 铅垂向下, 即 r_z < 0:
    #   pitch = atan2(r_x, −r_z),  roll = atan2(r_y, −z1)
    pitch = math.atan2(rx, -rz) if abs(rz) > 1e-9 else (
        math.copysign(math.pi / 2.0, rx) if rx else 0.0)
    sp, cp = math.sin(pitch), math.cos(pitch)
    z1 = sp * rx + cp * rz
    roll = math.atan2(ry, -z1) if abs(z1) > 1e-9 else (
        math.copysign(math.pi / 2.0, ry) if ry else 0.0)
    return pitch, roll


# ================================================================ 吊装核算
def _assembled_parts(inp, phi: float) -> Tuple[List[RigPart], List[RigPart]]:
    """返回 (动载部件列表, 静载部件列表), 含构件/配重/附加索具/吊梁。"""
    g = inp.g_mps2
    dyn = [RigPart(inp.component_weight_kn * phi, tuple(inp.cog_theory), "component")]
    sta = [RigPart(inp.component_weight_kn, tuple(inp.cog_theory), "component")]
    if inp.counterweight is not None and inp.counterweight.mass_kg > 0:
        w = inp.counterweight.mass_kg * g / 1000.0
        c = tuple(inp.counterweight.position)
        dyn.append(RigPart(w * phi, c, f"counterweight:{inp.counterweight.id}"))
        sta.append(RigPart(w, c, f"counterweight:{inp.counterweight.id}"))
    if inp.extra_rigging_mass_kg > 0:
        w = inp.extra_rigging_mass_kg * g / 1000.0
        c = tuple(inp.extra_rigging_cog)
        dyn.append(RigPart(w * phi, c, "extra_rigging"))
        sta.append(RigPart(w, c, "extra_rigging"))
    if inp.beam is not None and inp.beam.beam_mass_kg > 0:
        w = inp.beam.beam_mass_kg * g / 1000.0
        cx = inp.beam.beam_cg_x
        if cx is None:
            cx = 0.5 * (inp.beam.end_a_x + inp.beam.end_b_x)
        c = (cx, inp.beam.beam_y, inp.beam.beam_z)
        dyn.append(RigPart(w * phi, c, "spreader_beam"))
        sta.append(RigPart(w, c, "spreader_beam"))
    return dyn, sta


def _resultant(parts: List[RigPart]) -> Tuple[float, Vec3]:
    W = sum(p.weight_kn for p in parts)
    if W <= 0:
        return 0.0, (0.0, 0.0, 0.0)
    cog = (
        sum(p.weight_kn * p.cog[0] for p in parts) / W,
        sum(p.weight_kn * p.cog[1] for p in parts) / W,
        sum(p.weight_kn * p.cog[2] for p in parts) / W,
    )
    return W, cog


def _lower_parts(inp, phi: float) -> List[RigPart]:
    """下吊索承担的部件(不含吊梁自重; 梁自重走梁的隔离体)。"""
    g = inp.g_mps2
    parts = [RigPart(inp.component_weight_kn * phi, tuple(inp.cog_theory), "component")]
    if inp.counterweight is not None and inp.counterweight.mass_kg > 0:
        w = inp.counterweight.mass_kg * g / 1000.0
        parts.append(RigPart(w * phi, tuple(inp.counterweight.position), "counterweight"))
    if inp.extra_rigging_mass_kg > 0:
        w = inp.extra_rigging_mass_kg * g / 1000.0
        parts.append(RigPart(w * phi, tuple(inp.extra_rigging_cog), "extra_rigging"))
    return parts


def _leg_stiffness(leg, length_m: float) -> float:
    ea = leg.sling_ea_kn
    if ea is None or length_m <= 0:
        return 0.0
    return ea / length_m * leg.sling_count


def _sling_geometry(leg_d_b: Vec3) -> Dict[str, float]:
    vert = max(min(leg_d_b[2], 1.0), -1.0)
    angle_horiz = math.degrees(math.asin(max(vert, -1.0)))
    return {
        "vertical_component": vert,
        "angle_to_horizontal_deg": angle_horiz,
        "tension_multiplier": 1.0 / vert if vert > 1e-6 else None,  # type: ignore[dict-item]
    }


def analyze_lift(inp, inverse: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    正式起吊受力核算主入口。

    inverse: 试吊反算结果(若提供且成功, 用反算重心替换理论重心)。
    """
    g = inp.g_mps2
    phi = dynamic_factor(inp.lift_acceleration_mps2, g, inp.min_dynamic_factor)

    # ---------- 采用的实际重心
    cog_note = "theory"
    cog_actual: Optional[Vec3] = None
    if inp.cog_actual_override is not None:
        cog_actual = tuple(inp.cog_actual_override)  # type: ignore[assignment]
        cog_note = "override"
    elif inverse is not None and inverse.get("status") == "ok":
        cw = inverse.get("cog_component_world") or inverse.get("cog_component")
        if cw is not None:
            cog_actual = tuple(cw)  # type: ignore[assignment]
            cog_note = "inverse_from_trials"

    # 深拷贝式地构造“核算用输入”(替换重心/人工调整在外部派生新对象完成)
    shims: Dict[str, float] = {}
    if inp.manual_adjustment is not None:
        for s in inp.manual_adjustment.shims:
            shims[s.leg_id] = shims.get(s.leg_id, 0.0) + s.delta_length_m
    geo = build_geometry(inp, shims)

    # 本体坐标里的部件(替换实际重心)
    parts_dyn = _lower_parts(inp, phi)
    parts_sta = _lower_parts(inp, 1.0)
    if cog_actual is not None:
        for p in parts_dyn + parts_sta:
            if p.name == "component":
                p.cog = cog_actual

    # ---------- 预测姿态
    W_lower_dyn, cog_assembly = _resultant(parts_dyn)
    pitch, roll = predicted_tilt(cog_assembly, geo)
    tilt_mag = math.degrees(math.acos(
        max(-1.0, min(1.0, math.cos(pitch) * math.cos(roll)))
    ))

    # ---------- 下吊索张力: 用倾斜后本体方向
    dirs_b, n_b = tilted_body_directions(geo, pitch, roll, cog_assembly)
    # 外载荷需求(本体): 吊索需提供 +W n_b 的合力, 合力矩 q×(W n_b)
    Fext = v_mul(n_b, W_lower_dyn)
    Mext = v_cross(cog_assembly, Fext)
    lanes = []
    for i, leg in enumerate(inp.legs):
        lanes.append(ForceLane(
            id=leg.id, point=geo.pads[i], direction=dirs_b[i],
            stiffness=_leg_stiffness(leg, geo.lengths[i]),
        ))
    lower = solve_force_system(lanes, Fext, Mext, label="lower")

    # 静载(静止离地)张力, 供“静载利用率/称重复核”
    Fext0 = (0.0, 0.0, sum(p.weight_kn for p in parts_sta))
    W0, cog0 = _resultant(parts_sta)
    dirs0 = nominal_body_directions(geo)
    lanes0 = [ForceLane(l.id, l.point, d, l.stiffness)
              for l, d in zip(lanes, dirs0)]
    lower_static = solve_force_system(lanes0, Fext0, v_cross(cog0, Fext0), "lower_static")

    conflicts: List[Dict[str, Any]] = []
    evidence_gaps: List[Dict[str, Any]] = []
    warnings: List[str] = []

    if lower.slack:
        conflicts.append({
            "code": "SLACK_LEGS",
            "message": f"刚性平衡解要求以下吊索松弛(负张力), 几何/索长不自洽: {lower.slack}",
            "equations": ["ΣT d = −W n_b", "Σ r×T d = q×(−W n_b)"],
            "evidence": {"slack_legs": lower.slack,
                         "tensions_kn": dict(zip(geo.leg_ids, _round(lower.tensions)))},
        })
    if lower.nullspace_dim > 0:
        warnings.append(
            f"下吊索系统静不定(零空间维数 {lower.nullspace_dim}); "
            + ("按给定 EA/索长的弹性刚度分配张力"
               if all(l.stiffness > 0 for l in lanes)
               else "未提供 sling_ea_kn, 按等刚度(刚性吊梁/匀质索)经验分配, "
                    "正式方案应补测刚度或逐索测力计数据")
        )
    if lower.residual_norm_rel > CONFLICT_RESIDUAL:
        conflicts.append({
            "code": "LOWER_EQUILIBRIUM_RESIDUAL",
            "message": "下吊索平衡方程残差过大, 输入几何/载荷互不相容",
            "equations": ["force", "moment"],
            "evidence": {"residual": _round(lower.residual),
                         "relative_rms": lower.residual_norm_rel},
        })
    if cog_note == "theory":
        evidence_gaps.append({
            "code": "NO_VERIFIED_COG",
            "message": "未经试吊反算或人工确认实际重心, 仅按理论重心核算",
        })
    if tilt_mag > 0.5:
        warnings.append(f"预测倾斜 {tilt_mag:.2f}° (pitch {math.degrees(pitch):.2f}°, "
                        f"roll {math.degrees(roll):.2f}°), 需调平")

    # ---------- 逐索利用率
    leg_results = []
    for i, leg in enumerate(inp.legs):
        Td = lower.tensions[i]
        Ts = lower_static.tensions[i]
        # 每股张力(并联索股)
        per_strand_d = Td / leg.sling_count
        per_strand_s = Ts / leg.sling_count
        wll_sling = leg.sling_capacity_kn * leg.derating_factor
        wll_shackle = leg.shackle_capacity_kn  # 卸扣折减按其自身证书, 这里不重复折减
        geom = _sling_geometry(dirs_b[i])
        geom0 = _sling_geometry(dirs0[i])
        u_sling = per_strand_d / wll_sling
        u_shackle = per_strand_d / wll_shackle
        min_angle = inp.min_sling_angle_deg
        ang_warn = geom0["angle_to_horizontal_deg"] < min_angle - 1e-9
        leg_results.append({
            "leg_id": leg.id,
            "pad": _round(geo.pads[i]),
            "upper_point": _round(geo.lug_points[i]),
            "length_m": round(geo.lengths[i], 4),
            "sling_count": leg.sling_count,
            "tension_dynamic_kn": round(Td, 3),
            "tension_static_kn": round(Ts, 3),
            "tension_per_strand_dynamic_kn": round(per_strand_d, 3),
            "angle_to_horizontal_deg": round(geom0["angle_to_horizontal_deg"], 2),
            "angle_at_tilt_deg": round(geom["angle_to_horizontal_deg"], 2),
            "vertical_share": round(geom0["vertical_component"], 4),
            "sling": {
                "rated_kn": leg.sling_capacity_kn,
                "derating": leg.derating_factor,
                "effective_wll_kn": round(wll_sling, 3),
                "utilization": round(u_sling, 4),
                "safety_margin_kn": round(wll_sling - per_strand_d, 3),
                "safety_factor": round(la.safe_div(wll_sling, per_strand_d), 3),
            },
            "shackle": {
                "rated_kn": leg.shackle_capacity_kn,
                "utilization": round(u_shackle, 4),
                "safety_margin_kn": round(wll_shackle - per_strand_d, 3),
                "safety_factor": round(la.safe_div(wll_shackle, per_strand_d), 3),
            },
            "angle_below_minimum": ang_warn,
        })
        if u_sling > OVERLOAD_TOL or u_shackle > OVERLOAD_TOL:
            conflicts.append({
                "code": "LEG_OVERLOAD",
                "message": f"吊索 {leg.id} 动载利用率超限 "
                           f"(索 {u_sling:.2%}, 卸扣 {u_shackle:.2%})",
                "equations": ["T_dyn = φ·分配张力/股数", "u = T_dyn / (WLL·折减)"],
                "evidence": {"leg_id": leg.id, "sling_util": round(u_sling, 4),
                             "shackle_util": round(u_shackle, 4)},
            })
        if ang_warn:
            conflicts.append({
                "code": "SLING_ANGLE_LOW",
                "message": f"吊索 {leg.id} 水平夹角 {geom0['angle_to_horizontal_deg']:.1f}° "
                           f"低于最小允许 {min_angle}°",
                "equations": ["T = W_share / sin(β)"],
                "evidence": {"angle_deg": round(geom0["angle_to_horizontal_deg"], 2),
                             "minimum_deg": min_angle},
            })

    # ---------- 吊梁: 上吊索 + 梁内力
    beam_result = None
    hook_load_dyn = W_lower_dyn
    if inp.beam is not None:
        beam_result = _analyze_beam(inp, geo, lower, phi, parts_dyn, cog_assembly, conflicts)
        hook_load_dyn = beam_result["hook_load_dynamic_kn"]
        for c in beam_result.get("conflicts", []):
            conflicts.append(c)

    # ---------- 吊钩/起升机
    hoist = None
    if inp.hoist_capacity_kn:
        u = hook_load_dyn / inp.hoist_capacity_kn
        hoist = {"rated_kn": inp.hoist_capacity_kn,
                 "load_dynamic_kn": round(hook_load_dyn, 3),
                 "utilization": round(u, 4),
                 "safety_margin_kn": round(inp.hoist_capacity_kn - hook_load_dyn, 3)}
        if u > OVERLOAD_TOL:
            conflicts.append({
                "code": "HOIST_OVERLOAD",
                "message": f"吊钩/起升机动载载荷 {hook_load_dyn:.1f} kN 超过额定",
                "equations": ["P_hook = Σ 上吊索张力"],
                "evidence": hoist,
            })

    # ---------- 起重机工况(回转路径逐姿态校核, 复用吊钩动载) ----------
    crane_result = None
    if inp.crane is not None:
        from .crane import check_crane_duty  # 局部导入, 避免与 crane 模块循环依赖
        crane_result = check_crane_duty(inp.crane, hook_load_dyn)
        conflicts.extend(crane_result["conflicts"])
        evidence_gaps.extend(crane_result.get("evidence_gaps", []))

    # ---------- 载荷分配比例
    shares = {
        geo.leg_ids[i]: round(lower.tensions[i] * dirs_b[i][2] / W_lower_dyn, 4)
        for i in range(len(inp.legs))
    }

    out = {
        "status": "ok",
        "dynamic_factor": round(phi, 4),
        "cog_used": {"cog": _round(cog_actual or tuple(inp.cog_theory)),
                     "source": cog_note},
        "cog_offset_from_theory_m": (
            _round(v_sub(cog_actual, tuple(inp.cog_theory))) if cog_actual else [0, 0, 0]
        ),
        "weights_kn": {
            "component": inp.component_weight_kn,
            "lower_assembly_dynamic": round(W_lower_dyn, 3),
            "hook_dynamic": round(hook_load_dyn, 3),
            "beam_self": round(inp.beam.beam_mass_kg * g / 1000.0, 3) if inp.beam else 0.0,
            "counterweight": round(inp.counterweight.mass_kg * g / 1000.0, 3)
            if inp.counterweight else 0.0,
            "extra_rigging": round(inp.extra_rigging_mass_kg * g / 1000.0, 3),
        },
        "predicted_attitude": {
            "pitch_deg": round(math.degrees(pitch), 3),
            "roll_deg": round(math.degrees(roll), 3),
            "tilt_deg": round(tilt_mag, 3),
            "method": "rigid_cg_below_convergence" +
                      ("" if geo.top_concurrent else "_multi_top_mean_approx"),
        },
        "distribution": {
            "kind": lower.kind,
            "vertical_load_shares": shares,
            "max_share_ratio": round(max(shares.values()), 4),
            "equilibrium_residual_rel_rms": round(lower.residual_norm_rel, 6),
        },
        "legs": leg_results,
        "beam": beam_result,
        "hoist": hoist,
        "crane": crane_result,
        "conflicts": conflicts,
        "evidence_gaps": evidence_gaps,
        "warnings": warnings,
    }
    return out


# ================================================================ 吊梁
def _analyze_beam(inp, geo: Geometry, lower: ForceSolveResult,
                  phi: float, parts_dyn: List[RigPart], cog_assembly: Vec3) -> Dict[str, Any]:
    """
    吊梁隔离体:
      下吊索在吊耳处对梁的作用力 = −T_i d_b_i (向下压梁);
      上吊索拉力向上; 梁自重作用于梁重心。
    解上索张力后, 沿梁轴逐站计算 V/M 包络并校核截面。
    """
    beam = inp.beam
    conflicts: List[Dict[str, Any]] = []
    g = inp.g_mps2

    # 下吊索对梁的力(作用点为吊耳): 梁保持水平, 力在世界系(=水平本体架)中,
    # 方向为 lug − 倾斜后的 pad; 对梁的作用力与对构件的力反向。
    pitch_b, roll_b, cog_b = _attitude_for_beam(inp, geo)
    Rb = rot_matrix(pitch_b, roll_b)
    lower_forces: List[Tuple[float, Vec3]] = []
    for i, leg in enumerate(inp.legs):
        pad_w = tilted_pad_world(geo.pads[i], cog_b, Rb)
        d_w = v_unit(v_sub(geo.lug_points[i], pad_w))
        F = v_mul(d_w, -lower.tensions[i])
        lower_forces.append((geo.lug_points[i][0], F))

    beam_w = beam.beam_mass_kg * g / 1000.0 * phi
    cgx = beam.beam_cg_x if beam.beam_cg_x is not None else 0.5 * (beam.end_a_x + beam.end_b_x)

    # 上索
    top_dirs = [v_unit(v_sub(tuple(t.convergence), (t.at_x, beam.beam_y, beam.beam_z)))
                for t in beam.tops]
    lanes = []
    for t, d in zip(beam.tops, top_dirs):
        lanes.append(ForceLane(t.id, (t.at_x, beam.beam_y, beam.beam_z), d,
                               stiffness=1.0e6))  # 上索长度差异小, 等刚度近似
    # 外载合力/矩(梁隔离体)
    Fext = (0.0, 0.0, -(beam_w))
    for _, F in lower_forces:
        Fext = v_add(Fext, F)
    Mext = v_cross((cgx, beam.beam_y, beam.beam_z), (0.0, 0.0, -beam_w))
    for x, F in lower_forces:
        Mext = v_add(Mext, v_cross((x, beam.beam_y, beam.beam_z), F))
    upper = solve_force_system(lanes, Fext, Mext, "upper")

    top_results = []
    for i, t in enumerate(beam.tops):
        T = upper.tensions[i] / t.sling_count
        wll = t.sling_capacity_kn * t.derating_factor
        us = T / wll
        uk = T / t.shackle_capacity_kn
        ang = math.degrees(math.asin(max(min(top_dirs[i][2], 1), -1)))
        top_results.append({
            "top_sling_id": t.id,
            "at_x": t.at_x,
            "tension_dynamic_kn": round(upper.tensions[i], 3),
            "per_strand_kn": round(T, 3),
            "angle_deg": round(ang, 2),
            "sling_utilization": round(us, 4),
            "shackle_utilization": round(uk, 4),
            "sling_margin_kn": round(wll - T, 3),
        })
        if max(us, uk) > OVERLOAD_TOL:
            conflicts.append({"code": "TOP_SLING_OVERLOAD",
                              "message": f"上吊索 {t.id} 利用率超限 (索 {us:.2%}, 卸扣 {uk:.2%})",
                              "equations": ["梁隔离体力/矩平衡"],
                              "evidence": {"top_id": t.id, "sling_util": us, "shackle_util": uk}})
    if upper.slack:
        conflicts.append({"code": "TOP_SLACK",
                          "message": f"上吊索松弛解: {upper.slack}", "equations": [],
                          "evidence": {"slack": upper.slack}})
    if upper.residual_norm_rel > CONFLICT_RESIDUAL:
        conflicts.append({"code": "UPPER_EQUILIBRIUM_RESIDUAL",
                          "message": "上吊索平衡残差过大", "equations": [],
                          "evidence": {"relative_rms": upper.residual_norm_rel}})

    # ---------- 梁内力包络: 所有作用点设站, 截面左侧外力合成
    # ---------- 梁内力包络: 所有连接点设站, 截面左侧外力合成;
    #            梁自重按均布 w kN/m 处理
    point_loads: List[Tuple[float, Vec3]] = list(lower_forces)
    for i, t in enumerate(beam.tops):
        point_loads.append((t.at_x, v_mul(top_dirs[i], upper.tensions[i])))
    point_loads.sort(key=lambda s: s[0])

    length = beam.end_b_x - beam.end_a_x
    wdist = beam_w / length if length > 0 else 0.0

    envelope = {"axial_kn": 0.0, "shear_y_kn": 0.0, "shear_z_kn": 0.0,
                "moment_y_knm": 0.0, "moment_z_knm": 0.0, "torsion_x_knm": 0.0}
    critical_x = beam.end_a_x
    scan = sorted({beam.end_a_x, beam.end_b_x} | {x for x, _ in point_loads})
    # 截面 x 处, 取左侧隔离体: 内力 = −Σ左外力
    for x in scan:
        Fl = (0.0, 0.0, 0.0)
        Ml = (0.0, 0.0, 0.0)
        for px, F in point_loads:
            if px <= x + 1e-9:
                Fl = v_add(Fl, F)
                Ml = v_add(Ml, v_cross((px - x, 0.0, 0.0), F))
        # 均布自重(左段)
        if x > beam.end_a_x and wdist:
            seg = x - beam.end_a_x
            Fw = (0.0, 0.0, -wdist * seg)
            Fl = v_add(Fl, Fw)
            Ml = v_add(Ml, v_cross((beam.end_a_x + 0.5 * seg - x, 0.0, 0.0), Fw))
        N = -Fl[0]
        Vy, Vz = -Fl[1], -Fl[2]
        Mx, My, Mz = -Ml[0], -Ml[1], -Ml[2]
        if abs(My) > abs(envelope["moment_y_knm"]):
            envelope["moment_y_knm"] = My
            critical_x = x
        envelope["axial_kn"] = max(envelope["axial_kn"], abs(N))
        envelope["shear_y_kn"] = max(envelope["shear_y_kn"], abs(Vy))
        envelope["shear_z_kn"] = max(envelope["shear_z_kn"], abs(Vz))
        envelope["moment_z_knm"] = max(envelope["moment_z_knm"], abs(Mz))
        envelope["torsion_x_knm"] = max(envelope["torsion_x_knm"], abs(Mx))

    sec = beam.section
    stress = (abs(envelope["axial_kn"]) / sec.area_m2
              + abs(envelope["moment_y_knm"]) / sec.wy_m3
              + abs(envelope["moment_z_knm"]) / sec.wz_m3)
    util_stress = stress / sec.allowable_stress_kpa
    shear_util = None
    if sec.shear_capacity_kn:
        Vmax = math.hypot(envelope["shear_y_kn"], envelope["shear_z_kn"])
        shear_util = Vmax / sec.shear_capacity_kn
        if shear_util > OVERLOAD_TOL:
            conflicts.append({"code": "BEAM_SHEAR_OVER",
                              "message": "吊梁抗剪超限", "equations": ["V=ΣF_z(左段)"],
                              "evidence": {"vmax_kn": round(Vmax, 2),
                                           "capacity": sec.shear_capacity_kn}})
    if util_stress > OVERLOAD_TOL:
        conflicts.append({"code": "BEAM_STRESS_OVER",
                          "message": f"吊梁轴力+弯曲组合应力 {stress:.0f} kPa 超许用 "
                                     f"{sec.allowable_stress_kpa:.0f} kPa",
                          "equations": ["σ=|N|/A+|My|/Wy+|Mz|/Wz"],
                          "evidence": {"stress_kpa": round(stress, 1),
                                       "allowable_kpa": sec.allowable_stress_kpa,
                                       "at_x": round(critical_x, 3),
                                       **{k: round(v, 3) for k, v in envelope.items()}}})
    torsion_note = "理想共轴连接点, 扭转按 0 计; 偏心吊耳需另行核算"
    hook_load = sum(upper.tensions[i] * top_dirs[i][2] for i in range(len(beam.tops)))

    return {
        "top_slings": top_results,
        "solve_kind": upper.kind,
        "hook_load_dynamic_kn": round(hook_load, 3),
        "internal_force_envelope": {k: round(v, 3) for k, v in envelope.items()},
        "critical_section_x_m": round(critical_x, 3),
        "stress_kpa": round(stress, 1),
        "allowable_stress_kpa": sec.allowable_stress_kpa,
        "stress_utilization": round(util_stress, 4),
        "shear_utilization": round(shear_util, 4) if shear_util is not None else None,
        "torsion_note": torsion_note,
        "conflicts": conflicts,
    }


def _attitude_for_beam(inp, geo: Geometry) -> Tuple[float, float, Vec3]:
    """吊梁核算下吊索方向时采用的姿态与总重心: 与主分析一致的预测倾角。"""
    phi = dynamic_factor(inp.lift_acceleration_mps2, inp.g_mps2, inp.min_dynamic_factor)
    parts = _lower_parts(inp, phi)
    if inp.cog_actual_override is not None:
        cog_use = tuple(inp.cog_actual_override)
        for p in parts:
            if p.name == "component":
                p.cog = cog_use
    _W, cog = _resultant(parts)
    pitch, roll = predicted_tilt(cog, geo)
    return pitch, roll, cog


# ================================================================ 试吊反算
def inverse_cog_from_trials(inp) -> Dict[str, Any]:
    """
    由一次或多次低高度试吊的实测张力/倾角反算构件实际重心。

    堆叠方程(每次试吊, 本体坐标):
        Σ T_i d_i = W n_b                    (力)
        Σ r_i×T_i d_i + n_b × u = 0          (矩, u = W q_total)
    返回冲突方程、证据缺口与不可观测分量。
    """
    phi = dynamic_factor(inp.lift_acceleration_mps2, inp.g_mps2, inp.min_dynamic_factor)
    geo0 = build_geometry(inp)
    dirs_level = nominal_body_directions(geo0)

    rows: List[List[float]] = []
    rhs: List[float] = []
    row_meta: List[Tuple[str, str]] = []      # (trial_id, 行名)
    per_trial: Dict[str, Dict[str, Any]] = {}
    conflicts: List[Dict[str, Any]] = []
    evidence_gaps: List[Dict[str, Any]] = []
    angles_only_trials: List[str] = []

    rig_parts = []   # 附加在总体系上的非构件重量(动载)
    g = inp.g_mps2
    if inp.extra_rigging_mass_kg > 0:
        rig_parts.append((inp.extra_rigging_mass_kg * g / 1000.0 * phi,
                          tuple(inp.extra_rigging_cog), "extra_rigging"))
    if inp.beam is not None and inp.beam.beam_mass_kg > 0:
        cx = inp.beam.beam_cg_x
        if cx is None:
            cx = 0.5 * (inp.beam.end_a_x + inp.beam.end_b_x)
        rig_parts.append((inp.beam.beam_mass_kg * g / 1000.0 * phi,
                          (cx, inp.beam.beam_y, inp.beam.beam_z), "spreader_beam"))
    if inp.counterweight is not None and inp.counterweight.mass_kg > 0:
        rig_parts.append((inp.counterweight.mass_kg * g / 1000.0 * phi,
                          tuple(inp.counterweight.position), "counterweight"))

    total_measured_points = 0

    for tr in inp.trials:
        pitch = math.radians(tr.pitch_deg)
        roll = math.radians(tr.roll_deg)
        R = rot_matrix(pitch, roll)
        n_b = tuple(R[k][2] for k in range(3))
        m_by_id = {m.leg_id: m for m in tr.measurements}

        Fsum = [0.0, 0.0, 0.0]
        Msum = [0.0, 0.0, 0.0]
        measured = 0
        angle_checks = []

        for i, lid in enumerate(geo0.leg_ids):
            if lid not in m_by_id:
                continue
            m = m_by_id[lid]
            d_b, dir_source = _trial_direction(
                m, dirs_level[i], R, geo0.lug_points[i], geo0.pads[i]
            )
            if m.measured_angle_deg is not None or m.measured_direction is not None:
                ang_meas = m.measured_angle_deg
                if m.measured_direction is not None:
                    dw = v_unit(tuple(m.measured_direction))
                    ang_meas = math.degrees(math.asin(max(-1, min(1, dw[2]))))
                ang_nom = math.degrees(math.asin(max(-1, min(1, dirs_level[i][2], 1))))
                angle_checks.append((lid, ang_meas, ang_nom))
            if m.tension_kn is None:
                continue
            # 反算在“动载状态”方程中进行(右端 W 含 φ):
            #   动态张力(起升中测得, tension_is_static=False)直接采用;
            #   静态张力(静止离地测得)需先乘动载系数 φ 换算为等效动载。
            # φ=1.1 时: 输入 110 kN 动态 或 100 kN 静态, 进入方程均为 110 kN,
            # 最终反算静载总重一致(100 kN)。
            T = m.tension_kn * (phi if m.tension_is_static else 1.0)
            Fsum = [Fsum[k] + T * d_b[k] for k in range(3)]
            rx, ry, rz = geo0.pads[i]
            Msum[0] += T * (ry * d_b[2] - rz * d_b[1])
            Msum[1] += T * (rz * d_b[0] - rx * d_b[2])
            Msum[2] += T * (rx * d_b[1] - ry * d_b[0])
            measured += 1
            total_measured_points += 1

        if measured == 0:
            angles_only_trials.append(tr.id)
            evidence_gaps.append({
                "code": "NO_TENSION_DATA",
                "message": f"试吊 {tr.id} 无张力测点, 仅有角度/方向, 不能参与重心反算",
                "trial_id": tr.id,
            })
        elif measured < MIN_TRIAL_POINTS:
            evidence_gaps.append({
                "code": "INSUFFICIENT_TRIAL_POINTS",
                "message": f"试吊 {tr.id} 仅 {measured} 个张力测点, "
                           f"至少 {MIN_TRIAL_POINTS} 个才能稳定识别重量与重心",
                "trial_id": tr.id, "points": measured,
            })

        # 角度/方向与名义几何的一致性(2°)
        for lid, ang_m, ang_n in angle_checks:
            if ang_m is not None and abs(ang_m - ang_n) > 2.0 and abs(tr.pitch_deg) + abs(tr.roll_deg) < 0.5:
                conflicts.append({
                    "code": "ANGLE_MISMATCH",
                    "message": f"试吊 {tr.id} 吊索 {lid} 实测角 {ang_m:.1f}° 与名义几何 "
                               f"{ang_n:.1f}° 相差 >2°",
                    "equations": [f"β({lid}) = asin(d_z)"],
                    "evidence": {"trial_id": tr.id, "leg_id": lid,
                                 "measured_deg": round(ang_m, 2),
                                 "nominal_deg": round(ang_n, 2)},
                })

        if measured > 0:
            # 力行: [0,0,0 | -n_b] x = -Fsum  => -n_b W = -Fsum
            for k, name in enumerate(("fx", "fy", "fz")):
                rows.append([0.0, 0.0, 0.0, -n_b[k]])
                rhs.append(-Fsum[k])
                row_meta.append((tr.id, name))
            # 矩行: [skew(n_b) | 0] u = -Msum
            S = la.skew(n_b)
            for k, name in enumerate(("mx", "my", "mz")):
                rows.append([S[k][0], S[k][1], S[k][2], 0.0])
                rhs.append(-Msum[k])
                row_meta.append((tr.id, name))
            Wz = Fsum[2] / n_b[2] if abs(n_b[2]) > 1e-9 else None
            per_trial[tr.id] = {
                "measured_points": measured,
                "pitch_deg": tr.pitch_deg, "roll_deg": tr.roll_deg,
                "force_sum_body_kn": _round(Fsum),
                "moment_sum_body_knm": _round(Msum),
                "implied_total_weight_dynamic_kn": round(Wz, 3) if Wz is not None else None,
                "measured_total_weight_kn": tr.measured_total_weight_kn,
            }
            if tr.measured_total_weight_kn is not None and Wz is not None:
                ref = tr.measured_total_weight_kn * phi
                if abs(Wz - ref) / ref > WEIGHT_CHECK_TOL:
                    conflicts.append({
                        "code": "WEIGHT_MISMATCH",
                        "message": f"试吊 {tr.id} 张力竖向分量之和 {Wz:.1f} kN 与吊钩称重 "
                                   f"{tr.measured_total_weight_kn:.1f}×φ={ref:.1f} kN 偏差 >3%",
                        "equations": ["Σ T d_z = φ W_total"],
                        "evidence": {"trial_id": tr.id, "from_tensions": round(Wz, 2),
                                     "from_crane_scale": tr.measured_total_weight_kn,
                                     "dynamic_factor": round(phi, 3)},
                    })

    if not rows:
        return {"status": "insufficient_data",
                "conflicts": conflicts, "evidence_gaps": evidence_gaps,
                "message": "没有任何可用的张力测量", "per_trial": per_trial}

    sol = la.solve_linear(rows, rhs)
    ux, uy, uz, Wd = sol["x"]

    # 逐试吊残差(归一化), 找冲突方程
    res = sol["residual"]
    blocks: Dict[str, List[float]] = {}
    for (tid, name), r in zip(row_meta, res):
        blocks.setdefault(tid, []).append(r)
    for tid, rv in blocks.items():
        scale = per_trial[tid]["implied_total_weight_dynamic_kn"] or 1.0
        rel = la.rms(rv) / scale
        per_trial[tid]["residual_rel_rms"] = round(rel, 5)
        if rel > CONFLICT_RESIDUAL:
            worst = max(range(len(rv)), key=lambda i: abs(rv[i]))
            conflicts.append({
                "code": "TENSION_CONTRADICTION",
                "message": f"试吊 {tid} 的测量张力不自洽(归一化残差 {rel:.2%} > "
                           f"{CONFLICT_RESIDUAL:.0%}); 最冲突方程: "
                           f"{row_meta[[i for i,(t,_) in enumerate(row_meta) if t==tid][worst]][1]}",
                "equations": ["Σ T_i d_i − W n_b = 0",
                              "Σ r_i×T_i d_i + n_b×(Wq) = 0"],
                "evidence": {"trial_id": tid, "relative_rms": round(rel, 5),
                             "residual_components": _round(rv),
                             "component_order": ["fx", "fy", "fz", "mx", "my", "mz"]},
            })

    # 列可观测性(去掉某列后秩不变 => 该未知量不可观测)
    full_rank = sol["rank"]
    unobservable = []
    names = ["cog_x", "cog_y", "cog_z", "W"]
    for j in range(4):
        Ared = [row[:j] + row[j + 1:] for row in rows]
        if la.matrix_rank(Ared) == full_rank:
            unobservable.append(names[j])
    if unobservable:
        evidence_gaps.append({
            "code": "COG_COMPONENT_UNOBSERVABLE",
            "message": "实测数据的列秩不足, 以下量不可观测: "
                       + ", ".join(unobservable)
                       + "。水平姿态的试吊无法识别重心高度; 需增加带纵/横倾姿态的试吊",
            "unobservable": unobservable,
            "rank": full_rank, "unknowns": 4,
            "attitudes": [{"trial_id": t.id, "pitch_deg": t.pitch_deg, "roll_deg": t.roll_deg}
                          for t in inp.trials],
        })

    # 跨试吊重量一致性
    ws = [v["implied_total_weight_dynamic_kn"] for v in per_trial.values()
          if v["implied_total_weight_dynamic_kn"]]
    if len(ws) >= 2:
        spread = (max(ws) - min(ws)) / (sum(ws) / len(ws))
        if spread > WEIGHT_CHECK_TOL:
            conflicts.append({
                "code": "WEIGHT_SPREAD_BETWEEN_TRIALS",
                "message": f"各次试吊反算总重互差 {spread:.2%} > {WEIGHT_CHECK_TOL:.0%}",
                "equations": ["W = ΣT_d / n_b_z"],
                "evidence": {"weights_dynamic_kn": {k: v["implied_total_weight_dynamic_kn"]
                                                     for k, v in per_trial.items()}},
            })

    # 非构件重量扣除(总体系 -> 构件)
    W_total_static = Wd / phi
    cog_total = (ux / Wd, uy / Wd, uz / Wd)
    Wc = W_total_static - sum(w for w, _c, _n in rig_parts)
    if Wc <= 0:
        conflicts.append({"code": "NONPHYSICAL_WEIGHT",
                          "message": f"反算总重 {W_total_static:.1f} kN 小于索具自重之和",
                          "equations": [], "evidence": {"w_total": W_total_static}})
        status = "conflict"
        cog_component = cog_total
    else:
        cxm = Wd * cog_total[0] - sum(wd * c[0] for wd, c, _ in rig_parts)
        cym = Wd * cog_total[1] - sum(wd * c[1] for wd, c, _ in rig_parts)
        czm = Wd * cog_total[2] - sum(wd * c[2] for wd, c, _ in rig_parts)
        Wc_dyn = Wc * phi
        cog_component = (cxm / Wc_dyn, cym / Wc_dyn, czm / Wc_dyn)
        status = "conflict" if conflicts else ("partial" if evidence_gaps else "ok")

    # 重心高度在纯水平试吊下给不出可靠值: 显式标记而非返回无意义数字
    if "cog_z" in unobservable:
        cog_component_out = [cog_component[0], cog_component[1], None]
    else:
        cog_component_out = list(cog_component)

    offset = None
    if cog_component_out[0] is not None:
        offset = [round(cog_component_out[k] - inp.cog_theory[k], 4)
                  if cog_component_out[k] is not None else None for k in range(3)]

    return {
        "status": status,
        "rank": full_rank,
        "nullspace_dim": sol.get("nullspace_dim", 0),
        "dynamic_factor": round(phi, 4),
        "total_weight_solved_dynamic_kn": round(Wd, 3),
        "total_weight_solved_static_kn": round(W_total_static, 3),
        "component_weight_solved_static_kn": round(Wc, 3),
        "component_weight_declared_kn": inp.component_weight_kn,
        "cog_total_assembly": [round(v, 4) for v in cog_total],
        "cog_component": cog_component_out,
        "cog_component_world": cog_component_out,  # 本体=水平时世界, 倾斜姿态反算点映射后相同物理点
        "cog_theory": list(inp.cog_theory),
        "cog_offset_m": offset,
        "unobservable": unobservable,
        "per_trial": per_trial,
        "conflicts": conflicts,
        "evidence_gaps": evidence_gaps,
        "angles_only_trials": angles_only_trials,
        "total_measured_points": total_measured_points,
        "solve_kind": sol["kind"],
    }


def _trial_direction(m, d_level_b: Vec3, R: la.Matrix, upper: Vec3, pad: Vec3) \
        -> Tuple[Vec3, str]:
    """
    单次测量的索方向(本体坐标):
      实测世界方向 -> Rᵀ d_w
      仅实测水平夹角 -> 名义方位角 + 实测仰角组合世界向量, 再 Rᵀ 转回
      都没有       -> 上端点静止、下端点随刚体倾斜的名义几何方向
    """
    if m.measured_direction is not None:
        dw = v_unit(tuple(m.measured_direction))
        db = tuple(la.mat_vec(la.mat_T(R), list(dw)))
        return v_unit(db), "measured_direction"
    if m.measured_angle_deg is not None:
        d = d_level_b
        alpha = math.atan2(d[1], d[0])
        gamma = math.radians(m.measured_angle_deg)
        dw = (math.cos(gamma) * math.cos(alpha),
              math.cos(gamma) * math.sin(alpha),
              math.sin(gamma))
        db = tuple(la.mat_vec(la.mat_T(R), list(dw)))
        return v_unit(db), "measured_angle"
    # 上端点不动; 下端点按刚体转动(平移不影响方向, 取 pad 自身为旋转参考)
    pad_w = rotate(R, pad)
    d_w = v_unit(v_sub(upper, pad_w))
    return v_unit(tuple(la.mat_vec(la.mat_T(R), list(d_w)))), "nominal_geometry"


# ================================================================ 调整量
def recommend_adjustments(inp, analysis: Dict[str, Any], inverse: Optional[Dict[str, Any]]) \
        -> Dict[str, Any]:
    """
    三类调整:
      A 吊点移位: 把吊点几何中心移到重心正上方, δ = c_actual − p_centroid(水平)
      B 索长(垫片/调平):  按倾斜姿态各索需补偿的竖向差
      C 配重: 由重力矩反算质量×力臂, 在候选站位中选可行解
    """
    cog = analysis["cog_used"]["cog"]
    geo = build_geometry(inp)
    cx = sum(p[0] for p in geo.pads) / len(geo.pads)
    cy = sum(p[1] for p in geo.pads) / len(geo.pads)
    ex = cog[0] - cx
    ey = cog[1] - cy

    # ---- A 吊点整体平移
    max_shift = inp.max_pad_shift_m
    sx = la.clamp(ex, -max_shift, max_shift)
    sy = la.clamp(ey, -max_shift, max_shift)
    shift_feasible = abs(sx - ex) < 1e-6 and abs(sy - ey) < 1e-6
    pad_shifts = [{"leg_id": lid, "dx": round(sx, 4), "dy": round(sy, 4),
                   "new_pad": [round(geo.pads[i][0] + sx, 4),
                               round(geo.pads[i][1] + sy, 4),
                               round(geo.pads[i][2], 4)]}
                  for i, lid in enumerate(geo.leg_ids)]

    # ---- B 索长调平: 倾斜姿态下各索需要的长度与名义长度之差(放正为负差)
    pitch = math.radians(analysis["predicted_attitude"]["pitch_deg"])
    roll = math.radians(analysis["predicted_attitude"]["roll_deg"])
    R = rot_matrix(pitch, roll)
    shims = []
    for i, lid in enumerate(geo.leg_ids):
        pad_w = tilted_pad_world(geo.pads[i], tuple(cog), R)
        tilted_required = v_norm(v_sub(geo.lug_points[i], pad_w))
        dl = geo.lengths[i] - tilted_required  # 放长/收短量
        shims.append({"leg_id": lid,
                      "delta_length_m": round(dl, 4),
                      "nominal_length_m": round(geo.lengths[i], 4),
                      "tilted_required_m": round(tilted_required, 4)})

    # ---- C 配重力矩
    cw_options = []
    need_mx = inp.component_weight_kn * (cog[1] - cy)  # 绕 x 轴的矩(横向偏心)
    need_my = -inp.component_weight_kn * (cog[0] - cx)  # 绕 y
    chosen = None
    for st in inp.counterweight_stations:
        rx = st.position[0] - cx
        ry = st.position[1] - cy
        # m·g·r 抵消构件对吊点中心的矩: 需要 wx*? 解标量最近方向
        # 需要配重产生矩 (-need_mx, -need_my); 站位臂向量 (ry*w, -rx*w)
        # 最小二乘标量 w
        arm = (ry, -rx)
        need = (-need_mx, -need_my)
        denom = arm[0] ** 2 + arm[1] ** 2
        w = (arm[0] * need[0] + arm[1] * need[1]) / denom if denom > 1e-9 else 0.0
        w = max(w, 0.0)
        mass = w * 1000.0 / inp.g_mps2
        residual = math.hypot(arm[0] * w - need[0], arm[1] * w - need[1])
        feasible = mass <= st.capacity_kg and w > 0
        opt = {"station_id": st.id, "position": list(st.position),
               "required_mass_kg": round(mass, 1),
               "capacity_kg": st.capacity_kg,
               "residual_moment_knm": round(residual, 3),
               "fits_capacity": mass <= st.capacity_kg,
               "useful_direction": w > 0}
        cw_options.append(opt)
        if feasible and (chosen is None or residual < chosen["residual_moment_knm"]):
            chosen = opt
    cw_need = math.hypot(need_mx, need_my)

    return {
        "cog_offset_from_pad_centroid_m": {"dx": round(ex, 4), "dy": round(ey, 4),
                                           "moment_knm": round(cw_need, 3)},
        "option_a_shift_pickpoints": {
            "feasible_within_max": shift_feasible,
            "max_shift_m": max_shift,
            "pad_shifts": pad_shifts,
            "residual_offset_m": {"dx": round(ex - sx, 4), "dy": round(ey - sy, 4)},
        },
        "option_b_trim_slings": {"shims": shims,
                                 "max_abs_delta_m": round(max(abs(s["delta_length_m"]) for s in shims), 4)},
        "option_c_counterweight": {
            "required_balancing_moment_knm": round(cw_need, 3),
            "stations": cw_options,
            "recommended": chosen,
            "feasible": chosen is not None,
        },
    }


def evaluate_manual_adjustment(inp) -> Dict[str, Any]:
    """
    工程师人工调整复核: 应用 pad_shifts / shims / 配重后整体重算。
    不豁免任何检查; 返回新的核算结果与调整项清单。
    """
    ma = inp.manual_adjustment
    # 复制输入模型并修改
    data = inp.model_dump()
    legs = data["legs"]
    id2idx = {l["id"]: i for i, l in enumerate(legs)}
    applied = {"pad_shifts": [], "shims": [], "counterweight": None}

    for ps in ma.pad_shifts:
        i = id2idx[ps.leg_id]
        legs[i]["pad"] = [legs[i]["pad"][0] + ps.dx,
                          legs[i]["pad"][1] + ps.dy,
                          legs[i]["pad"][2]]
        applied["pad_shifts"].append(ps.model_dump())

    # 配重写入 counterweight
    if ma.counterweight_mass_kg > 0:
        pos = None
        if ma.counterweight_station_id:
            for st in inp.counterweight_stations:
                if st.id == ma.counterweight_station_id:
                    pos = list(st.position)
        if pos is None:
            raise ValueError("人工配重需要有效的 counterweight_station_id")
        data["counterweight"] = {"mass_kg": ma.counterweight_mass_kg,
                                 "position": pos, "id": "CW_MANUAL"}
        applied["counterweight"] = {"mass_kg": ma.counterweight_mass_kg,
                                    "station_id": ma.counterweight_station_id,
                                    "position": pos}

    from .models import LiftInput
    # shims 保留在 manual_adjustment 中
    new_inp = LiftInput.model_validate(data)
    inverse = inverse_cog_from_trials(new_inp) if new_inp.trials else None
    analysis = analyze_lift(new_inp, inverse)
    adjustment = recommend_adjustments(new_inp, analysis, inverse) \
        if new_inp.request_adjustment else None
    finalize_verdict(analysis, inverse, adjustment)
    analysis["manual_recheck"] = {
        "mode": ma.mode,
        "comment": ma.comment,
        "applied": applied,
        "shims": [s.model_dump() for s in ma.shims],
        "approvable": analysis["approvable"],
    }
    return analysis


# ================================================================ 汇总
def finalize_verdict(analysis: Dict[str, Any], inverse: Optional[Dict[str, Any]],
                     adjustment: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """汇总反算冲突/证据缺口, 给出 approvable 判定与利用率摘要。"""
    blocking_codes = {c["code"] for c in analysis["conflicts"]}
    gaps = list(analysis["evidence_gaps"])
    if inverse is not None:
        blocking_codes |= {c["code"] for c in inverse["conflicts"]}
        gaps += inverse["evidence_gaps"]
    # 无实测重心按证据缺口阻断(不能仅凭理论值批准)
    approvable = len(blocking_codes) == 0 and not any(
        g["code"] == "NO_VERIFIED_COG" for g in gaps
    )
    analysis["inverse"] = inverse
    analysis["adjustment_suggestion"] = adjustment
    analysis["approvable"] = approvable
    analysis["blocking_codes"] = sorted(blocking_codes)
    analysis["all_evidence_gaps"] = gaps
    analysis["utilization_summary"] = _util_summary(analysis)
    return analysis


def run_full_analysis(inp) -> Dict[str, Any]:
    """完整流程: 试吊反算 -> 正式核算 -> 调整建议; 给出 approvable 判定。"""
    inverse = inverse_cog_from_trials(inp) if inp.trials else None
    analysis = analyze_lift(inp, inverse)
    adjustment = recommend_adjustments(inp, analysis, inverse) if inp.request_adjustment else None
    return finalize_verdict(analysis, inverse, adjustment)


def _util_summary(analysis: Dict[str, Any]) -> Dict[str, Any]:
    legs = analysis["legs"]
    out = {
        "max_sling_utilization": max((l["sling"]["utilization"] for l in legs), default=0.0),
        "max_shackle_utilization": max((l["shackle"]["utilization"] for l in legs), default=0.0),
        "worst_sling_leg": None,
    }
    if legs:
        w = max(legs, key=lambda l: l["sling"]["utilization"])
        out["worst_sling_leg"] = w["leg_id"]
    if analysis.get("beam"):
        b = analysis["beam"]
        out["beam_stress_utilization"] = b["stress_utilization"]
        out["beam_shear_utilization"] = b["shear_utilization"]
        out["max_top_sling_utilization"] = max(
            (t["sling_utilization"] for t in b["top_slings"]), default=0.0)
    if analysis.get("hoist"):
        out["hoist_utilization"] = analysis["hoist"]["utilization"]
    if analysis.get("crane"):
        env = analysis["crane"]["envelope"]
        out["crane_max_load_utilization"] = env["max_load_utilization"]
        out["crane_max_ground_pressure_kpa"] = env["max_ground_pressure_kpa"]
        out["crane_min_outrigger_reaction_kn"] = env["min_outrigger_reaction_kn"]
        clr = analysis["crane"].get("clearance")
        if clr:
            out["clearance_min_conservative_m"] = clr["min_conservative_clearance_m"]
    return out


def _round(v, nd: int = 4):
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return [round(x, nd) if x is not None else None for x in v]
    return round(v, nd)
