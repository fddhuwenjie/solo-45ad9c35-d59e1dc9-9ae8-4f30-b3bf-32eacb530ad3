# -*- coding: utf-8 -*-
"""
路径净空校核
============

构件包络(定向长方体 OBB)沿吊钩路径的净空校核, 与起重机工况校核共用
同一份路径采样(crane.check_crane_duty 调用):

* 姿态(yaw/pitch/roll, °)随吊钩路径同步插值, 旋转矩阵约定
  R = Rz(yaw)·Ry(pitch)·Rx(roll), 本体系向量 v_b 的世界坐标为 R·v_b;
* 包络中心 = 吊钩位置 + R·(吊钩至包络中心偏移, 本体系);
* 障碍物与不可侵入区均为世界系轴对齐盒(AABB);
* 分离轴法(SAT, 15 条候选轴)给出有符号净空: 正值为两盒最小间距的
  保守下界, 负值为贯穿深度(近似); 逐点净空再减去扫掠修正(相邻采样间
  包络表面点最大位移的一半), 覆盖采样点之间未直接校核的区间;
* 包络退化(尺寸非正/非有限)、障碍几何无效(min>=max 或非有限)、
  姿态角跳变、采样上限不足均列为证据缺口, 由 crane 汇总上报。

纯 Python 实现, 不依赖 numpy。
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]
Mat3 = Tuple[Vec3, Vec3, Vec3]  # 行主序 3x3

_EPS = 1.0e-12


# ---------------------------------------------------------------- 角度与旋转
def wrap180(deg: float) -> float:
    """回绕到 (-180, 180]。"""
    while deg > 180.0:
        deg -= 360.0
    while deg <= -180.0:
        deg += 360.0
    return deg


def rot_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> Mat3:
    """R = Rz(yaw)·Ry(pitch)·Rx(roll); 本体系 -> 世界系。"""
    cy, sy = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    cp, sp = math.cos(math.radians(pitch_deg)), math.sin(math.radians(pitch_deg))
    cr, sr = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
    # Rz(yaw)·Ry(pitch)·Rx(roll) 展开
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def rot_vec(R: Mat3, v: Sequence[float]) -> Vec3:
    return (
        R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
        R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
        R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
    )


def rot_axes(R: Mat3) -> Tuple[Vec3, Vec3, Vec3]:
    """旋转矩阵的三列(本体轴在世界系中的方向)。"""
    return ((R[0][0], R[1][0], R[2][0]),
            (R[0][1], R[1][1], R[2][1]),
            (R[0][2], R[1][2], R[2][2]))


def geodesic_deg(Ra: Mat3, Rb: Mat3) -> float:
    """两姿态间的测地转角 °: θ = arccos((tr(Raᵀ·Rb) − 1)/2)。"""
    tr = sum(Ra[i][j] * Rb[i][j] for i in range(3) for j in range(3))
    c = max(-1.0, min(1.0, (tr - 1.0) / 2.0))
    return math.degrees(math.acos(c))


def unwrap_attitudes(attitudes: List[Vec3]) -> List[Vec3]:
    """逐轴按最短增量展开欧拉角序列, 消除 ±180° 跳变后再插值。"""
    out: List[Vec3] = []
    prev: Optional[Vec3] = None
    for ang in attitudes:
        if prev is None:
            out.append((float(ang[0]), float(ang[1]), float(ang[2])))
        else:
            out.append(tuple(prev[i] + wrap180(ang[i] - prev[i]) for i in range(3)))
        prev = out[-1]
    return out


# ---------------------------------------------------------------- 分离轴法
def _sat_gap(d: Vec3, u: Vec3, axes: Tuple[Vec3, Vec3, Vec3],
             half_a: Vec3, half_b: Vec3) -> Optional[float]:
    """候选分离轴 u 上的有符号间隙(>0 分离); u 近零(平行轴)返回 None。"""
    n = math.sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2])
    if n < 1.0e-9:
        return None
    ux, uy, uz = u[0] / n, u[1] / n, u[2] / n
    dist = abs(d[0] * ux + d[1] * uy + d[2] * uz)
    ra = (half_a[0] * abs(axes[0][0] * ux + axes[0][1] * uy + axes[0][2] * uz)
          + half_a[1] * abs(axes[1][0] * ux + axes[1][1] * uy + axes[1][2] * uz)
          + half_a[2] * abs(axes[2][0] * ux + axes[2][1] * uy + axes[2][2] * uz))
    rb = half_b[0] * abs(ux) + half_b[1] * abs(uy) + half_b[2] * abs(uz)
    return dist - ra - rb


def sat_clearance(center: Vec3, axes: Tuple[Vec3, Vec3, Vec3], half_a: Vec3,
                  bmin: Vec3, bmax: Vec3) -> float:
    """
    OBB(中心 center, 轴 axes, 半长 half_a) 与 AABB([bmin,bmax]) 的有符号净空。

    取 15 条候选分离轴(3 条世界轴 + 3 条 OBB 轴 + 9 条叉积轴)上间隙的
    最大值: >0 时为两盒最小间距的保守下界, 全部 <=0 时两盒相交,
    返回值(<=0)的绝对值为贯穿深度近似。
    """
    bc = ((bmin[0] + bmax[0]) / 2.0, (bmin[1] + bmax[1]) / 2.0,
          (bmin[2] + bmax[2]) / 2.0)
    bh = ((bmax[0] - bmin[0]) / 2.0, (bmax[1] - bmin[1]) / 2.0,
          (bmax[2] - bmin[2]) / 2.0)
    d = (center[0] - bc[0], center[1] - bc[1], center[2] - bc[2])
    world_axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    best = -math.inf
    for u in world_axes + axes:
        g = _sat_gap(d, u, axes, half_a, bh)
        if g is not None and g > best:
            best = g
    for a in axes:
        for e in world_axes:
            u = (a[1] * e[2] - a[2] * e[1],
                 a[2] * e[0] - a[0] * e[2],
                 a[0] * e[1] - a[1] * e[0])
            g = _sat_gap(d, u, axes, half_a, bh)
            if g is not None and g > best:
                best = g
    return best


# ---------------------------------------------------------------- 包络工具
def envelope_corners(offset: Vec3, half: Vec3) -> List[Vec3]:
    """包络 8 角点(相对吊钩, 本体系): offset + (±hx, ±hy, ±hz)。"""
    return [(offset[0] + sx * half[0], offset[1] + sy * half[1],
             offset[2] + sz * half[2])
            for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)]


def _hull2d(pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """二维凸包(单调链), 返回逆时针顶点序列。"""
    p = sorted(set(pts))
    if len(p) <= 2:
        return p

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    upper: List[Tuple[float, float]] = []
    for q in reversed(p):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return lower[:-1] + upper[:-1]


def footprint_hull_xy(center: Vec3, axes: Tuple[Vec3, Vec3, Vec3],
                      half: Vec3) -> List[Tuple[float, float]]:
    """OBB 在水平面上的投影凸包(站位图包络足迹用)。"""
    pts = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                wx = (center[0] + sx * half[0] * axes[0][0]
                      + sy * half[1] * axes[1][0] + sz * half[2] * axes[2][0])
                wy = (center[1] + sx * half[0] * axes[0][1]
                      + sy * half[1] * axes[1][1] + sz * half[2] * axes[2][1])
                pts.append((wx, wy))
    return _hull2d(pts)


# ---------------------------------------------------------------- 校核准备
def _box_valid(bmin: Vec3, bmax: Vec3) -> bool:
    return all(math.isfinite(bmin[i]) and math.isfinite(bmax[i])
               and bmin[i] < bmax[i] for i in range(3))


def prepare_clearance(clr) -> Dict[str, Any]:
    """
    校验包络与障碍几何并预计算常量, 返回逐点校核上下文。

    包络尺寸非正/非有限 -> 退化, 校核整体跳过(证据缺口);
    min>=max 或非有限的障碍盒 -> 无效, 跳过该盒(证据缺口), 其余照常。
    """
    gaps: List[Dict[str, Any]] = []
    dims = (clr.envelope.length_m, clr.envelope.width_m, clr.envelope.height_m)
    offset = tuple(float(v) for v in clr.hook_to_center_offset)
    degenerate = (any((not math.isfinite(d)) or d <= 0.0 for d in dims)
                  or any(not math.isfinite(v) for v in offset))
    if degenerate:
        gaps.append({
            "code": "CLEARANCE_ENVELOPE_DEGENERATE",
            "message": "构件包络退化(长/宽/高或吊钩偏移非正、非有限), "
                       "无法构成有效定向包络, 净空校核跳过",
            "evidence": {"length_m": dims[0], "width_m": dims[1],
                         "height_m": dims[2],
                         "hook_to_center_offset": list(offset)},
        })
    half = (max(dims[0], 0.0) / 2.0, max(dims[1], 0.0) / 2.0,
            max(dims[2], 0.0) / 2.0)
    corners = envelope_corners(offset, half)
    r_max = max(math.sqrt(c[0] ** 2 + c[1] ** 2 + c[2] ** 2) for c in corners)

    obstacles, invalid_obs = [], []
    for ob in clr.obstacles:
        bmin = tuple(float(v) for v in ob.min_corner)
        bmax = tuple(float(v) for v in ob.max_corner)
        if _box_valid(bmin, bmax):
            obstacles.append({"id": ob.id, "min": bmin, "max": bmax})
        else:
            invalid_obs.append(ob.id)
    zones, invalid_zones = [], []
    for z in clr.exclusion_zones:
        bmin = tuple(float(v) for v in z.min_corner)
        bmax = tuple(float(v) for v in z.max_corner)
        if _box_valid(bmin, bmax):
            zones.append({"id": z.id, "min": bmin, "max": bmax})
        else:
            invalid_zones.append(z.id)
    if invalid_obs or invalid_zones:
        gaps.append({
            "code": "CLEARANCE_OBSTACLE_INVALID",
            "message": "存在无效轴对齐盒(min>=max 或坐标非有限), 已从校核中剔除: "
                       f"障碍物 {invalid_obs or '无'}, 不可侵入区 {invalid_zones or '无'}",
            "evidence": {"invalid_obstacle_ids": invalid_obs,
                         "invalid_exclusion_ids": invalid_zones},
        })

    return {
        "degenerate": degenerate,
        "half": half,
        "offset": offset,
        "r_max": r_max,
        "obstacles": obstacles,
        "exclusion_zones": zones,
        "invalid_obstacle_ids": invalid_obs,
        "invalid_exclusion_ids": invalid_zones,
        "safety_margin_m": clr.safety_margin_m,
        "evidence_gaps": gaps,
    }


# ---------------------------------------------------------------- 逐点校核
def eval_sample(ctx: Dict[str, Any], hook_pos: Vec3,
                attitude: Vec3) -> Dict[str, Any]:
    """
    单个采样姿态的净空: 返回包络中心、对障碍物与不可侵入区的 SAT 净空。
    姿态为 (yaw, pitch, roll) °(未回绕亦可, 旋转矩阵等价)。
    """
    R = rot_matrix(attitude[0], attitude[1], attitude[2])
    off = rot_vec(R, ctx["offset"])
    center = (hook_pos[0] + off[0], hook_pos[1] + off[1], hook_pos[2] + off[2])
    axes = rot_axes(R)
    half = ctx["half"]

    best: Optional[float] = None
    worst: Optional[str] = None
    for ob in ctx["obstacles"]:
        c = sat_clearance(center, axes, half, ob["min"], ob["max"])
        if best is None or c < best:
            best, worst = c, ob["id"]
    zbest: Optional[float] = None
    zworst: Optional[str] = None
    for z in ctx["exclusion_zones"]:
        c = sat_clearance(center, axes, half, z["min"], z["max"])
        if zbest is None or c < zbest:
            zbest, zworst = c, z["id"]
    return {
        "envelope_center": center,
        "min_clearance_m": best,
        "worst_obstacle_id": worst,
        "exclusion_min_clearance_m": zbest,
        "worst_exclusion_id": zworst,
    }


def sweep_bound(d_hook: float, d_theta_deg: float, r_max: float) -> float:
    """
    相邻采样间包络表面点的最大位移上界:
    吊钩平移 + 绕吊钩的测地转角弦长(2·sin(Δθ/2)·R_max),
    R_max 为包络角点到吊钩的最大距离。
    """
    return d_hook + 2.0 * math.sin(math.radians(d_theta_deg) / 2.0) * r_max
