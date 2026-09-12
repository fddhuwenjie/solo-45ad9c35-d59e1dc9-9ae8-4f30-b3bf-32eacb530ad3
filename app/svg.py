# -*- coding: utf-8 -*-
"""
起重机站位图(SVG)
==================

平面布置图: 回转中心、支腿与垫板、吊钩回转路径与逐姿态采样点;
启用路径净空校核时, 叠加障碍物/不可侵入区足迹与构件包络逐姿态投影,
碰撞/间距不足采样点以红色标记。
输入仅为起重机工况校核结果 dict(crane.check_crane_duty 的返回值),
与 JSON 接口共用同一份逐姿态数据(姿态角与碰撞标记), 不重复计算。
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
from xml.sax.saxutils import escape

W, H = 1000.0, 780.0
LEGEND_H = 150.0
CLR_LEGEND_H = 44.0  # 净空校核启用时图例追加高度

_OK = "#1a9e50"
_BAD = "#d62728"
_PATH = "#1f6fc5"
_MAT_FILL = "#e8eefc"
_MAT_STROKE = "#3b6ea5"
_OBST = "#b8860b"
_EXCL = "#d62728"
_ENVELOPE = "#6a51a3"


def _fmt(v: Any, nd: int = 1) -> str:
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "-"


def _envelope_footprints(crane: Dict[str, Any],
                         samples: List[Dict[str, Any]]
                         ) -> List[Tuple[List[Tuple[float, float]], bool, bool]]:
    """逐姿态包络水平投影凸包: (世界坐标点列, 是否违规, 是否关键姿态)。

    仅绘制关键姿态点与净空违规点; 姿态角与包络尺寸取自 JSON 结果,
    旋转/投影由 clearance 模块同一套几何完成。
    """
    clr = crane.get("clearance")
    if not clr:
        return []
    from .clearance import footprint_hull_xy, rot_axes, rot_matrix
    env = clr["envelope"]
    half = (env["length_m"] / 2.0, env["width_m"] / 2.0, env["height_m"] / 2.0)
    if any(h <= 0 for h in half):
        return []
    out = []
    for s in samples:
        ce = s.get("clearance")
        if ce is None or "attitude_deg" not in s:
            continue
        violated = bool(ce["violations"])
        if not (s["at_pose"] or violated):
            continue
        att = s["attitude_deg"]
        axes = rot_axes(rot_matrix(att["yaw"], att["pitch"], att["roll"]))
        pts = footprint_hull_xy(tuple(ce["envelope_center"]), axes, half)
        out.append((pts, violated, s["at_pose"] is not None))
    return out


def crane_station_svg(crane: Dict[str, Any]) -> str:
    """由工况校核结果渲染站位平面图(世界 x 向右, y 向上)。"""
    setup = crane["setup"]
    samples: List[Dict[str, Any]] = crane["samples"]
    env = crane["envelope"]
    clr = crane.get("clearance")
    cx, cy = setup["slewing_center"]
    legend_h = LEGEND_H + (CLR_LEGEND_H if clr else 0.0)
    footprints = _envelope_footprints(crane, samples)

    # ---------------- 视图范围(含垫板外缘、路径、障碍与包络足迹) ----------------
    xs: List[float] = [cx]
    ys: List[float] = [cy]
    pad = 1.0
    for o in setup["outriggers"]:
        ox, oy = cx + o["position"][0], cy + o["position"][1]
        half = 0.5 * max(o["mat_length_m"], o["mat_width_m"])
        pad = max(pad, half)
        xs += [ox - half, ox + half]
        ys += [oy - half, oy + half]
    for s in samples:
        xs.append(s["position"][0])
        ys.append(s["position"][1])
    if clr:
        for box in clr["obstacles"] + clr["exclusion_zones"]:
            xs += [box["min"][0], box["max"][0]]
            ys += [box["min"][1], box["max"][1]]
    for pts, _v, _p in footprints:
        for wx, wy in pts:
            xs.append(wx)
            ys.append(wy)
    x0, x1 = min(xs) - pad, max(xs) + pad
    y0, y1 = min(ys) - pad, max(ys) + pad
    scale = min((W - 140.0) / max(x1 - x0, 1e-6),
                (H - legend_h - 120.0) / max(y1 - y0, 1e-6))
    ox = (W - (x1 - x0) * scale) / 2.0
    oy = (H - legend_h - (y1 - y0) * scale) / 2.0 + 40.0

    def X(x: float) -> float:
        return ox + (x - x0) * scale

    def Y(y: float) -> float:
        return oy + (y1 - y) * scale  # 世界 y 向上, SVG y 向下

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W:.0f} {H:.0f}" '
        f'font-family="monospace" font-size="12">',
        f'<rect x="0" y="0" width="{W:.0f}" height="{H:.0f}" fill="#ffffff" '
        f'stroke="#cccccc"/>',
        f'<text x="{W/2:.0f}" y="24" text-anchor="middle" font-size="16" '
        f'font-weight="bold">起重机站位图 {escape(str(crane["crane_id"]))} '
        f'— 回转路径工况校核</text>',
    ]

    # ---------------- 垫板与支腿 ----------------
    max_p_by_id: Dict[str, float] = {}
    for s in samples:
        for r in s["outriggers"]:
            max_p_by_id[r["id"]] = max(max_p_by_id.get(r["id"], 0.0), r["pressure_kpa"])
    for o in setup["outriggers"]:
        wx, wy = cx + o["position"][0], cy + o["position"][1]
        wpx = o["mat_length_m"] * scale
        hpx = o["mat_width_m"] * scale
        px, py = X(wx), Y(wy)
        ang = -o["mat_angle_deg"]  # y 翻转补偿, 保持世界系逆时针为正
        parts.append(
            f'<rect x="{px - wpx/2:.1f}" y="{py - hpx/2:.1f}" width="{wpx:.1f}" '
            f'height="{hpx:.1f}" fill="{_MAT_FILL}" stroke="{_MAT_STROKE}" '
            f'transform="rotate({ang:.2f} {px:.1f} {py:.1f})"/>'
        )
        parts.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3" fill="{_MAT_STROKE}"/>'
            f'<text x="{px:.1f}" y="{py - hpx/2 - 8:.1f}" text-anchor="middle" '
            f'font-size="11" fill="#333">{escape(o["id"])}</text>'
            f'<text x="{px:.1f}" y="{py + hpx/2 + 14:.1f}" text-anchor="middle" '
            f'font-size="10" fill="#333">max {_fmt(max_p_by_id.get(o["id"]), 0)} kPa</text>'
        )

    # ---------------- 障碍物与不可侵入区(水平足迹) ----------------
    if clr:
        for box in clr["obstacles"]:
            rx, ry = X(box["min"][0]), Y(box["max"][1])
            rw = (box["max"][0] - box["min"][0]) * scale
            rh = (box["max"][1] - box["min"][1]) * scale
            parts.append(
                f'<rect x="{rx:.1f}" y="{ry:.1f}" width="{rw:.1f}" height="{rh:.1f}" '
                f'fill="{_OBST}" fill-opacity="0.10" stroke="{_OBST}" '
                f'stroke-width="1.2" stroke-dasharray="5 3"/>'
                f'<text x="{rx + 3:.1f}" y="{ry + 13:.1f}" font-size="10" '
                f'fill="{_OBST}">障碍 {escape(box["id"])}</text>'
            )
        for box in clr["exclusion_zones"]:
            rx, ry = X(box["min"][0]), Y(box["max"][1])
            rw = (box["max"][0] - box["min"][0]) * scale
            rh = (box["max"][1] - box["min"][1]) * scale
            parts.append(
                f'<rect x="{rx:.1f}" y="{ry:.1f}" width="{rw:.1f}" height="{rh:.1f}" '
                f'fill="{_EXCL}" fill-opacity="0.14" stroke="{_EXCL}" '
                f'stroke-width="1.5"/>'
                f'<text x="{rx + 3:.1f}" y="{ry + 13:.1f}" font-size="10" '
                f'fill="{_EXCL}">禁入 {escape(box["id"])}</text>'
            )

    # ---------------- 半径标注(姿态点) ----------------
    for s in samples:
        if s["at_pose"] is None:
            continue
        px, py = X(s["position"][0]), Y(s["position"][1])
        mx, my = (px + X(cx)) / 2.0, (py + Y(cy)) / 2.0
        parts.append(
            f'<line x1="{X(cx):.1f}" y1="{Y(cy):.1f}" x2="{px:.1f}" y2="{py:.1f}" '
            f'stroke="#999999" stroke-width="0.8" stroke-dasharray="4 3"/>'
            f'<text x="{mx:.1f}" y="{my - 4:.1f}" text-anchor="middle" font-size="10" '
            f'fill="#555555">r={_fmt(s["radius_m"])}m</text>'
        )

    # ---------------- 构件包络逐姿态投影(与 JSON 共用姿态/碰撞标记) ----------------
    for pts, violated, _at_pose in footprints:
        poly = " ".join(f"{X(wx):.1f},{Y(wy):.1f}" for wx, wy in pts)
        color = _BAD if violated else _ENVELOPE
        parts.append(f'<polygon points="{poly}" fill="{color}" fill-opacity="0.10" '
                     f'stroke="{color}" stroke-width="1.2"/>')

    # ---------------- 回转路径与逐姿态采样点 ----------------
    pts = " ".join(f'{X(s["position"][0]):.1f},{Y(s["position"][1]):.1f}' for s in samples)
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{_PATH}" '
                 f'stroke-width="1.5"/>')
    for s in samples:
        px, py = X(s["position"][0]), Y(s["position"][1])
        color = _BAD if s["violations"] else _OK
        r = 5 if s["at_pose"] else 3
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{r}" fill="{color}" '
                     f'stroke="#ffffff" stroke-width="1"/>')
        if s["at_pose"] is not None:
            label = s["at_pose"]
            att = s.get("attitude_deg")
            if att is not None:
                label += f' yaw={_fmt(att["yaw"], 0)}°'
            parts.append(f'<text x="{px + 8:.1f}" y="{py - 8:.1f}" font-size="11" '
                         f'fill="#111111">{escape(label)}</text>')

    # ---------------- 回转中心 ----------------
    ccx, ccy = X(cx), Y(cy)
    parts.append(
        f'<line x1="{ccx - 9:.1f}" y1="{ccy:.1f}" x2="{ccx + 9:.1f}" y2="{ccy:.1f}" '
        f'stroke="#000000" stroke-width="1.5"/>'
        f'<line x1="{ccx:.1f}" y1="{ccy - 9:.1f}" x2="{ccx:.1f}" y2="{ccy + 9:.1f}" '
        f'stroke="#000000" stroke-width="1.5"/>'
        f'<text x="{ccx + 12:.1f}" y="{ccy + 16:.1f}" font-size="11">回转中心</text>'
    )

    # ---------------- 首个违规标记 ----------------
    fv = crane.get("first_violation")
    if fv:
        fx, fy = X(fv["interval"]["to"][0]), Y(fv["interval"]["to"][1])
        parts.append(
            f'<circle cx="{fx:.1f}" cy="{fy:.1f}" r="10" fill="none" '
            f'stroke="{_BAD}" stroke-width="2" stroke-dasharray="3 2"/>'
            f'<text x="{fx + 12:.1f}" y="{fy + 22:.1f}" font-size="11" '
            f'fill="{_BAD}">首个违规 {escape(fv["kind"])}</text>'
        )
    cfv = clr.get("first_violation") if clr else None
    if cfv and (fv is None or cfv["sample_index"] != fv["sample_index"]):
        fx, fy = X(cfv["interval"]["to"][0]), Y(cfv["interval"]["to"][1])
        parts.append(
            f'<circle cx="{fx:.1f}" cy="{fy:.1f}" r="13" fill="none" '
            f'stroke="{_ENVELOPE}" stroke-width="2" stroke-dasharray="2 2"/>'
            f'<text x="{fx + 14:.1f}" y="{fy - 12:.1f}" font-size="11" '
            f'fill="{_ENVELOPE}">首个净空违规 {escape(cfv["kind"])}</text>'
        )

    # ---------------- 图例/摘要(与 JSON 同一份数据) ----------------
    chart = crane["load_chart_summary"]
    path = crane["path_summary"]
    ly = H - legend_h + 18.0
    lines = [
        f'吊钩动载 {_fmt(crane["lifted_load_dynamic_kn"])} kN · '
        f'吊钩滑轮组 {_fmt(crane["hook_block_weight_kn"])} kN',
        f'臂长 {_fmt(crane["boom_length_m"])} m · '
        f'配置 {escape(str(crane.get("config", "STD")))} · '
        f'载荷表 {chart["fingerprint"]} (区段 {"、".join(chart["zones"]) or "-"})',
        f'最大载荷利用率 {env["max_load_utilization"]:.1%} · '
        f'最小支腿反力 {_fmt(env["min_outrigger_reaction_kn"])} kN',
        f'最大接地压力 {_fmt(env["max_ground_pressure_kpa"])} kPa / '
        f'限值 {_fmt(setup["ground_bearing_limit_kpa"])} kPa',
        f'路径 {path["poses"]} 姿态 · 采样 {path["samples"]} 点 · '
        f'总长 {_fmt(path["total_length_m"])} m',
    ]
    if clr:
        n_obs = clr["obstacles_total"] + clr["exclusion_zones_total"]
        lines.append(
            f'安全间距 {_fmt(clr["safety_margin_m"], 2)} m · '
            f'最小保守净空 {_fmt(clr["min_conservative_clearance_m"], 2)} m · '
            f'障碍 {n_obs} 个(违规采样 {clr["samples_with_violations"]} 点)'
        )
        if clr["status"] == "skipped":
            lines.append('净空校核跳过: 包络退化(见证据缺口)')
        elif clr["status"] == "ok":
            lines.append('净空校核通过, 无碰撞/间距不足区间')
        else:
            cf = clr["first_violation"]
            lines.append(f'首个净空违规 {cf["kind"]} @ {cf["segment"]} '
                         f'(保守净空 {_fmt(cf.get("conservative_clearance_m"), 2)} m)')
    lines.append(
        '校核通过, 无违规区间' if crane["status"] == "ok"
        else f'首个违规 {fv["kind"]} @ {fv["segment"]} (r={_fmt(fv["radius_m"])} m)'
        if fv else '校核未通过'
    )
    parts.append(f'<rect x="10" y="{H - legend_h:.0f}" width="{W - 20:.0f}" '
                 f'height="{legend_h - 16:.0f}" fill="#fafafa" stroke="#dddddd"/>')
    for i, ln in enumerate(lines):
        color = _BAD if (i == len(lines) - 1 and crane["status"] != "ok") else "#222222"
        if clr and i == len(lines) - 2 and clr["status"] != "ok":
            color = _BAD if clr["status"] == "violation" else "#8a6d00"
        parts.append(f'<text x="24" y="{ly + i * 20:.1f}" font-size="12" '
                     f'fill="{color}">{escape(ln)}</text>')

    parts.append('</svg>')
    return "".join(parts)
