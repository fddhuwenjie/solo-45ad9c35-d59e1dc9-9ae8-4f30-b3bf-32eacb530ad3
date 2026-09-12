# -*- coding: utf-8 -*-
"""
起重机站位图(SVG)
==================

平面布置图: 回转中心、支腿与垫板、吊钩回转路径与逐姿态采样点。
输入仅为起重机工况校核结果 dict(crane.check_crane_duty 的返回值),
与 JSON 接口共用同一份逐姿态数据, 不重复计算。
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple
from xml.sax.saxutils import escape

W, H = 1000.0, 780.0
LEGEND_H = 150.0

_OK = "#1a9e50"
_BAD = "#d62728"
_PATH = "#1f6fc5"
_MAT_FILL = "#e8eefc"
_MAT_STROKE = "#3b6ea5"


def _fmt(v: Any, nd: int = 1) -> str:
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "-"


def crane_station_svg(crane: Dict[str, Any]) -> str:
    """由工况校核结果渲染站位平面图(世界 x 向右, y 向上)。"""
    setup = crane["setup"]
    samples: List[Dict[str, Any]] = crane["samples"]
    env = crane["envelope"]
    cx, cy = setup["slewing_center"]

    # ---------------- 视图范围(含垫板外缘与路径) ----------------
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
    x0, x1 = min(xs) - pad, max(xs) + pad
    y0, y1 = min(ys) - pad, max(ys) + pad
    scale = min((W - 140.0) / max(x1 - x0, 1e-6),
                (H - LEGEND_H - 120.0) / max(y1 - y0, 1e-6))
    ox = (W - (x1 - x0) * scale) / 2.0
    oy = (H - LEGEND_H - (y1 - y0) * scale) / 2.0 + 40.0

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
            parts.append(f'<text x="{px + 8:.1f}" y="{py - 8:.1f}" font-size="11" '
                         f'fill="#111111">{escape(s["at_pose"])}</text>')

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

    # ---------------- 图例/摘要(与 JSON 同一份数据) ----------------
    chart = crane["load_chart_summary"]
    path = crane["path_summary"]
    ly = H - LEGEND_H + 18.0
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
        ('校核通过, 无违规区间' if crane["status"] == "ok"
         else f'首个违规 {fv["kind"]} @ {fv["segment"]} (r={_fmt(fv["radius_m"])} m)'
         if fv else '校核未通过'),
    ]
    parts.append(f'<rect x="10" y="{H - LEGEND_H:.0f}" width="{W - 20:.0f}" '
                 f'height="{LEGEND_H - 16:.0f}" fill="#fafafa" stroke="#dddddd"/>')
    for i, ln in enumerate(lines):
        color = _BAD if (i == len(lines) - 1 and crane["status"] != "ok") else "#222222"
        parts.append(f'<text x="24" y="{ly + i * 20:.1f}" font-size="12" '
                     f'fill="{color}">{escape(ln)}</text>')

    parts.append('</svg>')
    return "".join(parts)
