# -*- coding: utf-8 -*-
"""
FastAPI 应用入口
================

吊装受力核算 / 试吊重心反算 / 人工调整复核 的 HTTP 接口。
持久化(SQLite 版本库)由 storage 层提供; 未配置数据库时全部接口仍可计算,
版本相关端点返回 503 提示。

启动:
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse

from . import CALC_ENGINE_ID, __version__
from .models import DeriveIn, LiftInput, ReviewSubmitIn
from .physics import (
    analyze_lift,
    evaluate_manual_adjustment,
    inverse_cog_from_trials,
    run_full_analysis,
)

app = FastAPI(
    title="大型构件吊装核算服务",
    version=__version__,
    description="刚体静力学受力求解、试吊重心反算、吊索/卸扣/吊梁利用率与审批判定",
)

DB_PATH = os.environ.get("LIFTCALC_DB", "")
_storage = None


def get_storage():
    """惰性打开 SQLite 版本库; 未配置环境变量时返回 None(接口降级可用)。"""
    global _storage
    if DB_PATH and _storage is None:
        from .storage import PlanStore  # 局部导入, 无 DB 时不强制
        # FastAPI 在线程池中执行同步端点, 需允许跨线程使用连接
        _storage = PlanStore(DB_PATH, check_same_thread=False)
    return _storage


# ---------------------------------------------------------------- 异常
class CalcError(HTTPException):
    """计算阶段的工程性错误(几何退化等), 返回 422 + 结构化说明。"""


@app.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError):  # noqa: ANN001
    return JSONResponse(
        status_code=422,
        content={"error": "CALCULATION_ERROR", "detail": str(exc)},
    )


# ---------------------------------------------------------------- 接口
@app.get("/health", tags=["meta"])
def health() -> Dict[str, Any]:
    return {"status": "ok", "version": __version__, "engine": CALC_ENGINE_ID,
            "storage": "sqlite" if get_storage() else "disabled"}


@app.post("/api/lifts/analyze", tags=["lifts"])
def analyze(inp: LiftInput) -> Dict[str, Any]:
    """完整核算: 试吊反算(若有) -> 正式受力 -> 调整建议 -> 批准判定。"""
    result = run_full_analysis(inp)
    result["engine"] = CALC_ENGINE_ID
    return result


@app.post("/api/lifts/inverse-cog", tags=["lifts"])
def inverse_cog(inp: LiftInput) -> Dict[str, Any]:
    """仅执行试吊重心反算, 返回冲突方程与证据缺口。"""
    if not inp.trials:
        return {
            "status": "insufficient_data",
            "message": "没有任何试吊测量(trials 为空), 无法反算实际重心",
            "engine": CALC_ENGINE_ID,
            "conflicts": [],
            "evidence_gaps": [{"code": "NO_TRIALS",
                               "message": "至少需要一次带张力测点的低高度试吊"}],
        }
    result = inverse_cog_from_trials(inp)
    result["engine"] = CALC_ENGINE_ID
    return result


@app.post("/api/lifts/analyze-forward", tags=["lifts"])
def analyze_forward(inp: LiftInput) -> Dict[str, Any]:
    """仅执行正式起吊受力核算(不做试吊反算; 可用 cog_actual_override)。"""
    result = analyze_lift(inp, None)
    result["engine"] = CALC_ENGINE_ID
    return result


@app.post("/api/lifts/manual-recheck", tags=["lifts"])
def manual_recheck(inp: LiftInput) -> Dict[str, Any]:
    """
    工程师人工调整复核: 应用 manual_adjustment(吊点移位/索长/配重)后整体重算。
    系统不豁免任何检查; 调整后仍超载则 approvable=false 并给出冲突。
    """
    if inp.manual_adjustment is None:
        raise HTTPException(status_code=400,
                            detail="manual_adjustment 字段缺失, 无可复核的人工调整")
    result = evaluate_manual_adjustment(inp)
    result["engine"] = CALC_ENGINE_ID
    return result


@app.post("/api/lifts/validate", tags=["lifts"])
def validate_input(inp: LiftInput) -> Dict[str, Any]:
    """仅做参数校验回显(Pydantic 通过即返回规范化输入)。"""
    return {"valid": True, "name": inp.name,
            "n_legs": len(inp.legs), "n_trials": len(inp.trials),
            "has_beam": inp.beam is not None,
            "has_crane": inp.crane is not None}


@app.post("/api/lifts/crane-check", tags=["lifts"])
def crane_check(inp: LiftInput) -> Dict[str, Any]:
    """
    仅执行起重机工况校核: 回转路径按步长插值, 逐姿态给出作业半径、
    净额定载荷、支腿反力与接地压力, 并定位首个违规区间。
    吊钩载荷复用吊装核算的吊钩动载。
    """
    if inp.crane is None:
        raise HTTPException(status_code=400,
                            detail="输入缺少 crane 字段, 无起重机工况可校核")
    result = analyze_lift(inp, None)
    return {"engine": CALC_ENGINE_ID,
            "hook_load_dynamic_kn": result["weights_kn"]["hook_dynamic"],
            "crane": result["crane"]}


@app.post("/api/lifts/crane-station.svg", tags=["lifts"])
def crane_station_svg(inp: LiftInput) -> Response:
    """
    起重机站位图(SVG): 回转中心、支腿垫板、回转路径与逐姿态采样点。
    与 /api/lifts/crane-check 的 JSON 共用同一份逐姿态数据。
    """
    if inp.crane is None:
        raise HTTPException(status_code=400,
                            detail="输入缺少 crane 字段, 无起重机工况可绘制")
    from .svg import crane_station_svg as _render  # 局部导入, 保持启动轻量
    result = analyze_lift(inp, None)
    return Response(content=_render(result["crane"]), media_type="image/svg+xml")


@app.get("/api/plans", tags=["plans"])
def list_plans() -> Dict[str, Any]:
    store = get_storage()
    if store is None:
        raise HTTPException(status_code=503, detail="未配置 LIFTCALC_DB, 版本库未启用")
    return {"plans": store.list_plans()}


@app.get("/api/plans/{plan_id}/revisions", tags=["plans"])
def list_revisions(plan_id: str) -> Dict[str, Any]:
    store = get_storage()
    if store is None:
        raise HTTPException(status_code=503, detail="未配置 LIFTCALC_DB, 版本库未启用")
    revs = store.list_revisions(plan_id)
    if not revs:
        raise HTTPException(status_code=404, detail=f"方案 {plan_id} 不存在")
    return {"plan_id": plan_id, "revisions": revs}


@app.get("/api/plans/{plan_id}/revisions/{revision}", tags=["plans"])
def get_revision(plan_id: str, revision: int) -> Dict[str, Any]:
    store = get_storage()
    if store is None:
        raise HTTPException(status_code=503, detail="未配置 LIFTCALC_DB, 版本库未启用")
    rec = store.get_revision(plan_id, revision)
    if rec is None:
        raise HTTPException(status_code=404,
                            detail=f"版本 {plan_id} r{revision} 不存在")
    return rec


@app.post("/api/plans/{plan_id}/revisions", tags=["plans"], status_code=201)
def create_revision(plan_id: str, inp: LiftInput) -> Dict[str, Any]:
    """
    保存一次核算为新版本。系统自动计算 approvable:
      仅 approvable=true 的版本可在随后 /review 中被批准并锁定。
    """
    store = get_storage()
    if store is None:
        raise HTTPException(status_code=503, detail="未配置 LIFTCALC_DB, 版本库未启用")
    result = run_full_analysis(inp)
    try:
        rev = store.save_revision(plan_id, inp, result)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return rev


@app.post("/api/plans/{plan_id}/revisions/{revision}/review", tags=["plans"])
def review_revision(plan_id: str, revision: int, body: ReviewSubmitIn) -> Dict[str, Any]:
    """
    人工复核: action = approve / reject / request_changes。
    approve 仅在系统 approvable=true 时生效(方案不能带病批准);
    批准时锁定输入与计算摘要, 后续版本只能从它派生。
    """
    store = get_storage()
    if store is None:
        raise HTTPException(status_code=503, detail="未配置 LIFTCALC_DB, 版本库未启用")
    try:
        return store.review(plan_id, revision, body.action, body.reviewer, body.comment)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/api/plans/{plan_id}/derive", tags=["plans"], status_code=201)
def derive(plan_id: str, body: DeriveIn) -> Dict[str, Any]:
    """从指定版本(通常为批准版)派生新版本, 记录来源链与变更说明。"""
    store = get_storage()
    if store is None:
        raise HTTPException(status_code=503, detail="未配置 LIFTCALC_DB, 版本库未启用")
    result = run_full_analysis(body.input)
    try:
        return store.derive_revision(
            plan_id, body.from_revision, body.input, result, body.change_note
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/api/plans/{plan_id}/diff/{rev_a}/{rev_b}", tags=["plans"])
def diff_revisions(plan_id: str, rev_a: int, rev_b: int) -> Dict[str, Any]:
    """比较两版输入与计算摘要差异。"""
    store = get_storage()
    if store is None:
        raise HTTPException(status_code=503, detail="未配置 LIFTCALC_DB, 版本库未启用")
    return store.diff(plan_id, rev_a, rev_b)


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
