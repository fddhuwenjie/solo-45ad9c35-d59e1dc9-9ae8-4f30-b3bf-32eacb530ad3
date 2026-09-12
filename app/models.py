# -*- coding: utf-8 -*-
"""Pydantic v2 入参/出参模型与参数校验。

单位约定(全局 SI):
  长度 m, 力 kN, 质量 kg(内部按 g 换算为 kN), 角度 °, 加速度 m/s²。
坐标系(构件本体):
  x — 梁长方向(纵向), y — 横向, z — 竖直向上(水平姿态)。
"""
from __future__ import annotations

from typing import List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator

Vec3 = Tuple[float, float, float]

# ---------------------------------------------------------------- 基础构件
class LegIn(BaseModel):
    """一根下吊索: 构件吊点 pad -> 吊钩(无吊梁) 或吊梁下吊耳。"""

    id: str = Field(..., min_length=1, description="吊索编号, 如 L1")
    pad: Vec3 = Field(..., description="构件上吊点坐标 [x,y,z] m(本体坐标)")
    length_m: float = Field(
        0.0, ge=0, description="吊索公称长度 m; 0 表示由几何自动计算"
    )
    sling_capacity_kn: float = Field(..., gt=0, description="单根吊索额定工作载荷 WLL kN")
    shackle_capacity_kn: float = Field(..., gt=0, description="卸扣额定载荷 kN(按只, 受拉)")
    sling_count: int = Field(1, ge=1, description="该吊点并联手拉葫芦/索股数")
    derating_factor: float = Field(1.0, gt=0, le=1.0, description="额定载荷折减系数 (0,1]")
    lug_index: Optional[int] = Field(
        None, ge=0, description="带吊梁时: 连接的下吊耳序号(从0起); 不带吊梁留空"
    )
    sling_ea_kn: Optional[float] = Field(
        None, gt=0, description="吊索轴向刚度 EA kN; 缺省按索长倒数等刚度分配"
    )

    @field_validator("pad")
    @classmethod
    def _pad3(cls, v: Vec3) -> Vec3:
        if len(v) != 3 or any(not isinstance(x, (int, float)) for x in v):
            raise ValueError("pad 必须是 3 个数字 [x,y,z]")
        return (float(v[0]), float(v[1]), float(v[2]))


class BeamSectionIn(BaseModel):
    """吊梁截面特性(用于轴力+双向弯矩组合利用率, 箱型/管型截面通用输入)。

    梁轴沿 x: 弯矩 M_y 用截面模量 W_y, M_z 用 W_z。
    """

    area_m2: float = Field(..., gt=0)
    wy_m3: float = Field(..., gt=0, description="绕 y 轴截面模量 m³")
    wz_m3: float = Field(..., gt=0, description="绕 z 轴截面模量 m³")
    i_polar_m4: Optional[float] = Field(None, gt=0, description="极惯性矩 m⁴, 提供则核算扭转")
    torsion_lever_m: Optional[float] = Field(
        None, gt=0, description="开截面/弦杆梁的扭转计算力臂 m, 缺省按极惯性矩法"
    )
    allowable_stress_kpa: float = Field(..., gt=0, description="许用应力 kPa (=kN/m²)")
    shear_capacity_kn: Optional[float] = Field(None, gt=0, description="梁的抗剪承载力 kN")


class TopSlingIn(BaseModel):
    """上吊索(吊梁 -> 吊钩汇交点)。"""

    id: str = Field(..., min_length=1)
    at_x: float = Field(..., description="梁上的连接 x 坐标 m")
    convergence: Vec3 = Field(
        ..., description="上端汇交点(吊钩侧)本体坐标 [x,y,z] m, 多根上索通常汇交于同一点"
    )
    sling_capacity_kn: float = Field(..., gt=0)
    shackle_capacity_kn: float = Field(..., gt=0)
    sling_count: int = Field(1, ge=1)
    derating_factor: float = Field(1.0, gt=0, le=1.0)

    @field_validator("convergence")
    @classmethod
    def _c3(cls, v: Vec3) -> Vec3:
        return (float(v[0]), float(v[1]), float(v[2]))


class SpreaderBeamIn(BaseModel):
    """吊梁几何(梁轴沿本体 x, 位于 y=beam_y, z=beam_z)。"""

    end_a_x: float = Field(..., description="梁左端 x m")
    end_b_x: float = Field(..., description="梁右端 x m")
    beam_y: float = 0.0
    beam_z: float = Field(..., description="梁轴线高度 z m(本体坐标)")
    lower_lugs_x: List[float] = Field(..., min_length=1, description="下吊耳 x 坐标列表 m")
    tops: List[TopSlingIn] = Field(..., min_length=1)
    section: BeamSectionIn
    beam_mass_kg: float = Field(0.0, ge=0, description="吊梁自重 kg")
    beam_cg_x: Optional[float] = Field(None, description="梁重心 x, 缺省取梁中点")

    @model_validator(mode="after")
    def _check(self) -> "SpreaderBeamIn":
        if self.end_b_x <= self.end_a_x:
            raise ValueError("end_b_x 必须大于 end_a_x")
        for i, x in enumerate(self.lower_lugs_x):
            if not (self.end_a_x - 1e-6 <= x <= self.end_b_x + 1e-6):
                raise ValueError(f"下吊耳 {i} x={x} 超出梁端范围")
        for t in self.tops:
            if not (self.end_a_x - 1e-6 <= t.at_x <= self.end_b_x + 1e-6):
                raise ValueError(f"上吊索 {t.id} at_x 超出梁端范围")
            if t.convergence[2] <= self.beam_z:
                raise ValueError(f"上吊索 {t.id} 汇交点必须高于梁顶")
        if len({t.id for t in self.tops}) != len(self.tops):
            raise ValueError("上吊索 id 重复")
        return self


class CounterweightIn(BaseModel):
    """配重: 已有配重或拟加配重。"""

    mass_kg: float = Field(..., ge=0)
    position: Vec3 = Field(...)
    id: str = Field("CW1", min_length=1)

    @field_validator("position")
    @classmethod
    def _p3(cls, v: Vec3) -> Vec3:
        return (float(v[0]), float(v[1]), float(v[2]))


class CounterweightStationIn(BaseModel):
    """可加配重的候选位置(调整量求解时使用)。"""

    id: str
    position: Vec3
    capacity_kg: float = Field(..., ge=0)


# ---------------------------------------------------------------- 试吊实测
class MeasurementIn(BaseModel):
    leg_id: str
    tension_kn: Optional[float] = Field(None, ge=0, description="实测索张力 kN(动载状态)")
    measured_direction: Optional[Vec3] = Field(
        None, description="实测索方向(世界系, pad 指向吊钩, 无需归一化)"
    )
    measured_angle_deg: Optional[float] = Field(
        None, ge=0, le=90, description="实测索与水平面夹角 °(无方向向量时)"
    )
    tension_is_static: bool = Field(
        False, description="张力是否在静止(无加速度)状态测得; 为真则反算不乘动载系数"
    )


class TrialIn(BaseModel):
    """一次低高度试吊记录。"""

    id: str = Field(..., min_length=1)
    pitch_deg: float = Field(0.0, description="试吊时构件纵倾角 °(绕 y, 抬头为正)")
    roll_deg: float = Field(0.0, description="横倾角 °(绕 x)")
    measurements: List[MeasurementIn] = Field(..., min_length=1)
    measured_total_weight_kn: Optional[float] = Field(
        None, gt=0, description="吊钩处称出的总重(含索具) kN, 用于称重交叉核对"
    )
    notes: str = ""

    @model_validator(mode="after")
    def _chk(self) -> "TrialIn":
        ids = [m.leg_id for m in self.measurements]
        if len(set(ids)) != len(ids):
            raise ValueError(f"试吊 {self.id} 中同一吊索出现多条测量")
        return self


# ---------------------------------------------------------------- 人工调整
class PadShiftIn(BaseModel):
    leg_id: str
    dx: float = 0.0
    dy: float = 0.0


class ShimIn(BaseModel):
    leg_id: str
    delta_length_m: float = Field(..., description="索长调整量 m(正=放长, 负=收短)")


class ManualAdjustmentIn(BaseModel):
    """工程师提交的人工调整(复核时系统仍重新核算, 不豁免任何检查)。"""

    mode: Literal["shift_pickpoints", "trim_slings", "counterweight", "manual_mix"]
    pad_shifts: List[PadShiftIn] = []
    shims: List[ShimIn] = []
    counterweight_mass_kg: float = Field(0.0, ge=0)
    counterweight_station_id: Optional[str] = None
    comment: str = ""


# ---------------------------------------------------------------- 总输入
class LiftInput(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    component_weight_kn: float = Field(..., gt=0, description="构件图纸/称重重量 kN")
    cog_theory: Vec3 = Field(..., description="理论重心 [x,y,z] m")
    legs: List[LegIn] = Field(..., min_length=1)
    beam: Optional[SpreaderBeamIn] = None
    counterweight: Optional[CounterweightIn] = None
    counterweight_stations: List[CounterweightStationIn] = []
    extra_rigging_mass_kg: float = Field(0.0, ge=0, description="附加索具自重 kg(不含吊梁)")
    extra_rigging_cog: Optional[Vec3] = None
    lift_acceleration_mps2: float = Field(0.1, ge=0, description="起升加速度 m/s²")
    min_dynamic_factor: float = Field(1.1, ge=1.0, description="动载系数下限")
    g_mps2: float = Field(9.81, gt=0)
    min_sling_angle_deg: float = Field(45.0, ge=0, le=90)
    hoist_capacity_kn: Optional[float] = Field(None, gt=0, description="起升机/吊钩额定 kN")
    hook_point: Optional[Vec3] = Field(None, description="无吊梁时吊钩点本体坐标(缺省自动)")
    trials: List[TrialIn] = Field([], description="低高度试吊实测")
    cog_actual_override: Optional[Vec3] = Field(
        None, description="人工/批准版沿用的实测重心; 提供则优先于试吊反算"
    )
    manual_adjustment: Optional[ManualAdjustmentIn] = None
    max_pad_shift_m: float = Field(0.5, gt=0)
    request_adjustment: bool = Field(True, description="是否给出自动调整建议")

    @field_validator("cog_theory", "extra_rigging_cog", "hook_point", "cog_actual_override")
    @classmethod
    def _v3(cls, v):
        if v is None:
            return None
        return (float(v[0]), float(v[1]), float(v[2]))

    @model_validator(mode="after")
    def _chk(self) -> "LiftInput":
        ids = [lg.id for lg in self.legs]
        if len(set(ids)) != len(ids):
            raise ValueError("吊索 id 重复")
        known = set(ids)
        if self.beam is not None:
            for lg in self.legs:
                if lg.lug_index is not None:
                    if not (0 <= lg.lug_index < len(self.beam.lower_lugs_x)):
                        raise ValueError(f"吊索 {lg.id} 的 lug_index 越界")
            n_lugs = len(self.beam.lower_lugs_x)
            if len(self.legs) == n_lugs and all(lg.lug_index is None for lg in self.legs):
                pass  # 等数时按序号对应
            else:
                for lg in self.legs:
                    if lg.lug_index is None:
                        raise ValueError(
                            f"吊梁下吊耳数({n_lugs})与吊索数({len(self.legs)})不等, "
                            f"吊索 {lg.id} 必须显式指定 lug_index"
                        )
        else:
            for lg in self.legs:
                if lg.lug_index is None:
                    continue
                raise ValueError(f"吊索 {lg.id} 指定了 lug_index 但方案中没有吊梁")
        for t in self.trials:
            for m in t.measurements:
                if m.leg_id not in known:
                    raise ValueError(f"试吊 {t.id} 引用了不存在的吊索 {m.leg_id}")
                if m.tension_kn is None and m.measured_direction is None and m.measured_angle_deg is None:
                    raise ValueError(f"试吊 {t.id}/{m.leg_id} 缺少张力和倾角/方向, 测点无效")
        tids = [t.id for t in self.trials]
        if len(set(tids)) != len(tids):
            raise ValueError("试吊 id 重复")
        if self.extra_rigging_mass_kg > 0 and self.extra_rigging_cog is None:
            raise ValueError("提供附加索具质量时必须给出其重心 extra_rigging_cog")
        if self.manual_adjustment is not None:
            ma = self.manual_adjustment
            refs = {s.leg_id for s in ma.pad_shifts} | {s.leg_id for s in ma.shims}
            unknown = refs - known
            if unknown:
                raise ValueError(f"人工调整引用了不存在的吊索: {sorted(unknown)}")
        return self


# ---------------------------------------------------------------- 版本库相关
class ReviewSubmitIn(BaseModel):
    plan_id: str
    revision: int = Field(..., ge=1)
    reviewer: str = Field(..., min_length=1)
    action: Literal["approve", "reject", "request_changes"]
    comment: str = ""
    input_override: Optional[LiftInput] = Field(
        None, description="提交复核时可携带含 manual_adjustment 的调整后输入"
    )


class DeriveIn(BaseModel):
    plan_id: str
    revision: int = Field(..., ge=1, description="从此版本(通常是已批准版)派生")
    new_input: LiftInput
    change_note: str = ""
