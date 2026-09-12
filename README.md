# 构件试吊重心反演与索具核算 API

基于刚体静力学的大型构件吊装核算 FastAPI 服务：多索受力求解（静定 / 超定 / 静不定弹性分配）、
低高度试吊张力反算实际重心、吊索 / 卸扣 / 吊梁利用率核算、人工调整复核，以及 SQLite 方案版本与审批流。
起重机工况校核：吊钩自试吊点回转到安装点的路径上按步长插值，逐姿态核算作业半径、
载荷表净额定能力（同一臂长、配置与回转区段的相邻档位内插值，不跨配置、不外推）、
支腿反力与垫板接地压力，定位首个超载 / 支腿拔起 / 越出载荷表 / 地基超压区间；
路径净空校核：构件包络（定向长方体）随吊钩路径同步插值姿态（yaw/pitch/roll），
位置按最大位移步长、姿态按最大转角步长分别细分，分离轴法（SAT）逐点求
障碍物 / 不可侵入区轴对齐盒的保守净空，定位首个碰撞或间距不足区间；
结果进入 `approvable` 与版本差异，批准版冻结载荷表摘要与路径，
JSON 与 SVG 站位图共用同一份逐姿态数据（姿态角与碰撞标记）。
线性代数为纯 Python 实现（`app/linalg.py`），不依赖 numpy。

## 环境要求

- Python **3.11 及以上**（支持 3.11 / 3.12 / 3.13）
- 依赖：FastAPI、Pydantic v2、uvicorn；测试另需 pytest、httpx

## 安装

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"            # -e 可编辑安装; [dev] 装上 pytest/httpx
```

无网络环境时可改用离线依赖目录：

```bash
pip install -e . --no-index --find-links .pylibs
```

## 启动

入口固定为 `app.main:app`：

```bash
# 方式一: console 脚本
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 方式二: 模块方式
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# 方式三: python -m app.main（同样监听 0.0.0.0:8000）
python -m app.main
```

启动后：

- 健康检查：`GET http://localhost:8000/health`
- 交互式 API 文档：`http://localhost:8000/docs`

## 环境变量

| 变量 | 说明 |
| --- | --- |
| `LIFTCALC_DB` | SQLite 版本库文件路径。**未设置（或为空）时不建库、不读写任何文件**：`/health` 返回 `"storage": "disabled"`，纯计算接口（`/api/lifts/*`）完全可用，版本库接口（`/api/plans/*`）返回 503。 |

启用 SQLite 版本库（文件不存在会自动建表）：

```bash
LIFTCALC_DB=./liftcalc.db uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 测试

```bash
pytest                 # 配置已在 pytest.ini 中(testpaths=tests)
pytest -q tests/test_smoke.py     # 仅跑冒烟用例(不写固定数据库)
```

冒烟用例覆盖：模块导入、无 `LIFTCALC_DB` 时 `/health` 降级可用、一次对称四索核算
（四索等张力、吊钩动载 1100 kN）、临时 SQLite 库的版本保存与读回（`mkstemp` 路径，结束即删除）。

## 单位约定

SI 制：长度 m，力 kN，质量 kg（内部按 g 换算 kN），角度 °（接口）/ rad（内部）。
构件本体坐标：x 沿梁长纵向、y 横向、z 竖直向上。

## 起重机工况校核（`crane` 字段）

在任意吊装核算请求中加入 `crane` 对象即启用（回转中心、支腿与垫板几何、地基承压限值、
起重机部件重量与重心、配重、吊钩滑轮组重量、吊钩路径、按臂长 / 半径 / 回转区段给出的载荷表）：

- 路径按 `path_step_m` 在相邻姿态间插值，逐点计算作业半径、净额定载荷
  （载荷表毛能力 − 吊钩滑轮组重量）、支腿反力（刚性车体等刚度支座解）与垫板接地压力；
  吊钩载荷直接复用吊装核算的吊钩动载。
- 载荷表只在同一臂长、同一配置（`config`，缺省 `STD`）与同一回转区段的相邻半径档位间
  线性插值；同配置没有合法相邻档位即 `CRANE_OUT_OF_CHART`，不跨配置、不外推。
- 违规冲突码：`CRANE_CHART_OVERLOAD`（超载）、`CRANE_OUTRIGGER_UPLIFT`（支腿拔起）、
  `CRANE_OUT_OF_CHART`（越出载荷表）、`CRANE_GROUND_OVERPRESSURE`（地基超压）；
  任一出现即阻断 `approvable`，并进入版本 `diff` 的冲突与摘要差异。
- 相关接口：`POST /api/lifts/crane-check`（仅校核，JSON 逐姿态数据）、
  `POST /api/lifts/crane-station.svg`（站位图，与 JSON 共用同一份逐姿态数据）。
- 批准版随版本库冻结载荷表摘要（内容指纹）与吊钩路径；工况变化须从批准版 `/derive` 派生。

## 路径净空校核（`crane.clearance` 字段）

吊钩中心线避开厂房立柱，不代表长构件转向时还有足够净空。在 `crane` 中加入
`clearance` 对象即启用（与工况校核共用同一份路径采样）：

- 输入：构件包络长宽高（`envelope`，本体系长方体）、吊钩至包络中心偏移
  （`hook_to_center_offset`，本体系，随姿态旋转）、各关键姿态的
  `yaw_deg/pitch_deg/roll_deg`（挂在 `crane.path` 姿态上，旋转矩阵
  R = Rz(yaw)·Ry(pitch)·Rx(roll)）、障碍物轴对齐盒（`obstacles`）、
  不可侵入区（`exclusion_zones`，零容忍）、安全间距（`safety_margin_m`）。
- 插值：位置按 `path_step_m`、姿态按 `attitude_step_deg`（测地转角）分别细分，
  每段取两者较大分段数；欧拉角先逐轴展开（最短分支）再线性插值。
- 校核：包络中心 = 吊钩位置 + R·偏移，定向包络（OBB）与轴对齐盒（AABB）用
  分离轴法（15 条候选轴）求有符号净空，逐点再减去扫掠修正（相邻采样间包络
  表面点最大位移的一半），得到覆盖采样区间的保守净空。
- 冲突码：`CRANE_COLLISION`（保守净空 < 0）、`CRANE_CLEARANCE_INSUFFICIENT`
  （保守净空低于安全间距）、`CRANE_EXCLUSION_INTRUSION`（侵入不可侵入区）；
  任一出现即阻断 `approvable`，并进入版本 `diff` 的冲突与摘要差异。
- 证据缺口（不阻断但随版本记录）：`CLEARANCE_ENVELOPE_DEGENERATE`（包络尺寸
  非正/非有限，校核跳过）、`CLEARANCE_OBSTACLE_INVALID`（障碍盒 min>=max 或
  非有限，剔除该盒）、`CLEARANCE_ATTITUDE_JUMP`（相邻姿态转角超
  `attitude_jump_deg`，插值姿态未必代表实际转向）、`CLEARANCE_SAMPLING_CAP`
  （所需采样点超 `max_samples`，已等比放宽步长）。
- SVG 站位图叠加障碍物/不可侵入区足迹与包络逐姿态水平投影，
  碰撞采样点红色标记，与 JSON 共用同一份姿态角与碰撞标记。
