# -*- coding: utf-8 -*-
"""
纯 Python 线性代数工具(三维向量 / 矩阵 / 线性最小二乘)。

吊装问题的未知量不超过 10 个，无需 numpy；自实现也保证复算结果
在任何环境下逐位一致(同一份高斯消元代码)。
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]
Matrix = List[List[float]]  # m 行 n 列行主序

EPS = 1.0e-12


# ---------------------------------------------------------------- 三维向量
def v_add(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v_sub(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v_mul(a: Sequence[float], s: float) -> Vec3:
    return (a[0] * s, a[1] * s, a[2] * s)


def v_dot(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_cross(a: Sequence[float], b: Sequence[float]) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def v_norm(a: Sequence[float]) -> float:
    return math.sqrt(v_dot(a, a))


def v_unit(a: Sequence[float]) -> Vec3:
    n = v_norm(a)
    if n < EPS:
        raise ValueError("零向量不能归一化")
    return (a[0] / n, a[1] / n, a[2] / n)


def v_lerp(a: Sequence[float], b: Sequence[float], t: float) -> Vec3:
    return (
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


def skew(a: Sequence[float]) -> Matrix:
    """a × x 的反对称矩阵 [a]_x。"""
    ax, ay, az = a
    return [
        [0.0, -az, ay],
        [az, 0.0, -ax],
        [-ay, ax, 0.0],
    ]


# ---------------------------------------------------------------- 通用矩阵
def mat_zeros(m: int, n: int) -> Matrix:
    return [[0.0] * n for _ in range(m)]


def mat_mul(A: Matrix, B: Matrix) -> Matrix:
    m, k, n = len(A), len(A[0]), len(B[0])
    C = mat_zeros(m, n)
    for i in range(m):
        Ai = A[i]
        Ci = C[i]
        for t in range(k):
            a = Ai[t]
            if a == 0.0:
                continue
            Bt = B[t]
            for j in range(n):
                Ci[j] += a * Bt[j]
    return C


def mat_T(A: Matrix) -> Matrix:
    return [list(col) for col in zip(*A)]


def mat_vec(A: Matrix, x: Sequence[float]) -> List[float]:
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


def mat_identity(n: int) -> Matrix:
    I = mat_zeros(n, n)
    for i in range(n):
        I[i][i] = 1.0
    return I


# ---------------------------------------------------------------- 高斯消元
def _gauss_solve_square(A: Matrix, b: List[float], tol: float) -> List[float]:
    """带部分选主元的高斯消元，仅用于非奇异方阵；奇异返回 None。"""
    n = len(A)
    M = [A[i][:] + [b[i]] for i in range(n)]
    scale = max((abs(v) for row in M for v in row[:-1]), default=1.0)
    if scale < EPS:
        scale = 1.0
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) <= tol * scale:
            return None
        if piv != col:
            M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for r in range(col + 1, n):
            f = M[r][col] / pv
            if f == 0.0:
                continue
            for c in range(col, n + 1):
                M[r][c] -= f * M[col][c]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))
        piv = M[i][i]
        if abs(piv) <= tol * scale:
            return None
        x[i] = s / piv
    return x


def _rref(A: Matrix, b: Optional[List[float]] = None, tol: float = 1e-9):
    """
    全选主元 RREF(行间部分选主元, 消去主元列上全部行, 含已处理行)。
    不修改入参。返回 (pivot_row_indices, pivot_col_indices, R, br)。
    br 为同步消元后的右端(b 为 None 时返回 None)。
    """
    m, n = len(A), len(A[0]) if A else 0
    R = [row[:] for row in A]
    br = list(b) if b is not None else None
    scale = max((abs(v) for row in R for v in row), default=1.0) or 1.0
    pivot_rows: List[int] = []
    pivot_cols: List[int] = []
    # 记录当前第 r 行来自原矩阵的哪一行
    orig = list(range(m))
    rr = 0
    for col in range(n):
        if rr >= m:
            break
        piv = max(range(rr, m), key=lambda i: abs(R[i][col]))
        if abs(R[piv][col]) <= tol * scale:
            continue
        R[rr], R[piv] = R[piv], R[rr]
        orig[rr], orig[piv] = orig[piv], orig[rr]
        if br is not None:
            br[rr], br[piv] = br[piv], br[rr]
        pv = R[rr][col]
        for i in range(m):
            if i == rr:
                continue
            f = R[i][col] / pv
            if f == 0.0:
                continue
            for c in range(n):
                R[i][c] -= f * R[rr][c]
            if br is not None:
                br[i] -= f * br[rr]
        pivot_rows.append(orig[rr])
        pivot_cols.append(col)
        rr += 1
    return pivot_rows, pivot_cols, R, br


def _rank(A: Matrix, tol: float = 1e-9) -> int:
    pr, _pc, _R, _b = _rref(A, None, tol)
    return len(pr)



def matrix_rank(A: Matrix, tol: float = 1e-9) -> int:
    return _rank(A, tol)


def independent_system(A: Matrix, b: Sequence[float], tol: float = 1e-9) \
        -> Tuple[Matrix, List[float]]:
    """从 A x = b 中取一组线性无关方程(返回【原始】行与其右端)。"""
    prows, _pcols, _R, _br = _rref(A, list(b), tol)
    return [A[i][:] for i in prows], [b[i] for i in prows]


def residual(A: Matrix, x: Sequence[float], b: Sequence[float]) -> List[float]:
    return [Ax - bi for Ax, bi in zip(mat_vec(A, x), b)]


def rms(v: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in v) / max(len(v), 1))


def solve_linear(
    A: Matrix, b: Sequence[float], tol: float = 1e-9, weights: Sequence[float] | None = None
) -> dict:
    """
    求解 A x = b，自动判别三种适定性:

      列满秩 + 行数=秩  -> exact (唯一解)
      列满秩 + 超定矛盾 -> overdetermined_ls (法方程最小二乘)
      秩亏但方程一致     -> underdetermined_minnorm (最小范数特解)
      秩亏且矛盾        -> rank_deficient_ls (最小二乘 + 最小范数)

    返回 x / kind / residual(A x − b, 原始未加权方程) / rank / nullspace_dim。
    """
    m, n = len(A), len(A[0])
    if weights is not None:
        sqw = [math.sqrt(w) for w in weights]
        Aw = [[A[i][j] * sqw[i] for j in range(n)] for i in range(m)]
        bw = [b[i] * sqw[i] for i in range(m)]
    else:
        Aw, bw = [row[:] for row in A], list(b)

    rank = _rank(Aw, tol)
    norm_b = math.sqrt(sum(v * v for v in b)) or 1.0
    meta = dict(rank=rank, n=n, nullspace_dim=n - rank, norm_b=norm_b)

    if rank == n:
        if m == n:
            x = _gauss_solve_square(Aw, bw, tol)
            kind = "exact"
        else:
            At = mat_T(Aw)
            x = _gauss_solve_square(mat_mul(At, Aw), mat_vec(At, bw), tol)
            kind = "overdetermined_ls"
        if x is None:
            x = _ls_minnorm(Aw, bw)
            kind = "rank_deficient_ls"
        res = residual(A, x, b)
        return dict(x=x, kind=kind, residual=res, **meta)

    # 秩亏: RREF 上回代, 自由变量取 0(该特解即最小范数解)
    _prows, pcols, R, br2 = _rref(Aw, bw, tol)
    x = [0.0] * n
    for k, col in enumerate(pcols):
        x[col] = br2[k] / R[k][col]
    res_full = residual(Aw, x, bw)
    inconsistent = la_rms(res_full) > 1e-7 * (norm_b if weights is None else
                                              math.sqrt(sum(v * v for v in bw)) or 1.0)
    if inconsistent:
        # 矛盾秩亏: 最小二乘后再在零空间中取最小范数
        x = _ls_minnorm(Aw, bw)
        kind = "rank_deficient_ls"
    else:
        kind = "underdetermined_minnorm"
    res = residual(A, x, b)
    return dict(x=x, kind=kind, residual=res, **meta)


def la_rms(v: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in v) / max(len(v), 1))


def _ls_minnorm(A: Matrix, b: List[float]) -> List[float]:
    """矛盾系统最小二乘最小范数解: x = Aᵀ y, AAᵀ y = b (奇异时加微小正则)。"""
    m, n = len(A), len(A[0])
    AAt = mat_mul(A, mat_T(A))
    y = _gauss_solve_square(AAt, b, 1e-12)
    if y is None:
        scale = max((abs(v) for row in AAt for v in row), default=1.0) or 1.0
        for i in range(m):
            AAt[i][i] += 1e-10 * scale
        y = _gauss_solve_square(AAt, b, 1e-9) or [0.0] * m
    return mat_vec(mat_T(A), y)


def solve_constrained_minnorm(
    Aeq: Matrix, beq: Sequence[float], C: Matrix, tol: float = 1e-9
) -> Tuple[List[float], List[float]]:
    """
    在约束 Aeq x = beq 下最小化 ||C x||。
    返回 (x, lambda)。
    KKT 平稳条件 CᵀC x − Aeqᵀλ = 0:
      [ CᵀC  −Aeqᵀ ] [x]     [0]
      [ Aeq    0   ] [λ]  =  [beq]
    """
    n = len(C[0])
    CtC = mat_mul(mat_T(C), C)
    AeqT = mat_T(Aeq)
    me = len(Aeq)
    K = mat_zeros(n + me, n + me)
    for i in range(n):
        for j in range(n):
            K[i][j] = CtC[i][j]
    for i in range(me):
        for j in range(n):
            K[n + i][j] = Aeq[i][j]
            K[j][n + i] = -AeqT[j][i]
    rhs = [0.0] * n + list(beq)
    sol = _gauss_solve_square(K, rhs, tol)
    if sol is None:
        raise ValueError("约束矩阵奇异，无法求最小范数解")
    return sol[:n], sol[n:]


# ---------------------------------------------------------------- 小工具
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if abs(b) > EPS else default
