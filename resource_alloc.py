# -*- coding: utf-8 -*-
"""
多资源分配算法 (论文 Section V, Algorithm 1-3)
============================================================
给定缓存决策 delta 与卸载决策 lambda, 用拉格朗日对偶法求解
带宽分配 (BRA) 与 CPU 频率分配 (CRA) 两个凸子问题, 交替迭代收敛。

核心计算全部向量化 (对卸载任务数组整体运算)。

Algorithm 1: 带宽分配 (BRA)  -- 二分搜索 Lagrange 乘子 theta
Algorithm 2: 计算资源分配 (CRA) -- 拉格朗日对偶 + 次梯度
Algorithm 3: 多资源分配 (MRA) -- 交替求解 BRA/CRA
"""
import numpy as np
from scipy.special import lambertw

LN2 = np.log(2.0)


# =================================================================
# H(C0): 求带宽 b 使传输时延 d/(b log2(1+gp/(N0 b))) = C0 (论文式 24)
# Lambert-W 闭式解为主, 退化点用二分回退保证数值稳健。
# =================================================================
def H_of_C0(C0, d, p, g, N0):
    # 统一转为一维数组: 标量输入(如 bandwidth_allocation 里逐任务调用)也会得到
    # 1-d 的 a/b, 避免 0-d 数组无法下标赋值 (numpy.float64 不支持 b[i]=...)。
    C0 = np.atleast_1d(np.asarray(C0, dtype=np.float64))
    d = np.atleast_1d(np.asarray(d, dtype=np.float64))
    g = np.atleast_1d(np.asarray(g, dtype=np.float64))
    a = d * LN2 * N0 / (C0 * p * g)
    z = -a * np.exp(-a)
    W = np.full_like(a, np.nan)
    ok = (z >= -np.exp(-1.0) + 1e-12) & (z < -1e-12) & np.isfinite(a)
    if ok.any():
        W[ok] = lambertw(z[ok], k=-1).real
    b = -1.0 / ((W + a) * C0 / (d * LN2))
    bad = (~np.isfinite(b)) | (b <= 0.0)
    for i in np.flatnonzero(bad):
        b[i] = _bisect_H(float(C0[i]), float(d[i]), p, float(g[i]), N0)

    return b


def _bisect_H(C0, d, p, g, N0):
    """二分求 b 使传输时延 = C0 (鲁棒回退)。"""
    lo, hi = 1e-9, 1e12

    def trans(b):
        r = b * np.log2(1.0 + g * p / (N0 * b))
        return d / r

    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if trans(mid) > C0:     # 时延过大 -> 带宽过小 -> 增大 b
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# =================================================================
# theta(b) = dS'/db (论文式 27a, lambda=1), 向量化
# =================================================================
def theta_of_b(b, d, p, g, N0, A):
    """S' = A - d/r,  theta = d*r'/(r^2*S'),  r'=log2(1+x)-x/(ln2(1+x))。"""
    b = np.maximum(b, 1e-12)
    x = g * p / (N0 * b)
    r = b * np.log2(1.0 + x)
    rp = np.log2(1.0 + x) - x / (LN2 * (1.0 + x))
    Sp = A - d / r
    return np.where(Sp > 0, d * rp / (r * r * Sp), -np.inf)


def _bisect_theta(theta, d, p, g, N0, A, lo, hi, iters=80):
    """向量化二分: 求 b 使 theta_of_b(b)=theta (theta 随 b 递减)。"""
    lo = lo.copy(); hi = hi.copy()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        th = theta_of_b(mid, d, p, g, N0, A)
        gt = th > theta
        lo = np.where(gt, mid, lo)
        hi = np.where(gt, hi, mid)
    return 0.5 * (lo + hi)


# =================================================================
# Algorithm 1: 带宽分配 (BRA)
# =================================================================
def bandwidth_allocation(env, tasks, g, off_idx, f, prev_cache, delta):
    cfg = env.cfg
    d = tasks['d']; s = tasks['s']; lmax = tasks['lmax']
    service = tasks['service']
    N = len(d)
    b_all = np.zeros(N)
    if len(off_idx) == 0:
        return b_all

    o = np.asarray(off_idx)
    p, N0 = cfg.p_n, cfg.N0
    d_o = d[o]; s_o = s[o]; lmax_o = lmax[o]; g_o = g[o]
    svc_o = service[o]

    l_exe = s_o / np.maximum(f[o], 1e-12)
    remaining = lmax_o - l_exe
    beta = delta[svc_o]
    beta_prev = prev_cache[svc_o]
    newly = (beta - beta_prev) >= (1 - 1e-9)
    l_ser = env.h[svc_o] / cfg.r_cloud

    # ---- 分类: 固定分配 vs 可优化 ----
    # 固定: (a) 传输时延本身超预算(remaining<=0); (b) 新缓存服务下载时延超预算
    #       这两类任务无法在截止时间内完成, 分配尽力而为带宽(最小传输时延)。
    # 可优化: 其余任务在 [bmin, bmax] 内由 theta 二分求解。
    even_share = cfg.W / len(o)
    b_fixed = np.full(len(o), even_share)
    fixed = np.zeros(len(o), dtype=bool)
    bmin = np.full(len(o), cfg.W)
    bmax = np.full(len(o), cfg.W)

    for i in range(len(o)):
        if remaining[i] <= 0:
            fixed[i] = True
            continue
        bmin_i = float(H_of_C0(remaining[i], d_o[i], p, g_o[i], N0))
        if newly[i]:
            bmax_i = float(H_of_C0(l_ser[i], d_o[i], p, g_o[i], N0))
        else:
            bmax_i = cfg.W
        if bmax_i < bmin_i:            # 下载时延主导, 时延不可满足
            b_fixed[i] = bmax_i
            fixed[i] = True
            continue
        bmin[i] = bmin_i
        bmax[i] = bmax_i

    opt = ~fixed
    if not opt.any():
        b_all[o] = b_fixed
        if b_all.sum() > cfg.W:
            b_all[o] *= cfg.W / max(b_all[o].sum(), 1e-12)
        return b_all

    # 可优化任务的 theta 上下界
    A_o = 1.0 + cfg.phi + lmax_o - l_exe
    th_min = theta_of_b(bmax, d_o, p, g_o, N0, A_o)
    th_max = theta_of_b(bmin, d_o, p, g_o, N0, A_o)
    th_min = np.where(opt, th_min, np.inf)
    th_max = np.where(opt, th_max, -np.inf)

    theta_low = float(np.max(th_min[opt]))
    theta_high = float(np.min(th_max[opt]))
    budget = cfg.W - float(b_fixed[fixed].sum())   # 剩余带宽给可优化任务
    if (not np.isfinite(theta_low) or not np.isfinite(theta_high)
            or theta_low >= theta_high or budget <= 0):
        b_all[o] = np.where(opt, bmin, b_fixed)
        b_all[o] *= cfg.W / max(b_all[o].sum(), 1e-12)
        return b_all

    # 带宽约束松弛 (sum bmax <= budget) 时, Lagrange 乘子 theta=0, 直接给到 bmax
    if float(bmax[opt].sum()) <= budget:
        b_all[o] = np.where(opt, bmax, b_fixed)
        return b_all

    b_cur = np.zeros(len(o))
    for _ in range(cfg.max_iter_bra):
        if (theta_high - theta_low) <= cfg.tol_theta * max(abs(theta_high), 1e-30):
            break
        theta = 0.5 * (theta_low + theta_high)
        solved = _bisect_theta(theta, d_o, p, g_o, N0, A_o, bmin, bmax)
        b_cur = np.where(theta <= th_min, bmax,
                         np.where(theta >= th_max, bmin, solved))
        b_cur = np.where(opt, b_cur, 0.0)
        if b_cur.sum() > budget:
            theta_low = theta
        else:
            theta_high = theta

    theta = 0.5 * (theta_low + theta_high)
    solved = _bisect_theta(theta, d_o, p, g_o, N0, A_o, bmin, bmax)
    b_cur = np.where(theta <= th_min, bmax,
                     np.where(theta >= th_max, bmin, solved))
    b_cur = np.where(opt, b_cur, 0.0)
    b_all[o] = np.where(opt, b_cur, b_fixed)
    if b_all.sum() > cfg.W:
        b_all[o] *= cfg.W / max(b_all[o].sum(), 1e-12)
    return b_all


# =================================================================
# Algorithm 2: 计算资源分配 (CRA)
# =================================================================
def computing_allocation(env, tasks, g, off_idx, b, prev_cache, delta):
    cfg = env.cfg
    d = tasks['d']; s = tasks['s']; lmax = tasks['lmax']
    service = tasks['service']
    N = len(d)
    f_all = np.zeros(N)
    if len(off_idx) == 0:
        return f_all

    o = np.asarray(off_idx)
    d_o = d[o]; s_o = s[o]; lmax_o = lmax[o]; g_o = g[o]
    svc_o = service[o]

    r = env.rate(b[o], g_o)
    l_trs = d_o / np.maximum(r, 1e-12)
    beta = delta[svc_o]
    beta_prev = prev_cache[svc_o]
    update = np.maximum(beta - beta_prev, 0.0)
    l_ser = update * env.h[svc_o] / cfg.r_cloud
    l_trs_ser = np.maximum(l_trs, l_ser)

    remaining = lmax_o - l_trs_ser
    valid = remaining > 0
    v = np.where(valid)[0]
    if len(v) == 0:
        return f_all

    # 仅对有效任务求解频率
    s_v = s_o[v]; lmax_v = lmax_o[v]; l_ts_v = l_trs_ser[v]
    fmin_v = s_v / remaining[v]
    A_v = 1.0 + cfg.phi + lmax_v - l_ts_v     # S' = A_v - s_v/f
    fmax = cfg.F_e

    # 解析解 f(omega): 由 dS/df = omega 得 s/(f^2(A-s/f)) = omega
    #   -> f = s (1 + sqrt(1 + 4A/(omega s))) / (2A)   (omega > 0)
    def f_of_omega(omega):
        disc = 1.0 + 4.0 * A_v / (omega * s_v)
        f = s_v * (1.0 + np.sqrt(np.maximum(disc, 0.0))) / (2.0 * A_v)
        return np.clip(f, fmin_v, fmax)

    # 计算资源约束 sum f <= F_e
    if fmin_v.sum() > cfg.F_e:
        # 资源不足以满足所有时延 -> 按比例分配最小频率
        f_v = fmin_v * (cfg.F_e / fmin_v.sum())
    else:
        # 二分 omega 使 sum f(omega) = F_e
        lo, hi = 1e-30, 1.0
        while f_of_omega(hi).sum() > cfg.F_e and hi < 1e30:
            hi *= 2.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if f_of_omega(mid).sum() > cfg.F_e:
                lo = mid
            else:
                hi = mid
        f_v = f_of_omega(0.5 * (lo + hi))

    # 能量约束投影: E = sum delta*gamma + k_e sum s f^2 <= Q
    E_storage = float((delta * env.gamma).sum())
    E_exec = float(cfg.k_e * (s_v * f_v ** 2).sum())
    if E_storage + E_exec > cfg.Q:
        budget = max(cfg.Q - E_storage, 0.0)
        if budget <= 0.0 or E_exec <= 0.0:
            f_v = np.zeros_like(f_v)
        else:
            f_v = f_v * np.sqrt(budget / E_exec)

    f_all[o[v]] = f_v
    return f_all


# =================================================================
# Algorithm 3: 多资源分配 (MRA) -- 交替迭代 BRA/CRA
# =================================================================
def multi_resource_allocation(env, tasks, g, delta, lam, prev_cache, iters=3):
    cfg = env.cfg
    off_idx = np.where(lam > 0.5)[0]
    N = len(lam)
    b = np.zeros(N); f = np.zeros(N)
    if len(off_idx) > 0:
        b[off_idx] = cfg.W / len(off_idx)
        f[off_idx] = cfg.F_e / len(off_idx)
    for _ in range(iters):
        b_new = bandwidth_allocation(env, tasks, g, off_idx, f, prev_cache, delta)
        f_new = computing_allocation(env, tasks, g, off_idx, b_new, prev_cache, delta)
        b, f = b_new, f_new
    return b, f


def even_allocation(env, lam):
    """均匀资源分配 (AC-ea / AC-sv / AC-wd 基线与候选打分用)。"""
    cfg = env.cfg
    off_idx = np.where(lam > 0.5)[0]
    N = len(lam)
    b = np.zeros(N); f = np.zeros(N)
    if len(off_idx) > 0:
        b[off_idx] = cfg.W / len(off_idx)
        f[off_idx] = cfg.F_e / len(off_idx)
    return b, f
