# -*- coding: utf-8 -*-
"""
系统模型 (论文 Section III / Section IV)
============================================================
实现缓存辅助 MEC 场景: 任务生成、信道建模、服务请求转移、
本地/边缘执行时延、能耗与用户满意度函数。

对应论文公式:
    (1)  L_l       = s_n / f^l_n
    (2)  l_trs     = d_n / r_n
    (3)  r_n       = b_n log2(1 + g_n p_n / (N0 b_n))
    (4)  l_ser     = max(beta_n - beta_{n-1}, 0) * h_k / r_c
    (5)  l_exe     = s_n / f^e_n
    (6)  L_e       = max(l_trs, l_ser) + l_exe
    (7)  e_exe     = k_e s_n (f^e_n)^2
    (8)  E_e       = sum_k delta_k gamma_k + sum_n lambda_n e_exe
    (9)  S_n       = log(1 + phi + l_max - (lambda L_e + (1-lambda) L_l))
"""
import numpy as np


class MECEnvironment:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        self.K = cfg.K
        self.N = cfg.N
        self.phi = cfg.phi
        self.eps = cfg.eps

        # 服务: 存储需求 h (GB), 静态缓存能耗 gamma
        self.h = self.rng.uniform(cfg.h_min, cfg.h_max, self.K)
        self.gamma = self.rng.uniform(cfg.gamma_min, cfg.gamma_max, self.K)

        # UD 位置: 100x100 区域均匀分布, BS 位于中心
        self.bs = np.array([cfg.area_size / 2.0, cfg.area_size / 2.0])
        self.positions = self.rng.uniform(0, cfg.area_size, (self.N, 2))
        dist = np.linalg.norm(self.positions - self.bs, axis=1)
        self.dist = np.clip(dist, 1.0, None)   # 防止 d=0 导致信道增益发散
        self.path_loss = self.dist ** (-cfg.alpha) * cfg.ref_gain

        # 本地计算频率
        self.f_local = cfg.f_local

        # 服务请求状态(用于转移模型): 每个 UD 上一时隙请求的服务
        self.prev_service = self.rng.integers(0, self.K, self.N)

    # ----------------------------------------------------------
    def reset(self):
        self.rng = np.random.default_rng(self.cfg.seed)

    # ----------------------------------------------------------
    def generate_slot(self):
        """生成一个时隙的任务集合与信道增益。

        返回:
            tasks : dict, 含 'd'(bit), 's'(cycle), 'lmax'(s), 'service'(int)
            g     : (N,) 信道增益
        """
        cfg = self.cfg
        N, K = self.N, self.K

        # 输入数据大小 (bit), CPU 需求 (cycle), 截止时间 (s)
        d = self.rng.uniform(cfg.d_min_b, cfg.d_max_b, N)
        s = self.rng.uniform(cfg.s_min_c, cfg.s_max_c, N)
        lmax = self.rng.uniform(cfg.lmax_min, cfg.lmax_max, N)

        # 服务请求: 均匀转移模型 (给定上一请求 i -> 任意 k 等概率)
        if cfg.request_stickiness <= 0.0:
            service = self.rng.integers(0, K, N)
        else:
            service = self.prev_service.copy()
            switch = self.rng.random(N) > cfg.request_stickiness
            service[switch] = self.rng.integers(0, K, int(switch.sum()))
        self.prev_service = service.copy()

        # 信道增益: 大尺度路径损耗 + 可选 Rayleigh 快衰落
        g = self.path_loss.copy()
        if cfg.fading:
            g = g * self.rng.exponential(1.0, N)

        tasks = dict(d=d, s=s, lmax=lmax, service=service)
        return tasks, g

    # ----------------------------------------------------------
    def rate(self, b, g):
        """Shannon 速率 (bit/s), 公式 (3)。b>0 保护。"""
        b = np.maximum(b, 1e-12)
        snr = g * self.cfg.p_n / (self.cfg.N0 * b)
        return b * np.log2(1.0 + snr)

    # ----------------------------------------------------------
    def satisfaction(self, tasks, g, delta, lam, b, f):
        """给定缓存 delta、卸载 lam、带宽 b、频率 f, 计算完成时延与满意度。

        返回:
            completion : (N,) 每任务完成时延 (s)
            sat        : (N,) 每任务满意度 (公式 9)
            feasible   : (N,) bool, 是否满足截止时间约束
        """
        cfg = self.cfg
        d = tasks['d']; s = tasks['s']; lmax = tasks['lmax']
        service = tasks['service']
        N = len(d)

        # 本地执行
        L_local = s / self.f_local
        completion = L_local.copy()
        sat = self._safe_log(1.0 + cfg.phi + lmax - L_local)

        off = lam.astype(bool)
        if off.any():
            idx = np.where(off)[0]
            beta = delta[service[idx]]                 # 该任务服务是否已缓存
            # 传输时延
            r = self.rate(b[idx], g[idx])
            l_trs = d[idx] / np.maximum(r, 1e-12)
            # 服务下载时延: 仅当服务在本时隙新缓存时 (beta - beta_prev = 1)
            beta_prev = self._prev_cache[service[idx]]
            update = np.maximum(beta - beta_prev, 0.0)
            l_ser = update * self.h[service[idx]] / cfg.r_cloud
            # 执行时延
            l_exe = s[idx] / np.maximum(f[idx], 1e-12)
            L_e = np.maximum(l_trs, l_ser) + l_exe

            completion[idx] = L_e
            sat[idx] = self._safe_log(1.0 + cfg.phi + lmax[idx] - L_e)

        feasible = completion <= lmax + 1e-9
        return completion, sat, feasible

    # ----------------------------------------------------------
    def energy(self, delta, lam, s, f):
        """ES 单时隙总能耗 (公式 8)。"""
        e_str = float((delta * self.gamma).sum())
        e_exe = float((lam * self.cfg.k_e * s * f ** 2).sum())
        return e_str + e_exe

    # ----------------------------------------------------------
    def local_only_sat(self, tasks):
        """全部本地执行时的满意度 (R^{no-off}), 用于计算缓存奖励。"""
        lmax = tasks['lmax']; s = tasks['s']
        L_local = s / self.f_local
        return self._safe_log(1.0 + self.cfg.phi + lmax - L_local)

    # ----------------------------------------------------------
    def _safe_log(self, x):
        return np.log(np.maximum(x, self.eps))

    # ----------------------------------------------------------
    def set_prev_cache(self, prev_cache):
        """设置上一时隙缓存状态 (beta 计算用)。"""
        self._prev_cache = prev_cache.astype(np.float64)

    # ----------------------------------------------------------
    # 用于构建 A2C 状态输入的辅助函数
    # ----------------------------------------------------------
    def build_state(self, tasks, g, prev_cache):
        """构建缓存模块输入 sca_t: 2D 张量 (N+K, K+4) 的展平表示。

        每行 = [任务特征(one-hot 服务 + d + s + lmax + g) | 服务特征补零] 或
               [服务特征(prev_cache + gamma + h) 补零]。
        返回 2D numpy 数组 shape (N+K, K+4)。
        """
        cfg = self.cfg
        N, K = self.N, self.K
        feat = K + 4
        mat = np.zeros((N + K, feat), dtype=np.float32)

        # 归一化常量(避免特征尺度差异过大)
        d_norm = cfg.d_max_b
        s_norm = cfg.s_max_c
        lmax_norm = cfg.lmax_max
        g_norm = 1.0  # 信道增益量级

        for n in range(N):
            onehot = np.zeros(K)
            onehot[tasks['service'][n]] = 1.0
            mat[n, :K] = onehot
            mat[n, K + 0] = tasks['d'][n] / d_norm
            mat[n, K + 1] = tasks['s'][n] / s_norm
            mat[n, K + 2] = tasks['lmax'][n] / lmax_norm
            mat[n, K + 3] = g[n] * g_norm

        for k in range(K):
            mat[N + k, 0] = prev_cache[k]
            mat[N + k, 1] = self.gamma[k] / cfg.gamma_max
            mat[N + k, 2] = self.h[k] / cfg.h_max

        return mat

    def flat_state(self, tasks, g, prev_cache):
        """展平为 1D 状态向量 (含上一时隙缓存状态), 供全连接网络使用。"""
        mat = self.build_state(tasks, g, prev_cache)
        prev = prev_cache.astype(np.float32)
        return np.concatenate([mat.reshape(-1), prev])
