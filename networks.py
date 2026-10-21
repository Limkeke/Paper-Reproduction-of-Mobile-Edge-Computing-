# -*- coding: utf-8 -*-
"""
A2C 神经网络模块 (论文 Section VI)
============================================================
- CachingActor    : 服务缓存 actor, 输出 K 维 softmax 概率 Proca
- CachingCritic   : 多值 critic, 对每个服务输出状态-动作价值 V(sca, a_k)
- SingleValueCritic: AC-sv 基线的单值 critic
- OffloadingNet   : 任务卸载网络, 输出 [0,1]^N 连续值(后经序保持量化)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(dims, act=nn.ReLU, final_act=None):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act())
        elif final_act is not None:
            layers.append(final_act())
    return nn.Sequential(*layers)


class StateEncoder(nn.Module):
    """将 2D 状态 (N+K, K+4) 经层归一化后展平, 再拼接 prev_cache 编码。

    论文 Fig.3/4: 先 group/layer normalization 归一化异构特征, 再展平。
    """
    def __init__(self, cfg, state_dim2):
        super().__init__()
        self.K = cfg.K
        self.state_dim2 = state_dim2          # (N+K, K+4)
        flat = state_dim2[0] * state_dim2[1] + cfg.K
        self.norm = nn.LayerNorm(state_dim2[1])  # 逐行归一化(近似 group norm)
        self.enc = mlp([flat] + list(cfg.hidden))

    def forward(self, state2d, prev_cache):
        # state2d: (B, N+K, K+4), prev_cache: (B, K)
        x = self.norm(state2d)
        x = x.reshape(x.shape[0], -1)
        x = torch.cat([x, prev_cache], dim=1)
        return self.enc(x)


class CachingActor(nn.Module):
    def __init__(self, cfg, state_dim2):
        super().__init__()
        self.encoder = StateEncoder(cfg, state_dim2)
        self.head = mlp([cfg.hidden[-1], cfg.K], act=nn.ReLU)

    def forward(self, state2d, prev_cache):
        h = self.encoder(state2d, prev_cache)
        logits = self.head(h)
        return F.softmax(logits, dim=-1)      # Proca, sum=1


class CachingCritic(nn.Module):
    """多值 critic: 输出 V(sca_t, a_k) 向量 (size K), 论文 Fig.4。"""
    def __init__(self, cfg, state_dim2):
        super().__init__()
        self.K = cfg.K
        self.encoder = StateEncoder(cfg, state_dim2)
        self.value = mlp([cfg.hidden[-1] + cfg.K, 64, 1], act=nn.ReLU)

    def forward(self, state2d, prev_cache):
        h = self.encoder(state2d, prev_cache)   # (B, H)
        B = h.shape[0]
        vals = []
        eye = torch.eye(self.K, device=h.device)
        for k in range(self.K):
            ak = eye[k].unsqueeze(0).expand(B, -1)
            vals.append(self.value(torch.cat([h, ak], dim=1)))
        return torch.cat(vals, dim=1)            # (B, K)


class SingleValueCritic(nn.Module):
    """AC-sv 基线: 单一状态价值 V(sca_t)。"""
    def __init__(self, cfg, state_dim2):
        super().__init__()
        self.encoder = StateEncoder(cfg, state_dim2)
        self.value = mlp([cfg.hidden[-1], 64, 1], act=nn.ReLU)

    def forward(self, state2d, prev_cache):
        h = self.encoder(state2d, prev_cache)
        return self.value(h)                     # (B, 1)


class OffloadingNet(nn.Module):
    """任务卸载网络 (TOM), 输入 soff = sca + 缓存动作, 输出 [0,1]^N。"""
    def __init__(self, cfg, state_dim2):
        super().__init__()
        self.N = cfg.N
        self.K = cfg.K
        self.encoder = StateEncoder(cfg, state_dim2)
        # 缓存动作 aca (one-hot K) 与状态编码拼接
        self.head = mlp([cfg.hidden[-1] + cfg.K, 64, cfg.N], act=nn.ReLU, final_act=nn.Sigmoid)

    def forward(self, state2d, prev_cache, cache_action):
        h = self.encoder(state2d, prev_cache)
        h = torch.cat([h, cache_action], dim=1)
        return self.head(h)                      # (B, N) in [0,1]


def numpy_state_batch(state2d, prev_cache, device='cpu'):
    """numpy 状态 -> torch 张量 batch。"""
    s2 = torch.from_numpy(np.asarray(state2d, dtype=np.float32)).unsqueeze(0).to(device)
    pc = torch.from_numpy(np.asarray(prev_cache, dtype=np.float32)).unsqueeze(0).to(device)
    return s2, pc
