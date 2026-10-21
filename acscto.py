# -*- coding: utf-8 -*-
"""
ACSCTO 算法主体 (论文 Section VI, Algorithm 4)
============================================================
- 服务缓存模块 (SCM): actor 生成缓存决策, 多值 critic 评估每个服务价值
- 任务卸载模块 (TOM): 卸载网络输出 [0,1]^N, 序保持量化 + 掩码 + 贪心选择
- 奖励计算: Rca = 聚合 (Roff - Rno-off) 至每个服务
- A2C 训练: critic MSE 更新(式39/40), actor 策略梯度(式41), 卸载交叉熵(式42)
"""
import numpy as np
import torch
import torch.nn.functional as F

from networks import (CachingActor, CachingCritic, SingleValueCritic,
                      OffloadingNet)
from resource_alloc import multi_resource_allocation, even_allocation


# =================================================================
# 经验回放
# =================================================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, item):
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.pos % self.capacity] = item
        self.pos += 1

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch):
        idx = np.random.randint(0, len(self.buffer), batch)
        return [self.buffer[i] for i in idx]


# =================================================================
# 缓存决策生成: 按 softmax 概率降序贪心填充直到容量 (论文 VI-A2)
# =================================================================
def sample_caching(probs, h, C):
    """probs: (K,) 缓存概率, h: (K,) 存储需求, C: 容量。返回 delta (K,)。"""
    K = len(probs)
    order = np.argsort(-probs)
    delta = np.zeros(K, dtype=np.float32)
    storage = 0.0
    for k in order:
        if storage + h[k] <= C + 1e-9:
            delta[k] = 1.0
            storage += h[k]
        else:
            break
    return delta


# =================================================================
# 卸载决策生成 (论文 VI-B1): 序保持量化 + 掩码 + 满意度贪心
# =================================================================
def select_offloading(o, tasks, g, delta, prev_cache, env, M=None):
    """o: (N,) 连续输出 [0,1]。返回最优二进制卸载向量 aoff (N,)。"""
    N = len(o)
    M = M if M is not None else N
    M = min(M, N)
    mask = (delta[tasks['service']] > 0.5).astype(np.float32)

    order = np.argsort(-o)               # 按卸载倾向降序
    best_lam = np.zeros(N, dtype=np.float32)
    best_score = -np.inf
    for m in range(1, M + 1):
        lam = np.zeros(N, dtype=np.float32)
        lam[order[:m]] = 1.0
        lam = lam * mask                 # 仅服务已缓存的任务可卸载
        if lam.sum() < 0.5:
            continue
        score = _score_lam(lam, tasks, g, delta, prev_cache, env)
        if score > best_score:
            best_score = score
            best_lam = lam
    return best_lam


def _score_lam(lam, tasks, g, delta, prev_cache, env):
    """用均匀资源分配快速打分 (候选选择用; 最终奖励用 MRA)。"""
    b, f = even_allocation(env, lam)
    env.set_prev_cache(prev_cache)
    _, sat, _ = env.satisfaction(tasks, g, delta, lam, b, f)
    return float(sat.sum())


# =================================================================
# 奖励计算 (论文 VI-B2): Rca_k = sum_{task n requests k} (Roff_n - Rnoff_n)
# =================================================================
def compute_caching_reward(env, tasks, delta, lam, sat, local_sat):
    """sat: 卸载决策下每任务满意度, local_sat: 全本地满意度。返回 (K,) Rca。"""
    K = env.K
    service = tasks['service']
    rca = np.zeros(K)
    diff = sat - local_sat
    for k in range(K):
        rca[k] = float(diff[service == k].sum())
    return rca


# =================================================================
# ACSCTO Agent
# =================================================================
class ACSCTOAgent:
    def __init__(self, cfg, state_dim2, single_value_critic=False):
        self.cfg = cfg
        self.actor = CachingActor(cfg, state_dim2)
        if single_value_critic:
            self.critic = SingleValueCritic(cfg, state_dim2)
        else:
            self.critic = CachingCritic(cfg, state_dim2)
        self.offnet = OffloadingNet(cfg, state_dim2)

        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)
        self.opt_off = torch.optim.Adam(self.offnet.parameters(), lr=cfg.lr_off)

        self.mem_ca = ReplayBuffer(cfg.Mca)
        self.mem_off = ReplayBuffer(cfg.Moff)
        self.single_value = single_value_critic

    # ----------------------------------------------------------
    def act(self, state2d, prev_cache, tasks, g, env):
        """生成 (delta, lam, probs)。"""
        s2 = torch.from_numpy(state2d[None]).float()
        pc = torch.from_numpy(prev_cache[None]).float()
        with torch.no_grad():
            probs = self.actor(s2, pc)[0].numpy()
        # 缓存决策 (softmax 概率降序贪心填充)
        delta = sample_caching(probs, env.h, self.cfg.C)
        cache_action = delta.astype(np.float32)
        with torch.no_grad():
            o = self.offnet(s2, pc,
                            torch.from_numpy(cache_action[None]).float())[0].numpy()
        # 卸载决策 (序保持量化 + 掩码 + 满意度贪心)
        lam = select_offloading(o, tasks, g, delta, prev_cache, env, self.cfg.M_off)
        return delta, lam, probs

    # ----------------------------------------------------------
    # 网络更新
    # ----------------------------------------------------------
    def update_caching(self, batch):
        """更新缓存 actor 与 critic (式 39/40/41)。"""
        cfg = self.cfg
        s2 = torch.from_numpy(np.stack([x[0] for x in batch])).float()
        pc = torch.from_numpy(np.stack([x[1] for x in batch])).float()
        delta = torch.from_numpy(np.stack([x[2] for x in batch])).float()
        rca = torch.from_numpy(np.stack([x[3] for x in batch])).float()
        ns2 = torch.from_numpy(np.stack([x[4] for x in batch])).float()
        npc = torch.from_numpy(np.stack([x[5] for x in batch])).float()

        # ---- critic ----
        V = self.critic(s2, pc)                       # (B, K) 或 (B,1)
        with torch.no_grad():
            V_next = self.critic(ns2, npc)
        V_hat = rca + cfg.gamma * V_next               # (B, K)
        if self.single_value:
            rca_s = rca.sum(dim=1, keepdim=True)       # 单值: 用总奖励
            V_hat = rca_s + cfg.gamma * V_next
            critic_loss = F.mse_loss(V, V_hat.detach())
        else:
            critic_loss = F.mse_loss(V, V_hat.detach())

        self.opt_critic.zero_grad()
        critic_loss.backward()
        self.opt_critic.step()

        # ---- actor (策略梯度, 式 41) ----
        probs = self.actor(s2, pc)                     # (B, K)
        advantage = (V_hat - V).detach()
        if self.single_value:
            adv = advantage.expand(-1, cfg.K)          # 广播到每个服务
        else:
            adv = advantage
        eps = 1e-12
        logp = delta * torch.log(probs + eps) + (1 - delta) * torch.log(1 - probs + eps)
        actor_loss = -(adv * logp).mean()

        self.opt_actor.zero_grad()
        actor_loss.backward()
        self.opt_actor.step()

        return critic_loss.item(), actor_loss.item()

    def update_offloading(self, batch):
        """更新卸载网络 (式 42, 交叉熵)。"""
        s2 = torch.from_numpy(np.stack([x[0] for x in batch])).float()
        pc = torch.from_numpy(np.stack([x[1] for x in batch])).float()
        ca = torch.from_numpy(np.stack([x[2] for x in batch])).float()
        aoff = torch.from_numpy(np.stack([x[3] for x in batch])).float()
        o = self.offnet(s2, pc, ca)
        loss = F.binary_cross_entropy(o, aoff)
        self.opt_off.zero_grad()
        loss.backward()
        self.opt_off.step()
        return loss.item()
