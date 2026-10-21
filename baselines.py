# -*- coding: utf-8 -*-
"""
基线算法 (论文 Section VII-C)
============================================================
- AC-wd: A2C 无分解基线(单一网络直接输出缓存+卸载, 单值 critic)
- PC   : 流行度缓存 (缓存请求频率最高的服务, 优先卸载紧急任务)
- LC   : 全本地计算
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks import StateEncoder, mlp
from acscto import ReplayBuffer, sample_caching, select_offloading
from resource_alloc import even_allocation


# =================================================================
# AC-wd: 联合 actor + 单值 critic
# =================================================================
class JointActor(nn.Module):
    def __init__(self, cfg, state_dim2):
        super().__init__()
        self.encoder = StateEncoder(cfg, state_dim2)
        self.cache_head = mlp([cfg.hidden[-1], cfg.K])
        self.off_head = mlp([cfg.hidden[-1], 64, cfg.N], final_act=nn.Sigmoid)

    def forward(self, state2d, prev_cache):
        h = self.encoder(state2d, prev_cache)
        return F.softmax(self.cache_head(h), dim=-1), self.off_head(h)


class JointValue(nn.Module):
    def __init__(self, cfg, state_dim2):
        super().__init__()
        self.encoder = StateEncoder(cfg, state_dim2)
        self.value = mlp([cfg.hidden[-1], 64, 1])

    def forward(self, state2d, prev_cache):
        h = self.encoder(state2d, prev_cache)
        return self.value(h)


class JointAgent:
    """AC-wd: 单一网络联合输出缓存与卸载决策, 单值 critic。"""
    def __init__(self, cfg, state_dim2):
        self.cfg = cfg
        self.actor = JointActor(cfg, state_dim2)
        self.critic = JointValue(cfg, state_dim2)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)
        self.mem = ReplayBuffer(cfg.Mca)

    def act(self, state2d, prev_cache, tasks, g, env):
        s2 = torch.from_numpy(state2d[None]).float()
        pc = torch.from_numpy(prev_cache[None]).float()
        with torch.no_grad():
            probs, o = self.actor(s2, pc)
            probs = probs[0].numpy(); o = o[0].numpy()
        delta = sample_caching(probs, env.h, self.cfg.C)
        lam = select_offloading(o, tasks, g, delta, prev_cache, env, self.cfg.M_off)
        return delta, lam, probs

    def update(self, batch):
        cfg = self.cfg
        s2 = torch.from_numpy(np.stack([x[0] for x in batch])).float()
        pc = torch.from_numpy(np.stack([x[1] for x in batch])).float()
        delta = torch.from_numpy(np.stack([x[2] for x in batch])).float()
        aoff = torch.from_numpy(np.stack([x[3] for x in batch])).float()
        r = torch.from_numpy(np.stack([x[4] for x in batch])).float().unsqueeze(1)
        ns2 = torch.from_numpy(np.stack([x[5] for x in batch])).float()
        npc = torch.from_numpy(np.stack([x[6] for x in batch])).float()

        V = self.critic(s2, pc)
        with torch.no_grad():
            V_next = self.critic(ns2, npc)
        V_hat = r + cfg.gamma * V_next
        critic_loss = F.mse_loss(V, V_hat.detach())

        self.opt_critic.zero_grad(); critic_loss.backward(); self.opt_critic.step()

        probs, o = self.actor(s2, pc)
        adv = (V_hat - V).detach().expand(-1, cfg.K)
        eps = 1e-12
        logp_cache = delta * torch.log(probs + eps) + (1 - delta) * torch.log(1 - probs + eps)
        loss_cache = -(adv * logp_cache).mean()
        loss_off = F.binary_cross_entropy(o, aoff)
        actor_loss = loss_cache + loss_off

        self.opt_actor.zero_grad(); actor_loss.backward(); self.opt_actor.step()
        return critic_loss.item(), actor_loss.item()


# =================================================================
# PC: 流行度缓存
# =================================================================
class PopularityCaching:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env
        self.counts = np.zeros(cfg.K)

    def decide(self, tasks, g, prev_cache):
        cfg = self.cfg
        # 更新请求计数
        for k in tasks['service']:
            self.counts[k] += 1
        # 按流行度降序缓存直到容量
        order = np.argsort(-self.counts)
        delta = np.zeros(cfg.K, dtype=np.float32)
        storage = 0.0
        for k in order:
            if storage + self.env.h[k] <= cfg.C + 1e-9:
                delta[k] = 1.0
                storage += self.env.h[k]
            else:
                break
        # 优先卸载服务已缓存且更紧急的任务(剩余时延更小)
        mask = (delta[tasks['service']] > 0.5).astype(np.float32)
        local_time = tasks['s'] / self.env.f_local
        urgency = tasks['lmax'] - local_time        # 越小越紧急
        order_urgent = np.argsort(urgency)
        lam = np.zeros(cfg.N, dtype=np.float32)
        # 只卸载"本地无法满足时延"或服务已缓存且紧急的任务
        for n in order_urgent:
            if mask[n] > 0.5 and local_time[n] > tasks['lmax'][n]:
                lam[n] = 1.0
        # 若还无任务卸载, 卸载最紧急的已缓存任务
        if lam.sum() < 0.5:
            for n in order_urgent:
                if mask[n] > 0.5:
                    lam[n] = 1.0
                    break
        return delta, lam


# =================================================================
# LC: 全本地
# =================================================================
def local_computing(cfg):
    delta = np.zeros(cfg.K, dtype=np.float32)
    lam = np.zeros(cfg.N, dtype=np.float32)
    return delta, lam
