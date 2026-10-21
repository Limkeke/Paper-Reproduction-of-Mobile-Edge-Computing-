# -*- coding: utf-8 -*-
"""
主程序: 训练 + 评估 + 绘图
============================================================
运行方式:
    python main.py --algo acscto            # 训练 ACSCTO 并评估
    python main.py --algo all               # 训练全部基线并对比(较慢)
    python main.py --quick                  # 冒烟测试(极小规模跑通流程)
"""
import os
import sys
import argparse
import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from config import Config, make_config
from environment import MECEnvironment
from acscto import ACSCTOAgent, compute_caching_reward
from baselines import JointAgent, PopularityCaching, local_computing
from resource_alloc import multi_resource_allocation, even_allocation


# =================================================================
# 训练循环 (论文 Algorithm 4)
# =================================================================
def train_acscto(cfg, agent, env, use_mra=True, verbose=True):
    prev_cache = np.zeros(cfg.K)
    rewards = []
    pending = None
    for t in range(cfg.T_train):
        tasks, g = env.generate_slot()
        state2d = env.build_state(tasks, g, prev_cache)
        if pending is not None:
            agent.mem_ca.push(pending + (state2d, prev_cache))

        delta, lam, _ = agent.act(state2d, prev_cache, tasks, g, env)

        env.set_prev_cache(prev_cache)
        if use_mra:
            b, f = multi_resource_allocation(env, tasks, g, delta, lam, prev_cache)
        else:
            b, f = even_allocation(env, lam)
        completion, sat, feasible = env.satisfaction(tasks, g, delta, lam, b, f)
        local_sat = env.local_only_sat(tasks)
        rca = compute_caching_reward(env, tasks, delta, lam, sat, local_sat)
        rewards.append(float(rca.sum()))

        pending = (state2d, prev_cache, delta, rca)
        agent.mem_off.push((state2d, prev_cache, delta, lam))
        prev_cache = delta

        if len(agent.mem_ca) >= cfg.batch and len(agent.mem_off) >= cfg.batch:
            agent.update_caching(agent.mem_ca.sample(cfg.batch))
            agent.update_offloading(agent.mem_off.sample(cfg.batch))

        if verbose and (t + 1) % max(1, cfg.T_train // 10) == 0:
            avg = np.mean(rewards[-500:]) if rewards else 0.0
            print(f"  [train] step {t+1}/{cfg.T_train}  avg_reward(500)={avg:.4f}")
    return rewards


def train_joint(cfg, agent, env, verbose=True):
    prev_cache = np.zeros(cfg.K)
    rewards = []
    pending = None
    for t in range(cfg.T_train):
        tasks, g = env.generate_slot()
        state2d = env.build_state(tasks, g, prev_cache)
        if pending is not None:
            agent.mem.push(pending + (state2d, prev_cache))

        delta, lam, _ = agent.act(state2d, prev_cache, tasks, g, env)

        env.set_prev_cache(prev_cache)
        b, f = even_allocation(env, lam)
        completion, sat, feasible = env.satisfaction(tasks, g, delta, lam, b, f)
        # 论文 Fig.7 对四个 AC 算法统一用缓存奖励 sum_k R^c_{t,k} 作为训练奖励,
        # 因此 ac-wd 也需用 compute_caching_reward 而非原始满意度 sat.sum()。
        local_sat = env.local_only_sat(tasks)
        rca = compute_caching_reward(env, tasks, delta, lam, sat, local_sat)
        r = float(rca.sum())
        rewards.append(r)

        pending = (state2d, prev_cache, delta, lam, r)
        prev_cache = delta

        if len(agent.mem) >= cfg.batch:
            agent.update(agent.mem.sample(cfg.batch))

        if verbose and (t + 1) % max(1, cfg.T_train // 10) == 0:
            avg = np.mean(rewards[-500:]) if rewards else 0.0
            print(f"  [train] step {t+1}/{cfg.T_train}  avg_reward(500)={avg:.4f}")
    return rewards


# =================================================================
# 评估 (论文 Section VII-B 指标)
# =================================================================
def evaluate(cfg, env, decide_fn, use_mra=True):
    prev_cache = np.zeros(cfg.K)
    sats, resps, feass, energys, n_offs = [], [], [], [], []
    for _ in range(cfg.T_test):
        tasks, g = env.generate_slot()
        state2d = env.build_state(tasks, g, prev_cache)
        delta, lam = decide_fn(state2d, prev_cache, tasks, g)

        env.set_prev_cache(prev_cache)
        if use_mra:
            b, f = multi_resource_allocation(env, tasks, g, delta, lam, prev_cache)
        else:
            b, f = even_allocation(env, lam)
        completion, sat, feasible = env.satisfaction(tasks, g, delta, lam, b, f)
        E = env.energy(delta, lam, tasks['s'], f)

        sats.append(sat); resps.append(completion)
        feass.append(feasible); energys.append(E); n_offs.append(lam.sum())
        prev_cache = delta

    sat = np.concatenate(sats); resp = np.concatenate(resps)
    feas = np.concatenate(feass)
    resp = np.minimum(resp, cfg.response_cap)   # 截止时间外任务按封顶统计
    return {
        'satisfaction': float(sat.mean()),
        'response_time': float(resp.mean()),
        'completion_rate': float(feas.mean()),
        'fairness': float(resp.var()),
        'energy': float(np.mean(energys)),
        'avg_offload': float(np.mean(n_offs)),
    }


# =================================================================
# 各算法 decide 函数与训练入口
# =================================================================
def make_agent(cfg, state_dim2, algo):
    if algo == 'acscto':
        return ACSCTOAgent(cfg, state_dim2, single_value_critic=False)
    if algo == 'ac-ea':
        return ACSCTOAgent(cfg, state_dim2, single_value_critic=False)
    if algo == 'ac-sv':
        return ACSCTOAgent(cfg, state_dim2, single_value_critic=True)
    if algo == 'ac-wd':
        return JointAgent(cfg, state_dim2)
    raise ValueError(algo)


def use_mra_of(algo):
    return algo == 'acscto'


def train_algo(cfg, algo, env, state_dim2):
    """训练指定算法, 返回 (decide_fn, rewards)。"""
    print(f"\n=== 训练 {algo} (T_train={cfg.T_train}) ===")
    if algo in ('pc', 'lc'):
        return None, []   # 无需训练
    agent = make_agent(cfg, state_dim2, algo)
    if algo == 'ac-wd':
        rewards = train_joint(cfg, agent, env)
    else:
        rewards = train_acscto(cfg, agent, env, use_mra=use_mra_of(algo))
    return agent, rewards


def decide_fn_of(cfg, env, algo, agent):
    if algo == 'pc':
        pc = PopularityCaching(cfg, env)
        return lambda s2, pc_, tasks, g: pc.decide(tasks, g, pc_)
    if algo == 'lc':
        return lambda s2, pc_, tasks, g: local_computing(cfg)
    # RL 算法
    return lambda s2, pc_, tasks, g: agent.act(s2, pc_, tasks, g, env)[:2]


# =================================================================
# 绘图
# =================================================================
def plot_metrics(results, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = list(results.keys())
    metrics = ['satisfaction', 'response_time', 'completion_rate', 'fairness']
    titles = ['Average user satisfaction', 'Average response time (s)',
              'Task completion rate', 'Inter-user fairness (variance)']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, m, title in zip(axes.ravel(), metrics, titles):
        vals = [results[n][m] for n in names]
        ax.bar(names, vals, color='tab:blue')
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, 'comparison.png')
    fig.savefig(path, dpi=120)
    print(f"图已保存: {path}")


def plot_convergence(rewards_dict, outdir, window=500):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    for name, r in rewards_dict.items():
        r = np.asarray(r)
        ma = np.convolve(r, np.ones(window) / window, mode='valid')
        ax.plot(ma, label=name, linewidth=1.5)
    ax.set_xlabel('time slot')
    ax.set_ylabel('reward (moving avg)')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, 'convergence.png')
    fig.savefig(path, dpi=120)
    print(f"图已保存: {path}")


# =================================================================
# 主入口
# =================================================================
def main():
    parser = argparse.ArgumentParser(description='ACSCTO 复现')
    parser.add_argument('--algo', default='acscto',
                        choices=['acscto', 'ac-ea', 'ac-sv', 'ac-wd', 'pc', 'lc', 'all'])
    parser.add_argument('--quick', action='store_true', help='冒烟测试(极小规模)')
    parser.add_argument('--N', type=int, default=None, help='用户数量')
    parser.add_argument('--C', type=float, default=None, help='ES 缓存容量 GB')
    parser.add_argument('--T-train', type=int, default=None)
    parser.add_argument('--T-test', type=int, default=None)
    args = parser.parse_args()

    overrides = {'quick': args.quick}
    if args.N is not None:
        overrides['N'] = args.N
    if args.C is not None:
        overrides['C'] = args.C
    if args.T_train is not None:
        overrides['T_train'] = args.T_train
    if args.T_test is not None:
        overrides['T_test'] = args.T_test

    cfg = make_config(**overrides)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    state_dim2 = (cfg.N + cfg.K, cfg.K + 4)
    outdir = os.path.dirname(os.path.abspath(__file__))

    algos = (['acscto', 'ac-ea', 'ac-sv', 'ac-wd', 'pc', 'lc']
             if args.algo == 'all' else [args.algo])

    results = {}
    rewards_dict = {}
    agents = {}
    for algo in algos:
        # 每个算法用相同种子重建环境 -> 服务/位置/任务序列完全一致, 保证公平对比
        env = MECEnvironment(cfg)
        agent, rewards = train_algo(cfg, algo, env, state_dim2)
        agents[algo] = agent
        if rewards:
            rewards_dict[algo] = rewards
        decide = decide_fn_of(cfg, env, algo, agent)
        print(f"  评估 {algo} ...")
        metrics = evaluate(cfg, env, decide, use_mra=use_mra_of(algo))
        results[algo] = metrics
        print(f"    satisfaction={metrics['satisfaction']:.4f} "
              f"resp_time={metrics['response_time']:.4f} "
              f"completion_rate={metrics['completion_rate']:.4f} "
              f"fairness={metrics['fairness']:.4f} "
              f"energy={metrics['energy']:.2f} offload={metrics['avg_offload']:.2f}")

    if len(results) > 0:
        plot_metrics(results, outdir)
    if len(rewards_dict) > 1:
        plot_convergence(rewards_dict, outdir, window=cfg.moving_avg_window)

    # 打印汇总表
    print('\n=== 结果汇总 ===')
    header = f"{'algo':<10}" + "".join(f"{m:<16}" for m in
             ['satisfaction', 'resp_time', 'completion', 'fairness', 'energy', 'offload'])
    print(header)
    for algo in algos:
        m = results[algo]
        print(f"{algo:<10}{m['satisfaction']:<16.4f}{m['response_time']:<16.4f}"
              f"{m['completion_rate']:<16.4f}{m['fairness']:<16.4f}"
              f"{m['energy']:<16.2f}{m['avg_offload']:<16.2f}")


if __name__ == '__main__':
    main()
