# 复现: 缓存辅助移动边缘计算中的协作服务缓存、任务卸载与资源分配

本文档对应论文:

> C. Tang, Y. Ding, S. Xiao, Z. Huang, H. Wu, **"Collaborative Service
> Caching, Task Offloading, and Resource Allocation in Caching-Assisted
> Mobile Edge Computing"**, *IEEE Transactions on Services Computing*,
> vol. 18, no. 4, 2025.

复现目标是实现论文提出的 **ACSCTO** 算法(基于 Advantage Actor-Critic 的
服务缓存 + 任务卸载决策, 结合拉格朗日对偶法的带宽/CPU 频率资源分配),
并与论文中的基线算法对比。

---

## 一、问题模型 (论文 Section III / IV)

系统由远端云、一个边缘服务器(ES)和 N 个用户设备(UD)组成。优化周期被划分为
T 个时隙。每个时隙 UD n 生成至多一个任务 `t_n = (d_n, s_n, l_n^max)`:

- `d_n`: 输入数据量(bit)
- `s_n`: 所需计算量(CPU cycle)
- `l_n^max`: 最大可容忍时延(s)

任务与 K 个服务(数据库/库)相关联, 服务 k 有存储需求 `h_k` 与缓存静态能耗 `γ_k`。
ES 在每个时隙决定缓存哪些服务(`δ_k ∈ {0,1}`)、卸载哪些任务(`λ_n ∈ {0,1}`),
并为卸载任务分配带宽 `b_n` 与 CPU 频率 `f_n`。

关键时延/能耗(与论文公式一一对应):

| 公式 | 含义 |
|---|---|
| (1) `L_l = s_n / f_n^l` | 本地执行时延 |
| (3) `r_n = b_n log2(1 + g_n p_n/(N0 b_n))` | 上行速率 |
| (4) `l_ser = max(β_n - β_{n-1},0)·h_k/r_c` | 服务下载时延(新缓存时) |
| (6) `L_e = max(l_trs, l_ser) + l_exe` | 边缘执行总时延 |
| (7) `e_exe = k_e s_n (f_n)^2` | 边缘执行能耗 |
| (9) `S_n = log(1 + φ + l_n^max - L_n)` | 用户满意度(对数比例公平) |

优化目标(式 10)为最大化平均用户满意度, 受时延、能耗、存储、带宽、计算资源约束。

---

## 二、求解方法

论文将 MINLP 问题分解为两个子问题:

1. **多资源分配 (MRA, 论文 Section V)** —— 给定缓存与卸载决策, 用拉格朗日对偶法:
   - **带宽分配 (BRA, Algorithm 1)**: 基于 Lambert-W 函数闭式解 + 二分 Lagrange
     乘子 `θ` 求解最优带宽;
   - **计算资源分配 (CRA, Algorithm 2)**: 二分 Lagrange 乘子 `ω` 满足
     `Σf ≤ F_e`, 并用能量投影满足 `E ≤ Q`。
2. **服务缓存与任务卸载 (SC&TO, 论文 Section VI)** —— A2C 框架:
   - **服务缓存模块 (SCM)**: actor 输出 K 维 softmax 概率, 按概率降序贪心填充缓存
     直到容量; 多值 critic 对每个服务输出价值 `V(sca, a_k)`;
   - **任务卸载模块 (TOM)**: 输出 `[0,1]^N`, 经序保持量化 + 服务可用性掩码,
     选择满意度最高的卸载动作。

---

## 三、运行

```bash
pip install -r requirements.txt

# 冒烟测试(极小规模, 快速验证流程)
python main.py --algo acscto --quick
python main.py --algo all --quick

# 完整复现(论文默认规模: N=30, T_train=20000, T_test=10000)
python main.py --algo acscto
python main.py --algo all

# 指定参数(对应论文 Fig.9 / Fig.10 的规模扫描)
python main.py --algo all --N 20          # 用户数 20
python main.py --algo acscto --C 12       # 缓存容量 12 GB
```

运行结束后生成两张图:
- `comparison.png`: 各算法在满意度 / 响应时间 / 完成率 / 公平性上的对比;
- `convergence.png`: A2C 类算法的训练收敛曲线(移动平均)。

---

## 四、参数设置

### 4.1 论文明确给出的参数 (Section VII-A)

| 参数 | 取值 | 说明 |
|---|---|---|
| K | 10 | 服务数量 |
| `h_k` | U(1, 4) GB | 服务存储需求 |
| `γ_k` | U(2, 5) | 缓存静态能耗 |
| `d_n` | U(1, 6) MB | 任务输入数据量 |
| `s_n` | U(1, 5) GHz | 计算需求(按 Giga-cycle 理解) |
| `l_n^max` | U(0.7, 3) s | 任务截止时间 |
| C | 10 GB | ES 存储容量 |
| `F_e` | 40 GHz | ES 计算能力 |
| W | 30 GHz | 上行总带宽 |
| 区域 | 100×100, BS 居中 | UD 均匀分布 |
| 服务请求 | 均匀转移 | 请求转移模型 |
| N | 30(默认) | Fig.9 用 10~40, Fig.10 用 20 |

### 4.2 论文未给出、复现中采用的合理假设 (`config.py` 标注 `[假设]`)

| 参数 | 取值 | 说明 |
|---|---|---|
| `f_n^l` | 2.0 GHz | 本地计算频率(取 2GHz 使 N=30 时系统紧而不爆, 见 config.py 注释) |
| `p_n` | 0.1 W | UD 发射功率 |
| `N0` | 1e-19 W/Hz | 噪声功率谱密度 |
| 路径损耗 | `g = d^{-3}` + Rayleigh 衰落 | 信道模型 |
| `r_c` | 5 GB/s | ES-云下载速率 |
| `k_e` | 1e-27 | 开关电容系数 |
| Q | 2000 J | 单时隙能量预算(设宽松, 非主导) |
| φ | 0 | 满意度标定参数 |
| 学习率等 | 1e-3 等 | A2C 超参数 |

---

## 五、代码结构

```
config.py            # 全部参数(区分[论文]/[假设])
environment.py       # 系统模型: 任务生成、信道、时延、能耗、满意度
resource_alloc.py    # Algorithm 1(BRA)/2(CRA)/3(MRA) 拉格朗日对偶资源分配
networks.py          # A2C 网络(缓存 actor/critic、卸载网络)
acscto.py            # Algorithm 4: 缓存采样、卸载选择、奖励计算、训练
baselines.py         # AC-wd / PC / LC 基线
main.py              # 训练 + 评估 + 绘图入口
```

---

## 六、实现说明与简化

1. **CRA 的求解方式**: 论文用拉格朗日乘子 `μ, ω, ν` 的次梯度更新(式 37)。由于
   问题在 SI 单位下数值尺度差异巨大(频率 ~1e9 Hz, 满意度梯度 ~1e-10), 固定步长
   次梯度难以收敛, 复现改用**二分乘子 `ω`**(解析二次方程解)精确满足计算资源约束,
   并用能量投影处理能量约束。二者数学上等价, 但数值上更稳健。
2. **缓存决策生成**: 论文的"无重复采样直到容量"按 softmax 概率降序贪心填充实现。
3. **卸载候选打分**: TOM 在多个候选卸载动作中选择满意度最高者, 复现中用均匀资源
   分配快速打分(最终奖励仍用 MRA 精确分配), 以控制计算开销。
4. **缓存 actor 更新**: 采用多值 critic 得到的**逐服务优势** `A_k = V̂_k - V_k`
   加权 softmax 策略梯度(式 41 的逐服务形式)。
5. 响应时间指标对错过截止时间的任务按 `response_cap=10s` 封顶统计, 避免无穷大。

## 七、预期结果

与论文 Fig.7/Fig.8 一致: ACSCTO 应取得最高的平均用户满意度、最短的平均响应时间、
最高的任务完成率与最好的用户间公平性; PC(流行度缓存)表现次差, LC(全本地)最差。
AC-ea 由于多值 critic 优于单值 critic 的 AC-sv; AC-wd 因无法理解服务-任务对应关系
而表现最差(在 A2C 类方法中)。
