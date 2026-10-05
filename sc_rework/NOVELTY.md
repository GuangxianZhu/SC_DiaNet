# 创新点与相关工作（2026-10 检索，部分作者和年份来自搜索摘要，投稿前需逐条核对）

## 可能的贡献 vs 最接近的已有工作
1. **深层 FSM 级联 SC 网络的崩溃机制分析（相关性 + 噪声累积，随深度变化）**：部分已有。
   - Baker & Hayes, ISVLSI 2019：自相关会降低时序 SC 电路的精度，但只在电路层面。https://ieeexplore.ieee.org/document/8839424/
   - Ma & Lilja, ISQED 2018：FSM 的误差随自相关增大。
   - Wang et al., arXiv 2025（ASL）：分析了逐层截断噪声，但只针对浅层 MLP，不涉及 FSM。https://arxiv.org/abs/2508.09163
   - **没找到**：研究深层 SC 神经网络中级联 Btanh 自相关问题的工作。这是最有希望的创新点。
2. **用解析噪声 (1-a²)/L 做 SC 感知训练**：部分已有。
   - REX-SC（UCLA, Li/Gupta 组）：用拟合的均值和方差做误差注入，但针对 OR 累加。https://nanocad.ee.ucla.edu/wp-content/papercite-data/pdf/j74.pdf
   - 结论：只能定位为“面向 FSM 的解析版误差注入”，不能当作主创新点。
3. **洗牌缓存去相关**：电路本身已有，见 Lee, Alaghi, Ceze, DATE 2018。https://arxiv.org/abs/1803.04862
   - 能说的新意只有“用在级联 Btanh 之间”，并且要给出代价（预热延迟、触发器数量）。
4. **DiaNet 拓扑**：NAIST 张任远、中岛康彦组的工作（Wu et al., Neurocomputing 2021, https://www.sciencedirect.com/science/article/pii/S0925231221012492 ；NEWCAS 2020）。
   - 必须引用为来源。
   - 还没找到 DiaNet 的 SC 实现。分块（类 CNN）的做法是否已经被该组发表过，需要确认。

## 可投会议
- ISCAS 2027：10 月 13 日截稿（已核实），时间来不及。
- DAC 2027：11 月 17 日截稿（已核实），门槛太高。
- AICAS 2027：一般 11 到 12 月截稿（未核实）。
- GLSVLSI 2027：一般 2 月截稿（未核实）。
- ISVLSI 2027：一般 3 月截稿（未核实）。
- IEEE TCAS-II 短文、Electronics Letters：随时可投。
