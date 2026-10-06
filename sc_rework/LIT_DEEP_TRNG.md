# 深层 SC 与真随机源（TRNG）文献调研（2026-10-06）

标 [未核实] 的条目来自搜索摘要或记忆，没有读到原文；IEEE 和 ACM 的页面无法打开。

## 结论
1. **做得深、精度又好的 SC 网络确实存在，但都不是层与层之间一直用比特流级联。** ResNet-18/34 在 ImageNet 上比定点低 2% 到 5%，VGG-16 在 CIFAR-10 上低约 2 个百分点。它们的共同做法是：每层或部分累加后转回二进制或定点，再重新生成比特流；ReLU 和 BN 在二进制域里做，或者干脆用确定性的温度计码加排序网络。都没有用 FSM 级联。
2. **真随机源没有被证明对深度有帮助。** 唯一的对照实验（GEO, DATE 2021）显示，配合 SC 感知训练时，共享 LFSR 反而比不共享的 TRNG 高最多 6.1 个百分点。用 MTJ TRNG 的 NIST 工作也仍然要加去相关电路。
3. **空白：** 还没有工作端到端保持比特流做深（8 层以上，CIFAR 或 ImageNet），无论是否用器件 TRNG；也没有人量化 FSM 激活的自相关在级联中如何累积，或者 TRNG 加再生能否修复它。

**和我们实验的关系：** 我们的仿真用的本来就是理想的独立随机源（相当于完美 TRNG），16 层仍然崩到 9.9%。所以崩溃不是伪随机造成的，而是 Btanh FSM 自身产生的时间自相关；换成真随机也救不了。

## A. 深层且精度好的 SC（全部是伪随机或确定性编码，层间转回二进制）
- **ACOUSTIC**（Romaszkan, Li, Gupta, DATE 2020）：AlexNet、VGG-16、ResNet-18 跑 ImageNet；CIFAR-10 自定义 CNN 78.0%，8 位定点 79.9%。LFSR 共享；每层转二进制并重新生成比特流，作者说这“彻底消除了相关性问题”。https://nanocad.ee.ucla.edu/wp-content/papercite-data/pdf/c109.pdf
- **GEO**（同组，DATE 2021）：VGG-16 在 CIFAR-10 上 88.7%，定点 90.9%。共享 LFSR 加 SC 感知训练比 TRNG 高最多 6.1 个百分点。https://nanocad.ee.ucla.edu/wp-content/papercite-data/pdf/c112.pdf
- **REX-SC**（Li, Gupta 等）：ResNet-18/34 在 ImageNet 上比定点低 2% 到 5%。层间用计数器转回定点。https://nanocad.ee.ucla.edu/wp-content/papercite-data/pdf/j74.pdf
- **SASCHA**（CASES 2022）：VGG-16、ResNet-18/34，精度接近 8 位定点。LFSR、Sobol 或 Halton。https://nanocad.ee.ucla.edu/wp-content/papercite-data/pdf/c120.pdf
- **Hu et al.**（DATE 2023，北大）：ResNet-18/34 跑 CIFAR-10/100。确定性温度计码加排序网络，残差保持高精度。https://past.date-conference.com/proceedings-archive/2023/DATA/600.pdf
- **End-to-End SC**（arXiv 2024）和 ASCEND（ViT）：所谓“端到端”指的是确定性温度计比特流，激活靠排序网络选位实现，不用 FSM。https://arxiv.org/abs/2401.15332 ，https://arxiv.org/abs/2402.12820
- **Lee, Alaghi et al.**（DATE 2017）：只在第 1 层用 SC，明确说是为了避免误差逐层叠加。https://arxiv.org/pdf/1706.02344
- **层间保持比特流的工作都很浅**：SC-DCNN（ASPLOS 2017），LeNet-5，96.6%，浮点约 98.5%，用 APC 加 Btanh。https://arxiv.org/pdf/1611.05939
- **Li et al. 2017**：定性指出激活误差会被后续神经元放大。https://arxiv.org/pdf/1703.04135

## B. 用真随机源的 SC 网络（全部很浅）
- **Daniels et al.**（NIST，超顺磁 MTJ，Phys. Rev. Applied 2020 [未核实]）：6 层类 LeNet，MNIST 97%，浮点 98.9%。全程比特流，但每层后仍要加随机延迟的“隔离器”来去相关。https://arxiv.org/pdf/1911.11204
- **Shao, Khalili Amiri et al.**（IEEE Magn. Lett. 2020）：MTJ TRNG，1 个隐藏层，MNIST 95%。https://par.nsf.gov/servlets/purl/10250874
- **Sabyasachi, Atulasimha et al.**（Nanotechnology 2025）：随机 MTJ，最多 3 个隐藏层，MNIST 96.8%。https://arxiv.org/abs/2504.06414
- **Weller et al.**（DATE 2021）：印刷电子，亚稳态 TRNG，9-3-2 的小网络。https://passat.crhc.illinois.edu/date21.pdf
- **Parmar, Querlioz, Suri**（Frontiers 2022）：OxRAM 噪声只用于输入层采样，其余是二值 CNN；CIFAR-10 85.6%。https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2021.781786/full
- **不属于比特流 SC 的**：StoX-Net（SOT-MTJ 当 ADC 或激活，ResNet-20 在 CIFAR-10 上 86.6%）https://arxiv.org/html/2407.12378v2 ；p-bit（Camsari 等，用于采样而不是前馈推理）。

## C. 去相关电路
CoMix-D（DATE 2026）https://past.date-conference.com/proceedings-archive/2026/DATA/465.pdf ；CORLD（ICCAD 2021）；洗牌缓存（Lee, Alaghi, Ceze, DATE 2018）。
