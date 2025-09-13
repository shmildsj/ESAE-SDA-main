1. 稀疏自编码器
网络结构：编码器-解码器对称式架构，编码器包含 4 层全连接层（input→ 1152 → 576 → 288 → 144），解码器与编码器完全对称。
激活函数：Sigmoid。
优化器：Adam。
学习率 (lr)：0.01（训练过程中使用 ReduceLROnPlateau 动态调整）。
训练轮数 (epochs)：150。
KL-divergence 稀疏约束：目标稀疏度 p = 0.05。权重系数 β = 1e-7（代码中常量 BETA）。
是否使用稀疏约束：开启（use_sparse=True）。
损失函数：重构损失 (MSE) + 稀疏性约束 (KL 散度)。

2. 基学习器
优化器：Adam。
学习率 (lr)：0.001–0.005（根据子模块不同而设置）。
辍学率 (dropout)：GraphNetEncoder 层：0.1。Global Attention 层：0.5。
归一化：BatchNorm 和 LayerNorm 结合。
注意力机制：多头注意力（n_head=1–4），以及全局注意力 (GlobalAttn)。
