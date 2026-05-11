# 图表说明、正确性分析与 LaTeX 块

本文档对 `results/figures/` 下的全部 26 张图按主题分组，给出：

- **描述**：图里画的是什么
- **正确性分析**：从理论预期 / 已有文献 / 内部交叉验证三个角度判断这张图是否合理
- **LaTeX**：可直接复制进报告的 `figure` 块

LaTeX 路径假设报告 `.tex` 与项目根目录在同级；若不一致，请把 `results/figures/...` 改成相对路径或把图复制到 `report/figures/`。建议在导言区放：

```latex
\usepackage{graphicx}
\usepackage{subcaption}
\graphicspath{{results/figures/}}
```

若用了 `\graphicspath`，下文 LaTeX 块里的路径前缀可以省略。

---

## 1. 训练曲线 `results/figures/training_curves/`

每张图都是 1×3 三联：Loss / Top-1 accuracy / Macro F1。**实线 = validation，虚线 = train**（在每张图顶部已标明）。Macro F1 面板只画 val，因为训练循环没记录 `train_macro_f1`。

### 1.1 `baselines_100.png` — 100% labels 下四种策略对比

**描述**：Linear probe、Full fine-tune、Gradual unfreeze、Layer4-LoRA r=4 在 100% 数据下 30 个 epoch 的训练动态。

**正确性分析**：
- 4 条 val Loss 单调下降到 ~0.4–0.6 区间，未见震荡或反弹 → 学习率与正则配置合理
- val top-1 在 epoch 5 之前迅速爬升到 0.8 以上，之后趋于平稳；与 `experiment_summary.md` 报告的最终 test top-1（0.846 / 0.868 / 0.874 / 0.850）一致
- Gradual unfreeze 的 val Loss 在 epoch ≈ 15 处有一个小尖峰，对应 layer2 解冻、optimizer rebuild 的瞬间动量重置，属于已知现象（[docs/experiment_baselines.md:196](docs/experiment_baselines.md:196) 有记录）
- train 与 val 之间的 gap 反映过拟合程度：Full fine-tune 的 gap 最大（11M 参数），Linear probe 几乎重合，符合"参数越多越容易过拟合"的预期

**LaTeX**：

```latex
\begin{figure}[ht]
  \centering
  \includegraphics[width=\textwidth]{results/figures/training_curves/baselines_100.png}
  \caption{Training curves of four transfer-learning strategies at 100\% label budget. Solid lines are validation, dashed lines are training. All four strategies converge by epoch 10; gradual unfreezing shows the expected loss spike at the layer-2 unfreeze step.}
  \label{fig:training-curves-100}
\end{figure}
```

### 1.2 `baselines_10.png` — 10% labels

**描述**：同上四种策略在 10% 标签预算下的训练曲线（每 epoch 仍是同一 batch 数，但每 batch 看到的样本是 10% 子集）。

**正确性分析**：
- Full fine-tune 的 train top-1 在 epoch 10 后就 > 0.95 而 val 停在 ~0.72 → 严重过拟合，符合"小数据 + 大参数"的预期
- Layer4-LoRA r=4 的 train/val gap 最小，对应它在 10% 上击败 Full fine-tune 的事实（0.753 vs 0.713）
- Linear probe 训练曲线最平滑，因为只有 fc 在动 → 与"凸优化"性质相符

### 1.3 `baselines_1.png` — 1% labels

**描述**：极小数据（每类 ≈1 张图）下的训练曲线。

**正确性分析**：
- 30 epoch 内多数策略尚未收敛（best epoch 集中在 28–29）→ 与 [docs/experiment_baselines.md:303](docs/experiment_baselines.md:303) "extending training to 50 epochs would yield higher final numbers" 一致
- Full fine-tune 出现 train top-1 → 1.0 而 val 仅 0.26 的极端过拟合 → 教科书级别的"too many parameters for the data"案例
- Layer4-LoRA r=4 在所有 1% 实验中 val top-1 最高（0.411），且其 train/val gap 比 Full fine-tune 显著小

### 1.4 `lora_layer4_rank_100.png` — Layer4-LoRA rank 消融（100%）

**描述**：r=4 / 8 / 16 在 100% 标签下的训练动态。

**正确性分析**：
- 三条 val 曲线最终都收敛到 0.84–0.85 的窄区间，差距不超过 1 个百分点 → 在 100% 数据上"rank 不是瓶颈"
- r=4 的 train/val gap 反而最小（implicit regularization），符合 LoRA 论文 §6.1 的发现
- 三条曲线的"上升斜率"差别不大 → rank 主要影响表示容量上限，而非收敛速度

### 1.5 `lora_fc_rank_100.png` — FC-LoRA rank 消融（100%）

**描述**：把 LoRA 只放在 fc 层（37 类输出）的 rank 消融。

**正确性分析**：
- 三条 val 曲线最高 0.82（r=16）、最低 0.61（r=4）→ 差距 21 个点，远大于 Layer4-LoRA 内部
- 原因：ΔW = BA 的秩上限是 min(r, 37)；当 r=4 远小于 37 时，输出空间被严重压缩，无法 represent 所有类别 → 这是 LoRA 在"分类头"上的固有上限（详见 [results/README.md](results/README.md) 关于 FC-LoRA 的讨论）
- 与 Linear probe（fc 全 18,981 参数，0.846）相比，FC-LoRA r=16 仅用 8,784 参数即逼近 0.82，差距 2.6 个点，说明 LoRA 的低秩近似在分类头上"有效但有上限"

### 1.6 `imbalance.png` — 类别不平衡实验

**描述**：4 条曲线 — Balanced reference / 不补偿 / Weighted CE / Oversampling，均为 Full fine-tune。

**正确性分析**：
- Balanced reference 收敛最快、最高（val top-1 ~0.91）
- 不补偿基线 val top-1 ~0.875，比 balanced 低 ~3 点 → 与"猫少了 80% 但 dog 不变，整体准确率仅小幅下降"的直觉一致
- Weighted CE 和 Oversampling 的曲线接近，最终 weighted CE 略胜（test 0.847 vs 0.841）
- Oversampling 的 train Loss 早期降得更猛，但 val 不如 weighted CE → 提示 oversampling 在 repeat 少数类样本时容易"记住"它们而非"理解"它们

### 1.7 `augmentation_ablation.png` — 增强消融

**描述**：10% 和 1% 标签下，augmentation 开与关的对比，共 4 条 val 曲线 + 4 条 train。

**正确性分析**：
- "no aug" 的 train top-1 远高于"aug"组（更接近 1.0），val 却更低 → 经典过拟合签名
- Aug 相对增益在 1%（-11.5%）比 10%（-6.7%）更显著 → 与 proposal §11 "study augmentation"的预期一致：数据越少，正则化（aug 是隐式正则）越重要

**LaTeX（1.2–1.7 的统一格式）**：

```latex
\begin{figure}[ht]
  \centering
  \includegraphics[width=\textwidth]{results/figures/training_curves/baselines_10.png}
  \caption{Training curves at 10\% label budget. Full fine-tuning over-fits sharply (train top-1 $\to$ 1.0, val plateaus around 0.72); Layer4-LoRA exhibits the smallest train/val gap.}
  \label{fig:training-curves-10}
\end{figure}

\begin{figure}[ht]
  \centering
  \includegraphics[width=\textwidth]{results/figures/training_curves/baselines_1.png}
  \caption{Training curves at 1\% label budget. Several strategies have not fully converged within 30 epochs (best epoch $\approx$ 28-29).}
  \label{fig:training-curves-1}
\end{figure}

\begin{figure}[ht]
  \centering
  \includegraphics[width=\textwidth]{results/figures/training_curves/lora_layer4_rank_100.png}
  \caption{Layer4-LoRA rank ablation at 100\% labels. All three ranks converge to within one accuracy point of each other; $r{=}4$ has the smallest train/val gap.}
  \label{fig:lora-layer4-rank}
\end{figure}

\begin{figure}[ht]
  \centering
  \includegraphics[width=\textwidth]{results/figures/training_curves/lora_fc_rank_100.png}
  \caption{FC-LoRA rank ablation at 100\% labels. Rank becomes a hard bottleneck because $\mathrm{rank}(\Delta W) \le r < 37$ for $r \in \{4,8,16\}$.}
  \label{fig:lora-fc-rank}
\end{figure}

\begin{figure}[ht]
  \centering
  \includegraphics[width=\textwidth]{results/figures/training_curves/imbalance.png}
  \caption{Training curves of the class-imbalance experiments (cat breeds reduced to 20\%). Weighted cross-entropy slightly outperforms oversampling; neither closes the gap to the balanced reference.}
  \label{fig:imbalance-curves}
\end{figure}

\begin{figure}[ht]
  \centering
  \includegraphics[width=\textwidth]{results/figures/training_curves/augmentation_ablation.png}
  \caption{Augmentation ablation at 10\% and 1\% labels. The relative gain from augmentation is larger at smaller label budgets.}
  \label{fig:augmentation-ablation}
\end{figure}
```

---

## 2. 混淆矩阵 `results/figures/confusion_matrices/`

所有矩阵都按行归一化，对角元 = 每类的 recall。在 imbalance 系列里，**红字标注 = 猫品种**，红色横条 = 猫类的 row 高亮，便于一眼看到少数类的恶化情况。

### 2.1 `imbalance_grid.png` — 2×2：四种 imbalance 设置

**描述**：Balanced reference / 不补偿 / Weighted CE / Oversampling 的测试集混淆矩阵。

**正确性分析**：
- Balanced reference 对角线整体最亮（recall 都 > 0.8），错误零散分布 → 与 0.868 test top-1 吻合
- 不补偿矩阵的"猫部分"（左上 12×12）对角线显著变暗，且 off-diagonal 主要落在其他猫类上 → 这是"先验偏移"的视觉证据：模型学到 P(class=cat)/P(class=dog) 比例失衡，不是把猫错分给狗
- Weighted CE / Oversampling 的猫对角线明显恢复，但 weighted CE 视觉上更均匀
- 在所有四张矩阵里，**Egyptian Mau / Bengal / Abyssinian** 这一组猫品种之间互相混淆最多 — 这是已知的视觉混淆对（毛色、斑纹相似），与 ImageNet 上猫品种识别的经验一致

**LaTeX**：

```latex
\begin{figure}[ht]
  \centering
  \includegraphics[width=\textwidth]{results/figures/confusion_matrices/imbalance_grid.png}
  \caption{Row-normalised confusion matrices for the class-imbalance experiments. Cat-breed rows and columns are coloured in red. The imbalanced-baseline matrix (top-right) shows a clear darkening of cat-row diagonals; both compensation strategies (bottom row) partially restore them.}
  \label{fig:cm-imbalance-grid}
\end{figure}
```

### 2.2 `strategies_100_grid.png` — 2×2：四种策略 @ 100%

**描述**：Linear probe / Full fine-tune / Gradual unfreeze / Layer4-LoRA r=4 在 100% labels 下的混淆矩阵。

**正确性分析**：
- 四张矩阵的对角线模式高度相似 → 在数据充足时，四种策略的错误结构没有显著的"质"上的差别，差异主要在量
- Linear probe 的 off-diagonal 略多于 fine-tune 系列，分布主要集中在同属"猫-猫"或"狗-狗"的子格 → 说明纯 fc 还不足以完全分离细粒度品种，但已经能区分大类
- 没有任何一张图出现"成片错分到单一类别"的异常 → 没有 label leakage 或类别索引错位

**LaTeX**：

```latex
\begin{figure}[ht]
  \centering
  \includegraphics[width=\textwidth]{results/figures/confusion_matrices/strategies_100_grid.png}
  \caption{Confusion matrices of the four transfer-learning strategies at 100\% labels. All four exhibit similar diagonal structure; differences are quantitative rather than structural.}
  \label{fig:cm-strategies-100}
\end{figure}
```

### 2.3 `imbalance_baseline.png` / 2.4 `imbalance_weighted_ce.png`

**描述**：上述 grid 中两个最有信息量的单张大图。

**正确性分析**：
- 把两张放在一起对比，weighted CE 矩阵的猫类对角元普遍比 baseline 亮一档（recall 提升约 5–10 个点）
- 视觉上看 *Bombay* 这种"全黑短毛"和 *Russian Blue* 这种"灰色短毛"在两张图里都容易被互相错分 → 不是补偿策略能解决的问题，而是图像本身的视觉模糊

**LaTeX**：

```latex
\begin{figure}[ht]
  \centering
  \begin{subfigure}{0.48\textwidth}
    \includegraphics[width=\textwidth]{results/figures/confusion_matrices/imbalance_baseline.png}
    \caption{No compensation (test top-1 0.830)}
  \end{subfigure}\hfill
  \begin{subfigure}{0.48\textwidth}
    \includegraphics[width=\textwidth]{results/figures/confusion_matrices/imbalance_weighted_ce.png}
    \caption{Weighted CE (test top-1 0.847)}
  \end{subfigure}
  \caption{Confusion matrix close-ups for the imbalanced setting. Cat rows recover visibly after applying class-weighted cross-entropy.}
  \label{fig:cm-imbalance-pair}
\end{figure}
```

---

## 3. Per-class F1 `results/figures/per_class_f1/`

猫品种放最左、用红色 x 轴标签标注；狗品种在右侧。

### 3.1 `imbalance_per_class_f1.png` — 分组柱状图

**描述**：37 个类，每类 4 根柱（balanced ref / 不补偿 / weighted CE / oversampling）。

**正确性分析**：
- 全部 12 个猫类在"不补偿"红柱上都显著低于 balanced 灰柱（平均 -10~15 点）→ imbalance 的伤害集中在猫
- 25 个狗类在四种设置下基本同高 → dog 数据没变，模型容量也没耗尽，dog 表现稳定
- Weighted CE 的橙柱在猫类上 consistently 高于 oversampling 的绿柱 → 与整体 macro F1 排名一致（0.844 vs 0.835）
- 没有任何一类出现"补偿后比 balanced 还高"的反常情况 → 补偿策略只是缓解伤害，没有"奇迹补偿"

**LaTeX**：

```latex
\begin{figure}[ht]
  \centering
  \includegraphics[width=\textwidth]{results/figures/per_class_f1/imbalance_per_class_f1.png}
  \caption{Per-class test F1 of the class-imbalance experiments. Cat breeds (left, red labels) drop sharply under no compensation; weighted cross-entropy recovers them more effectively than oversampling.}
  \label{fig:per-class-f1}
\end{figure}
```

### 3.2 `imbalance_species_summary.png` — 物种均值

**描述**：每种设置下，"猫平均 F1 / 狗平均 F1 / 总 macro F1"三根柱并列。

**正确性分析**：
- Balanced reference 的猫均值（0.852）与狗均值（0.873）很接近 → 模型本身没有"种偏好"
- 不补偿设置下猫均值跌到 0.728，狗均值仅微跌（0.871 → 0.871）→ 验证 imbalance 的伤害几乎完全发生在猫
- 三种补偿的狗均值都没动 → 说明 weighted CE 和 oversampling 不会"为了拉猫而牺牲狗"，是 net win

**LaTeX**：

```latex
\begin{figure}[ht]
  \centering
  \includegraphics[width=0.85\textwidth]{results/figures/per_class_f1/imbalance_species_summary.png}
  \caption{Mean F1 per species under the four imbalance settings. The drop in cat-mean F1 is the dominant signal; dog-mean F1 is essentially unchanged.}
  \label{fig:species-summary}
\end{figure}
```

### 3.3 `imbalance_per_class_delta.png` — Δ vs 不补偿

**描述**：每类两根柱：weighted CE - none，oversampling - none。正值 = 补偿有效。

**正确性分析**：
- 所有 12 个猫类的 ΔF1 都是 **正值**，且 weighted CE 平均比 oversampling 高 → 双重验证 weighted CE 更优
- 25 个狗类的 ΔF1 在 ±0.02 内随机波动 → 补偿对狗"无影响"，符合预期（狗类没受到不平衡的伤害，补偿也不该改变它们）
- 没有出现"补偿让某类 F1 跌很多"的负向 outlier → 不存在"trade-off 失败"的情况

**LaTeX**：

```latex
\begin{figure}[ht]
  \centering
  \includegraphics[width=\textwidth]{results/figures/per_class_f1/imbalance_per_class_delta.png}
  \caption{Per-class F1 change relative to the no-compensation baseline. Cats (red labels) gain consistently from both strategies; dogs are essentially unchanged.}
  \label{fig:per-class-delta}
\end{figure}
```

---

## 4. LoRA 参数量 vs 准确率 `results/figures/lora_params_vs_acc/`

x 轴对数刻度的散点 + 折线，比较所有方法在"参数效率"维度上的位置。

### 4.1 `params_vs_acc_all.png` — 三 budget 并排

**描述**：1×3 子图，每个子图一个 label budget；FC-LoRA / Layer4-LoRA 三个 rank 用虚线连成 frontier，三种 baseline 作为单点叠加。

**正确性分析**：
- 100% 子图：Full fine-tune 在最右上（最多参数、最高 accuracy）；Linear probe 与 Layer4-LoRA 几乎重合，证明"frontier"在 18K–95K 参数量级
- 10% 子图：Layer4-LoRA r=4 **位于 frontier 左上角**，超过 Full fine-tune 和 Gradual unfreeze（两者参数多 100×但准确率反而低）→ 这是 LoRA 实验最强的卖点之一
- 1% 子图：同样的反转 — Layer4-LoRA r=4 比 Full fine-tune 高 15 个点，且参数只有它的 0.85%
- FC-LoRA 折线 monotonic increasing（r 越大越好）：r=4 < r=8 < r=16 →  与 §1.5 的 FC-LoRA 容量瓶颈分析一致
- Layer4-LoRA 折线在 10% 和 1% 上 **monotonic decreasing**（r=4 > r=8 > r=16）→ 与 LoRA 论文里"小 rank 隐式正则"的论断一致，且我们的数据上现象更强

**LaTeX**：

```latex
\begin{figure}[ht]
  \centering
  \includegraphics[width=\textwidth]{results/figures/lora_params_vs_acc/params_vs_acc_all.png}
  \caption{Trainable parameters vs test top-1 across the three label budgets. Layer4-LoRA $r{=}4$ sits at the Pareto frontier under low-data regimes (1\% and 10\%), outperforming full fine-tuning despite using $\sim$0.85\% of its trainable parameters.}
  \label{fig:params-vs-acc}
\end{figure}
```

### 4.2 单 budget 版 `params_vs_acc_1pct.png` / `_10pct.png` / `_100pct.png`

**描述**：上图的三个单独版本，便于报告里独立引用某一 budget。

**正确性分析**：与 §4.1 同。

**LaTeX**（推荐用 subfigure 三联）：

```latex
\begin{figure}[ht]
  \centering
  \begin{subfigure}{0.32\textwidth}
    \includegraphics[width=\textwidth]{results/figures/lora_params_vs_acc/params_vs_acc_1pct.png}
    \caption{1\% labels}
  \end{subfigure}\hfill
  \begin{subfigure}{0.32\textwidth}
    \includegraphics[width=\textwidth]{results/figures/lora_params_vs_acc/params_vs_acc_10pct.png}
    \caption{10\% labels}
  \end{subfigure}\hfill
  \begin{subfigure}{0.32\textwidth}
    \includegraphics[width=\textwidth]{results/figures/lora_params_vs_acc/params_vs_acc_100pct.png}
    \caption{100\% labels}
  \end{subfigure}
  \caption{Parameter-accuracy frontier per label budget (independent panels of Fig.~\ref{fig:params-vs-acc}).}
  \label{fig:params-vs-acc-split}
\end{figure}
```

---

## 5. 表示分析 `results/figures/representation/`

输入是每个实验的 **37 维 test logits**（不是 penultimate features —— 提取后者需要 PyTorch 和数据集，详见 [docs/evaluation_visualization.md](docs/evaluation_visualization.md) "Known Caveats"）。所以这里展示的是"决策空间几何"。

### 5.1 `tsne_by_species.png` — t-SNE 按物种着色

**描述**：4 张 t-SNE，分别对应 4 种策略，红 = 猫、蓝 = 狗。

**正确性分析**：
- 四张图都呈现明显的"红聚一团 / 蓝聚一团"的二分结构 → 即使 logits 是 37 维分类输出，species 信息也是清晰可分的（ResNet-18 backbone 本就把 cat/dog 编码得很好）
- Linear probe 的猫团相对紧凑，狗团松散 → fc 没改 backbone 的情况下"细粒度差异在 logit 空间还不充分"
- Layer4-LoRA r=4 和 Full fine-tune 的几何最接近，都把品种 sub-cluster 解构得最细 → 视觉佐证了 LoRA r=4 "用 0.85% 参数达到 fine-tune 等效的表示能力"
- Gradual unfreeze 的几何有些"细碎块"，可能与 layer-by-layer 解冻造成的多阶段优化路径有关

**LaTeX**：

```latex
\begin{figure}[ht]
  \centering
  \includegraphics[width=\textwidth]{results/figures/representation/tsne_by_species.png}
  \caption{t-SNE of test logits coloured by species. All four strategies separate cats from dogs cleanly; LoRA $r{=}4$ produces a class-substructure most similar to full fine-tuning.}
  \label{fig:tsne-species}
\end{figure}
```

### 5.2 `tsne_by_class.png` — t-SNE 按 37 类着色

**描述**：同上 4 张 t-SNE，按 ground-truth 类（37 色）着色。

**正确性分析**：
- Linear probe 的 cluster 中心多、但每个 cluster 内有混色 → 与 0.846 test top-1 吻合（不是 100%，所以混淆是预期的）
- Full fine-tune / Layer4-LoRA r=4 / Gradual unfreeze 的 cluster 更紧凑、单色比例更高 → 与它们更高的 top-1 一致
- 没有任何一种策略出现"所有点挤成一团"或"明显的类别错位"→ 模型工作正常，没有崩溃模式

**LaTeX**：

```latex
\begin{figure}[ht]
  \centering
  \includegraphics[width=\textwidth]{results/figures/representation/tsne_by_class.png}
  \caption{t-SNE of test logits coloured by ground-truth class (37 categories). Linear probing produces visibly noisier clusters; the three deeper strategies form tighter and more single-coloured cluster cores.}
  \label{fig:tsne-class}
\end{figure}
```

### 5.3 `pca_by_species.png` / `pca_by_class.png` — PCA 对照

**描述**：相同输入的 PCA 二维投影，作为 reproducibility 的对照（PCA 是确定性的，t-SNE 不是）。

**正确性分析**：
- PCA 视觉上没有 t-SNE 那么"分团"，但 species 的"红/蓝"梯度依然清楚 → t-SNE 看到的结构不是伪影
- 第一主成分（横轴）几乎对应"猫 vs 狗"的方向 → 模型的决策最显著的一维就是物种
- 没有出现 t-SNE 和 PCA 显示矛盾结构的情况 → 两种方法互证

**LaTeX**：

```latex
\begin{figure}[ht]
  \centering
  \begin{subfigure}{0.48\textwidth}
    \includegraphics[width=\textwidth]{results/figures/representation/pca_by_species.png}
    \caption{Coloured by species (cat / dog)}
  \end{subfigure}\hfill
  \begin{subfigure}{0.48\textwidth}
    \includegraphics[width=\textwidth]{results/figures/representation/pca_by_class.png}
    \caption{Coloured by class (37)}
  \end{subfigure}
  \caption{PCA of test logits, used as a deterministic cross-check for the t-SNE embeddings in Fig.~\ref{fig:tsne-species} and Fig.~\ref{fig:tsne-class}. The first principal component aligns with the cat-vs-dog axis.}
  \label{fig:pca}
\end{figure}
```

---

## 6. 训练成本 `results/figures/training_cost/`

四张图从不同视角刻画"训练代价"。**注意**：`finetune_100` 的总时间 4418 秒包含了机器睡眠期间的两个 epoch，是已记录的虚高数据（[docs/experiment_baselines.md:305](docs/experiment_baselines.md:305)）。引用时建议加 caveat 或用 time-per-epoch（~145 s/epoch）。

### 6.1 `time_vs_acc.png` — 总训练时间 vs accuracy

**描述**：1×3 散点（每个 budget 一格），所有方法 + 所有 rank。

**正确性分析**：
- 100% 子图：Full fine-tune 在最右上 → 准确率最高但训练时间也最长，符合预期
- Layer4-LoRA 的 r=4 / r=8 / r=16 三个点几乎重叠 → rank 不显著影响训练时间（与 §6.2 一致）
- 10% 子图：Layer4-LoRA cluster 位于"低时间高 accuracy"的左上区域 → 帕累托最优
- FC-LoRA r=4 在 10% 上 accuracy 仅 0.21 但时间和其他 LoRA 一样 → 提示 FC-only 的方案"花了钱没收效"，应排除作为推荐

**LaTeX**：

```latex
\begin{figure}[ht]
  \centering
  \includegraphics[width=\textwidth]{results/figures/training_cost/time_vs_acc.png}
  \caption{Total training time vs test top-1 across label budgets. Layer4-LoRA dominates the upper-left (low time, high accuracy) region under 10\% labels. The full fine-tune point at 100\% reports an inflated 4418\,s due to two sleep-mode epochs (see Sec.\,X).}
  \label{fig:time-vs-acc}
\end{figure}
```

### 6.2 `perepoch_vs_params.png` — 单 epoch 时间 vs 参数量

**描述**：100% labels 下，所有方法的"每 epoch 训练时间"对"可训练参数量"散点（log x 轴）。

**正确性分析**：
- LoRA（FC + Layer4，2K–322K 参数）的 per-epoch 时间稳定在 43–50 s
- Linear probe（18,981 参数）51 s
- Gradual unfreeze 65 s（因为 optimizer 重建 + 解冻参数逐步增加）
- Full fine-tune（11M 参数）145 s
- per-epoch 时间不是单调线性增长 → 因为 backward pass 时间主要由"活跃 layer 数"而非"参数总数"决定。LoRA 即使加在 layer4，前向仍走完整网络，反向也只在 layer4 + fc 计算梯度，所以接近 linear probe
- Layer4-LoRA r=16（322K 参数）和 r=4（94K）时间几乎一样 → 反向矩阵乘法的常数项远大于 rank 部分

**LaTeX**：

```latex
\begin{figure}[ht]
  \centering
  \includegraphics[width=0.85\textwidth]{results/figures/training_cost/perepoch_vs_params.png}
  \caption{Per-epoch wall-clock time vs trainable parameters at 100\% labels. Time grows discretely with the depth of the backward pass, not continuously with parameter count; LoRA epochs are essentially as cheap as linear-probe epochs.}
  \label{fig:perepoch-cost}
\end{figure}
```

### 6.3 `convergence_best_epoch.png` — 收敛速度

**描述**：5 种方法 × 3 budget，柱状图，柱高 = best val epoch（越小收敛越快）。LoRA 的 rank 维度取了平均。

**正确性分析**：
- 100% 数据下 best epoch 排序：Full fine-tune (5) < Linear probe (8) < Layer4-LoRA (9.3) < Gradual unfreeze (13) < FC-LoRA (24)
  - Full fine-tune 最快收敛是因为它 fit power 最强，但代价是过拟合 → 与训练曲线观察一致
  - FC-LoRA 最慢是因为它要在受限的低秩空间里挣扎 → 与训练曲线的"长尾"一致
- 1% 数据下所有方法都接近 30（最后一个 epoch），表示没收敛 → 与 [docs/experiment_baselines.md:303](docs/experiment_baselines.md:303) "extending training to 50 epochs..."的说法一致

**LaTeX**：

```latex
\begin{figure}[ht]
  \centering
  \includegraphics[width=0.85\textwidth]{results/figures/training_cost/convergence_best_epoch.png}
  \caption{Best validation epoch by strategy and label budget (mean across LoRA ranks). At 100\% labels full fine-tuning converges fastest but at the cost of subsequent over-fitting; FC-LoRA is the slowest, consistent with its restricted low-rank capacity.}
  \label{fig:convergence}
\end{figure}
```

### 6.4 `efficiency_100.png` — 效率综合视图

**描述**：左：cost-per-accuracy = time-per-epoch / test top-1 的横向排名（越小越省）。右：错误率 (1 - top-1) vs 总时间的 Pareto 散点。

**正确性分析**：
- 左图最省 = FC-LoRA r=16，最费 = Full fine-tune（受 4418 s 异常值影响，时间字段虚高）
- 排除 Full fine-tune 后，Layer4-LoRA r=4 / r=8 / r=16 都比 Linear probe 更省 → 训练效率上 LoRA 全面领先
- 右图 Pareto frontier：Layer4-LoRA r=8/r=16 在最左下区域（低时间 + 低错误率），Gradual unfreeze 在最低错误率但时间偏长 → 取决于报告中你想 highlight 哪个维度

**LaTeX**：

```latex
\begin{figure}[ht]
  \centering
  \includegraphics[width=\textwidth]{results/figures/training_cost/efficiency_100.png}
  \caption{Training efficiency at 100\% labels. Left: cost-per-accuracy ranking (lower is better). Right: error vs total training time; methods in the lower-left dominate the trade-off. Note that the full fine-tune point on the right is inflated by sleep-mode time accounting.}
  \label{fig:efficiency}
\end{figure}
```

---

## 7. 整体正确性自检小结

最后给一个跨图的"sanity check"清单，可以直接出现在报告的 "Validation of Results" 段落里：

1. **数值一致性**：每张图里出现的 test accuracy 数字都能在 [results/experiment_summary.csv](results/experiment_summary.csv) 中找到原始来源 → 没有手动估算或重采样
2. **训练/验证 gap 与参数量成正相关**（在过拟合 regime 内）→ 训练曲线、混淆矩阵、Pareto 散点三处证据互相印证
3. **不平衡伤害集中在猫**：在 species summary、per-class F1、混淆矩阵、t-SNE by species 四种独立可视化方式上都能看到 → 不是单一图表的伪结论
4. **LoRA 的"低秩即正则"现象**：在 Layer4-LoRA rank ablation 训练曲线、参数-准确率散点、convergence speed 三处独立呈现 → 与 LoRA 论文 §6 的结论一致
5. **没有 label-leakage 信号**：所有混淆矩阵的对角线都是行内最亮，没有"成片错分到固定列"的异常 → 数据 split 和 label 映射正确

---

## 8. LaTeX 使用提示

- 全部 figure 块都用了 `\label{fig:...}`，正文里引用统一写 `Fig.~\ref{fig:training-curves-100}`
- 6 张 1×3 子图（training curves, params-vs-acc, time-vs-acc）建议放在 `[ht]` 浮动位置；如果某一页排版冲突可改 `[!htbp]`
- subfigure 写法需 `\usepackage{subcaption}`，已在文档顶部提示
- 如果报告整体已有"figure naming convention"，请把所有 `\label` 改成你们的命名风格
