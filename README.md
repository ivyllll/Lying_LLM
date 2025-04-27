# Lying_LLM

4/22;

Fetch the activations of all specific topics based on three prompts (honest, neurtral, deceptive).

Run **step1** to examine the probing accuracy on several topics (on truthful prompt).

4/23;

Continue to run **step1** for deceptive and neutral prompts.

Get the pkl file for all three prompts, containing the average accuracies and the standard deviations of two probing methods TTPD and LR among all layers.

Run **step2**:

**layer 15:**
| Prompt Type | Probe   | Mean Accuracy (%) | Std. Dev (%) |
|-------------|---------|-------------------|--------------|
| Truthful    | TTPD    | 79.99              | 0.42         |
|             | LRProbe | 78.12              | 2.76         |
| Deceptive   | TTPD    | 80.36              | 0.14         |
|             | LRProbe | 68.36              | 5.25         |
| Neutral     | TTPD    | 84.88              | 0.46         |
|             | LRProbe | 75.43              | 4.53         |

4/26；

Add SAE on the collected activations to do further analysis:

1. Feature Shift Analysis for truthful vs neutral / truthful vs deceptive / neutral vs deceptive
        using Cosine Similarity, Overlap Ratio, and L2 score to evaluate the tendency through layers
2. 


4/27；
**step2: **
| Layer | Truthful-TTPD (Mean ± Std) | Truthful-LR (Mean ± Std) | Neutral-TTPD (Mean ± Std) | Neutral-LR (Mean ± Std) | Deceptive-TTPD (Mean ± Std) | Deceptive-LR (Mean ± Std) |
|:-----:|:--------------------------:|:-----------------------:|:--------------------------:|:-----------------------:|:--------------------------:|:-----------------------:|
| 0 | 50.69 ± 1.10 | 49.50 ± 0.00 | 50.43 ± 0.65 | 49.50 ± 0.00 | 50.63 ± 1.00 | 49.50 ± 0.00 |
| 1 | 50.17 ± 0.35 | 49.56 ± 0.35 | 50.36 ± 0.66 | 49.48 ± 0.11 | 50.16 ± 0.36 | 49.48 ± 0.17 |
| 2 | 50.07 ± 0.08 | 49.64 ± 0.27 | 50.21 ± 0.19 | 49.79 ± 0.43 | 50.20 ± 0.36 | 49.76 ± 0.45 |
| 3 | 50.23 ± 0.31 | 50.62 ± 0.88 | 50.49 ± 0.50 | 50.74 ± 0.96 | 50.14 ± 0.24 | 50.11 ± 0.68 |
| 4 | 49.83 ± 0.63 | 51.20 ± 0.64 | 49.88 ± 0.40 | 51.08 ± 0.83 | 49.91 ± 0.56 | 50.95 ± 0.62 |
| 5 | 49.91 ± 0.55 | 50.92 ± 0.18 | 49.88 ± 0.47 | 50.93 ± 0.28 | 49.87 ± 0.24 | 50.84 ± 0.33 |
| 6 | 52.38 ± 1.31 | 52.02 ± 0.64 | 51.58 ± 1.48 | 52.63 ± 1.06 | 51.01 ± 1.70 | 52.52 ± 1.12 |
| 7 | 52.90 ± 1.26 | 52.13 ± 0.61 | 52.04 ± 0.86 | 52.94 ± 1.02 | 52.46 ± 0.75 | 52.12 ± 0.67 |
| 8 | 58.66 ± 1.96 | 53.09 ± 1.60 | 57.93 ± 1.99 | 56.53 ± 3.05 | 57.77 ± 2.41 | 53.88 ± 2.34 |
| 9 | 59.58 ± 4.57 | 61.37 ± 3.67 | 57.23 ± 4.96 | 62.20 ± 4.01 | 58.79 ± 3.42 | 62.05 ± 3.08 |
| 10 | 68.41 ± 0.74 | 57.16 ± 3.11 | 64.66 ± 2.49 | 59.57 ± 3.65 | 60.11 ± 3.96 | 57.07 ± 3.38 |
| 11 | 69.69 ± 0.63 | 63.11 ± 3.09 | 69.75 ± 0.83 | 62.33 ± 4.54 | 69.13 ± 0.73 | 58.76 ± 3.67 |
| 12 | 77.20 ± 0.23 | 65.37 ± 4.95 | 80.10 ± 0.24 | 62.92 ± 5.64 | 73.43 ± 0.83 | 62.25 ± 4.66 |
| 13 | 78.22 ± 0.23 | 75.70 ± 2.88 | 82.25 ± 0.53 | 71.66 ± 4.21 | 79.29 ± 0.30 | 70.44 ± 3.74 |
| 14 | 80.04 ± 0.47 | 75.97 ± 3.46 | 84.86 ± 0.47 | 75.49 ± 3.53 | 80.42 ± 0.13 | 70.07 ± 6.07 |
| 15 | 82.58 ± 0.60 | 77.15 ± 4.43 | 85.62 ± 0.14 | 74.91 ± 4.07 | 79.23 ± 0.11 | 71.33 ± 4.35 |
| 16 | 83.62 ± 0.41 | 76.97 ± 3.27 | 85.42 ± 0.15 | 74.03 ± 5.79 | 79.80 ± 0.29 | 69.46 ± 6.76 |
| 17 | 83.55 ± 0.35 | 74.53 ± 4.70 | 85.01 ± 0.30 | 74.43 ± 4.92 | 76.25 ± 0.35 | 67.11 ± 6.53 |
| 18 | 83.24 ± 0.22 | 73.62 ± 5.45 | 83.75 ± 0.34 | 73.71 ± 4.70 | 78.69 ± 0.18 | 67.65 ± 5.26 |
| 19 | 83.07 ± 0.41 | 74.90 ± 4.18 | 83.62 ± 0.22 | 71.85 ± 4.34 | 78.69 ± 0.48 | 66.16 ± 7.26 |
| 20 | 82.84 ± 0.27 | 69.61 ± 5.96 | 83.41 ± 0.36 | 76.20 ± 4.87 | 79.50 ± 0.34 | 66.87 ± 6.48 |
| 21 | 82.53 ± 0.35 | 71.29 ± 5.17 | 83.61 ± 0.21 | 73.31 ± 5.80 | 79.22 ± 0.23 | 66.44 ± 7.18 |
| 22 | 82.25 ± 0.29 | 73.93 ± 5.95 | 83.39 ± 0.24 | 72.20 ± 4.89 | 79.06 ± 0.37 | 67.68 ± 6.19 |
| 23 | 82.11 ± 0.37 | 71.52 ± 5.36 | 83.22 ± 0.24 | 72.48 ± 6.43 | 78.54 ± 0.33 | 68.94 ± 5.61 |
| 24 | 82.10 ± 0.29 | 73.07 ± 6.04 | 83.30 ± 0.33 | 72.60 ± 6.11 | 78.24 ± 0.19 | 67.35 ± 6.50 |
| 25 | 81.96 ± 0.29 | 71.62 ± 6.31 | 83.26 ± 0.19 | 73.14 ± 5.08 | 78.33 ± 0.26 | 70.58 ± 4.26 |
| 26 | 81.55 ± 0.28 | 71.34 ± 6.16 | 82.97 ± 0.23 | 73.90 ± 5.83 | 77.45 ± 0.56 | 69.06 ± 5.07 |
| 27 | 81.42 ± 0.33 | 73.38 ± 5.56 | 82.90 ± 0.24 | 75.66 ± 5.13 | 76.81 ± 0.30 | 68.20 ± 5.64 |
| 28 | 81.36 ± 0.18 | 75.02 ± 3.82 | 82.70 ± 0.25 | 73.47 ± 4.71 | 76.43 ± 0.38 | 68.37 ± 6.25 |
| 29 | 81.31 ± 0.22 | 70.84 ± 6.26 | 82.87 ± 0.34 | 72.73 ± 4.14 | 76.51 ± 0.60 | 69.63 ± 7.97 |
| 30 | 81.62 ± 0.23 | 71.57 ± 5.66 | 82.89 ± 0.33 | 71.96 ± 5.07 | 76.17 ± 0.71 | 63.63 ± 7.32 |
| 31 | 81.57 ± 0.33 | 64.56 ± 4.66 | 83.87 ± 0.19 | 72.36 ± 5.25 | 77.86 ± 1.23 | 68.55 ± 5.39 |

## 🔍 Project Summary: Probing and Visualizing Internal Representations of LLaMA

### 🔗 Function Call Overview

```text
main()
├── if run_step["step1"]:
│   └── run_step1(...)             # Train & evaluate TTPD & LR probes on 12 topic-specific datasets
│       └── collect_training_data(...)   # Load activations, labels, polarities from local storage
│       └── TTPD.from_data(...), LRProbe.from_data(...)  # Fit probes
│       └── DataManager.add_dataset(...) → probe.pred(...)  # Evaluate probes on held-out datasets
│       └── compute_statistics(...), compute_average_accuracies(...)
│       └── Save accuracy summary as chat_probe_accuracies_layerwise.pkl
│
├── visualize_layerwise_probe_accuracy()  # Plot probe accuracy curves across all layers (TTPD vs LR)
│   └── Load chat_probe_accuracies_layerwise.pkl → generate line plot → save as PNG
│
├── for layer in range(0, 32):
│   └── visualize_latent_space(...)       # Project and plot activations at a fixed layer (13/15/32)
│       └── collect_training_data(...)    # Load activations
│       └── t-SNE / UMAP / PCA / Isomap → save 2D plots as PNG
│
└── if run_step["step2"]:
    └── run_step2(...)             # Evaluate probes on logical conjunctions/disjunctions
        └── Similar to run_step1, but different test sets
```


### 📌 Function Descriptions

- **main()**: The main entrypoint to run experiments, visualization, or evaluations.
- **run_step1(...)**: Train and cross-validate TTPD/LR probes on factual datasets.
- **run_step2(...)**: Evaluate generalization of probes to logical combinations (conj/disj).
- **visualize_layerwise_probe_accuracy()**: Visualize probing accuracy across all layers.
- **visualize_latent_space(...)**: Visualize activations (layer 12) in 2D using four projection methods.

---

## 🔍 项目总结：使用探量器分析 LLaMA 内部表征

### 🔗 函数调用路径

```text
main()
├── if run_step["step1"]:
│   └── run_step1(...)             # 培训 TTPD 和 LR 探量器，对 12 个主题进行分类添加
│       └── collect_training_data(...)   # 从本地文件读取激活值、标签、极性
│       └── TTPD.from_data(...), LRProbe.from_data(...)  # 调用探量器进行训练
│       └── DataManager.add_dataset(...) → probe.pred(...)  # 在留出集进行分类测试
│       └── compute_statistics(...), compute_average_accuracies(...)
│       └── 将添加结果保存为 chat_probe_accuracies_layerwise.pkl
│
├── visualize_layerwise_probe_accuracy()  # 同时绘制 TTPD 和 LR 探量类型的层级分类精度曲线
│   └── 读取 chat_probe_accuracies_layerwise.pkl 进行绘图并保存 PNG
│
├── for layer in range(0, 32):
│   └── visualize_latent_space(...)       # 在给定层 (13/15/32) 对激活值进行缩线成 2D 进行可视化
│       └── collect_training_data(...)    # 读取数据
│       └── t-SNE / UMAP / PCA / Isomap 进行降维绘图
│
└── if run_step["step2"]:
    └── run_step2(...)             # 评估 TTPD / LR 在逻辑关系数据集上的迁化性
        └── 和 run_step1 类似，但是测试集不同
```


### 📌 函数介绍

- **main()**: 程序进入点，用于运行实验、后处理、或表征可视化
- **run_step1(...)**: 对 TTPD 和 LR 探量器进行分类训练，重复交叉验证
- **run_step2(...)**: 评估探量器对逻辑形式统计值的迁化性
- **visualize_layerwise_probe_accuracy()**: 演示各层探量类型的分类精度较动图
- **visualize_latent_space(...)**: 对一层的激活值进行 t-SNE/UMAP/PCA/Isomap 降维和绘图

