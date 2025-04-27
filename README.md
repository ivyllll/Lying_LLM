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

