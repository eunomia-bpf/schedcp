#!/usr/bin/env python3
"""
FAISS 实验结果可视化脚本
生成两个子图：索引构建时间进度 和 搜索延迟改进（折线图）
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# 结果目录
results_dir = Path(__file__).parent

# 读取所有结果文件
result_files = list(results_dir.glob("*.json"))

def parse_filename(filename):
    name = filename.stem
    if "SIFT100M" in name:
        dataset = "SIFT100M"
    elif "SIFT50M" in name:
        dataset = "SIFT50M"
    elif "SIFT20M" in name:
        dataset = "SIFT20M"
    else:
        dataset = "Unknown"

    if "prefetch_adaptive" in name:
        config = "UVM+BPF Prefetch"
    elif "uvm_baseline" in name:
        config = "UVM Baseline"
    elif "cpu" in name:
        config = "CPU"
    elif "baseline" in name:
        config = "GPU"
    else:
        config = "Other"

    return dataset, config

# 加载所有数据
data_by_dataset = {}
for f in result_files:
    dataset, config = parse_filename(f)
    with open(f) as fp:
        content = json.load(fp)
    if dataset not in data_by_dataset:
        data_by_dataset[dataset] = {}
    data_by_dataset[dataset][config] = content

# 配色
config_colors = {
    "CPU": "tab:blue",
    "GPU": "tab:green",
    "UVM Baseline": "tab:orange",
    "UVM+BPF Prefetch": "tab:red",
}

dataset_colors = {
    "SIFT50M": "tab:blue",
    "SIFT100M": "tab:red",
}

dataset_markers = {
    "SIFT50M": "o",
    "SIFT100M": "s",
}

# ==================== 创建两个子图 ====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# ==================== 左图: Build Index 进度 ====================
for dataset in sorted(data_by_dataset.keys()):
    for config, d in data_by_dataset[dataset].items():
        progress = d["index_add"]["progress"]
        vectors = [p["vectors_added"] / 1e6 for p in progress]
        times = [p["time"] for p in progress]
        label = f"{dataset} {config}"
        ax1.plot(vectors, times, marker='o', markersize=4, label=label,
                 color=config_colors.get(config, "tab:gray"))

ax1.set_xlabel("Vectors Added (millions)", fontsize=18)
ax1.set_ylabel("Time (seconds)", fontsize=18)
ax1.set_title("(a) Index Build Time", fontsize=20, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='both', labelsize=14)

# ==================== 右图: Search Latency 改进 (折线图) ====================
uvm_datasets = [ds for ds in sorted(data_by_dataset.keys())
                if "UVM Baseline" in data_by_dataset[ds] and "UVM+BPF Prefetch" in data_by_dataset[ds]]

all_nprobes = sorted(set(
    s["nprobe"]
    for ds in uvm_datasets
    for s in data_by_dataset[ds]["UVM Baseline"]["search"]
))

for ds in uvm_datasets:
    base_search = {s["nprobe"]: s for s in data_by_dataset[ds]["UVM Baseline"]["search"]}
    pref_search = {s["nprobe"]: s for s in data_by_dataset[ds]["UVM+BPF Prefetch"]["search"]}

    nprobes = []
    normalized_latencies = []

    for nprobe in all_nprobes:
        if nprobe in base_search and nprobe in pref_search:
            base_latency = base_search[nprobe]["search_time"] / base_search[nprobe]["num_queries"] * 1000
            pref_latency = pref_search[nprobe]["search_time"] / pref_search[nprobe]["num_queries"] * 1000
            nprobes.append(nprobe)
            normalized_latencies.append(pref_latency / base_latency)

    ax2.plot(nprobes, normalized_latencies,
             marker=dataset_markers.get(ds, 'o'),
             markersize=14,
             linewidth=3,
             label=ds,
             color=dataset_colors.get(ds, "tab:gray"))

    for np_val, norm_lat in zip(nprobes, normalized_latencies):
        change = (norm_lat - 1.0) * 100
        ax2.annotate(f'{change:.1f}%',
                    xy=(np_val, norm_lat),
                    xytext=(0, -20 if ds == "SIFT50M" else 15),
                    textcoords='offset points',
                    ha='center', va='top' if ds == "SIFT50M" else 'bottom',
                    fontsize=16, fontweight='bold',
                    color=dataset_colors.get(ds, "tab:gray"))

ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, linewidth=2)
ax2.set_xticks(all_nprobes)
ax2.set_xticklabels([str(n) for n in all_nprobes])
ax2.set_xlabel("nprobe", fontsize=18)
ax2.set_ylabel("Normalized Latency (Baseline = 1.0)", fontsize=18)
ax2.set_title("(b) Search Latency", fontsize=20, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.75, 1.1)
ax2.tick_params(axis='both', labelsize=14)

ax2.fill_between(all_nprobes, 0, 1.0, alpha=0.1, color='green', label='_nolegend_')
ax2.text(all_nprobes[-1], 0.77, 'Lower is Better', ha='right', fontsize=14, color='green', alpha=0.8)

# ==================== 合并图例放在底部 ====================
# 创建统一的图例元素
legend_elements = [
    # 配置类型
    Line2D([0], [0], color='tab:blue', marker='o', markersize=8, linewidth=2, label='CPU'),
    Line2D([0], [0], color='tab:green', marker='o', markersize=8, linewidth=2, label='GPU'),
    Line2D([0], [0], color='tab:orange', marker='o', markersize=8, linewidth=2, label='UVM Baseline'),
    Line2D([0], [0], color='tab:red', marker='o', markersize=8, linewidth=2, label='UVM+BPF Prefetch'),
    # 数据集（右图）
    Line2D([0], [0], color='tab:blue', marker='o', markersize=10, linewidth=2, linestyle='-', label='SIFT50M (right)'),
    Line2D([0], [0], color='tab:red', marker='s', markersize=10, linewidth=2, linestyle='-', label='SIFT100M (right)'),
    # Baseline 参考线
    Line2D([0], [0], color='gray', linestyle='--', linewidth=2, label='Baseline (1.0)'),
]

fig.legend(handles=legend_elements, loc='lower center', ncol=7, fontsize=12,
           bbox_to_anchor=(0.5, -0.02), frameon=True, fancybox=True)

# 调整布局，为底部图例留出空间
plt.subplots_adjust(bottom=0.18)

# 保存图片
output_path = results_dir / 'faiss_benchmark_results.png'
fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"图片已保存: {output_path}")

output_pdf = results_dir / 'faiss_benchmark_results.pdf'
fig.savefig(output_pdf, bbox_inches='tight', facecolor='white')
print(f"PDF 已保存: {output_pdf}")

plt.show()

# ==================== 打印详细数据 ====================
print("\n" + "=" * 60)
print("Search Latency 详细数据:")
print("=" * 60)
print(f"{'Dataset':<12} {'nprobe':<8} {'Baseline(ms)':<14} {'Prefetch(ms)':<14} {'变化':<10}")
print("-" * 60)

for ds in uvm_datasets:
    base_search = {s["nprobe"]: s for s in data_by_dataset[ds]["UVM Baseline"]["search"]}
    pref_search = {s["nprobe"]: s for s in data_by_dataset[ds]["UVM+BPF Prefetch"]["search"]}

    for nprobe in all_nprobes:
        if nprobe in base_search and nprobe in pref_search:
            base_lat = base_search[nprobe]["search_time"] / base_search[nprobe]["num_queries"] * 1000
            pref_lat = pref_search[nprobe]["search_time"] / pref_search[nprobe]["num_queries"] * 1000
            change = (pref_lat - base_lat) / base_lat * 100
            print(f"{ds:<12} {nprobe:<8} {base_lat:<14.3f} {pref_lat:<14.3f} {change:>+.1f}%")
