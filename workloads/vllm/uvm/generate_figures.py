#!/usr/bin/env python3
"""
Generate figures for vLLM UVM benchmark results.
Data extracted from test_uvm.md benchmark results.
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Benchmark data from test_uvm.md
data = {
    "configurations": [
        "CPU Offload\n(8GB)",
        "UVM\nBaseline",
        "UVM +\nPrefetch",
        "LMCache"
    ],
    # Time to First Token (ms)
    "ttft_mean": [8387.80, 9642.27, 5042.22, 5401.71],
    "ttft_median": [9066.78, 10330.52, 5419.23, 4961.94],
    "ttft_p99": [14937.16, 16549.02, 7933.22, 10072.64],
    # Time per Output Token (ms)
    "tpot_mean": [324.13, 374.23, 235.68, 222.24],
    "tpot_median": [215.65, 270.06, 207.69, 149.84],
    "tpot_p99": [1288.01, 1288.79, 583.74, 817.94],
    # Inter-token Latency (ms)
    "itl_mean": [191.40, 247.34, 194.62, 131.20],
    "itl_median": [180.39, 214.44, 173.72, 122.30],
    "itl_p99": [1299.17, 1077.62, 467.74, 833.54],
    # Throughput metrics
    "output_throughput": [190.40, 149.56, 183.28, 278.21],
    "total_throughput": [391.14, 307.26, 376.53, 571.54],
    "benchmark_duration": [115.87, 147.50, 120.36, 79.30],
}

# Output directory
output_dir = os.path.dirname(os.path.abspath(__file__))

# Color palette
colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']


def plot_ttft_comparison():
    """Plot Time to First Token comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(data["configurations"]))
    width = 0.25

    bars1 = ax.bar(x - width, data["ttft_mean"], width, label='Mean', color=colors[0], alpha=0.8)
    bars2 = ax.bar(x, data["ttft_median"], width, label='Median', color=colors[1], alpha=0.8)
    bars3 = ax.bar(x + width, data["ttft_p99"], width, label='P99', color=colors[2], alpha=0.8)

    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Time to First Token (ms)', fontsize=12)
    ax.set_title('vLLM TTFT Comparison: Qwen3-30B-A3B-FP8 on RTX 5090', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(data["configurations"], fontsize=10)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ttft_comparison.png'), dpi=150)
    plt.close()
    print("Generated: ttft_comparison.png")


def plot_tpot_comparison():
    """Plot Time per Output Token comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(data["configurations"]))
    width = 0.25

    bars1 = ax.bar(x - width, data["tpot_mean"], width, label='Mean', color=colors[0], alpha=0.8)
    bars2 = ax.bar(x, data["tpot_median"], width, label='Median', color=colors[1], alpha=0.8)
    bars3 = ax.bar(x + width, data["tpot_p99"], width, label='P99', color=colors[2], alpha=0.8)

    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Time per Output Token (ms)', fontsize=12)
    ax.set_title('vLLM TPOT Comparison: Qwen3-30B-A3B-FP8 on RTX 5090', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(data["configurations"], fontsize=10)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tpot_comparison.png'), dpi=150)
    plt.close()
    print("Generated: tpot_comparison.png")


def plot_throughput_comparison():
    """Plot throughput comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(data["configurations"]))
    width = 0.35

    bars1 = ax.bar(x - width/2, data["output_throughput"], width,
                   label='Output Token Throughput', color=colors[0], alpha=0.8)
    bars2 = ax.bar(x + width/2, data["total_throughput"], width,
                   label='Total Token Throughput', color=colors[2], alpha=0.8)

    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Throughput (tok/s)', fontsize=12)
    ax.set_title('vLLM Throughput Comparison: Qwen3-30B-A3B-FP8 on RTX 5090', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(data["configurations"], fontsize=10)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'throughput_comparison.png'), dpi=150)
    plt.close()
    print("Generated: throughput_comparison.png")


def plot_latency_summary():
    """Plot summary of all latency metrics (median values)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(data["configurations"]))
    width = 0.25

    bars1 = ax.bar(x - width, data["ttft_median"], width, label='TTFT', color=colors[0], alpha=0.8)
    bars2 = ax.bar(x, data["tpot_median"], width, label='TPOT', color=colors[1], alpha=0.8)
    bars3 = ax.bar(x + width, data["itl_median"], width, label='ITL', color=colors[2], alpha=0.8)

    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('vLLM Median Latency Summary: Qwen3-30B-A3B-FP8 on RTX 5090', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(data["configurations"], fontsize=10)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latency_summary.png'), dpi=150)
    plt.close()
    print("Generated: latency_summary.png")


def plot_benchmark_duration():
    """Plot benchmark duration comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(data["configurations"]))

    bars = ax.bar(x, data["benchmark_duration"], color=colors, alpha=0.8)

    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Benchmark Duration (s)', fontsize=12)
    ax.set_title('vLLM Benchmark Duration: 100 Prompts on Qwen3-30B-A3B-FP8', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(data["configurations"], fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}s',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmark_duration.png'), dpi=150)
    plt.close()
    print("Generated: benchmark_duration.png")


def plot_combined_overview():
    """Create a combined overview figure with multiple subplots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    x = np.arange(len(data["configurations"]))
    width = 0.25

    # Plot 1: TTFT
    ax1 = axes[0, 0]
    ax1.bar(x - width, data["ttft_mean"], width, label='Mean', color=colors[0], alpha=0.8)
    ax1.bar(x, data["ttft_median"], width, label='Median', color=colors[1], alpha=0.8)
    ax1.bar(x + width, data["ttft_p99"], width, label='P99', color=colors[2], alpha=0.8)
    ax1.set_ylabel('TTFT (ms)', fontsize=11)
    ax1.set_title('Time to First Token', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(data["configurations"], fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: TPOT
    ax2 = axes[0, 1]
    ax2.bar(x - width, data["tpot_mean"], width, label='Mean', color=colors[0], alpha=0.8)
    ax2.bar(x, data["tpot_median"], width, label='Median', color=colors[1], alpha=0.8)
    ax2.bar(x + width, data["tpot_p99"], width, label='P99', color=colors[2], alpha=0.8)
    ax2.set_ylabel('TPOT (ms)', fontsize=11)
    ax2.set_title('Time per Output Token', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(data["configurations"], fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: Throughput
    ax3 = axes[1, 0]
    width_t = 0.35
    ax3.bar(x - width_t/2, data["output_throughput"], width_t,
            label='Output Throughput', color=colors[0], alpha=0.8)
    ax3.bar(x + width_t/2, data["total_throughput"], width_t,
            label='Total Throughput', color=colors[2], alpha=0.8)
    ax3.set_ylabel('Throughput (tok/s)', fontsize=11)
    ax3.set_title('Token Throughput', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(data["configurations"], fontsize=9)
    ax3.legend(fontsize=8)
    ax3.grid(axis='y', alpha=0.3)

    # Plot 4: Benchmark Duration
    ax4 = axes[1, 1]
    bars = ax4.bar(x, data["benchmark_duration"], color=colors, alpha=0.8)
    ax4.set_ylabel('Duration (s)', fontsize=11)
    ax4.set_title('Benchmark Duration (100 prompts)', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(data["configurations"], fontsize=9)
    ax4.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax4.annotate(f'{height:.1f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    fig.suptitle('vLLM UVM Benchmark Results: Qwen3-30B-A3B-FP8 on RTX 5090 (32GB)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_overview.png'), dpi=150)
    plt.close()
    print("Generated: combined_overview.png")


def plot_speedup_comparison():
    """Plot speedup relative to UVM baseline."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate speedup (lower is better for latency, so we invert)
    baseline_ttft = data["ttft_median"][1]  # UVM Baseline
    baseline_tpot = data["tpot_median"][1]
    baseline_throughput = data["total_throughput"][1]

    configs = data["configurations"]

    # Speedup for TTFT (baseline / value, higher = better)
    ttft_speedup = [baseline_ttft / v for v in data["ttft_median"]]
    # Speedup for TPOT
    tpot_speedup = [baseline_tpot / v for v in data["tpot_median"]]
    # Speedup for throughput (value / baseline, higher = better)
    throughput_speedup = [v / baseline_throughput for v in data["total_throughput"]]

    x = np.arange(len(configs))
    width = 0.25

    bars1 = ax.bar(x - width, ttft_speedup, width, label='TTFT Speedup', color=colors[0], alpha=0.8)
    bars2 = ax.bar(x, tpot_speedup, width, label='TPOT Speedup', color=colors[1], alpha=0.8)
    bars3 = ax.bar(x + width, throughput_speedup, width, label='Throughput Speedup', color=colors[2], alpha=0.8)

    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='Baseline (1.0x)')

    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Speedup (relative to UVM Baseline)', fontsize=12)
    ax.set_title('Performance Speedup vs UVM Baseline', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=10)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}x',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speedup_comparison.png'), dpi=150)
    plt.close()
    print("Generated: speedup_comparison.png")


if __name__ == "__main__":
    print(f"Output directory: {output_dir}")
    print("Generating figures...")

    plot_ttft_comparison()
    plot_tpot_comparison()
    plot_throughput_comparison()
    plot_latency_summary()
    plot_benchmark_duration()
    plot_combined_overview()
    plot_speedup_comparison()

    print("\nAll figures generated successfully!")
