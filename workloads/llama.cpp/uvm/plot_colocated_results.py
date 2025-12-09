#!/usr/bin/env python3
"""
Plot co-located evaluation results: llama.cpp (LC) + GNN Training (BE)
Compares Baseline UVM vs BPF Policy with Single baseline reference.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# OSDI-style formatting
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 20,
    'font.family': 'serif',
    'axes.labelsize': 24,
    'axes.titlesize': 24,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'legend.fontsize': 18,
    'figure.dpi': 150,
    'axes.linewidth': 1,
})

OUTPUT_DIR = Path(__file__).parent

# =============================================================================
# Raw Data from test-record-co-located.md
# =============================================================================

# Single Baseline
SINGLE_LLAMA = {
    'ttft_mean': 63.70,
    'ttft_median': 70.27,
    'ttft_p99': 98.48,
    'tpot_mean': 3.67,
    'tpot_median': 3.70,
    'tpot_p99': 3.91,
    'itl_mean': 3.74,
    'itl_p99': 3.93,
    'throughput': 251.83,
}

SINGLE_GNN = {
    'avg_epoch': 8.975,
    'median_epoch': 9.023,
}

# Co-located Baseline UVM
COLOC_UVM_LLAMA = {
    'ttft_mean': 428.24,
    'ttft_median': 323.23,
    'ttft_p99': 1391.61,
    'tpot_mean': 19.73,
    'tpot_median': 16.97,
    'tpot_p99': 56.06,
    'itl_mean': 17.79,
    'itl_p99': 116.63,
    'throughput': 39.73,
}

COLOC_UVM_GNN = {
    'avg_epoch': 23.230,
    'median_epoch': 21.657,
}

# Co-located with BPF Policy
COLOC_BPF_LLAMA = {
    'ttft_mean': 341.48,
    'ttft_median': 209.25,
    'ttft_p99': 1202.97,
    'tpot_mean': 10.86,
    'tpot_median': 9.72,
    'tpot_p99': 33.37,
    'itl_mean': 9.66,
    'itl_p99': 73.53,
    'throughput': 44.55,
}

COLOC_BPF_GNN = {
    'avg_epoch': 16.718,
    'median_epoch': 15.527,
}


def calc_improvement(baseline, improved):
    """Calculate improvement percentage (for latency: lower is better)."""
    return (baseline - improved) / baseline * 100


def calc_normalized(value, baseline):
    """Normalize value to baseline (baseline = 1.0)."""
    return value / baseline


def print_analysis():
    """Print detailed analysis with normalized values."""
    print("=" * 80)
    print("Co-located Performance Analysis")
    print("=" * 80)

    print("\n### llama.cpp (LC) - Normalized to Single Baseline ###\n")
    print(f"{'Metric':<15} {'Single':<12} {'UVM':<12} {'BPF':<12} {'UVM/Single':<12} {'BPF/Single':<12} {'BPF Impr':<10}")
    print("-" * 85)

    for metric in ['tpot_mean', 'tpot_p99', 'ttft_mean', 'ttft_p99']:
        single = SINGLE_LLAMA[metric]
        uvm = COLOC_UVM_LLAMA[metric]
        bpf = COLOC_BPF_LLAMA[metric]
        uvm_norm = calc_normalized(uvm, single)
        bpf_norm = calc_normalized(bpf, single)
        bpf_impr = calc_improvement(uvm, bpf)
        print(f"{metric:<15} {single:<12.2f} {uvm:<12.2f} {bpf:<12.2f} {uvm_norm:<12.2f}x {bpf_norm:<12.2f}x {bpf_impr:<10.1f}%")

    print("\n### GNN Training (BE) - Normalized to Single Baseline ###\n")
    print(f"{'Metric':<15} {'Single':<12} {'UVM':<12} {'BPF':<12} {'UVM/Single':<12} {'BPF/Single':<12} {'BPF Impr':<10}")
    print("-" * 85)

    single = SINGLE_GNN['avg_epoch']
    uvm = COLOC_UVM_GNN['avg_epoch']
    bpf = COLOC_BPF_GNN['avg_epoch']
    uvm_norm = calc_normalized(uvm, single)
    bpf_norm = calc_normalized(bpf, single)
    bpf_impr = calc_improvement(uvm, bpf)
    print(f"{'avg_epoch':<15} {single:<12.2f} {uvm:<12.2f} {bpf:<12.2f} {uvm_norm:<12.2f}x {bpf_norm:<12.2f}x {bpf_impr:<10.1f}%")

    print("\n" + "=" * 80)
    print("Summary for OSDI Claims:")
    print("=" * 80)

    # Key metrics
    tpot_mean_impr = calc_improvement(COLOC_UVM_LLAMA['tpot_mean'], COLOC_BPF_LLAMA['tpot_mean'])
    tpot_p99_impr = calc_improvement(COLOC_UVM_LLAMA['tpot_p99'], COLOC_BPF_LLAMA['tpot_p99'])
    ttft_mean_impr = calc_improvement(COLOC_UVM_LLAMA['ttft_mean'], COLOC_BPF_LLAMA['ttft_mean'])
    gnn_impr = calc_improvement(COLOC_UVM_GNN['avg_epoch'], COLOC_BPF_GNN['avg_epoch'])

    tpot_bpf_vs_single = calc_normalized(COLOC_BPF_LLAMA['tpot_mean'], SINGLE_LLAMA['tpot_mean'])
    ttft_bpf_vs_single = calc_normalized(COLOC_BPF_LLAMA['ttft_mean'], SINGLE_LLAMA['ttft_mean'])
    gnn_bpf_vs_single = calc_normalized(COLOC_BPF_GNN['avg_epoch'], SINGLE_GNN['avg_epoch'])

    print(f"\n1. TPOT Mean: {tpot_mean_impr:.1f}% improvement (BPF vs UVM)")
    print(f"   - BPF is {tpot_bpf_vs_single:.1f}x of single baseline (goal: ~1.0x)")

    print(f"\n2. TPOT P99: {tpot_p99_impr:.1f}% improvement")

    print(f"\n3. TTFT Mean: {ttft_mean_impr:.1f}% improvement")
    print(f"   - BPF is {ttft_bpf_vs_single:.1f}x of single baseline")

    print(f"\n4. GNN Epoch: {gnn_impr:.1f}% improvement")
    print(f"   - BPF is {gnn_bpf_vs_single:.1f}x of single baseline")

    return {
        'tpot_mean_impr': tpot_mean_impr,
        'tpot_p99_impr': tpot_p99_impr,
        'ttft_mean_impr': ttft_mean_impr,
        'gnn_impr': gnn_impr,
        'tpot_bpf_vs_single': tpot_bpf_vs_single,
        'gnn_bpf_vs_single': gnn_bpf_vs_single,
    }


def plot_main_figure():
    """Create 3-subplot figure: (a) TTFT, (b) TPOT, (c) GNN Epoch Time."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    colors_uvm = '#e74c3c'  # Red for baseline UVM
    colors_bpf = '#2ecc71'  # Green for BPF

    policies = ['Default\nUVM', 'eBPF\nUVM']
    x = np.arange(len(policies))
    width = 0.35

    # ==========================================================================
    # (a) TTFT (Mean & P99)
    # ==========================================================================
    ax1 = axes[0]

    uvm_vals = [COLOC_UVM_LLAMA['ttft_mean'], COLOC_UVM_LLAMA['ttft_p99']]
    bpf_vals = [COLOC_BPF_LLAMA['ttft_mean'], COLOC_BPF_LLAMA['ttft_p99']]
    single_vals = [SINGLE_LLAMA['ttft_mean'], SINGLE_LLAMA['ttft_p99']]

    # Group by policy: each policy has Mean and P99
    bars1 = ax1.bar(x - width/2, [uvm_vals[0], bpf_vals[0]], width, label='Mean',
                    color=colors_uvm, alpha=0.85, edgecolor='black', linewidth=1)
    bars2 = ax1.bar(x + width/2, [uvm_vals[1], bpf_vals[1]], width, label='P99',
                    color=colors_bpf, alpha=0.85, edgecolor='black', linewidth=1)

    # Add single baseline reference lines
    ax1.axhline(y=single_vals[0], color='#3498db', linestyle='--', linewidth=2.5,
                label='Single (Mean)')
    ax1.axhline(y=single_vals[0] * 2, color='#3498db', linestyle=':', linewidth=2,
                label='2× Single (Mean)')
    ax1.axhline(y=single_vals[1], color='#9b59b6', linestyle='--', linewidth=2.5,
                label='Single (P99)')
    ax1.axhline(y=single_vals[1] * 2, color='#9b59b6', linestyle=':', linewidth=2,
                label='2× Single (P99)')

    # Add value labels
    for bar in bars1:
        val = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, val + max(uvm_vals)*0.02, f'{val:.0f}',
                 ha='center', va='bottom', fontsize=14)
    for bar in bars2:
        val = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, val + max(uvm_vals)*0.02, f'{val:.0f}',
                 ha='center', va='bottom', fontsize=14)

    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('(a) TTFT')
    ax1.set_xticks(x)
    ax1.set_xticklabels(policies)
    ax1.set_ylim(0, max(uvm_vals) * 1.25)

    # ==========================================================================
    # (b) TPOT (Mean & P99)
    # ==========================================================================
    ax2 = axes[1]

    uvm_tpot = [COLOC_UVM_LLAMA['tpot_mean'], COLOC_UVM_LLAMA['tpot_p99']]
    bpf_tpot = [COLOC_BPF_LLAMA['tpot_mean'], COLOC_BPF_LLAMA['tpot_p99']]
    single_tpot = [SINGLE_LLAMA['tpot_mean'], SINGLE_LLAMA['tpot_p99']]

    bars3 = ax2.bar(x - width/2, [uvm_tpot[0], bpf_tpot[0]], width, label='Mean',
                    color=colors_uvm, alpha=0.85, edgecolor='black', linewidth=1)
    bars4 = ax2.bar(x + width/2, [uvm_tpot[1], bpf_tpot[1]], width, label='P99',
                    color=colors_bpf, alpha=0.85, edgecolor='black', linewidth=1)

    # Add single baseline reference lines
    ax2.axhline(y=single_tpot[0], color='#3498db', linestyle='--', linewidth=2.5,
                label='Single (Mean)')
    ax2.axhline(y=single_tpot[0] * 2, color='#3498db', linestyle=':', linewidth=2,
                label='2× Single (Mean)')
    ax2.axhline(y=single_tpot[1], color='#9b59b6', linestyle='--', linewidth=2.5,
                label='Single (P99)')
    ax2.axhline(y=single_tpot[1] * 2, color='#9b59b6', linestyle=':', linewidth=2,
                label='2× Single (P99)')

    # Add value labels
    for bar in bars3:
        val = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, val + max(uvm_tpot)*0.02, f'{val:.1f}',
                 ha='center', va='bottom', fontsize=14)
    for bar in bars4:
        val = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, val + max(uvm_tpot)*0.02, f'{val:.1f}',
                 ha='center', va='bottom', fontsize=14)

    ax2.set_ylabel('Latency (ms)')
    ax2.set_title('(b) TPOT')
    ax2.set_xticks(x)
    ax2.set_xticklabels(policies)
    ax2.set_ylim(0, max(uvm_tpot) * 1.25)

    # ==========================================================================
    # (c) GNN Training Epoch Time
    # ==========================================================================
    ax3 = axes[2]

    uvm_gnn = COLOC_UVM_GNN['avg_epoch']
    bpf_gnn = COLOC_BPF_GNN['avg_epoch']
    single_gnn = SINGLE_GNN['avg_epoch']

    bars5 = ax3.bar(x, [uvm_gnn, bpf_gnn], width * 1.5,
                    color=[colors_uvm, colors_bpf], alpha=0.85,
                    edgecolor='black', linewidth=1)

    # Single baseline line
    ax3.axhline(y=single_gnn, color='#3498db', linestyle='--', linewidth=2.5,
                label='Single')
    # 2x Single baseline (theoretical worst case for fair sharing)
    ax3.axhline(y=single_gnn * 2, color='#9b59b6', linestyle=':', linewidth=2,
                label='2× Single')

    # Add value labels
    for bar in bars5:
        val = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val:.1f}s',
                 ha='center', va='bottom', fontsize=14)

    ax3.set_ylabel('Epoch Time (s)')
    ax3.set_title('(c) GNN Training')
    ax3.set_xticks(x)
    ax3.set_xticklabels(policies)
    ax3.set_ylim(0, max(uvm_gnn, bpf_gnn) * 1.35)

    # Shared legend at the bottom
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles3, labels3 = ax3.get_legend_handles_labels()
    # Combine unique handles
    fig.legend(handles1 + handles3, labels1 + labels3,
               loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05),
               frameon=True, fontsize=20)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)

    # Save
    output_path = OUTPUT_DIR / 'fig_colocated_results'
    plt.savefig(f'{output_path}.pdf', bbox_inches='tight')
    plt.savefig(f'{output_path}.png', bbox_inches='tight', dpi=150)
    print(f"\nSaved: {output_path}.pdf/png")
    plt.close()


def plot_normalized_figure():
    """Plot normalized comparison (all values relative to single baseline)."""

    fig, ax = plt.subplots(figsize=(12, 6))

    metrics = ['TPOT\nMean', 'TPOT\nP99', 'TTFT\nMean', 'TTFT\nP99', 'GNN\nEpoch']

    # Normalized values (Single = 1.0)
    uvm_norm = [
        COLOC_UVM_LLAMA['tpot_mean'] / SINGLE_LLAMA['tpot_mean'],
        COLOC_UVM_LLAMA['tpot_p99'] / SINGLE_LLAMA['tpot_p99'],
        COLOC_UVM_LLAMA['ttft_mean'] / SINGLE_LLAMA['ttft_mean'],
        COLOC_UVM_LLAMA['ttft_p99'] / SINGLE_LLAMA['ttft_p99'],
        COLOC_UVM_GNN['avg_epoch'] / SINGLE_GNN['avg_epoch'],
    ]

    bpf_norm = [
        COLOC_BPF_LLAMA['tpot_mean'] / SINGLE_LLAMA['tpot_mean'],
        COLOC_BPF_LLAMA['tpot_p99'] / SINGLE_LLAMA['tpot_p99'],
        COLOC_BPF_LLAMA['ttft_mean'] / SINGLE_LLAMA['ttft_mean'],
        COLOC_BPF_LLAMA['ttft_p99'] / SINGLE_LLAMA['ttft_p99'],
        COLOC_BPF_GNN['avg_epoch'] / SINGLE_GNN['avg_epoch'],
    ]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, uvm_norm, width, label='Baseline UVM',
                   color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, bpf_norm, width, label='BPF Policy',
                   color='#2ecc71', alpha=0.85, edgecolor='black', linewidth=1)

    # Single baseline reference line
    ax.axhline(y=1.0, color='#3498db', linestyle='--', linewidth=2.5, label='Single Baseline (1.0×)')

    # Add value labels
    for bar, val in zip(bars1, uvm_norm):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.3, f'{val:.1f}×',
                ha='center', va='bottom', fontsize=12)
    for bar, val in zip(bars2, bpf_norm):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.3, f'{val:.1f}×',
                ha='center', va='bottom', fontsize=12)

    ax.set_ylabel('Normalized to Single Baseline')
    ax.set_title('Co-located Performance (Lower is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, max(uvm_norm) * 1.25)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = OUTPUT_DIR / 'fig_colocated_normalized'
    plt.savefig(f'{output_path}.pdf', bbox_inches='tight')
    plt.savefig(f'{output_path}.png', bbox_inches='tight', dpi=150)
    print(f"Saved: {output_path}.pdf/png")
    plt.close()


def main():
    stats = print_analysis()
    plot_main_figure()
    plot_normalized_figure()
    print("\nDone!")


if __name__ == "__main__":
    main()
