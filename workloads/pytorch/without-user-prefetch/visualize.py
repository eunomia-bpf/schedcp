#!/usr/bin/env python3
"""
Visualize GCN benchmark results comparing:
- No UVM (PyTorch default)
- UVM Baseline (without prefetch)
- UVM + eBPF scheduler
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt

# Data directories
BASE_DIR = Path(__file__).parent
RESULT_DIRS = {
    "No UVM": BASE_DIR / "result_no_uvm1",
    "UVM Baseline": BASE_DIR / "result_uvm_baseline1",
    "UVM + eBPF": BASE_DIR / "result_uvm_ebpf1",
}

# GPU memory for reference
GPU_MEMORY_GB = 31.36


def load_results(result_dir: Path) -> dict:
    """Load all JSON results from a directory."""
    results = {}
    if not result_dir.exists():
        print(f"Warning: {result_dir} does not exist")
        return results

    for json_file in result_dir.glob("*.json"):
        try:
            with open(json_file) as f:
                data = json.load(f)
                num_nodes = data.get("num_nodes", int(json_file.stem))
                results[num_nodes] = data
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load {json_file}: {e}")

    return results


def main():
    # Load all results
    all_results = {}
    for name, result_dir in RESULT_DIRS.items():
        all_results[name] = load_results(result_dir)
        print(f"Loaded {len(all_results[name])} results from {name}")

    # Collect data for plotting
    plot_data = {}
    for name, results in all_results.items():
        nodes = sorted(results.keys())
        times = [results[n]["avg_epoch_time_s"] for n in nodes]
        plot_data[name] = {"nodes": nodes, "times": times}

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color scheme and markers
    styles = {
        "No UVM": {"color": "#2ecc71", "marker": "o", "linestyle": "-"},       # Green solid
        "UVM Baseline": {"color": "#e74c3c", "marker": "s", "linestyle": "-"}, # Red solid
        "UVM + eBPF": {"color": "#3498db", "marker": "^", "linestyle": "-"},   # Blue solid
    }

    # Plot each condition
    for name, data in plot_data.items():
        if data["nodes"]:
            nodes_m = [n / 1e6 for n in data["nodes"]]
            style = styles[name]
            ax.plot(nodes_m, data["times"],
                    marker=style["marker"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=2.5,
                    markersize=10,
                    label=name)

    # Add vertical line for GPU memory limit (OOM threshold without UVM)
    ax.axvline(x=8, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.text(8.2, ax.get_ylim()[1] * 0.95, 'OOM\n(No UVM)', fontsize=10, color='gray', va='top')

    # Labels and title
    ax.set_xlabel("Number of Nodes (Millions)", fontsize=14)
    ax.set_ylabel("Epoch Time (seconds)", fontsize=14)
    ax.set_title("GCN Training: UVM Oversubscription Performance\n(Without cudaMemPrefetchAsync)", fontsize=14)

    # Legend
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)

    # Grid
    ax.grid(True, alpha=0.3)

    # Axis limits
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # Tight layout
    plt.tight_layout()

    # Save figure
    output_path = BASE_DIR / "benchmark_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved figure to: {output_path}")

    # Also save as PDF
    pdf_path = BASE_DIR / "benchmark_results.pdf"
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved PDF to: {pdf_path}")

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Nodes':<12} {'No UVM':<15} {'UVM Baseline':<15} {'UVM+eBPF':<15} {'eBPF Speedup':<12}")
    print("-"*80)

    all_nodes = set()
    for data in plot_data.values():
        all_nodes.update(data["nodes"])

    for n in sorted(all_nodes):
        row = f"{n/1e6:.0f}M".ljust(12)

        for name in ["No UVM", "UVM Baseline", "UVM + eBPF"]:
            results = all_results.get(name, {})
            if n in results:
                time = results[n]["avg_epoch_time_s"]
                row += f"{time:.2f}s".ljust(15)
            else:
                row += "OOM".ljust(15)

        # Calculate eBPF speedup
        uvm_results = all_results.get("UVM Baseline", {})
        ebpf_results = all_results.get("UVM + eBPF", {})
        if n in uvm_results and n in ebpf_results:
            speedup = uvm_results[n]["avg_epoch_time_s"] / ebpf_results[n]["avg_epoch_time_s"]
            row += f"{speedup:.2f}×"
        else:
            row += "-"

        print(row)

    print("="*80)

    plt.show()


if __name__ == "__main__":
    main()
