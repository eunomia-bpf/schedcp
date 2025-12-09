#!/usr/bin/env python3
"""
Visualize GCN benchmark results comparing prefetch and eBPF scheduler impact.

5 Lines:
1. No UVM (Baseline)
2. UVM without prefetch (Baseline)
3. UVM without prefetch + eBPF
4. UVM with prefetch (Baseline)
5. UVM with prefetch + eBPF
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_DIR = Path(__file__).parent

# Data directories
RESULT_DIRS = {
    # Baseline - No UVM
    "No UVM (Baseline)": BASE_DIR / "without-user-prefetch" / "result_no_uvm1",
    # Without prefetch
    "UVM (no prefetch)": BASE_DIR / "without-user-prefetch" / "result_uvm_baseline1",
    "UVM (no prefetch) eBPF": BASE_DIR / "without-user-prefetch" / "result_uvm_ebpf1",
    # With prefetch
    "UVM (prefetch)": BASE_DIR / "with-user-prefetch" / "result_uvm_baseline",
    "UVM (prefetch) eBPF": BASE_DIR / "with-user-prefetch" / "result_uvm_ebpf",
}


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

    # Create figure - adjusted height (1.3x again)
    fig, ax = plt.subplots(figsize=(14, 8.5))

    # Style definitions - thicker lines, larger markers
    styles = {
        "No UVM (Baseline)": {
            "color": "#2ecc71", "marker": "o", "linestyle": "-", "linewidth": 4
        },
        "UVM (no prefetch)": {
            "color": "#e74c3c", "marker": "s", "linestyle": "-", "linewidth": 3.5
        },
        "UVM (no prefetch) eBPF": {
            "color": "#e74c3c", "marker": "s", "linestyle": "--", "linewidth": 3.5
        },
        "UVM (prefetch)": {
            "color": "#3498db", "marker": "^", "linestyle": "-", "linewidth": 3.5
        },
        "UVM (prefetch) eBPF": {
            "color": "#3498db", "marker": "^", "linestyle": "--", "linewidth": 3.5
        },
    }

    # Plot order for legend
    plot_order = [
        "No UVM (Baseline)",
        "UVM (no prefetch)",
        "UVM (no prefetch) eBPF",
        "UVM (prefetch)",
        "UVM (prefetch) eBPF",
    ]

    # Plot each condition
    for name in plot_order:
        data = plot_data.get(name, {"nodes": [], "times": []})
        if data["nodes"]:
            nodes_m = [n / 1e6 for n in data["nodes"]]
            style = styles[name]
            ax.plot(nodes_m, data["times"],
                    marker=style["marker"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    linewidth=style["linewidth"],
                    markersize=14,
                    label=name)

    # Add vertical line for GPU memory limit
    ax.axvline(x=8, color='gray', linestyle=':', alpha=0.8, linewidth=2)
    ax.text(8.3, ax.get_ylim()[1] * 0.92, 'GPU Memory\nLimit (~8M)',
            fontsize=24, color='gray', va='top')

    # Labels - enlarged fonts (3x), no title
    ax.set_xlabel("Number of Nodes (Millions)", fontsize=36)
    ax.set_ylabel("Epoch Time (seconds)", fontsize=36)

    # Legend at bottom - below x-axis label
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
              fontsize=22, framealpha=0.95, ncol=3)

    # Enlarge tick labels
    ax.tick_params(axis='both', which='major', labelsize=28)

    # Grid
    ax.grid(True, alpha=0.3)

    # Axis limits
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    # Save figure
    output_path = BASE_DIR / "uvm_benchmark_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved figure to: {output_path}")

    pdf_path = BASE_DIR / "uvm_benchmark_comparison.pdf"
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved PDF to: {pdf_path}")

    # Print comprehensive summary table
    print("\n" + "="*100)
    print("COMPREHENSIVE SUMMARY TABLE")
    print("="*100)

    header = f"{'Nodes':<8} {'No UVM':<10} {'UVM':<12} {'UVM+eBPF':<12} {'UVM(pf)':<12} {'UVM(pf)+eBPF':<14} {'Prefetch':<10} {'eBPF(pf)':<10}"
    print(header)
    print(f"{'':8} {'':10} {'(no pf)':12} {'(no pf)':12} {'':12} {'':14} {'Speedup':10} {'Speedup':10}")
    print("-"*100)

    all_nodes = set()
    for data in plot_data.values():
        all_nodes.update(data["nodes"])

    for n in sorted(all_nodes):
        row = f"{n/1e6:.0f}M".ljust(8)

        # No UVM
        if n in all_results.get("No UVM (Baseline)", {}):
            t = all_results["No UVM (Baseline)"][n]["avg_epoch_time_s"]
            row += f"{t:.2f}s".ljust(10)
        else:
            row += "OOM".ljust(10)

        # UVM no prefetch
        uvm_nopf = all_results.get("UVM (no prefetch)", {})
        if n in uvm_nopf:
            t = uvm_nopf[n]["avg_epoch_time_s"]
            row += f"{t:.2f}s".ljust(12)
        else:
            row += "-".ljust(12)

        # UVM no prefetch + eBPF
        uvm_nopf_ebpf = all_results.get("UVM (no prefetch) eBPF", {})
        if n in uvm_nopf_ebpf:
            t = uvm_nopf_ebpf[n]["avg_epoch_time_s"]
            row += f"{t:.2f}s".ljust(12)
        else:
            row += "-".ljust(12)

        # UVM prefetch
        uvm_pf = all_results.get("UVM (prefetch)", {})
        if n in uvm_pf:
            t = uvm_pf[n]["avg_epoch_time_s"]
            row += f"{t:.2f}s".ljust(12)
        else:
            row += "-".ljust(12)

        # UVM prefetch + eBPF
        uvm_pf_ebpf = all_results.get("UVM (prefetch) eBPF", {})
        if n in uvm_pf_ebpf:
            t = uvm_pf_ebpf[n]["avg_epoch_time_s"]
            row += f"{t:.2f}s".ljust(14)
        else:
            row += "-".ljust(14)

        # Prefetch speedup (UVM no prefetch vs UVM prefetch)
        if n in uvm_nopf and n in uvm_pf:
            speedup = uvm_nopf[n]["avg_epoch_time_s"] / uvm_pf[n]["avg_epoch_time_s"]
            row += f"{speedup:.2f}x".ljust(10)
        else:
            row += "-".ljust(10)

        # eBPF speedup (with prefetch)
        if n in uvm_pf and n in uvm_pf_ebpf:
            speedup = uvm_pf[n]["avg_epoch_time_s"] / uvm_pf_ebpf[n]["avg_epoch_time_s"]
            row += f"{speedup:.2f}x".ljust(10)
        else:
            row += "-".ljust(10)

        print(row)

    print("="*100)

    # Key insights
    print("\nKEY INSIGHTS:")
    print("-" * 50)

    # Calculate average speedups
    prefetch_speedups = []
    ebpf_nopf_speedups = []
    ebpf_pf_speedups = []

    for n in sorted(all_nodes):
        uvm_nopf = all_results.get("UVM (no prefetch)", {})
        uvm_nopf_ebpf = all_results.get("UVM (no prefetch) eBPF", {})
        uvm_pf = all_results.get("UVM (prefetch)", {})
        uvm_pf_ebpf = all_results.get("UVM (prefetch) eBPF", {})

        if n in uvm_nopf and n in uvm_pf:
            prefetch_speedups.append(uvm_nopf[n]["avg_epoch_time_s"] / uvm_pf[n]["avg_epoch_time_s"])

        if n in uvm_nopf and n in uvm_nopf_ebpf:
            ebpf_nopf_speedups.append(uvm_nopf[n]["avg_epoch_time_s"] / uvm_nopf_ebpf[n]["avg_epoch_time_s"])

        if n in uvm_pf and n in uvm_pf_ebpf:
            ebpf_pf_speedups.append(uvm_pf[n]["avg_epoch_time_s"] / uvm_pf_ebpf[n]["avg_epoch_time_s"])

    if prefetch_speedups:
        print(f"1. Prefetch speedup (avg): {sum(prefetch_speedups)/len(prefetch_speedups):.2f}x")
    if ebpf_nopf_speedups:
        print(f"2. eBPF speedup without prefetch (avg): {sum(ebpf_nopf_speedups)/len(ebpf_nopf_speedups):.2f}x")
    if ebpf_pf_speedups:
        print(f"3. eBPF speedup with prefetch (avg): {sum(ebpf_pf_speedups)/len(ebpf_pf_speedups):.2f}x")

    plt.show()


if __name__ == "__main__":
    main()
