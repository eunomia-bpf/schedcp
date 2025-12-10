#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 28})

configs = [
    "ncmoe=64",
    "ncmoe=32",
    "UVM only",
    "UVM user hint",
    "UVM eBPF",
]

pp512 = [245.63, 260.14, 238.48, 144.00, 229.67]
tg128 = [16.34, 18.18, 7.72, 49.31, 86.89]

x = np.arange(len(configs))
width = 0.6

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# Left plot: Prefill throughput (pp512)
rects1 = ax1.bar(x, pp512, width, color="tab:blue", alpha=0.8)
ax1.set_ylabel("tokens/s", fontsize=28)
ax1.set_title("Prefill Throughput (pp512)", fontsize=28)
ax1.set_xticks(x)
ax1.set_xticklabels(configs, rotation=20, ha="right", fontsize=22)
ax1.tick_params(axis='y', labelsize=24)
ax1.grid(axis='y', alpha=0.3)

# Right plot: Decode throughput (tg128)
rects2 = ax2.bar(x, tg128, width, color="tab:orange", alpha=0.8)
ax2.set_ylabel("tokens/s", fontsize=28)
ax2.set_title("Decode Throughput (tg128)", fontsize=28)
ax2.set_xticks(x)
ax2.set_xticklabels(configs, rotation=20, ha="right", fontsize=22)
ax2.tick_params(axis='y', labelsize=24)
ax2.grid(axis='y', alpha=0.3)

fig.tight_layout()
fig.savefig("llama_uvm_combined_color.pdf")
print("saved to llama_uvm_combined_color.pdf")
