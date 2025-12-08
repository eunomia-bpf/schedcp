#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 28})

configs = [
    "ncmoe=64",
    "ncmoe=32",
    "UVM only",
    "UVM + user hint",
    "UVM + eBPF",
]

pp512 = [245.63, 260.14, 238.48, 144.00, 229.67]
tg128 = [16.34, 18.18, 7.72, 49.31, 86.89]

x = np.arange(len(configs))
width = 0.35

fig, ax1 = plt.subplots(figsize=(14, 8))

# left axis: pp512 (blue)
rects1 = ax1.bar(
    x - width/2, pp512, width,
    label="prefill tokens/s",
    color="tab:blue",
)
ax1.set_ylabel("tokens/s (pp512)")
ax1.set_xticks(x)
ax1.set_xticklabels(configs, rotation=20, ha="right")

# right axis: tg128 (orange)
ax2 = ax1.twinx()
rects2 = ax2.bar(
    x + width/2, tg128, width,
    label="decoding tokens/s",
    color="tab:orange",
)
ax2.set_ylabel("tokens/s (tg128)")

fig.suptitle("gpt-oss-120B MXFP4 MoE - throughput vs config")

handles = [rects1, rects2]
labels = ["pp512 tokens/s", "tg128 tokens/s"]
fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))

fig.tight_layout()
fig.subplots_adjust(bottom=0.25)
fig.savefig("llama_uvm_combined_color.pdf")
print("saved to llama_uvm_combined_color.pdf")
