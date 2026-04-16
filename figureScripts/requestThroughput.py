import matplotlib.pyplot as plt
import numpy as np

# Data Extraction from throughput&e2eLatency.md (Request throughput (req/s))
# We average the multiple runs provided in the markdown file.

datasets = ["ShareGPT", "WildGPT"]
req_rates = [4, 8, 32]

# Parallax Data (Request throughput (req/s))
parallax_throughput = {
    "ShareGPT": {
        4: np.mean([0.13, 0.13]),
        8: np.mean([0.22, 0.24]),
        32: np.mean([0.88, 0.86, 0.92])
    },
    "WildGPT": {
        4: np.mean([0.13, 0.13, 0.15]),
        8: np.mean([0.23, 0.18, 0.23, 0.19]),
        32: np.mean([0.69, 0.67, 0.81, 0.79])
    }
}

# ADDIS Data (Request throughput (req/s))
addis_throughput = {
    "ShareGPT": {
        4: np.mean([0.23, 0.23]),
        8: np.mean([0.45, 0.45]),
        32: np.mean([1.68, 1.65])
    },
    "WildGPT": {
        4: np.mean([0.22, 0.22, 0.22]),
        8: np.mean([0.43, 0.44, 0.44]),
        32: np.mean([1.64, 0.95, 1.68, 1.64])
    }
}

fig, axes = plt.subplots(len(req_rates), len(datasets), figsize=(12, 15), sharey=False)

x = np.arange(1)  # Only one group per subplot
width = 0.35

for i, rate in enumerate(req_rates):
    for j, dataset in enumerate(datasets):
        ax = axes[i, j]

        p_val = parallax_throughput[dataset][rate]
        c_val = addis_throughput[dataset][rate]

        # Calculate percentage difference (Improvement of ADDIS over Parallax)
        percent_diff = ((c_val - p_val) / c_val) * 100

        rects1 = ax.bar(x - 0.2, [p_val], width, label='Parallax', color='skyblue', edgecolor='black', linewidth=1.2, alpha=0.8)
        rects2 = ax.bar(x + 0.2, [c_val], width, label='ADDIS', color='orange', edgecolor='black', linewidth=1.2, alpha=0.8)

        ax.text(x[0] - 0.2, p_val, f"↓{percent_diff:.1f}%", ha='center', va='bottom', color='black', fontsize=26)
        ax.set_title(f"{dataset} (req rate={rate})", fontsize=26)
        ax.set_xticks([])
        if j == 0:
            ax.set_ylabel("Throughput (req/s)", fontsize=26)
        ax.tick_params(axis='y', labelsize=26)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add a single legend
fig.legend([rects1, rects2], ["Parallax", "ADDIS"], loc='upper center', ncol=2, fontsize=26)

plt.tight_layout(rect=[0.01, 0.01, 1, 0.92], h_pad=4.0)
plt.savefig("request_throughput.png", bbox_inches='tight')
