import matplotlib.pyplot as plt
import numpy as np

# Data Extraction from throughput&e2eLatency.md
# Units are converted from milliseconds to seconds (/1000)
# We show: Mean, P95, P96, P97, P98, P99, P100 (Excluding Median)

datasets = ["ShareGPT", "WildGPT"]
req_rates = [4, 8, 32]
metrics = ["Mean", "P95", "P96", "P97", "P98", "P99", "P100"]

# Parallax Data
parallax_data = {
    "ShareGPT": {
        4: [30.717, 30.796, 30.796, 30.796, 30.797, 30.798, 30.800], # Mean, P95-P100 (Averaged)
        8: [29.603, 32.674, 33.073, 33.472, 33.871, 34.271, 34.670],
        32: [33.143, 33.182, 33.182, 33.384, 34.266, 35.148, 36.035]
    },
    "WildGPT": {
        4: [26.013, 29.440, 29.442, 29.443, 29.445, 29.446, 29.448],
        8: [30.823, 36.429, 36.925, 37.420, 37.915, 38.411, 38.906],
        32: [37.710, 38.147, 38.149, 38.528, 40.226, 41.924, 43.622]
    }
}

# CADIS Data
cadis_data = {
    "ShareGPT": {
        4: [17.687, 17.691, 17.691, 17.691, 17.691, 17.691, 17.691],
        8: [17.892, 17.895, 17.895, 17.895, 17.895, 17.895, 17.895],
        32: [18.947, 19.203, 19.203, 19.203, 19.204, 19.205, 19.206]
    },
    "WildGPT": {
        4: [16.699, 18.194, 18.194, 18.194, 18.194, 18.194, 18.194],
        8: [17.384, 18.264, 18.264, 18.264, 18.264, 18.265, 18.265],
        32: [22.590, 22.925, 22.925, 22.925, 22.925, 22.925, 22.925]
    }
}

fig, axes = plt.subplots(len(req_rates), len(datasets), figsize=(12, 15), sharey=True)

x = np.arange(len(metrics))

for i, rate in enumerate(req_rates):
    for j, dataset in enumerate(datasets):
        ax = axes[i, j]

        p_vals = parallax_data[dataset][rate]
        c_vals = cadis_data[dataset][rate]
        line1, = ax.plot(x, p_vals, marker='o', label='Parallax', color='skyblue', linestyle='-', markeredgecolor='black', linewidth=3.5)
        line2, = ax.plot(x, c_vals, marker='s', label='CADIS', color='orange', linestyle='-', markeredgecolor='black', linewidth=3.5)

        ax.set_title(f"{dataset} (req rate={rate})", fontsize=26)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=23, rotation=45, ha='right', rotation_mode='anchor')
        if j == 0:
            ax.set_ylabel("Latency (s)", fontsize=26)
        if i == len(req_rates) - 1:
            ax.set_xlabel("Tail Latency Metrics", fontsize=26)
        ax.tick_params(axis='y', labelsize=26)
        ax.grid(True, linestyle='--', alpha=0.6)

# Add a single legend for the entire figure in the layout, often near the title
fig.legend([line1, line2], ["Parallax", "CADIS"], loc='upper center', ncol=2, fontsize=26)

# Use h_pad to increase vertical spacing and rect[0] to add room for the y-label
plt.tight_layout(rect=[0.05, 0.01, 1, 0.92], h_pad=4.0)
plt.savefig("end2end_latency.png", bbox_inches='tight')
