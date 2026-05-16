import matplotlib.pyplot as plt
import numpy as np

models = ["Qwen3-8B", "Qwen3-14B"]
datasets = ["ShareGPT", "WildGPT"]
req_rates = [4, 8, 32]

parallax_throughput = {
    "Qwen3-8B": {
        "ShareGPT": {4: 0.16, 8: 0.28, 32: 0.60},
        "WildGPT":  {4: 0.16, 8: 0.29, 32: 0.67}
    },
    "Qwen3-14B": {
        "ShareGPT": {4: 0.12, 8: 0.21, 32: 0.45},
        "WildGPT":  {4: 0.12, 8: 0.22, 32: 0.51}
    }
}

addis_throughput = {
    "Qwen3-8B": {
        "ShareGPT": {4: 0.20, 8: 0.38, 32: 1.42},
        "WildGPT":  {4: 0.20, 8: 0.37, 32: 1.39}
    },
    "Qwen3-14B": {
        "ShareGPT": {4: 0.14, 8: 0.26, 32: 0.96},
        "WildGPT":  {4: 0.14, 8: 0.25, 32: 0.94}
    }
}

fig, axes = plt.subplots(len(models), len(datasets), figsize=(18, 14), sharey=False)

x = np.arange(len(req_rates))
width = 0.35
x_offset = 0.2  # Increase this value to move labels further left

for i, model in enumerate(models):
    for j, dataset in enumerate(datasets):
        ax = axes[i, j]

        p_vals = [parallax_throughput[model][dataset][r] for r in req_rates]
        c_vals = [addis_throughput[model][dataset][r] for r in req_rates]

        rects1 = ax.bar(x - width/2, p_vals, width, label='Parallax', color='skyblue', edgecolor='black', linewidth=1.2, alpha=0.8)
        rects2 = ax.bar(x + width/2, c_vals, width, label='ADDIS', color='orange', edgecolor='black', linewidth=1.2, alpha=0.8)

        # Add percentage labels
        for k in range(len(req_rates)):
            p_val = p_vals[k]
            c_val = c_vals[k]

            percent_decrease = ((c_val - p_val) / c_val) * 100

            ax.text(x[k] - width/2 - x_offset, p_val + (max(c_vals)*0.02),
                    f"↓{percent_decrease:.0f}%",
                    ha='center', va='bottom', color='black', fontsize=28)

        ax.set_xlim(-0.8, len(req_rates) - 0.2)

        ax.set_xticks(x)
        ax.set_xticklabels([])

        if i == 0:
            ax.set_title(f"{dataset}", fontsize=30, pad=20)

        if j == 0:
            ax.set_ylabel("Throughput (req/s)", fontsize=28)

        if j == len(datasets) - 1:
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(f"{model}", rotation=90, fontsize=30)

        if i == len(models) - 1:
            ax.set_xticks(x)
            ax.set_xticklabels([f"Rate={r}" for r in req_rates], fontsize=28)


        ax.tick_params(axis='y', labelsize=28)

fig.legend([rects1, rects2], ["Parallax", "ADDIS"], loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=28)

# Adjust layout: 'left' parameter in rect ensures the whole subplot block shifts right
plt.tight_layout(rect=[0.05, 0.03, 0.95, 0.9], h_pad=5.0, w_pad=4.0)

plt.savefig("request_throughput.png", bbox_inches='tight')
print("Saved request_throughput.png")