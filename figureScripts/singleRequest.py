import matplotlib.pyplot as plt

# Data Extraction from singleRequest.md
latencies = [0, 100, 200, 400, 800]

parallax_throughput = [27.22, 11.49, 7.30, 4.22, 2.29]
addis_throughput = [28.82, 11.80, 7.41, 4.25, 2.30]

plt.figure(figsize=(10, 4))

plt.plot(latencies, parallax_throughput, marker='o', label='Parallax', color='skyblue', linestyle='-', markeredgecolor='black')
plt.plot(latencies, ADDIS_throughput, marker='s', label='ADDIS', color='orange', linestyle='-', markeredgecolor='black')

# Add numbers next to each point
for i, (latency, p_val, c_val) in enumerate(zip(latencies, parallax_throughput, addis_throughput)):
    # Parallax numbers below the point
    plt.text(latency - 20, p_val - 0.3, f'{p_val:.2f}', ha='center', va='top', fontsize=10, color='darkblue')
    # ADDIS numbers above the point
    plt.text(latency + 25, c_val - 0.1, f'{c_val:.2f}', ha='center', va='bottom', fontsize=10, color='darkorange')

plt.xlabel("Total Cluster Latency (ms)", fontsize=14)
plt.ylabel("Single Request Throughput (tok/s)", fontsize=14)

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=14)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig("single_request.png")