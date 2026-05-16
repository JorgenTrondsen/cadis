import re
import os
from collections import defaultdict

def extract_and_average():
    input_path = '/home/jorg1/cadis/benchmarks/throughput.md'
    output_path = '/home/jorg1/cadis/benchmarks/averagedData.md'

    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    with open(input_path, 'r') as f:
        lines = f.readlines()

    current_model = ""
    current_system = ""
    current_dataset = ""
    current_req_rate = ""

    results = []
    current_result = None

    metrics_to_extract = [
        "Request throughput (req/s)",
        "Mean E2EL (ms)",
        "P95 E2EL (ms)",
        "P96 E2EL (ms)",
        "P97 E2EL (ms)",
        "P98 E2EL (ms)",
        "P99 E2EL (ms)",
        "P100 E2EL (ms)"
    ]

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("## "):
            current_model = line.replace("## ", "").strip()
        elif line.startswith("### "):
            current_system = line.replace("### ", "").strip()
        elif line.startswith("#### "):
            current_dataset = line.replace("#### ", "").strip()
        elif line.startswith("##### "):
            current_req_rate = line.replace("##### ", "").strip()

        if "============ Serving Benchmark Result ============" in line:
            current_result = {
                "model": current_model,
                "system": current_system,
                "dataset": current_dataset,
                "req_rate": current_req_rate,
                "metrics": {}
            }

        if current_result is not None:
            for metric in metrics_to_extract:
                if metric in line:
                    # Match both integer and decimal values
                    match = re.search(rf"{re.escape(metric)}:\s+([\d.]+)", line)
                    if match:
                        current_result["metrics"][metric] = float(match.group(1))

            if "==================================================" in line:
                results.append(current_result)
                current_result = None

    # Group by model, system, dataset, req_rate
    grouped_data = defaultdict(list)
    for res in results:
        key = (res["model"], res["system"], res["dataset"], res["req_rate"])
        grouped_data[key].append(res["metrics"])

    # Prepare output
    output_lines = ["# Averaged Throughput Benchmarks\n"]

    # Custom sorting function for req-rate
    def get_req_rate_val(s):
        match = re.search(r'(\d+)', s)
        return int(match.group(1)) if match else 0

    sorted_keys = sorted(grouped_data.keys(), key=lambda x: (x[0], x[1], x[2], get_req_rate_val(x[3])))

    last_model = None
    last_system = None
    last_dataset = None

    for key in sorted_keys:
        model, system, dataset, req_rate = key

        if model != last_model:
            output_lines.append(f"\n## {model}")
            last_model = model
            last_system = None
            last_dataset = None

        if system != last_system:
            output_lines.append(f"\n### {system}")
            last_system = system
            last_dataset = None

        if dataset != last_dataset:
            output_lines.append(f"\n#### {dataset}")
            last_dataset = dataset

        output_lines.append(f"\n##### {req_rate}")

        metrics_list = grouped_data[key]
        num_samples = len(metrics_list)

        for metric in metrics_to_extract:
            values = [m[metric] for m in metrics_list if metric in m]
            if values:
                avg = sum(values) / len(values)
                output_lines.append(f"{metric}: {avg:.2f}")
            else:
                output_lines.append(f"{metric}: N/A")

    with open(output_path, 'w') as f:
        f.write("\n".join(output_lines) + "\n")

    print(f"Averaged data written to {output_path}")

if __name__ == "__main__":
    extract_and_average()
