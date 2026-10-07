import os
import glob
import json
import pandas as pd

BENCHMARK_MAP = {
    "gsm8k": "GSM8K (Target - Math)",
    "mmlu": "MMLU (Broad Academia)",
    "arc_challenge": "ARC-Challenge (Science Reasoning)",
    "hellaswag": "HellaSwag (Commonsense)",
    "winogrande": "Winogrande (Language Understanding)"
}

def extract_scores(results_dir):
    files = glob.glob(os.path.join(results_dir, "**/*.json"), recursive=True)
    if not files:
        files = glob.glob(os.path.join(results_dir, "*.json"))
    if not files:
        raise FileNotFoundError(f"No result JSON files found in {results_dir}")

    latest_file = max(files, key=os.path.getmtime)
    with open(latest_file, "r") as f:
        data = json.load(f)

    results = data.get("results", {})
    scores = {}
    for task_key in BENCHMARK_MAP.keys():
        if task_key in results:
            task_dict = results[task_key]
            raw_acc = task_dict.get("exact_match,none",
                      task_dict.get("exact_match,flexible-extract",
                      task_dict.get("acc_norm,none",
                      task_dict.get("acc,none", 0.0))))
            scores[task_key] = round(float(raw_acc) * 100, 2)
    return scores

if __name__ == "__main__":
    base_scores = extract_scores("/content/drive/MyDrive/catastrophic_slm/raw_evals/eval_base")
    step100_scores = extract_scores("/content/drive/MyDrive/catastrophic_slm/raw_evals/eval_step100")
    step300_scores = extract_scores("/content/drive/MyDrive/catastrophic_slm/raw_evals/eval_step300_final")

    records = []
    for task_key, label in BENCHMARK_MAP.items():
        s_base = base_scores.get(task_key, 0.0)
        s_100 = step100_scores.get(task_key, 0.0)
        s_300 = step300_scores.get(task_key, 0.0)

        records.append({
            "Benchmark": label,
            "Task_ID": task_key,
            "Base (%)": s_base,
            "Step100 (%)": s_100,
            "Δ_100": round(s_100 - s_base, 2),
            "CRR_100": round(s_100 / s_base, 3) if s_base > 0 else 1.0,
            "Step300 (%)": s_300,
            "Δ_300": round(s_300 - s_base, 2),
            "CRR_300": round(s_300 / s_base, 3) if s_base > 0 else 1.0
        })

    summary_df = pd.DataFrame(records)
    out_csv = "results_summary/layer1_dynamics.csv"
    summary_df.to_csv(out_csv, index=False)
    print("Parsed dynamics table successfully.")
