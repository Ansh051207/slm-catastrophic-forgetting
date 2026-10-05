import os
import glob
import json
import pandas as pd

def extract_benchmark_scores(results_dir):
    json_pattern = os.path.join(results_dir, "**/*.json")
    files = glob.glob(json_pattern, recursive=True)
    if not files:
        files = glob.glob(os.path.join(results_dir, "*.json"))
        
    if not files:
        raise FileNotFoundError(f"No result JSON files found in {results_dir}")

    latest_file = max(files, key=os.path.getmtime)
    print(f"Reading evaluation results from: {latest_file}")
    
    with open(latest_file, "r") as f:
        data = json.load(f)
        
    results = data.get("results", {})
    records = []
    
    benchmark_map = {
        "gsm8k": "GSM8K (Target Task - Math)",
        "mmlu": "MMLU (Broad Academia)",
        "arc_challenge": "ARC-Challenge (Science Reasoning)",
        "hellaswag": "HellaSwag (Commonsense Inference)",
        "winogrande": "Winogrande (Language Understanding)"
    }
    
    for task_key, task_label in benchmark_map.items():
        if task_key in results:
            task_dict = results[task_key]
            raw_acc = task_dict.get("exact_match,none", 
                      task_dict.get("acc_norm,none", 
                      task_dict.get("acc,none", 0.0)))
            
            if isinstance(raw_acc, (int, float)):
                acc_pct = round(raw_acc * 100, 2)
            else:
                acc_pct = 0.0
                
            records.append({
                "Benchmark": task_label,
                "Task_ID": task_key,
                "Baseline_Accuracy (%)": acc_pct
            })
            
    return pd.DataFrame(records)

if __name__ == "__main__":
    eval_folder = "/content/drive/MyDrive/catastrophic_slm/raw_evals/eval_base"
    df = extract_benchmark_scores(eval_folder)
    
    print("\n" + "="*50)
    print("      STEP 0: BASELINE EVALUATION SCORES")
    print("="*50)
    print(df.to_string(index=False))
    print("="*50)
    
    output_path = "results_summary/step0_baseline.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved summary table to: {output_path}")
