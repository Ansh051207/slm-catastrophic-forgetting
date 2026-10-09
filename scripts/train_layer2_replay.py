import os
import gc
import torch
from datasets import load_dataset, concatenate_datasets
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
SEED = 3407
MAX_SEQ_LENGTH = 1024

OUTPUT_DIR = "/content/drive/MyDrive/catastrophic_slm/adapters/replay_lora"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EOS = "<|endoftext|>"

# 1. 80% GSM8K (480 samples)
gsm_ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=SEED)
gsm_selected = gsm_ds.select(range(480))

def fmt_gsm(ex):
    return {"text": f"Question: {ex['question']}\nAnswer: {ex['answer']}{EOS}"}

gsm_formatted = gsm_selected.map(fmt_gsm, remove_columns=gsm_selected.column_names)

# 2. 20% Alpaca Replay (120 samples)
alpaca_ds = load_dataset("tatsu-lab/alpaca", split="train").shuffle(seed=SEED)
alpaca_selected = alpaca_ds.select(range(120))

def fmt_alpaca(ex):
    inp = f"\nInput: {ex['input']}" if ex.get("input") and len(ex["input"].strip()) > 0 else ""
    return {"text": f"Instruction: {ex['instruction']}{inp}\nResponse: {ex['output']}{EOS}"}

alpaca_formatted = alpaca_selected.map(fmt_alpaca, remove_columns=alpaca_selected.column_names)

# 3. Concatenate and Shuffle
mixed_dataset = concatenate_datasets([gsm_formatted, alpaca_formatted]).shuffle(seed=SEED)
print(f"Total training dataset size: {len(mixed_dataset)} (480 Math + 120 General Replay)")

# 4. Model Setup
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=False
)

# 5. LoRA Configuration (r=16, alpha=16)
LORA_R = 16
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_R,
    lora_dropout=0.0,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=SEED
)

# 6. SFTTrainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=mixed_dataset,
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=300,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        optim="adamw_8bit",
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        seed=SEED,
        report_to="none"
    )
)

print("Starting Layer 2 Experience Replay Training...")
trainer.train()

final_path = os.path.join(OUTPUT_DIR, "final")
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)
print(f"Replay training complete. Adapter saved to: {final_path}")
