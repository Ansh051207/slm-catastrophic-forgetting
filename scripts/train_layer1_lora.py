import os
import gc
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
SEED = 3407
MAX_SEQ_LENGTH = 1024

OUTPUT_DIR = "/content/drive/MyDrive/catastrophic_slm/adapters/gsm8k_lora"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load GSM8K Dataset
ds = load_dataset("openai/gsm8k", "main")
EOS = "<|endoftext|>"

def fmt(ex):
    return {"text": f"Question: {ex['question']}\nAnswer: {ex['answer']}{EOS}"}

train_ds = ds["train"].shuffle(seed=SEED)
train_ds = train_ds.select(range(600)).map(fmt, remove_columns=train_ds.column_names)

# 2. Initialize Model with Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=False
)

# 3. LoRA Configuration (r=16, alpha=16)
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

# 4. SFTTrainer with intermediate checkpoints
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
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

trainer.train()
final_path = os.path.join(OUTPUT_DIR, "final")
model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)
print(f"Training completed. Weights saved to: {final_path}")
