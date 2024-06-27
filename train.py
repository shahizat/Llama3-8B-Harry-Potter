# 1. Importing and configurations 
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer
import wandb
import gc
from huggingface_hub import login
login(token="YOUR_HUGGING_FACE_KEY")

new_model = "Llama-3-HarryPotter"
max_seq_length = 2048

base_model = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

chat_prompt = """
### Instruction:
{}


### Input:
{}


### Response:
{}"""


EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
   instruction = ""
   inputs       = examples["question"]
   outputs      = examples["answer"]
   texts = []
   for input, output in zip(inputs, outputs):
       # Must add EOS_TOKEN, otherwise your generation will go on forever!
       text = chat_prompt.format(instruction, input, output) + EOS_TOKEN
       texts.append(text)
   return { "text" : texts, }

train_dataset = load_dataset('json', data_files='train.jsonl', split='train')
eval_dataset = load_dataset('json', data_files='val.jsonl', split='train')

train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)

compute_dtype = getattr(torch, "bfloat16")

# 2. Load Llama3 model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    use_cache=False,
    trust_remote_code=True
)
model.config.use_cache = True 
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# 3 Before training
def generate_text(text):
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
    outputs = model.generate(**inputs, max_new_tokens=20)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

print("Before training\n")
generate_text("<question>: What are the key differences between Python and C++?\n<answer>: ")

# 4. Do model patching and add fast LoRA weights and training
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)
model = get_peft_model(model, peft_config)

# Monitering the LLM
wandb.login(key = "YOUR_WANDB_KEY")
run = wandb.init(project='Fine tuning of LLAMA3 8B', job_type="training", anonymous="allow")

trainer = SFTTrainer(
    model = model,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    dataset_text_field="text",
    peft_config=peft_config,
    max_seq_length = max_seq_length,
    tokenizer = tokenizer,
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps = 4,
        evaluation_strategy="steps",
        do_eval=True,
        eval_steps=25,
        save_steps=50,
        logging_steps = 25,
        max_steps = 100,
        output_dir = "outputs",
        optim = "adamw_8bit",
        learning_rate=2e-4,
        weight_decay=0.001,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="wandb"
    ),
)
trainer.train()

# 5. After training
print("\n ######## \nAfter training\n")
generate_text("<question>: What are the key differences between Python and C++?\n<answer>: ")

# 6. Save the model
output_dir = "lora_model"
hf_model_repo = "fine-tuned_llama3-8b" 
trainer.save_model(output_dir)

output_dir = os.path.join(output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
tokenizer.push_to_hub(hf_model_repo)