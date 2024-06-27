import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

# Load the fine-tuned model
output_dir = "./lora_model"  # Path where your fine-tuned model is saved
device_map = "auto"  # Adjust this according to your device setup

model = AutoPeftModelForCausalLM.from_pretrained(
    output_dir,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.bfloat16,
    device_map=device_map,
)

# Merge LoRA and base model
merged_model = model.merge_and_unload()

# Save the merged model
merged_model.save_pretrained("merged_model", safe_serialization=True)

# Load the tokenizer and save it
tokenizer = AutoTokenizer.from_pretrained(output_dir)
tokenizer.save_pretrained("merged_model")

# Optionally, push the merged model to the Hugging Face Hub
hf_model_repo = "fine-tuned_llama3-8b"  # Replace with your actual Hugging Face repository name

#merged_model.push_to_hub(hf_model_repo)
merged_model.push_to_hub(hf_model_repo)
tokenizer.push_to_hub(hf_model_repo)
