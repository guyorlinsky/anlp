# Model Inference

This directory contains files used to generate data for our analysis.

> As our new models have a larger input length and are more flexible,
> adaptation was needed to run inference with the older model, `alberta-v2`.
> The relevant sections were commented out to be used twice.

# Model Training

Model training was done using Hugging Face's [`trl`](https://github.com/huggingface/trl) script.
No quantization was used.

[WANDB REPORT](https://api.wandb.ai/links/shahaf-vl/h3tphrgv)

**Gemma 2 2B Instruct:**

```
python sft.py 
--model_name google/gemma-2-2b-it \
--dataset_name shahafvl/fake_news \
--dataset_text_field="text" \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 4 \
--learning_rate 2e-4 \
--report_to wandb \
--bf16 \
--max_seq_length 1024 \
--lora_r 16 \
--lora_alpha 32 \
--lora_task_type SEQ_CLS \
--lora_target_modules q_proj k_proj v_proj o_proj \
--use_peft \
--attn_implementation eager \
--logging_steps=10 \
--gradient_checkpointing \
--output_dir models/gemma-2-2b-it-fake-news \
--push_to_hub
```

The code for running the base model is the same, swapping the base model and `GonzaloA/fake_news` as the dataset.

**Llama 3.1 8B Instruct:**

```
python sft.py
--model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
--dataset_name shahafvl/fake_news \
--dataset_text_field="text" \
--per_device_train_batch_size 2 \ 
--gradient_accumulation_steps 2 \
--learning_rate 3e-4 \
--report_to wandb \
--bf16 \
--max_seq_length 2048 \
--lora_r 16 \
--lora_alpha 32 \
--lora_task_type SEQ_CLS \
--lora_target_modules q_proj k_proj v_proj o_proj up_proj down_proj gate_proj \
--use_peft \
--attn_implementation eager \
--logging_steps=10 \
--gradient_checkpointing \
--output_dir models/llama-3_1-8b-instruct-fake-news \
--push_to_hub
```
