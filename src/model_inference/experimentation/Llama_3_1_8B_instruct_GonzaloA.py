import os
import torch

from transformers import pipeline


os.environ['CUDA_VISIBLE_DEVICES'] = '7'


def main():
    model_id: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    print("Activating model...\n")
    pipe = pipeline("text-generation",
                    model=model_id,
                    model_kwargs={"torch_dtype": torch.bfloat16},
                    cache_dir="/cs/labs/tomhope/shahaf_levy/.cache",
                    device_map="auto")
    print(f"Model activated! {pipe.device = }")
    
    messages: list[dict] = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
    ]

    print("Running prompt...\n")
    outputs = pipeline(
        messages,
        max_new_tokens=256,
    )
    print(outputs[0]["generated_text"][-1])



if __name__ == '__main__':
    main()

