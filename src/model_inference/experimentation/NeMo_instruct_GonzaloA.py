import os
from transformers import pipeline

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

def main():
    messages: list[dict] = [
        {"role": "user", "content": "Who are you?"},
    ]
    print("Activating model...\n")
    pipe = pipeline("text-generation",
                    model="mistralai/Mistral-Nemo-Instruct-2407",
                    device_map="auto")
    print(f"Model activated! {pipe.device = }")
    print("Running prompt...\n")
    print(f"{pipe(messages) = }")


if __name__ == '__main__':
    main()

