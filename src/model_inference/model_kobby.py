import os
import torch
from datasets import load_dataset, DatasetDict, Dataset, tqdm
from transformers import pipeline, AutoTokenizer

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_names: dict = {
    'gemma_it_ft': 'shahafvl/gemma-2-2b-it-fake-news',
    # 'albert_v2_ft': 'XSY/albert-base-v2-fakenews-discriminator',
    'llama_instruct': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'llama_it_ft': 'shahafvl/llama-3_1-8b-instruct-fake-news',
    'gemma_ft': 'shahafvl/gemma-2-2b-fake-news',
    'gemma_instruct': 'google/gemma-2-2b-it'
}

datasets_names: dict = {
        'kobby': 'ikekobby/40-percent-cleaned-preprocessed-fake-real-news',
}

prompt: str = (
    "Your task is to classify the provided input as real or fake news. "
    "Label 0 is meant for fake news, Label 1 is for real news."
)

label_mapping: dict = {
    'LABEL_0': 0,
    'LABEL_1': 1,
}


def preprocess_row(row: dict) -> dict:
    row['input'] = f"{prompt} TEXT: {row['article']}"
    row['clean_input'] = f"{prompt} TEXT: {row['clean_article']}"
    return row


def main():
    for model_name, model_hf_path in model_names.items():
        pipe = pipeline(
            'text-classification',
            model=model_hf_path,
            torch_dtype=torch.bfloat16,
            device='cuda',
        )
        # tokenizer = AutoTokenizer.from_pretrained(model_hf_path)
        # max_length: int = tokenizer.model_max_length

        for dataset_name, dataset_hf_path in tqdm(
                datasets_names.items(),
                total=len(datasets_names),
        ):
            dataset: DatasetDict = load_dataset(dataset_hf_path)
            for key in tqdm(dataset.keys(), leave=False):
                curr_dataset: Dataset = dataset[key].map(preprocess_row)

                results = pipe(
                    curr_dataset['input'],
                    # max_length=max_length,
                    # truncation=True,
                    # padding=True,
                )
                curr_dataset = curr_dataset.add_column(
                    'pred',
                    [label_mapping[row['label']] for row in results],
                )
                curr_dataset = curr_dataset.add_column(
                    'score',
                    [row['score'] for row in results],
                )

                results = pipe(
                    curr_dataset['clean_input'],
                    # max_length=max_length,
                    # truncation=True,
                    # padding=True,
                )
                curr_dataset = curr_dataset.add_column(
                    'pred_clean_input',
                    [label_mapping[row['label']] for row in results],
                )
                curr_dataset = curr_dataset.add_column(
                    'score_clean_input',
                    [row['score'] for row in results],
                )

                results = pipe(
                    curr_dataset['article'],
                    # max_length=max_length,
                    # truncation=True,
                    # padding=True,
                )
                curr_dataset = curr_dataset.add_column(
                    'pred_article',
                    [label_mapping[row['label']] for row in results],
                )
                curr_dataset = curr_dataset.add_column(
                    'score_article',
                    [row['score'] for row in results],
                )

                results = pipe(
                    curr_dataset['clean_article'],
                    # max_length=max_length,
                    # truncation=True,
                    # padding=True,
                )
                curr_dataset = curr_dataset.add_column(
                    'pred_clean_article',
                    [label_mapping[row['label']] for row in results],
                )
                curr_dataset = curr_dataset.add_column(
                    'score_clean_article',
                    [row['score'] for row in results],
                )

                curr_dataset.to_parquet(
                    f'../output/datasets/{dataset_name}_{key}_{model_name}.parquet'
                )


if __name__ == '__main__':
    main()
