import torch
from datasets import load_dataset, DatasetDict, Dataset, tqdm
from transformers import pipeline, AutoTokenizer
import os
import re

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

model_names: dict = {
    'gemma_it_ft': 'shahafvl/gemma-2-2b-it-fake-news',
    'albert_v2_ft': 'XSY/albert-base-v2-fakenews-discriminator',
    'llama_instruct': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'llama_it_ft': 'shahafvl/llama-3_1-8b-instruct-fake-news',
    'gemma_ft': 'shahafvl/gemma-2-2b-fake-news',
    'gemma_instruct': 'google/gemma-2-2b-it',
    'gemma_it': 'google/gemma-2-2b-it',
    'llama_it': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
}
pattern = re.compile(r'|'.join(re.escape(key) for key in model_names.keys()))

label_mapping: dict = {
    'LABEL_0': 0,
    'LABEL_1': 1,
}

dir_path: str = '../data/files_without_or_masked_common_words_to_predict'

def main():
    for filename in os.listdir(dir_path):
        if not filename.endswith('.parquet'):
            continue

        match = pattern.search(filename)

        model_name: str = 'Unknown model'
        if match:
            model_identifier = match.group(0)
            model_name: str = model_names[model_identifier]
        if model_name == 'Unkown model':
            continue
        print(f'For {filename} loading model: {model_name}')
        pipe = pipeline(
            'text-classification',
            model=model_name,
            torch_dtype=torch.bfloat16,
            device='cuda',
        )
        print(f"Model loaded! {pipe.device = }\nLoading dataset...")
        dataset: Dataset = load_dataset("parquet", data_files=(dir_path + '/' + filename))['train']
        results: list[dict]
        results_masked: list[disct]
        if model_name == 'XSY/albert-base-v2-fakenews-discriminator':
            max_length: int = AutoTokenizer.from_pretrained(model_name).model_max_length
            results = pipe(
                dataset['input_without_common_words_to_predict'],
                max_length=max_length,
                truncation=True,
                padding=True,
            )
            results_masked = pipe(
                dataset['input_masked_common_words_to_predict'],
                max_length=max_length,
                truncation=True,
                padding=True,
            )
        else:
            results = pipe(dataset['input_without_common_words_to_predict'])
            results_masked = pipe(dataset['input_masked_common_words_to_predict'])

        dataset = dataset.add_column(
            'pred_without_common_words',
            [label_mapping[row['label']] for row in results],
        )
        dataset = dataset.add_column(
            'score_without_common_words',
            [row['score'] for row in results],
        )

        dataset = dataset.add_column(
            'pred_masked_common_words',
            [label_mapping[row['label']] for row in results_masked],
        )
        dataset = dataset.add_column(
            'score_masked_common_words',
            [row['score'] for row in results_masked],
        )
        dataset.to_parquet(f'../output/masked_words_2/{filename}')


if __name__ == '__main__':
    main()
