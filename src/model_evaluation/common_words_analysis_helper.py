from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from collections import Counter
import pandas as pd
from wordcloud import WordCloud
import re
from huggingface_hub import HfApi
from datasets import load_dataset
import os
import matplotlib.pyplot as plt
import numpy as np


def get_all_datasets():
    # Initialize the API
    api = HfApi()

    # List datasets and filter by user or organization
    user = "shahafvl"
    datasets_info = api.list_datasets(author=f"{user}")

    # Convert generator to list and print
    datasets_info = list(datasets_info)
    datasets = {}
    for dataset_info in datasets_info:
        if 'john' in dataset_info.id or 'kobby' in dataset_info.id:
            continue
        data = load_dataset(dataset_info.id)
        if dataset_info.id.endswith("ft"):
            datasets[dataset_info.id] = data['test'].to_pandas()
        else:
            datasets[dataset_info.id] = pd.concat([data[k].to_pandas() for k in data.keys()])

    directory_john_fake_news = 'data/john_fake_news/'
    directory_kobby_fake_news = 'data/kobby_fake_news/'
    files = [f"{directory_john_fake_news}/{f}" for f in os.listdir(directory_john_fake_news) if os.path.isfile(os.path.join(directory_john_fake_news, f))] + [f"{directory_kobby_fake_news}/{f}" for f in os.listdir(directory_kobby_fake_news) if os.path.isfile(os.path.join(directory_kobby_fake_news, f))]

    for file in files:
        datasets[file.split("/")[-1].replace(".parquet", "")] = pd.read_parquet(file)
    return datasets

def get_filtered_common_words_by_classification_all(df, n_examples=100, m_words=100, distance_factor=2, pred='pred',
                                                    text_name='text'):
    df['pred_type'] = None
    df.loc[(df['label'] == 0) & (df[pred] == 1), 'pred_type'] = 'FP'
    df.loc[(df['label'] == 1) & (df[pred] == 0), 'pred_type'] = 'FN'
    df.loc[(df['label'] == 1) & (df[pred] == 1), 'pred_type'] = 'TP'
    df.loc[(df['label'] == 0) & (df[pred] == 0), 'pred_type'] = 'TN'

    classification_types = pd.unique(df['pred_type']).tolist()

    # classification_types = ['TP', 'TN', 'FP', 'FN']
    results = {}

    # Normalization methods
    normalization_methods = ['per_doc', 'length', 'tfidf']

    for normalization in normalization_methods:
        common_words_dict = {}

        # Generate word counts for each classification type
        for class_type in classification_types:
            texts = df[df['pred_type'] == class_type][text_name].tolist()
            combined_text = " ".join(texts)
            if normalization == 'tfidf':
                # TF-IDF Transformation
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(texts)
                words = vectorizer.get_feature_names_out()
                freqs = tfidf_matrix.sum(axis=0).A1
                word_freq = dict(zip(words, freqs))

            else:
                # CountVectorizer for word counts
                vectorizer = CountVectorizer(stop_words='english')
                word_count = vectorizer.fit_transform([combined_text])
                words = vectorizer.get_feature_names_out()
                freqs = word_count.toarray().sum(axis=0)
                word_freq = dict(zip(words, freqs))

                if normalization == 'per_doc':
                    # Per-Document Normalization
                    num_docs = len(texts)
                    word_freq = {word: freq / num_docs for word, freq in word_freq.items()}

                elif normalization == 'length':
                    # Length Normalization
                    total_words = len(combined_text.split())
                    word_freq = {word: freq / total_words for word, freq in word_freq.items()}

            common_words_dict[class_type] = Counter(word_freq)

        # Filter the words for each classification type based on comparison with other types
        filtered_common_words_dict = {}

        for class_type in classification_types:
            filtered_words = {}
            current_word_freq = common_words_dict[class_type]

            # Combine the word counts from the other classes
            other_word_freq = Counter()
            for other_class in classification_types:
                if other_class != class_type:
                    other_word_freq.update(common_words_dict[other_class])

            # Apply the filtering logic
            for word, freq in current_word_freq.items():
                if len([other_class for other_class in classification_types if
                        other_class != class_type and freq > common_words_dict[other_class].get(word,
                                                                                                0) * distance_factor]) == len(
                    classification_types) - 1:
                    filtered_words[word] = freq

            filtered_common_words_dict[class_type] = {word: freq for word, freq in
                                                      Counter(filtered_words).most_common(n=m_words)}

        # Function to count common words in text
        def count_common_words(text, common_words):
            text_words = set(text.split())
            return [word for word in common_words if word in text_words]

        # Apply the function to each classification type
        for class_type in classification_types:
            df.loc[df['pred_type'] == class_type, 'common_word_found'] = df.loc[
                df['pred_type'] == class_type, text_name].apply(
                lambda x: count_common_words(x, filtered_common_words_dict[class_type])
            )
            df.loc[df['pred_type'] == class_type, 'common_word_count'] = df.loc[
                df['pred_type'] == class_type, 'common_word_found'].apply(lambda x: len(x))
        df['common_word_count'] = df['common_word_count'].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

        # Return the top n_examples for each classification type
        top_examples = pd.DataFrame()
        for class_type in classification_types:
            top_examples = pd.concat([top_examples, df[df['pred_type'] == class_type].nlargest(n_examples,
                                                                                               'common_word_count').sort_values(
                'common_word_count', ascending=False)], axis=0, ignore_index=True)
        top_examples = top_examples[top_examples['common_word_count'] > 0].reset_index(drop=True)

        # Store results in the dictionary
        results[normalization] = {
            'top_examples': top_examples,
            'filtered_common_words_dict': filtered_common_words_dict,
            'common_words_dict': common_words_dict
        }

    return results


def get_filtered_common_words_by_classification_normalized(df, n_examples=100, m_words=100, distance_factor=2,
                                                pred='pred', text_name='text', normalization='length'):
    df['pred_type'] = None
    df.loc[(df['label'] == 0) & (df[pred] == 1), 'pred_type'] = 'FP'
    df.loc[(df['label'] == 1) & (df[pred] == 0), 'pred_type'] = 'FN'
    df.loc[(df['label'] == 1) & (df[pred] == 1), 'pred_type'] = 'TP'
    df.loc[(df['label'] == 0) & (df[pred] == 0), 'pred_type'] = 'TN'

    classification_types = pd.unique(df['pred_type']).tolist()
    common_words_dict = {}

    # Generate word counts for each classification type
    for class_type in classification_types:
        texts = df[df['pred_type'] == class_type][text_name].tolist()
        combined_text = " ".join(texts)

        if normalization == 'tfidf':
            # TF-IDF Transformation
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(texts)
            words = vectorizer.get_feature_names_out()
            freqs = tfidf_matrix.sum(axis=0).A1
            word_freq = dict(zip(words, freqs))

        else:
            # CountVectorizer for word counts
            vectorizer = CountVectorizer(stop_words='english')
            word_count = vectorizer.fit_transform([combined_text])
            words = vectorizer.get_feature_names_out()
            freqs = word_count.toarray().sum(axis=0)
            word_freq = dict(zip(words, freqs))

            if normalization == 'per_doc':
                # Per-Document Normalization
                num_docs = len(texts)
                word_freq = {word: freq / num_docs for word, freq in word_freq.items()}

            elif normalization == 'length':
                # Length Normalization
                total_words = len(combined_text.split())
                word_freq = {word: freq / total_words for word, freq in word_freq.items()}

        common_words_dict[class_type] = Counter(word_freq)

    # Filter the words for each classification type based on comparison with other types
    filtered_common_words_dict = {}

    for class_type in classification_types:
        filtered_words = {}
        current_word_freq = common_words_dict[class_type]

        # Combine the word counts from the other classes
        other_word_freq = Counter()
        for other_class in classification_types:
            if other_class != class_type:
                other_word_freq.update(common_words_dict[other_class])

        # Apply the filtering logic
        for word, freq in current_word_freq.items():
            if len([other_class for other_class in classification_types if
                    other_class != class_type and freq > common_words_dict[other_class].get(word,
                                                                                            0) * distance_factor]) == len(
                    classification_types) - 1:
                filtered_words[word] = freq

        filtered_common_words_dict[class_type] = {word: freq for word, freq in
                                                  Counter(filtered_words).most_common(n=m_words)}

    # Function to count common words in text
    def count_common_words(text, common_words):
        text_words = set(text.split())
        return [word for word in common_words if word in text_words]

    # Apply the function to each classification type
    for class_type in classification_types:
        df.loc[df['pred_type'] == class_type, 'common_word_found'] = df.loc[
            df['pred_type'] == class_type, text_name].apply(
            lambda x: count_common_words(x, filtered_common_words_dict[class_type])
        )
        df.loc[df['pred_type'] == class_type, 'common_word_count'] = df.loc[
            df['pred_type'] == class_type, 'common_word_found'].apply(lambda x: len(x))

    # Return the top n_examples for each classification type
    df['common_word_count'] = df['common_word_count'].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    top_examples = pd.DataFrame()
    for class_type in classification_types:
        top_examples = pd.concat([top_examples, df[df['pred_type'] == class_type].nlargest(n_examples,
                                                                                           'common_word_count').sort_values(
            'common_word_count', ascending=False)], axis=0, ignore_index=True)
    top_examples = top_examples[top_examples['common_word_count'] > 0].reset_index(drop=True)

    return top_examples, filtered_common_words_dict, common_words_dict


def get_filtered_common_words_by_prediction_normalized(df, n_examples=100, m_words=100, distance_factor=2,
                                                           pred='pred', text_name='text', normalization='length'):
    common_words_dict = {}
    unique_labels = df[pred].unique()

    # Generate word counts for each classification type based on the model's prediction label
    for label in unique_labels:
        texts = df[df[pred] == label][text_name].tolist()
        combined_text = " ".join(texts)

        if normalization == 'tfidf':
            # TF-IDF Transformation
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(texts)
            words = vectorizer.get_feature_names_out()
            freqs = tfidf_matrix.sum(axis=0).A1
            word_freq = dict(zip(words, freqs))

        else:
            # CountVectorizer for word counts
            vectorizer = CountVectorizer(stop_words='english')
            word_count = vectorizer.fit_transform([combined_text])
            words = vectorizer.get_feature_names_out()
            freqs = word_count.toarray().sum(axis=0)
            word_freq = dict(zip(words, freqs))

            if normalization == 'per_doc':
                # Per-Document Normalization
                num_docs = len(texts)
                word_freq = {word: freq / num_docs for word, freq in word_freq.items()}

            elif normalization == 'length':
                # Length Normalization
                total_words = len(combined_text.split())
                word_freq = {word: freq / total_words for word, freq in word_freq.items()}

        common_words_dict[label] = Counter(word_freq)

    # Filter the words for each classification type based on comparison with other types
    filtered_common_words_dict = {}

    for label in unique_labels:
        filtered_words = {}
        current_word_freq = common_words_dict[label]

        # Combine the word counts from the other labels
        other_word_freq = Counter()
        for other_label in unique_labels:
            if other_label != label:
                other_word_freq.update(common_words_dict[other_label])

        # Apply the filtering logic
        for word, freq in current_word_freq.items():
            if len([other_label for other_label in unique_labels if
                    other_label != label and freq > common_words_dict[other_label].get(word,
                                                                                      0) * distance_factor]) == len(
                    unique_labels) - 1:
                filtered_words[word] = freq

        filtered_common_words_dict[label] = {word: freq for word, freq in
                                             Counter(filtered_words).most_common(n=m_words)}

    # Function to count common words in text
    def count_common_words(text, common_words):
        text_words = set(text.split())
        return [word for word in common_words if word in text_words]

    # Apply the function to each classification type
    for label in unique_labels:
        df.loc[df[pred] == label, 'common_word_found'] = df.loc[
            df[pred] == label, text_name].apply(
            lambda x: count_common_words(x, filtered_common_words_dict[label])
        )
        df.loc[df[pred] == label, 'common_word_count'] = df.loc[
            df[pred] == label, 'common_word_found'].apply(lambda x: len(x))

    # Return the top n_examples for each classification type
    df['common_word_count'] = df['common_word_count'].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    top_examples = pd.DataFrame()
    for label in unique_labels:
        top_examples = pd.concat([top_examples, df[df[pred] == label].nlargest(n_examples,
                                                                               'common_word_count').sort_values(
            'common_word_count', ascending=False)], axis=0, ignore_index=True)
    top_examples = top_examples[top_examples['common_word_count'] > 0].reset_index(drop=True)

    return top_examples, filtered_common_words_dict, common_words_dict



def display_comparison_word_clouds(common_words_dict, filtered_common_words_dict, df_and_model_name='',
                                   should_show=False):
    classification_types = sorted(list(common_words_dict.keys()))

    # Set up the 2x4 subplot grid
    fig, axes = plt.subplots(4, 2, figsize=(20, 20))
    fig.suptitle(f'Word Clouds Comparison - {df_and_model_name.split("/")[-1]}', fontsize=20)
    for i, class_type in enumerate(classification_types):
        # Generate the word cloud for the unfiltered common words
        wordcloud_unfiltered = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(
            common_words_dict[class_type])
        axes[i, 0].imshow(wordcloud_unfiltered, interpolation='bilinear')
        axes[i, 0].set_title(f'Unfiltered {class_type}', fontsize=16)
        axes[i, 0].axis('off')

        # Generate the word cloud for the filtered common words
        wordcloud_filtered = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(
            filtered_common_words_dict[class_type])
        axes[i, 1].imshow(wordcloud_filtered, interpolation='bilinear')
        axes[i, 1].set_title(f'Filtered {class_type}', fontsize=16)
        axes[i, 1].axis('off')

    plt.tight_layout()
    # Save the plot as a JPG file
    plt.savefig(f'plots/wordclouds_by_classification_types/wordcloud_{df_and_model_name.split("/")[-1]}.jpg', format='jpg', bbox_inches='tight')

    # if should_show:
    #     plt.show()
    plt.close()

def display_comparison_word_clouds_by_prediction(common_words_dict, filtered_common_words_dict, df_and_model_name='',
                                   should_show=False):
    labels = sorted(list(common_words_dict.keys()))

    # Set up the 2x4 subplot grid (adjust the number of subplots based on the number of labels)
    fig, axes = plt.subplots(len(labels), 2, figsize=(20, 5 * len(labels)))
    fig.suptitle(f'Word Clouds Comparison - {df_and_model_name.split("/")[-1]}', fontsize=20)

    for i, label in enumerate(labels):
        # Generate the word cloud for the unfiltered common words
        wordcloud_unfiltered = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(
            common_words_dict[label])
        axes[i, 0].imshow(wordcloud_unfiltered, interpolation='bilinear')
        axes[i, 0].set_title(f'Unfiltered {label}', fontsize=16)
        axes[i, 0].axis('off')

        # Generate the word cloud for the filtered common words
        wordcloud_filtered = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(
            filtered_common_words_dict[label])
        axes[i, 1].imshow(wordcloud_filtered, interpolation='bilinear')
        axes[i, 1].set_title(f'Filtered {label}', fontsize=16)
        axes[i, 1].axis('off')

    plt.tight_layout()

    # Save the plot as a JPG file
    plt.savefig(f'plots/wordclouds_by_predicted_label/wordcloud_{df_and_model_name.split("/")[-1]}.jpg', format='jpg', bbox_inches='tight')

    if should_show:
        plt.show()

    plt.close()


def display_comparison_word_clouds_all_morm_methods_combined(results, df_and_model_name='', should_show=False):
    normalization_methods = ['per_doc', 'length', 'tfidf']
    classification_types = list(
        results[normalization_methods[0]]['filtered_common_words_dict'].keys())  # ['TP', 'TN', 'FP', 'FN']

    # Set up the 3x4 subplot grid (3 normalization methods x 4 classification types)
    fig, axes = plt.subplots(len(classification_types), len(normalization_methods) * 2, figsize=(30, 20))
    fig.suptitle(f'Word Clouds Comparison - {df_and_model_name.split("/")[-1]}', fontsize=20)

    for i, class_type in enumerate(classification_types):
        for j, norm_method in enumerate(normalization_methods):
            common_words_dict = results[norm_method]['common_words_dict']
            filtered_common_words_dict = results[norm_method]['filtered_common_words_dict']

            # Generate the word cloud for the unfiltered common words
            wordcloud_unfiltered = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(
                common_words_dict[class_type])
            axes[i, j * 2].imshow(wordcloud_unfiltered, interpolation='bilinear')
            axes[i, j * 2].set_title(f'{norm_method} Unfiltered {class_type}', fontsize=16)
            axes[i, j * 2].axis('off')

            # Generate the word cloud for the filtered common words
            wordcloud_filtered = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(
                filtered_common_words_dict[class_type])
            axes[i, j * 2 + 1].imshow(wordcloud_filtered, interpolation='bilinear')
            axes[i, j * 2 + 1].set_title(f'{norm_method} Filtered {class_type}', fontsize=16)
            axes[i, j * 2 + 1].axis('off')

    plt.tight_layout()

    # Save the plot as a JPG file
    plt.savefig(f'plots/wordclouds_all_morm_methods_combined/wordcloud_{df_and_model_name.split("/")[-1]}.jpg', format='jpg',
                bbox_inches='tight')

    # if should_show:
    #     plt.show()
    plt.close()


def get_input_without_common_words(row, text_col_to_mask='input', mask=False):
    cleaned_text = row[text_col_to_mask].lower()
    for word in row['common_word_found']:
        # Use word boundaries to match whole words only
        if mask:
            cleaned_text = re.sub(rf'\b{re.escape(word)}\b', "[MASK]", cleaned_text)
        else:
            cleaned_text = re.sub(rf'\b{re.escape(word)}\b', "", cleaned_text)
    return cleaned_text


def plot_changes_in_prediction(df_dict):
    labels = []
    masked_common_words_error_rates = []
    without_common_words_error_rates = []
    label_changes = {}

    for model_name, df in df_dict.items():
        total = df.shape[0]

        # Calculate the error rates
        masked_common_words_error_rate = (df['pred_masked_common_words'] != df['pred']).sum() / total
        without_common_words_error_rate = (df['pred_without_common_words'] != df['pred']).sum() / total

        masked_common_words_error_rates.append(masked_common_words_error_rate)
        without_common_words_error_rates.append(without_common_words_error_rate)

        labels.append(model_name)

        # Calculate the number of preds changed from 1 to 0 and 0 to 1 for each pred type
        changes_masked_common_words_1_to_0 = ((df['pred'] == 1) & (df['pred_masked_common_words'] == 0)).sum()
        changes_masked_common_words_0_to_1 = ((df['pred'] == 0) & (df['pred_masked_common_words'] == 1)).sum()

        changes_without_common_words_1_to_0 = ((df['pred'] == 1) & (df['pred_without_common_words'] == 0)).sum()
        changes_without_common_words_0_to_1 = ((df['pred'] == 0) & (df['pred_without_common_words'] == 1)).sum()

        label_changes[model_name] = {
            'masked_common_words_1_to_0': changes_masked_common_words_1_to_0,
            'masked_common_words_0_to_1': changes_masked_common_words_0_to_1,
            'without_common_words_1_to_0': changes_without_common_words_1_to_0,
            'without_common_words_0_to_1': changes_without_common_words_0_to_1,
        }

    # Plot the error rates
    plt.figure(figsize=(10, 6))

    x = range(len(labels))
    plt.plot(x, masked_common_words_error_rates, label="Masked Common Words Change in prediction Rate", marker='o')
    plt.plot(x, without_common_words_error_rates, label="Without Common Words Change in prediction Rate", marker='o')

    plt.xticks(x, labels, rotation=90, size=10)
    plt.ylabel("Change in prediction Rate")
    plt.title("Change in prediction Rate Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/label_change_by_masked_words_plots/{"Change in prediction Rate Comparison".replace(" ", "_")}.jpg', format='jpg', bbox_inches='tight')
    plt.show()


def plot_metrics(df_dict, accuracies):
    plot_changes_in_prediction(df_dict)
    labels = list(df_dict.keys())
    half_point = len(labels) // 2

    first_half_labels = labels[:half_point]
    second_half_labels = labels[half_point:]

    label_changes = {}

    for model_name in labels:
        df = df_dict[model_name]
        total = df.shape[0]

        # Calculate the number of preds changed from 1 to 0 and 0 to 1 for each pred type
        changes_masked_common_words_1_to_0 = ((df['pred'] == 1) & (df['pred_masked_common_words'] == 0)).sum()
        changes_masked_common_words_0_to_1 = ((df['pred'] == 0) & (df['pred_masked_common_words'] == 1)).sum()

        changes_without_common_words_1_to_0 = ((df['pred'] == 1) & (df['pred_without_common_words'] == 0)).sum()
        changes_without_common_words_0_to_1 = ((df['pred'] == 0) & (df['pred_without_common_words'] == 1)).sum()

        label_changes[model_name] = {
            'masked_common_words_1_to_0': changes_masked_common_words_1_to_0,
            'masked_common_words_0_to_1': changes_masked_common_words_0_to_1,
            'without_common_words_1_to_0': changes_without_common_words_1_to_0,
            'without_common_words_0_to_1': changes_without_common_words_0_to_1,
        }

    # Colors for each category
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
    width = 0.2  # the width of the bars

    fig, axs = plt.subplots(2, 1, figsize=(18, 12))

    # Plot for the first half of the DataFrames
    x1 = np.arange(len(first_half_labels))
    for i, model_name in enumerate(first_half_labels):
        axs[0].bar(x1[i] - 1.5 * width, label_changes[model_name]['masked_common_words_1_to_0'], width,
                   label='Masked 1 to 0' if i == 0 else "", color=colors[0])
        axs[0].bar(x1[i] - 0.5 * width, label_changes[model_name]['masked_common_words_0_to_1'], width,
                   label='Masked 0 to 1' if i == 0 else "", color=colors[1])
        axs[0].bar(x1[i] + 0.5 * width, label_changes[model_name]['without_common_words_1_to_0'], width,
                   label='Without 1 to 0' if i == 0 else "", color=colors[2])
        axs[0].bar(x1[i] + 1.5 * width, label_changes[model_name]['without_common_words_0_to_1'], width,
                   label='Without 0 to 1' if i == 0 else "", color=colors[3])

    axs[0].set_title('Number of Predictions Changed by Label (First Half)')
    axs[0].set_xticks(x1)
    axs[0].set_xticklabels([f"{i}\nAccuracy: {round(accuracies.get(i), 2)}" for i in first_half_labels], rotation=75,
                           size=15)
    axs[0].set_ylabel("Number of Predictions Changed")
    axs[0].legend()
    axs[0].grid(True)

    # Plot for the second half of the DataFrames
    x2 = np.arange(len(second_half_labels))
    for i, model_name in enumerate(second_half_labels):
        axs[1].bar(x2[i] - 1.5 * width, label_changes[model_name]['masked_common_words_1_to_0'], width,
                   label='Masked 1 to 0' if i == 0 else "", color=colors[0])
        axs[1].bar(x2[i] - 0.5 * width, label_changes[model_name]['masked_common_words_0_to_1'], width,
                   label='Masked 0 to 1' if i == 0 else "", color=colors[1])
        axs[1].bar(x2[i] + 0.5 * width, label_changes[model_name]['without_common_words_1_to_0'], width,
                   label='Without 1 to 0' if i == 0 else "", color=colors[2])
        axs[1].bar(x2[i] + 1.5 * width, label_changes[model_name]['without_common_words_0_to_1'], width,
                   label='Without 0 to 1' if i == 0 else "", color=colors[3])

    axs[1].set_title('Number of Predictions Changed by Label (Second Half)')
    axs[1].set_xticks(x2)
    axs[1].set_xticklabels([f"{i}\nAccuracy: {round(accuracies.get(i), 2)}" for i in second_half_labels], rotation=75,
                           size=15)
    axs[1].set_ylabel("Number of Predictions Changed")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig(f'plots/label_change_by_masked_words_plots/{"Number of Predictions Changed by Label".replace(" ", "_")}.jpg', format='jpg', bbox_inches='tight')
    plt.show()


def plot_normalized_label_changes_with_avg_word_count(df_dict):
    # Initialize dictionaries to accumulate counts
    masked_1_to_0_counts = {}
    masked_0_to_1_counts = {}
    without_1_to_0_counts = {}
    without_0_to_1_counts = {}

    total_word_count = 0
    total_examples = 0

    for df in df_dict.values():
        # Calculate the total number of words across all examples
        total_word_count += df[(df['pred'] != df['pred_masked_common_words']) | (df['pred'] != df['pred_without_common_words'])]['input'].apply(lambda x: len(x.split())).sum()
        total_examples += df[(df['pred'] != df['pred_masked_common_words']) | (df['pred'] != df['pred_without_common_words'])]['input'].shape[0]

        for count in df['common_word_count'].unique():
            if count not in masked_1_to_0_counts:
                masked_1_to_0_counts[count] = 0
                masked_0_to_1_counts[count] = 0
                without_1_to_0_counts[count] = 0
                without_0_to_1_counts[count] = 0

            masked_1_to_0_counts[count] += ((df['common_word_count'] == count) & (df['pred'] == 1) & (df['pred_masked_common_words'] == 0)).sum()
            masked_0_to_1_counts[count] += ((df['common_word_count'] == count) & (df['pred'] == 0) & (df['pred_masked_common_words'] == 1)).sum()
            without_1_to_0_counts[count] += ((df['common_word_count'] == count) & (df['pred'] == 1) & (df['pred_without_common_words'] == 0)).sum()
            without_0_to_1_counts[count] += ((df['common_word_count'] == count) & (df['pred'] == 0) & (df['pred_without_common_words'] == 1)).sum()

    # Calculate the average word count per example
    avg_word_count = total_word_count / total_examples

    # Calculate the total changes for normalization
    total_masked_1_to_0 = sum(masked_1_to_0_counts.values())
    total_masked_0_to_1 = sum(masked_0_to_1_counts.values())
    total_without_1_to_0 = sum(without_1_to_0_counts.values())
    total_without_0_to_1 = sum(without_0_to_1_counts.values())

    # Calculate normalized ratios
    common_word_counts = sorted(masked_1_to_0_counts.keys())
    normalized_masked_1_to_0 = [masked_1_to_0_counts[count] / total_masked_1_to_0 for count in common_word_counts]
    normalized_masked_0_to_1 = [masked_0_to_1_counts[count] / total_masked_0_to_1 for count in common_word_counts]
    normalized_without_1_to_0 = [without_1_to_0_counts[count] / total_without_1_to_0 for count in common_word_counts]
    normalized_without_0_to_1 = [without_0_to_1_counts[count] / total_without_0_to_1 for count in common_word_counts]

    # Plotting
    plt.figure(figsize=(14, 8))

    plt.plot(common_word_counts, normalized_masked_1_to_0, label='Masked 1 to 0', color='#FF9999', marker='o')
    plt.plot(common_word_counts, normalized_masked_0_to_1, label='Masked 0 to 1', color='#66B2FF', marker='o')
    plt.plot(common_word_counts, normalized_without_1_to_0, label='Without 1 to 0', color='#99FF99', marker='o')
    plt.plot(common_word_counts, normalized_without_0_to_1, label='Without 0 to 1', color='#FFCC99', marker='o')

    plt.title('Normalized Label Changes by Total Changes vs Common Word Count (Aggregated Across All Models)')
    plt.xlabel('Common Word Count')
    plt.ylabel('Ratio of Total Label Changes')
    plt.legend()
    plt.grid(True)

    # Add a box with the average word count
    plt.text(0.95, 0.95, f'Avg. Words per Input: {avg_word_count:.2f}',
             transform=plt.gca().transAxes,
             fontsize=12,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.5'))

    plt.tight_layout()
    plt.savefig(f'{"plots/label_change_by_masked_words_plots/Normalized Label Changes by Total Changes vs Common Word Count (Aggregated Across All Models)".replace(" ", "_")}.jpg', format='jpg', bbox_inches='tight')
    plt.show()
