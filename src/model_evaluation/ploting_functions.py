import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns



def plot_accuracies(accuracies, title):
    """
    Plots a bar chart of accuracies from a dictionary, with model names displayed vertically and accuracy values on top of each bar.

    Parameters:
    accuracies (dict): A dictionary where keys are model names and values are their accuracies.
    title (str): The title of the plot.
    """
    models = list(accuracies.keys())
    accuracy_values = list(accuracies.values())

    plt.figure(figsize=(20, 15))
    bars = plt.bar(models, accuracy_values, color='skyblue')

    # Add text labels on top of each bar with vertical model names
    for bar, model_name in zip(bars, models):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{round(yval, 2)}',
                 ha='center', va='bottom', fontsize=10, color='black', rotation=90)
        plt.text(bar.get_x() + bar.get_width()/2, 0.05, f'{model_name.split("/")[-1]}',
                 ha='center', va='bottom', fontsize=10, color='black', rotation=90)
    plt.xticks([])
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.ylim(0, 1)  # Assuming accuracies are between 0 and 1
    plt.savefig(f'plots/accuracies/{title.replace(" ", "_")}.jpg', format='jpg', bbox_inches='tight')
    plt.show()



def plot_multiple_accuracies(title_accuracies, full_article_with_prompt_accuracies,
                             full_article_without_prompt_accuracies, title='Accuracies', ):
    """
    Plots a grouped bar chart for three types of accuracies from different dictionaries.

    Parameters:
    title_accuracies (dict): Accuracies for predictions based on the title only.
    full_article_with_prompt_accuracies (dict): Accuracies for predictions based on the full article with an instruction prompt.
    full_article_without_prompt_accuracies (dict): Accuracies for predictions based on the full article without an instruction prompt.
    title (str): The title of the plot.
    """
    # Combine all model names
    all_models = set(title_accuracies.keys()) & set(full_article_with_prompt_accuracies.keys()) & set(
        full_article_without_prompt_accuracies.keys())
    all_models = sorted(all_models)  # Sort for consistency

    # Prepare data for plotting
    title_values = [title_accuracies.get(model, 0) for model in all_models]
    with_prompt_values = [full_article_with_prompt_accuracies.get(model, 0) for model in all_models]
    without_prompt_values = [full_article_without_prompt_accuracies.get(model, 0) for model in all_models]

    bar_width = 0.25
    index = np.arange(len(all_models))

    plt.figure(figsize=(12, 8))

    # Plot each type of accuracy
    bars1 = plt.bar(index, title_values, bar_width, label='Title Only')
    bars2 = plt.bar(index + bar_width, with_prompt_values, bar_width, label='Full Article with Prompt')
    bars3 = plt.bar(index + 2 * bar_width, without_prompt_values, bar_width, label='Full Article without Prompt')

    # Add model names and accuracy values on top of each group of bars
    for i, model_name in enumerate(all_models):
        plt.text(i + bar_width, 0.01, f'{model_name.split("/")[-1]}', ha='center', va='bottom', fontsize=8,
                 color='black', rotation=90)
        plt.text(i, title_values[i] + 0.01, f'{round(title_values[i], 2)}', ha='center', va='bottom', fontsize=8,
                 color='black', rotation=90)
        plt.text(i + bar_width, with_prompt_values[i] + 0.01, f'{round(with_prompt_values[i], 2)}', ha='center',
                 va='bottom', fontsize=8, color='black', rotation=90)
        plt.text(i + 2 * bar_width, without_prompt_values[i] + 0.01, f'{round(without_prompt_values[i], 2)}',
                 ha='center', va='bottom', fontsize=8, color='black', rotation=90)

    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.xticks(index + bar_width, all_models)
    plt.ylim(0, 1)  # Assuming accuracies are between 0 and 1
    plt.xticks([])  # Hide the x-axis labels (model names under bars)
    plt.legend()

    # Save the plot as a JPG file
    plt.savefig(f'plots/accuracies/{title.replace(" ", "_")}.jpg', format='jpg', bbox_inches='tight')

    # Display the plot
    plt.show()


def plot_accuracy_and_confusion_matrix(df, dataset_name=""):
    # 1. Identify Misclassified Instances
    df['is_misclassified'] = df['label'] != df['pred']
    misclassified = df[df['is_misclassified']]

    # 2. Confusion Matrix
    labels = df['label'].unique()  # Assuming 'label' and 'pred' have the same unique values
    cm = confusion_matrix(df['label'], df['pred'], labels=labels)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    labels = df['label'].astype(str).unique()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix. dataset: {dataset_name}')
    plt.savefig(f'plots/confusion_matrices/{dataset_name.split("/")[-1]}.jpg', format='jpg', bbox_inches='tight')
    plt.show()

    # Analyze Misclassifications
    misclassified_grouped = misclassified.groupby(['label', 'pred']).size().reset_index(name='count')
    print(f"Misclassified Instances: dataset: {dataset_name}")
    print(misclassified_grouped)

    # 4. Error Metrics
    report = classification_report(df['label'], df['pred'], target_names=labels)
    accuracy = accuracy_score(df['label'], df['pred'])
    print(f"dataset: {dataset_name}")
    print("Classification Report:\n", report)
    print("Accuracy:", accuracy)

    # 5. Examine Score Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df[df['is_misclassified']]['score'], color='red', label='Misclassified', kde=True)
    sns.histplot(df[~df['is_misclassified']]['score'], color='green', label='Correctly Classified', kde=True)
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution')
    plt.savefig(f'plots/score_distributions/{dataset_name.split("/")[-1]}.jpg', format='jpg', bbox_inches='tight')
    plt.legend()
    plt.show()
    return accuracy

