# TASKS 4 AND 5
# 4-classifier robustness + analysis script
# -----------------------------

import os
import csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from itertools import combinations

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



# Set this to None to use the full dataset.
MAX_SAMPLES_PER_CLASS = None

# All images will be resized to the same size with this.
TARGET_SIZE = (200, 200)

# These are the augmentation names.
AUGMENTATION_TYPES = ['noise', 'jpeg', 'downsample', 'crop', 'bitdepth']

# These are the class names and their numeric labels.
CLASS_NAMES = ['Word', 'Google', 'Python']


# LOAD ORIGINAL DATA

def load_original_dataset(
    word_dir='word_pdfs_png',
    google_dir='google_docs_pdfs_png',
    python_dir='python_pdfs_png',
    target_size=(200, 200),
    max_samples_per_class=None
):
    # X will store the image features
    # y will store the class labels
    X = []
    y = []

    # This stores folder path, numeric label, and readable class name
    folder_info = [
        (word_dir, 0, "Word"),
        (google_dir, 1, "Google"),
        (python_dir, 2, "Python")
    ]

    # Go through each class folder one at a time
    for folder, label, name in folder_info:
        files = [f for f in os.listdir(folder) if f.endswith('.png')]

        # Optionally limit the number of samples for quick testing
        if max_samples_per_class is not None:
            files = files[:max_samples_per_class]

        print(f"Loading {len(files)} original {name} images...")

        for filename in files:
            path = os.path.join(folder, filename)

            # Open image, convert to grayscale, resize, then flatten to 1D vector
            img = Image.open(path).convert('L')
            img = img.resize(target_size, Image.LANCZOS)
            img_array = np.array(img).flatten()

            X.append(img_array)
            y.append(label)

    # Convert lists to NumPy arrays
    X = np.array(X)
    y = np.array(y)

    print(f"Original dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class counts: Word={np.sum(y==0)}, Google={np.sum(y==1)}, Python={np.sum(y==2)}")

    return X, y



# LOAD ONE AUGMENTATION TYPE

def load_augmentation_dataset(
    augmentation_type,
    augmented_root='augmented_images',
    target_size=(200, 200)
):
    X = []
    y = []

    # These are the augmented class folders
    folder_info = [
        (os.path.join(augmented_root, 'word_pdfs_png'), 0, "Word"),
        (os.path.join(augmented_root, 'google_docs_pdfs_png'), 1, "Google"),
        (os.path.join(augmented_root, 'python_pdfs_png'), 2, "Python")
    ]

    # This goes through each augmented class folder
    for folder, label, name in folder_info:
        # This keeps only files with the chosen augmentation in the filename
        files = [f for f in os.listdir(folder) if f"__{augmentation_type}" in f]

        print(f"Loading {len(files)} {augmentation_type} {name} images...")

        for filename in files:
            path = os.path.join(folder, filename)

            # This opens and convert the image to grayscale, resizes and flattens it.
            img = Image.open(path).convert('L')
            img = img.resize(target_size, Image.LANCZOS)
            img_array = np.array(img).flatten()

            X.append(img_array)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    print(f"{augmentation_type} dataset loaded: {X.shape[0]} samples")
    return X, y


# TRAIN ALL MODELS

def train_models(X_train, y_train):
    print("\nTraining models on ORIGINAL images only...")

    models = {}

    # SVM
    print("Training SVM...")
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm_model.fit(X_train, y_train)
    models["SVM"] = svm_model

    # SGD
    print("Training SGD...")
    sgd_model = SGDClassifier(
        loss='hinge',
        alpha=0.01,
        max_iter=1000,
        tol=1e-3,
        random_state=42
    )
    sgd_model.fit(X_train, y_train)
    models["SGD"] = sgd_model

    # Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    models["RandomForest"] = rf_model

    # MLP
    print("Training MLP...")
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(100,),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        learning_rate_init=0.001,
        max_iter=200,
        random_state=42
    )
    mlp_model.fit(X_train, y_train)
    models["MLP"] = mlp_model

    return models


# 4. EVALUATE ONE MODEL

def evaluate_model(model, X_test, y_test, label):
    # predicts labels for this test set
    y_pred = model.predict(X_test)

    # Gives basic accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Full classification report in dictionary form
    report = classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # This prints a readable summary
    print(f"\n--- {label} ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(
        y_test,
        y_pred,
        target_names=CLASS_NAMES,
        zero_division=0
    ))
    print("Confusion Matrix:")
    print(cm)

    # This returns everything needed for later analysis
    metrics = {
        "accuracy": accuracy,
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
        "f1_macro": report["macro avg"]["f1-score"],
        "precision_weighted": report["weighted avg"]["precision"],
        "recall_weighted": report["weighted avg"]["recall"],
        "f1_weighted": report["weighted avg"]["f1-score"],
        "confusion_matrix": cm,
        "y_pred": y_pred
    }

    return metrics


# SAVE A CONFUSION MATRIX IMAGE

def save_confusion_matrix_plot(cm, class_names, title, output_file):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Puts the raw counts inside each square
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


# SAVE METRICS TO A CSV

def save_metrics_csv(results, output_file='performance_metrics.csv'):
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header row
        writer.writerow([
            'condition',
            'model',
            'accuracy',
            'precision_macro',
            'recall_macro',
            'f1_macro',
            'precision_weighted',
            'recall_weighted',
            'f1_weighted'
        ])

        # One row per model per condition
        for condition, model_results in results.items():
            for model_name, metrics in model_results.items():
                writer.writerow([
                    condition,
                    model_name,
                    metrics["accuracy"],
                    metrics["precision_macro"],
                    metrics["recall_macro"],
                    metrics["f1_macro"],
                    metrics["precision_weighted"],
                    metrics["recall_weighted"],
                    metrics["f1_weighted"]
                ])


# PLOT ROBUSTNESS FOR ALL MODELS

def plot_robustness(results, output_file='robustness_plot_all_models.png'):
    conditions = list(results.keys())
    model_names = list(results[conditions[0]].keys())

    plt.figure(figsize=(12, 7))

    # This draws one line for each model
    for model_name in model_names:
        scores = [results[c][model_name]["accuracy"] for c in conditions]
        plt.plot(conditions, scores, marker='o', label=model_name)

    plt.xlabel('Condition')
    plt.ylabel('Accuracy')
    plt.title('Robustness Analysis Across Augmentations')
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    print(f"\nRobustness plot saved to {output_file}")



# BOOTSTRAP SIGNIFICANCE TESTING

def bootstrap_accuracy_diff(y_true, y_pred_a, y_pred_b, n_bootstrap=1000, random_state=42):
    rng = np.random.default_rng(random_state)
    n = len(y_true)

    diffs = []

    # This resamples the prediction results a lot
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)

        acc_a = np.mean(y_pred_a[idx] == y_true[idx])
        acc_b = np.mean(y_pred_b[idx] == y_true[idx])

        diffs.append(acc_a - acc_b)

    diffs = np.array(diffs)

    # 95% confidence interval
    mean_diff = np.mean(diffs)
    lower = np.percentile(diffs, 2.5)
    upper = np.percentile(diffs, 97.5)

    return mean_diff, lower, upper


def save_bootstrap_csv(results, y_true_by_condition, output_file='bootstrap_significance.csv'):
    model_names = list(next(iter(results.values())).keys())

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'condition',
            'model_a',
            'model_b',
            'mean_accuracy_difference',
            'ci_lower_95',
            'ci_upper_95'
        ])

        # Compares every pair of models within each condition
        for condition, model_results in results.items():
            y_true = y_true_by_condition[condition]

            for model_a, model_b in combinations(model_names, 2):
                y_pred_a = model_results[model_a]["y_pred"]
                y_pred_b = model_results[model_b]["y_pred"]

                mean_diff, lower, upper = bootstrap_accuracy_diff(y_true, y_pred_a, y_pred_b)

                writer.writerow([
                    condition,
                    model_a,
                    model_b,
                    mean_diff,
                    lower,
                    upper
                ])


# MAIN PIPELINE!!! WOOP WOOP!!!

def main():
    print("=" * 60)
    print("TASKS 4 AND 5 - ADDITIONAL CLASSIFIERS + ANALYSIS")
    print("=" * 60)

    # Creates output folders 
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/confusion_matrices", exist_ok=True)


    # LOAD ORIGINAL DATA

    X, y = load_original_dataset(
        target_size=TARGET_SIZE,
        max_samples_per_class=MAX_SAMPLES_PER_CLASS
    )

    
    # SCALE FEATURES

    # This learns scaling from the original dataset only
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

  
    # TRAIN / TEST SPLIT
   
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Original test set size: {X_test.shape[0]}")

    
    # TRAIN MODELS
  
    models = train_models(X_train, y_train)

    # This dictionary will store all results
    results = {}

    # This stores the true labels for each condition
    y_true_by_condition = {}

   
    # EVALUATE ON ORIGINAL TEST SET

    results["original"] = {}
    y_true_by_condition["original"] = y_test

    for model_name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, f"{model_name} on original test set")
        results["original"][model_name] = metrics

        save_confusion_matrix_plot(
            metrics["confusion_matrix"],
            CLASS_NAMES,
            f"{model_name} - original",
            os.path.join("results/confusion_matrices", f"{model_name}_original.png")
        )

    
    # EVALUATE ON EACH AUGMENTATION TYPE
   
    for aug_type in AUGMENTATION_TYPES:
        print("\n" + "=" * 60)
        print(f"TESTING AUGMENTATION: {aug_type}")
        print("=" * 60)

        X_aug, y_aug = load_augmentation_dataset(
            aug_type,
            target_size=TARGET_SIZE
        )

        X_aug_scaled = scaler.transform(X_aug)

        results[aug_type] = {}
        y_true_by_condition[aug_type] = y_aug

        for model_name, model in models.items():
            metrics = evaluate_model(model, X_aug_scaled, y_aug, f"{model_name} on {aug_type}")
            results[aug_type][model_name] = metrics

            save_confusion_matrix_plot(
                metrics["confusion_matrix"],
                CLASS_NAMES,
                f"{model_name} - {aug_type}",
                os.path.join("results/confusion_matrices", f"{model_name}_{aug_type}.png")
            )

  
    # SAVE METRICS CSV
   
    save_metrics_csv(results, output_file='results/performance_metrics.csv')

   
    # SAVE ROBUSTNESS PLOT
   
    plot_robustness(results, output_file='results/robustness_plot_all_models.png')

   
    # SAVE BOOTSTRAP SIGNIFICANCE CSV

    save_bootstrap_csv(
        results,
        y_true_by_condition,
        output_file='results/bootstrap_significance.csv'
    )

    
    # J. PRINT A SHORT SUMMARY
   
    print("\n" + "=" * 60)
    print("SUMMARY OF ACCURACY")
    print("=" * 60)

    for condition, model_results in results.items():
        print(f"\nCondition: {condition}")
        for model_name, metrics in model_results.items():
            print(f"  {model_name:14s} {metrics['accuracy']:.4f}")

    print("\nFinished.")
    print("Saved files:")
    print("- results/performance_metrics.csv")
    print("- results/bootstrap_significance.csv")
    print("- results/robustness_plot_all_models.png")
    print("- results/confusion_matrices/*.png")


# RUN MAIN!!!!!!!!!!!!!!!!

if __name__ == "__main__":
    main()