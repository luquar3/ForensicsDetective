import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def load_original_dataset(
    word_dir='word_pdfs_png',
    google_dir='google_docs_pdfs_png',
    python_dir='python_pdfs_png',
    target_size=(200, 200),
    max_samples_per_class=None
):
    X = []
    y = []

    # This stores the three class folders in one structure
    folder_info = [
        (word_dir, 0, "Word"),
        (google_dir, 1, "Google"),
        (python_dir, 2, "Python")
    ]

    # This keeps all the png files only
    for folder, label, name in folder_info:
        files = [f for f in os.listdir(folder) if f.endswith('.png')]

        if max_samples_per_class is not None:
            files = files[:max_samples_per_class]

        print(f"Loading {len(files)} original {name} images...")

        # Build path, convert to grayscale, resize, flatten
        for filename in files:
            path = os.path.join(folder, filename)
            img = Image.open(path).convert('L')
            img = img.resize(target_size, Image.LANCZOS)
            img_array = np.array(img).flatten()

            X.append(img_array)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    print(f"Original dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class counts: Word={np.sum(y==0)}, Google={np.sum(y==1)}, Python={np.sum(y==2)}")

    return X, y


def load_augmentation_dataset(
    augmentation_type,
    augmented_root='augmented_images',
    target_size=(200, 200)
):
    X = []
    y = []

    folder_info = [
        (os.path.join(augmented_root, 'word_pdfs_png'), 0, "Word"),
        (os.path.join(augmented_root, 'google_docs_pdfs_png'), 1, "Google"),
        (os.path.join(augmented_root, 'python_pdfs_png'), 2, "Python")
    ]

    for folder, label, name in folder_info:
        files = [f for f in os.listdir(folder) if f"__{augmentation_type}" in f]

        print(f"Loading {len(files)} {augmentation_type} {name} images...")

        for filename in files:
            path = os.path.join(folder, filename)
            img = Image.open(path).convert('L')
            img = img.resize(target_size, Image.LANCZOS)
            img_array = np.array(img).flatten()

            X.append(img_array)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    print(f"{augmentation_type} dataset loaded: {X.shape[0]} samples")
    return X, y


def train_models(X_train, y_train):
    print("\nTraining models on ORIGINAL images only...")

    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm_model.fit(X_train, y_train)

    sgd_model = SGDClassifier(
        loss='hinge',
        alpha=0.01,
        max_iter=1000,
        tol=1e-3,
        random_state=42
    )
    sgd_model.fit(X_train, y_train)

    return svm_model, sgd_model


def evaluate_model(model, X_test, y_test, label):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n--- {label} ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Word', 'Google', 'Python']))
    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))

    return accuracy


def plot_robustness(results, output_file='robustness_plot.png'):
    conditions = list(results.keys())
    svm_scores = [results[c]['svm'] for c in conditions]
    sgd_scores = [results[c]['sgd'] for c in conditions]

    plt.figure(figsize=(10, 6))
    plt.plot(conditions, svm_scores, marker='o', label='SVM')
    plt.plot(conditions, sgd_scores, marker='o', label='SGD')
    plt.xlabel('Condition')
    plt.ylabel('Accuracy')
    plt.title('Augmentations Robustness Analysis')
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    print(f"\nRobustness plot saved to {output_file}")


def main():
    print("=" * 60)
    print("TASK 3 - ROBUSTNESS ANALYSIS")
    print("=" * 60)

    # 1. Load original images only
    X, y = load_original_dataset()

    # 2. Scale original features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Split originals into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Original test set size: {X_test.shape[0]}")

    # 4. Train models on original training data only
    svm_model, sgd_model = train_models(X_train, y_train)

    # 5. Store results
    results = {}

    # 6. Baseline evaluation on original test set
    results['original'] = {
        'svm': evaluate_model(svm_model, X_test, y_test, 'SVM on original test set'),
        'sgd': evaluate_model(sgd_model, X_test, y_test, 'SGD on original test set')
    }

    # 7. Evaluate on each augmentation type separately
    for aug_type in ['noise', 'jpeg', 'downsample', 'crop', 'bitdepth']:
        print("\n" + "=" * 60)
        print(f"TESTING AUGMENTATION: {aug_type}")
        print("=" * 60)

        X_aug, y_aug = load_augmentation_dataset(aug_type)

        # IMPORTANT: use the scaler fitted on original data
        X_aug_scaled = scaler.transform(X_aug)

        results[aug_type] = {
            'svm': evaluate_model(svm_model, X_aug_scaled, y_aug, f'SVM on {aug_type}'),
            'sgd': evaluate_model(sgd_model, X_aug_scaled, y_aug, f'SGD on {aug_type}')
        }

    # 8. Print Summary
    print("\n" + "=" * 60)
    print("ROBUSTNESS SUMMARY")
    print("=" * 60)

    for condition, scores in results.items():
        print(f"{condition:12s} SVM={scores['svm']:.4f} SGD={scores['sgd']:.4f}")

    # 9. Save robustness plot
    plot_robustness(results)


if __name__ == "__main__":
    main()