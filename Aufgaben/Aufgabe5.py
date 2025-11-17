import numpy as np
from collections import Counter


def parse_mnist_data(
    idx_file_training_samples: str,
    idx_file_training_labels: str,
    idx_file_test_samples: str,
    idx_file_test_labels: str,
    num_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    training_samples = parse_mnist_images(idx_file_training_samples)
    training_labels = parse_mnist_labels(idx_file_training_labels)

    test_samples = parse_mnist_images(idx_file_test_samples)
    test_labels = parse_mnist_labels(idx_file_test_labels)

    return (
        training_samples[:num_samples],
        training_labels[:num_samples],
        test_samples[:num_samples],
        test_labels[:num_samples],
    )


def parse_mnist_images(idx_file_path: str) -> np.ndarray:
    with open(idx_file_path, "rb") as f:

        # read magic number
        f.read(4)
        num_img = int.from_bytes(f.read(4), "big")
        num_rows = int.from_bytes(f.read(4), "big")
        num_cols = int.from_bytes(f.read(4), "big")

        data = f.read()
        out = np.ndarray((num_img, num_rows, num_cols), np.uint8, data) / 255.0
        return out


def parse_mnist_labels(idx_file_path: str) -> np.ndarray:
    with open(idx_file_path, "rb") as f:

        # read magic number
        f.read(4)
        num_item = int.from_bytes(f.read(4), "big")

        data = f.read()
        out = np.ndarray((num_item, 1), np.uint8, data)
        return out


def knn(test_sample, training_samples, training_labels, k):
    distances = np.linalg.norm(training_samples - test_sample, axis=1)

    k_lowest_idx = np.argsort(distances)[:k]

    k_labels = training_labels[k_lowest_idx]
    predicted = Counter(k_labels).most_common(1)[0][0]
    return predicted


if __name__ == "__main__":
    k = 1
    num_samples = 1000
    training_samples, training_labels, test_samples, test_labels = parse_mnist_data(
        "./assets/train_img.idx",
        "./assets/train_label.idx",
        "./assets/test_img.idx",
        "./assets/test_label.idx",
        num_samples,
    )
    training_samples_flat = training_samples.reshape(num_samples, -1)
    test_samples_flat = test_samples.reshape(num_samples, -1)
    training_labels = training_labels.flatten()
    test_labels = test_labels.flatten()

    correct = 0
    total = len(test_samples)

    for i in range(total):
        if i % 100 == 0:
            print(f"Case: {i} / {total}")
        predicted = knn(test_samples_flat[i], training_samples_flat, training_labels, k)
        true = test_labels[i]
        if predicted == true:
            correct += 1

    accuracy = correct / total
    print(f"Accuracy: {accuracy}")
