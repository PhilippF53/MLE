import numpy as np


def parse_mnist_data(
    idx_file_training_samples: str,
    idx_file_training_labels: str,
    idx_file_test_samples: str,
    idx_file_test_labels: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    training_samples = parse_mnist_images(idx_file_training_samples)
    training_labels = parse_mnist_labels(idx_file_training_labels)

    test_samples = parse_mnist_images(idx_file_test_samples)
    test_labels = parse_mnist_labels(idx_file_test_labels)

    return training_samples, training_labels, test_samples, test_labels


def parse_mnist_images(idx_file_path: str) -> np.ndarray:
    with open(idx_file_path, "rb") as f:

        # read magic number
        f.read(4)
        num_img = int.from_bytes(f.read(4), "big")
        num_rows = int.from_bytes(f.read(4), "big")
        num_cols = int.from_bytes(f.read(4), "big")

        data = f.read()
        out = np.ndarray((num_img, num_rows, num_cols), np.uint8, data)
        return out


def parse_mnist_labels(idx_file_path: str) -> np.ndarray:
    with open(idx_file_path, "rb") as f:

        # read magic number
        f.read(4)
        num_item = int.from_bytes(f.read(4), "big")

        data = f.read()
        out = np.ndarray((num_item, 1), np.uint8, data)
        return out


if __name__ == "__main__":
    training_samples, training_labels, test_samples, test_labels = parse_mnist_data(
        "./assets/train_img.idx",
        "./assets/train_label.idx",
        "./assets/test_img.idx",
        "./assets/test_label.idx",
    )
