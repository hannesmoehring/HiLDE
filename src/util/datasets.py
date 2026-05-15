import numpy as np


def train_val_split(X, Y, val_ratio=0.2, seed=0):
    assert X.shape[1] == Y.shape[1]

    rng = np.random.default_rng(seed)
    N = X.shape[1]

    idx = rng.permutation(N)
    n_val = int(N * val_ratio)

    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    X_train = X[:, train_idx]
    Y_train = Y[:, train_idx]

    X_val = X[:, val_idx]
    Y_val = Y[:, val_idx]

    return X_train, Y_train, X_val, Y_val


def one_hot_array(digits):
    result = []
    for d in digits:
        d = int(d)
        if not isinstance(d, int) or not (0 <= d <= 9):
            raise ValueError(f"not in bounds for mnist data: {d}")

        arr = [0] * 10
        arr[d] = 1
        result.append(arr)
    return result


def load_mnist_images(path):
    with open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic} in {path}")

        n_images = int.from_bytes(f.read(4), "big")
        rows = int.from_bytes(f.read(4), "big")
        cols = int.from_bytes(f.read(4), "big")

        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(n_images, rows, cols)

    return data


def load_mnist_labels(path):
    with open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} in {path}")

        n_labels = int.from_bytes(f.read(4), "big")
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels


def load_mnist_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train = load_mnist_images("datasets/mnist/raw/train-images-idx3-ubyte")
    y_train = load_mnist_labels("datasets/mnist/raw/train-labels-idx1-ubyte")
    X_test = load_mnist_images("datasets/mnist/raw/t10k-images-idx3-ubyte")
    y_test = load_mnist_labels("datasets/mnist/raw/t10k-labels-idx1-ubyte")

    X_train_flat = X_train.reshape(X_train.shape[0], 784)
    X_test_flat = X_test.reshape(X_test.shape[0], 784)
    X_train_flat = X_train_flat.astype(np.float64) / 255.0
    X_test_flat = X_test_flat.astype(np.float64) / 255.0
    return X_train_flat, one_hot_array(y_train), X_test_flat, one_hot_array(y_test)
