"""
표 형태 데이터 → 이탈 여부 예측 ANN/DNN 예제.
TensorFlow 사용 가능 시 Keras 사용, 불가 시 NumPy만으로 동일 구조 학습.
"""
import numpy as np

# TensorFlow 사용 가능 여부 (DLL 등 ImportError 시 NumPy fallback)
_USE_TF = False
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    _USE_TF = True
except ImportError as e:
    print(f"[안내] TensorFlow를 불러올 수 없어 NumPy 버전으로 실행합니다. ({e})\n")


# ---------- NumPy 전용 구현 (입력층 - 은닉층 - 출력층 구조 동일) ----------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def _relu_derivative(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(np.float64)


class NumpyANN:
    """단일 은닉층 ANN: 입력층 - 은닉층1 - 출력층."""

    def __init__(self, input_dim: int, hidden: int = 16, lr: float = 0.01):
        self.lr = lr
        self.W1 = np.random.randn(input_dim, hidden) * 0.1
        self.b1 = np.zeros((1, hidden))
        self.W2 = np.random.randn(hidden, 1) * 0.1
        self.b2 = np.zeros((1, 1))

    def forward(self, X: np.ndarray):
        self._X = X
        self._Z1 = X @ self.W1 + self.b1
        self._A1 = _relu(self._Z1)
        self._Z2 = self._A1 @ self.W2 + self.b2
        self._A2 = _sigmoid(self._Z2)
        return self._A2

    def backward(self, y: np.ndarray):
        m = y.shape[0]
        dZ2 = self._A2 - y
        dW2 = (self._A1.T @ dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * _relu_derivative(self._Z1)
        dW1 = (self._X.T @ dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 5, batch_size: int = 32):
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
        n = X.shape[0]
        for ep in range(epochs):
            perm = np.random.permutation(n)
            loss_sum, correct = 0.0, 0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                Xb, yb = X[idx], y[idx]
                pred = self.forward(Xb)
                self.backward(yb)
                loss_sum += np.sum(-yb * np.log(pred + 1e-8) - (1 - yb) * np.log(1 - pred + 1e-8))
                correct += np.sum((pred >= 0.5).astype(np.float64) == yb)
            loss = loss_sum / n
            acc = correct / n
            print(f"  Epoch {ep + 1}/{epochs} - loss: {loss:.4f} - accuracy: {acc:.4f}")

    def evaluate(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        pred = self.forward(X)
        loss = np.mean(-y * np.log(pred + 1e-8) - (1 - y) * np.log(1 - pred + 1e-8))
        acc = np.mean((pred >= 0.5).astype(np.float64) == y)
        return float(loss), float(acc)


class NumpyDNN:
    """은닉층 2개 DNN: 입력층 - 은닉층1 - 은닉층2 - 출력층."""

    def __init__(self, input_dim: int, hidden1: int = 32, hidden2: int = 16, lr: float = 0.01):
        self.lr = lr
        self.W1 = np.random.randn(input_dim, hidden1) * 0.1
        self.b1 = np.zeros((1, hidden1))
        self.W2 = np.random.randn(hidden1, hidden2) * 0.1
        self.b2 = np.zeros((1, hidden2))
        self.W3 = np.random.randn(hidden2, 1) * 0.1
        self.b3 = np.zeros((1, 1))

    def forward(self, X: np.ndarray):
        self._X = X
        self._Z1 = X @ self.W1 + self.b1
        self._A1 = _relu(self._Z1)
        self._Z2 = self._A1 @ self.W2 + self.b2
        self._A2 = _relu(self._Z2)
        self._Z3 = self._A2 @ self.W3 + self.b3
        self._A3 = _sigmoid(self._Z3)
        return self._A3

    def backward(self, y: np.ndarray):
        m = y.shape[0]
        dZ3 = self._A3 - y
        dW3 = (self._A2.T @ dZ3) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m
        dA2 = dZ3 @ self.W3.T
        dZ2 = dA2 * _relu_derivative(self._Z2)
        dW2 = (self._A1.T @ dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * _relu_derivative(self._Z1)
        dW1 = (self._X.T @ dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 5, batch_size: int = 32):
        X, y = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64)
        n = X.shape[0]
        for ep in range(epochs):
            perm = np.random.permutation(n)
            loss_sum, correct = 0.0, 0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                Xb, yb = X[idx], y[idx]
                pred = self.forward(Xb)
                self.backward(yb)
                loss_sum += np.sum(-yb * np.log(pred + 1e-8) - (1 - yb) * np.log(1 - pred + 1e-8))
                correct += np.sum((pred >= 0.5).astype(np.float64) == yb)
            loss = loss_sum / n
            acc = correct / n
            print(f"  Epoch {ep + 1}/{epochs} - loss: {loss:.4f} - accuracy: {acc:.4f}")

    def evaluate(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        pred = self.forward(X)
        loss = np.mean(-y * np.log(pred + 1e-8) - (1 - y) * np.log(1 - pred + 1e-8))
        acc = np.mean((pred >= 0.5).astype(np.float64) == y)
        return float(loss), float(acc)


def _run_numpy_main(num_samples: int, num_features: int, epochs: int, batch_size: int) -> None:
    np.random.seed(42)
    X = np.random.rand(num_samples, num_features).astype(np.float64)
    y = np.random.randint(0, 2, size=(num_samples, 1)).astype(np.float64)

    print("=== 단일 은닉층 ANN (NumPy) ===")
    print("  구조: 입력층(%d) -> 은닉층1(16, ReLU) -> 출력층(1, Sigmoid)" % num_features)
    ann = NumpyANN(input_dim=num_features, hidden=16, lr=0.01)
    ann.fit(X, y, epochs=epochs, batch_size=batch_size)
    loss, acc = ann.evaluate(X, y)
    print(f"  ANN 최종 loss: {loss:.4f}, accuracy: {acc:.4f}\n")

    print("=== 은닉층 2개 DNN (NumPy) ===")
    print("  구조: 입력층(%d) -> 은닉층1(32) -> 은닉층2(16) -> 출력층(1, Sigmoid)" % num_features)
    dnn = NumpyDNN(input_dim=num_features, hidden1=32, hidden2=16, lr=0.01)
    dnn.fit(X, y, epochs=epochs, batch_size=batch_size)
    loss, acc = dnn.evaluate(X, y)
    print(f"  DNN 최종 loss: {loss:.4f}, accuracy: {acc:.4f}")


# ---------- TensorFlow/Keras 구현 ----------

def _build_ann_model(input_dim: int):
    return models.Sequential(
        [
            layers.Input(shape=(input_dim,), name="input_layer"),
            layers.Dense(16, activation="relu", name="hidden_layer_1"),
            layers.Dense(1, activation="sigmoid", name="output_layer"),
        ]
    )


def _build_dnn_model(input_dim: int):
    return models.Sequential(
        [
            layers.Input(shape=(input_dim,), name="input_layer"),
            layers.Dense(32, activation="relu", name="hidden_layer_1"),
            layers.Dense(16, activation="relu", name="hidden_layer_2"),
            layers.Dense(1, activation="sigmoid", name="output_layer"),
        ]
    )


def _run_tf_main(num_samples: int, num_features: int, epochs: int, batch_size: int) -> None:
    np.random.seed(42)
    tf.random.set_seed(42)
    X = np.random.rand(num_samples, num_features).astype("float32")
    y = np.random.randint(0, 2, size=(num_samples, 1)).astype("float32")

    print("=== 단일 은닉층 ANN (Keras) ===")
    ann = _build_ann_model(num_features)
    ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    ann.summary()
    ann.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
    loss, acc = ann.evaluate(X, y, verbose=0)
    print(f"ANN 최종 loss: {loss:.4f}, accuracy: {acc:.4f}\n")

    print("=== 은닉층 2개 DNN (Keras) ===")
    dnn = _build_dnn_model(num_features)
    dnn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    dnn.summary()
    dnn.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)
    loss, acc = dnn.evaluate(X, y, verbose=0)
    print(f"DNN 최종 loss: {loss:.4f}, accuracy: {acc:.4f}")


def main() -> None:
    num_samples = 1000
    num_features = 10
    epochs = 5
    batch_size = 32

    if _USE_TF:
        _run_tf_main(num_samples, num_features, epochs, batch_size)
    else:
        _run_numpy_main(num_samples, num_features, epochs, batch_size)


if __name__ == "__main__":
    main()
