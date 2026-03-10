# -*- coding: utf-8 -*-
"""
표 형태 데이터(2만+, 원인변수 10개) → 이탈 여부 예측.
ANN(은닉 1층) / DNN(은닉 2층) / DNN(은닉 5층) 샘플 모델 3개 비교 및 시각화.
"""
import numpy as np
import os

_USE_TF = False
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    _USE_TF = True
except ImportError as e:
    print("[안내] TensorFlow 미사용, NumPy 버전으로 실행합니다:", str(e)[:80], "\n")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False
    print("[안내] matplotlib 없음. 시각화는 건너뜁니다.\n")

# 상수
NUM_SAMPLES = 20_000
NUM_FEATURES = 10
EPOCHS = 15
BATCH_SIZE = 64
VAL_SPLIT = 0.2
RANDOM_SEED = 42
OUTPUT_DIR = "churn_results"


def generate_churn_data(n_samples: int, n_features: int, seed: int) -> tuple:
    """학습 가능한 패턴이 있는 이탈 데이터 생성 (원인변수 n_features개)."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n_samples, n_features)).astype(np.float64)
    # 이탈 확률: 특징 선형결합 + 노이즈 → sigmoid
    w_true = rng.standard_normal(n_features) * 0.5
    logit = X @ w_true + rng.standard_normal(n_samples) * 0.3
    p = 1.0 / (1.0 + np.exp(-np.clip(logit, -20, 20)))
    y = (rng.random(n_samples) < p).astype(np.float64).reshape(-1, 1)
    return X, y


# ---------- TensorFlow: 입력층 / 은닉층 / 출력층 명확한 3개 모델 ----------

def build_model_1hidden(input_dim: int):
    """샘플 1: ANN - 입력층 → 은닉층 1개 → 출력층."""
    return models.Sequential([
        layers.Input(shape=(input_dim,), name="input_layer"),
        layers.Dense(64, activation="relu", name="hidden_layer_1"),
        layers.Dense(1, activation="sigmoid", name="output_layer"),
    ], name="ANN_1hidden")


def build_model_2hidden(input_dim: int):
    """샘플 2: DNN - 입력층 → 은닉층 2개 → 출력층."""
    return models.Sequential([
        layers.Input(shape=(input_dim,), name="input_layer"),
        layers.Dense(64, activation="relu", name="hidden_layer_1"),
        layers.Dense(32, activation="relu", name="hidden_layer_2"),
        layers.Dense(1, activation="sigmoid", name="output_layer"),
    ], name="DNN_2hidden")


def build_model_5hidden(input_dim: int):
    """샘플 3: DNN - 입력층 → 은닉층 5개 → 출력층."""
    return models.Sequential([
        layers.Input(shape=(input_dim,), name="input_layer"),
        layers.Dense(64, activation="relu", name="hidden_layer_1"),
        layers.Dense(48, activation="relu", name="hidden_layer_2"),
        layers.Dense(32, activation="relu", name="hidden_layer_3"),
        layers.Dense(24, activation="relu", name="hidden_layer_4"),
        layers.Dense(16, activation="relu", name="hidden_layer_5"),
        layers.Dense(1, activation="sigmoid", name="output_layer"),
    ], name="DNN_5hidden")


def run_tensorflow_models(X_tr, y_tr, X_val, y_val, epochs, batch_size):
    """TF로 3개 모델 학습, history 반환."""
    tf.random.set_seed(RANDOM_SEED)
    results = []
    builders = [
        ("ANN (은닉 1층)", build_model_1hidden),
        ("DNN (은닉 2층)", build_model_2hidden),
        ("DNN (은닉 5층)", build_model_5hidden),
    ]
    for name, build_fn in builders:
        model = build_fn(X_tr.shape[1])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        print("\n---", name, "---")
        model.summary()
        hist = model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
        )
        loss, acc = model.evaluate(X_val, y_val, verbose=0)
        results.append({
            "name": name,
            "history": hist.history,
            "val_loss": float(loss),
            "val_accuracy": float(acc),
        })
    return results


# ---------- NumPy: 다층 MLP (은닉 1/2/5 지원) ----------

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def _relu(z):
    return np.maximum(0, z)


def _relu_derivative(z):
    return (z > 0).astype(np.float64)


class NumpyMLP:
    """은닉층 개수/유닛 수를 리스트로 받는 MLP. fit 시 history 반환."""

    def __init__(self, input_dim: int, hidden_units: list, lr: float = 0.01):
        self.lr = lr
        dims = [input_dim] + hidden_units + [1]
        self.Ws = [np.random.randn(dims[i], dims[i + 1]) * 0.1 for i in range(len(dims) - 1)]
        self.bs = [np.zeros((1, dims[i + 1])) for i in range(len(dims) - 1)]
        self.n_hidden = len(hidden_units)

    def forward(self, X: np.ndarray):
        self._cache = [X]
        a = X
        for i in range(len(self.Ws) - 1):
            z = a @ self.Ws[i] + self.bs[i]
            a = _relu(z)
            self._cache.append((z, a))
        z_out = a @ self.Ws[-1] + self.bs[-1]
        a_out = _sigmoid(z_out)
        self._cache.append((z_out, a_out))
        return a_out

    def backward(self, y: np.ndarray):
        m = y.shape[0]
        d = self._cache
        # 출력층 그래디언트: dL/d(z_L) = (a_L - y) / m
        dZ = (d[-1][1] - y) / m
        for i in range(len(self.Ws) - 1, -1, -1):
            a_prev = d[i][1] if i > 0 else d[0]
            self.Ws[i] -= self.lr * (a_prev.T @ dZ)
            self.bs[i] -= self.lr * np.sum(dZ, axis=0, keepdims=True)
            if i > 0:
                dZ = (dZ @ self.Ws[i].T) * _relu_derivative(d[i][0])

    def fit_with_history(self, X, y, epochs, batch_size, X_val, y_val):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        X_val = np.asarray(X_val, dtype=np.float64)
        y_val = np.asarray(y_val, dtype=np.float64)
        n = X.shape[0]
        hist = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
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
            pred_val = self.forward(X_val)
            v_loss = np.mean(-y_val * np.log(pred_val + 1e-8) - (1 - y_val) * np.log(1 - pred_val + 1e-8))
            v_acc = np.mean((pred_val >= 0.5).astype(np.float64) == y_val)
            hist["loss"].append(float(loss))
            hist["accuracy"].append(float(acc))
            hist["val_loss"].append(float(v_loss))
            hist["val_accuracy"].append(float(v_acc))
            if (ep + 1) % 3 == 0 or ep == 0:
                print(f"  Epoch {ep + 1}/{epochs} - loss: {loss:.4f} - acc: {acc:.4f} - val_loss: {v_loss:.4f} - val_acc: {v_acc:.4f}")
        return hist

    def evaluate(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        pred = self.forward(X)
        loss = np.mean(-y * np.log(pred + 1e-8) - (1 - y) * np.log(1 - pred + 1e-8))
        acc = np.mean((pred >= 0.5).astype(np.float64) == y)
        return float(loss), float(acc)


def run_numpy_models(X_tr, y_tr, X_val, y_val, epochs, batch_size):
    """NumPy MLP로 3개 모델 학습, TF와 동일한 이름/순서."""
    np.random.seed(RANDOM_SEED)
    configs = [
        ("ANN (은닉 1층)", [64]),
        ("DNN (은닉 2층)", [64, 32]),
        ("DNN (은닉 5층)", [64, 48, 32, 24, 16]),
    ]
    results = []
    for name, hidden in configs:
        print("\n---", name, "---")
        print("  구조: 입력층(%d) -> " % X_tr.shape[1] + " -> ".join(f"은닉층{i+1}({u})" for i, u in enumerate(hidden)) + " -> 출력층(1)")
        model = NumpyMLP(X_tr.shape[1], hidden, lr=0.01)
        hist = model.fit_with_history(X_tr, y_tr, epochs, batch_size, X_val, y_val)
        val_loss, val_acc = model.evaluate(X_val, y_val)
        results.append({"name": name, "history": hist, "val_loss": val_loss, "val_accuracy": val_acc})
    return results


# ---------- 학습 실행 및 검증 분리 ----------

def train_val_split(X, y, val_ratio, seed):
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(n * val_ratio)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]


# ---------- 결과 분석 및 시각화 ----------

def print_analysis(results):
    print("\n" + "=" * 60)
    print("결과 요약 (검증 집합 기준)")
    print("=" * 60)
    for r in results:
        print(f"  {r['name']}: val_loss={r['val_loss']:.4f}, val_accuracy={r['val_accuracy']:.4f}")
    print("=" * 60)
    best = max(results, key=lambda x: x["val_accuracy"])
    print(f"  최고 검증 정확도: {best['name']} ({best['val_accuracy']:.4f})")
    print("=" * 60)


def save_results_html(results, backend: str, save_dir: str):
    """matplotlib 없을 때 HTML 표/막대로 요약 저장."""
    if not results:
        return
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "churn_comparison.html")
    rows = "".join(
        f"<tr><td>{r['name']}</td><td>{r['val_loss']:.4f}</td><td>{r['val_accuracy']:.4f}</td>"
        f"<td><div style='width:{r['val_accuracy']*100:.0f}%;background:#3498db;height:20px'></div></td></tr>"
        for r in results
    )
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Churn 비교</title></head><body>
<h2>이탈 예측 모델 비교 ({backend}, n={NUM_SAMPLES}, features={NUM_FEATURES})</h2>
<table border="1" cellpadding="8">
<thead><tr><th>모델</th><th>Val Loss</th><th>Val Accuracy</th><th>Accuracy 막대</th></tr></thead>
<tbody>{rows}</tbody></table>
</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n결과 요약 HTML 저장: {os.path.abspath(path)}")


def plot_results(results, backend: str, save_dir: str):
    if not results:
        return
    os.makedirs(save_dir, exist_ok=True)
    if not _HAS_MATPLOTLIB:
        save_results_html(results, backend, save_dir)
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    labels = ["ANN (1 hidden)", "DNN (2 hidden)", "DNN (5 hidden)"]
    # 1) Loss curves
    ax = axes[0, 0]
    for i, r in enumerate(results):
        h = r["history"]
        lb = labels[i] if i < len(labels) else r["name"]
        ax.plot(h["loss"], label=lb + " train", alpha=0.8)
        ax.plot(h["val_loss"], "--", label=lb + " val", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss (train / val)")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)

    # 2) Accuracy curves
    ax = axes[0, 1]
    for i, r in enumerate(results):
        h = r["history"]
        lb = labels[i] if i < len(labels) else r["name"]
        ax.plot(h["accuracy"], label=lb + " train", alpha=0.8)
        ax.plot(h["val_accuracy"], "--", label=lb + " val", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy (train / val)")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)

    # 3) Final validation loss bar
    ax = axes[1, 0]
    names = [r["name"] for r in results]
    val_losses = [r["val_loss"] for r in results]
    ax.bar(range(len(names)), val_losses, color=["#2ecc71", "#3498db", "#9b59b6"], alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(["ANN (1H)", "DNN (2H)", "DNN (5H)"])
    ax.set_ylabel("Val Loss")
    ax.set_title("Final validation loss")

    # 4) Final validation accuracy bar
    ax = axes[1, 1]
    val_accs = [r["val_accuracy"] for r in results]
    ax.bar(range(len(names)), val_accs, color=["#2ecc71", "#3498db", "#9b59b6"], alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(["ANN (1H)", "DNN (2H)", "DNN (5H)"])
    ax.set_ylabel("Val Accuracy")
    ax.set_title("Final validation accuracy")

    plt.suptitle(f"Churn prediction comparison ({backend}, n={NUM_SAMPLES}, features={NUM_FEATURES})", fontsize=11)
    plt.tight_layout()
    path = os.path.join(save_dir, "churn_comparison.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n시각화 저장: {os.path.abspath(path)}")


def main():
    print("데이터 생성: n=%d, 원인변수=%d" % (NUM_SAMPLES, NUM_FEATURES))
    X, y = generate_churn_data(NUM_SAMPLES, NUM_FEATURES, RANDOM_SEED)
    X_tr, y_tr, X_val, y_val = train_val_split(X, y, VAL_SPLIT, RANDOM_SEED)
    print("학습 %d / 검증 %d" % (X_tr.shape[0], X_val.shape[0]))

    if _USE_TF:
        X_tr, y_tr = X_tr.astype("float32"), y_tr.astype("float32")
        X_val, y_val = X_val.astype("float32"), y_val.astype("float32")
        results = run_tensorflow_models(X_tr, y_tr, X_val, y_val, EPOCHS, BATCH_SIZE)
        backend = "TensorFlow"
    else:
        results = run_numpy_models(X_tr, y_tr, X_val, y_val, EPOCHS, BATCH_SIZE)
        backend = "NumPy"

    print_analysis(results)
    plot_results(results, backend, OUTPUT_DIR)


if __name__ == "__main__":
    main()
