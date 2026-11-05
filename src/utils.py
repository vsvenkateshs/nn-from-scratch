import numpy as np
import matplotlib.pyplot as plt

def make_spiral(n=200, k=3, noise=0.2, seed=0):
    rng = np.random.RandomState(seed)
    X = np.zeros((n * k, 2))
    y = np.zeros(n * k, dtype=int)
    for c in range(k):
        ix = slice(n * c, n * (c + 1))
        r = np.linspace(0.0, 1, n)
        t = np.linspace(c * 4, (c + 1) * 4, n) + rng.randn(n) * noise
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = c
    return X, y

def load_mnist():
    from torchvision import datasets
    tr = datasets.MNIST("data", train=True, download=True)
    te = datasets.MNIST("data", train=False, download=True)
    X_tr = tr.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    X_te = te.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    return X_tr, tr.targets.numpy(), X_te, te.targets.numpy()

def plot_decision_boundary(predict_fn, X, y, title="", path=None):
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 0.1, X[:, 0].max() + 0.1, 300),
        np.linspace(X[:, 1].min() - 0.1, X[:, 1].max() + 0.1, 300),
    )
    Z = predict_fn(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.figure(figsize=(5, 5))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=10, edgecolors="k", linewidths=0.2)
    plt.title(title)
    if path:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()

def plot_curves(histories, key, ylabel, title, path=None):
    """histories: dict of {label: history_dict}"""
    plt.figure(figsize=(6, 4))
    for label, h in histories.items():
        plt.plot(h[key], label=label)
    plt.xlabel("Epoch"); plt.ylabel(ylabel); plt.title(title); plt.legend()
    plt.grid(alpha=0.3)
    if path:
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()