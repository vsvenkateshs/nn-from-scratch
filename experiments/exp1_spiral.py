import sys, os
sys.path.append("src")
import numpy as np
from numpy_nn import build_numpy_model, Tanh, ReLU
from utils import make_spiral, plot_decision_boundary, plot_curves
from train import train_numpy

os.makedirs("results", exist_ok=True)
np.random.seed(0)

# ---------- 1. Gradient check (proves your backprop is correct) ----------
def grad_check(model, X, y, eps=1e-5):
    layer = model.layers[0]
    model.train_step(X, y, lr=0.0)          # lr=0 -> computes grads, no update
    analytic = layer.grad_W.copy()
    numeric = np.zeros_like(layer.W)
    for i in range(layer.W.shape[0]):
        for j in range(layer.W.shape[1]):
            old = layer.W[i, j]
            layer.W[i, j] = old + eps
            lp = model.loss_fn.forward(model.forward(X), y)
            layer.W[i, j] = old - eps
            lm = model.loss_fn.forward(model.forward(X), y)
            layer.W[i, j] = old
            numeric[i, j] = (lp - lm) / (2 * eps)
    rel_err = np.abs(analytic - numeric).max() / (np.abs(analytic) + np.abs(numeric)).max()
    return rel_err

Xs, ys = make_spiral(n=20, k=3)
check_model = build_numpy_model([2, 8, 3], act=Tanh)
print("Gradient check relative error:", grad_check(check_model, Xs, ys))
# Should print something below 1e-6. If not, you have a backprop bug.

# ---------- 2. Train on spiral ----------
X, y = make_spiral(n=200, k=3, noise=0.2)
idx = np.random.permutation(len(X))
split = int(0.8 * len(X))
tr, va = idx[:split], idx[split:]

model = build_numpy_model([2, 64, 64, 3], act=ReLU)
h = train_numpy(model, X[tr], y[tr], X[va], y[va], epochs=300, lr=0.5, batch_size=64)

plot_decision_boundary(model.predict, X, y, "NumPy NN on spiral", "results/spiral_boundary.png")
plot_curves({"NumPy": h}, "loss", "Loss", "Spiral training loss", "results/spiral_loss.png")