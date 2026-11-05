import sys, os
sys.path.append("src")
import numpy as np, torch
from numpy_nn import build_numpy_model, ReLU, Sigmoid, Tanh
from torch_nn import make_torch_model
from utils import load_mnist, plot_curves
from train import train_numpy, train_torch

os.makedirs("results", exist_ok=True)
X, y, _, _ = load_mnist()
# Smaller subset keeps the many runs fast
X_tr, y_tr, X_val, y_val = X[:10000], y[:10000], X[50000:52000], y[50000:52000]
SIZES, EPOCHS = [784, 128, 64, 10], 8

def run_np(act=ReLU, init="he", lr=0.1, bs=64):
    np.random.seed(0)
    m = build_numpy_model(SIZES, act, init)
    return train_numpy(m, X_tr, y_tr, X_val, y_val, EPOCHS, lr, bs)

# A) Activation functions
res = {n: run_np(act=a) for n, a in [("ReLU", ReLU), ("Tanh", Tanh), ("Sigmoid", Sigmoid)]}
plot_curves(res, "val_acc", "Val accuracy", "Activation comparison", "results/abl_activation.png")

# B) Learning rates
res = {f"lr={lr}": run_np(lr=lr) for lr in [0.001, 0.01, 0.1, 0.5]}
plot_curves(res, "loss", "Train loss", "Learning rate comparison", "results/abl_lr.png")

# C) Weight initialization (zeros should fail: every neuron learns the same thing)
res = {i: run_np(init=i) for i in ["he", "small", "zeros"]}
plot_curves(res, "val_acc", "Val accuracy", "Weight init comparison", "results/abl_init.png")

# D) Batch size
res = {f"bs={b}": run_np(bs=b) for b in [16, 64, 256]}
plot_curves(res, "val_acc", "Val accuracy", "Batch size comparison", "results/abl_batch.png")

# E) Optimizers (PyTorch)
res = {}
for name, lr in [("sgd", 0.1), ("momentum", 0.05), ("adam", 0.001)]:
    torch.manual_seed(0)
    res[name] = train_torch(make_torch_model(SIZES), X_tr, y_tr, X_val, y_val,
                            EPOCHS, lr, 64, optimizer=name)
plot_curves(res, "val_acc", "Val accuracy", "Optimizer comparison (PyTorch)", "results/abl_optim.png")

# F) Overfitting and regularization: tiny training set, with and without L2
Xs, ys = X[:500], y[:500]
res = {}
for name, wd in [("no regularization", 0.0), ("L2 weight decay", 1e-3)]:
    torch.manual_seed(0)
    res[name] = train_torch(make_torch_model(SIZES), Xs, ys, X_val, y_val,
                            epochs=60, lr=0.1, batch_size=32, weight_decay=wd)
plot_curves(res, "val_acc", "Val accuracy", "Overfitting: L2 effect", "results/abl_l2.png")