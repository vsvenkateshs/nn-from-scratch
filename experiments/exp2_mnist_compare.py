import sys, os
sys.path.append("src")
import numpy as np, torch
from numpy_nn import build_numpy_model, ReLU
from torch_nn import make_torch_model
from utils import load_mnist, plot_curves
from train import train_numpy, train_torch, copy_weights_numpy_to_torch

os.makedirs("results", exist_ok=True)
np.random.seed(0); torch.manual_seed(0)

X, y, X_test, y_test = load_mnist()
X_tr, y_tr, X_val, y_val = X[:50000], y[:50000], X[50000:], y[50000:]

SIZES = [784, 128, 64, 10]
EPOCHS, LR, BS = 10, 0.1, 64

np_model = build_numpy_model(SIZES, ReLU, init="he")
torch_model = make_torch_model(SIZES)
copy_weights_numpy_to_torch(np_model, torch_model)   # identical start

h_np = train_numpy(np_model, X_tr, y_tr, X_val, y_val, EPOCHS, LR, BS)
h_pt = train_torch(torch_model, X_tr, y_tr, X_val, y_val, EPOCHS, LR, BS, optimizer="sgd")

# Final test accuracy
np_test = (np_model.predict(X_test) == y_test).mean()
torch_model.eval()
with torch.no_grad():
    pt_test = (torch_model(torch.tensor(X_test)).argmax(1).numpy() == y_test).mean()

print("\n===== RESULTS =====")
print(f"NumPy  : test acc {np_test:.4f} | time {h_np['time']:.1f}s")
print(f"PyTorch: test acc {pt_test:.4f} | time {h_pt['time']:.1f}s")

histories = {"NumPy": h_np, "PyTorch": h_pt}
plot_curves(histories, "loss", "Train loss", "NumPy vs PyTorch: loss", "results/mnist_loss.png")
plot_curves(histories, "val_acc", "Val accuracy", "NumPy vs PyTorch: val accuracy", "results/mnist_val_acc.png")