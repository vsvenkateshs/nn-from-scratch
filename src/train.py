import time
import numpy as np
import torch
import torch.nn as nn

def iterate_minibatches(X, y, batch_size):
    idx = np.random.permutation(len(X))
    for i in range(0, len(X), batch_size):
        b = idx[i:i + batch_size]
        yield X[b], y[b]

def train_numpy(model, X_tr, y_tr, X_val, y_val, epochs=10, lr=0.1, batch_size=64):
    h = {"loss": [], "train_acc": [], "val_acc": [], "time": 0.0}
    start = time.perf_counter()
    for ep in range(epochs):
        losses = [model.train_step(xb, yb, lr)
                  for xb, yb in iterate_minibatches(X_tr, y_tr, batch_size)]
        h["loss"].append(float(np.mean(losses)))
        h["train_acc"].append((model.predict(X_tr) == y_tr).mean())
        h["val_acc"].append((model.predict(X_val) == y_val).mean())
        print(f"[NumPy] epoch {ep+1}/{epochs} loss={h['loss'][-1]:.4f} val_acc={h['val_acc'][-1]:.4f}")
    h["time"] = time.perf_counter() - start
    return h

def train_torch(model, X_tr, y_tr, X_val, y_val, epochs=10, lr=0.1,
                batch_size=64, optimizer="sgd", weight_decay=0.0):
    Xt = torch.tensor(X_tr, dtype=torch.float32); yt = torch.tensor(y_tr, dtype=torch.long)
    Xv = torch.tensor(X_val, dtype=torch.float32); yv = torch.tensor(y_val, dtype=torch.long)
    if optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "momentum":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    h = {"loss": [], "train_acc": [], "val_acc": [], "time": 0.0}
    start = time.perf_counter()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        losses = []
        for i in range(0, len(Xt), batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            h["loss"].append(float(np.mean(losses)))
            h["train_acc"].append((model(Xt).argmax(1) == yt).float().mean().item())
            h["val_acc"].append((model(Xv).argmax(1) == yv).float().mean().item())
        print(f"[Torch] epoch {ep+1}/{epochs} loss={h['loss'][-1]:.4f} val_acc={h['val_acc'][-1]:.4f}")
    h["time"] = time.perf_counter() - start
    return h

def copy_weights_numpy_to_torch(np_model, torch_model):
    """Give PyTorch the exact same starting weights for a fair comparison."""
    from numpy_nn import Dense
    dense = [l for l in np_model.layers if isinstance(l, Dense)]
    linear = [m for m in torch_model if isinstance(m, nn.Linear)]
    with torch.no_grad():
        for t, n in zip(linear, dense):
            t.weight.copy_(torch.tensor(n.W.T, dtype=torch.float32))
            t.bias.copy_(torch.tensor(n.b.ravel(), dtype=torch.float32))