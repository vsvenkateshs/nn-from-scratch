import numpy as np


class Dense:
    def __init__(self, n_in, n_out, init="he"):
        if init == "he":
            self.W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
        elif init == "small":
            self.W = np.random.randn(n_in, n_out) * 0.01
        elif init == "zeros":
            self.W = np.zeros((n_in, n_out))
        self.b = np.zeros((1, n_out))

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_out, lr):
        self.grad_W = self.x.T @ grad_out / self.x.shape[0]
        self.grad_b = grad_out.mean(axis=0, keepdims=True)
        grad_in = grad_out @ self.W.T
        self.W -= lr * self.grad_W
        self.b -= lr * self.grad_b
        return grad_in


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad_out, lr):
        return grad_out * self.mask


class Sigmoid:
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, grad_out, lr):
        return grad_out * self.out * (1 - self.out)


class Tanh:
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, grad_out, lr):
        return grad_out * (1 - self.out ** 2)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class SoftmaxCrossEntropy:
    def forward(self, logits, y):
        self.probs = softmax(logits)
        self.y = y
        n = len(y)
        return -np.log(self.probs[np.arange(n), y] + 1e-12).mean()

    def backward(self):
        grad = self.probs.copy()
        grad[np.arange(len(self.y)), self.y] -= 1
        return grad


class NeuralNet:
    def __init__(self, layers):
        self.layers = layers
        self.loss_fn = SoftmaxCrossEntropy()

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def train_step(self, x, y, lr):
        logits = self.forward(x)
        loss = self.loss_fn.forward(logits, y)
        grad = self.loss_fn.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)
        return loss

    def predict(self, x):
        return self.forward(x).argmax(axis=1)


def build_numpy_model(sizes, act=ReLU, init="he"):
    layers = []
    for i in range(len(sizes) - 2):
        layers += [Dense(sizes[i], sizes[i + 1], init), act()]
    layers.append(Dense(sizes[-2], sizes[-1], init))
    return NeuralNet(layers)