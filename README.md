\# Neural Network from Scratch + Deep Learning Experiments



A fully connected neural network built \*\*from scratch in NumPy\*\* (forward pass, backpropagation, mini-batch SGD), verified with gradient checking, and compared against an equivalent \*\*PyTorch\*\* model on MNIST. The project also includes ablation studies on activations, learning rates, weight initialization, batch size, optimizers, and regularization.



\## Highlights



\- Layers, activations (ReLU, Sigmoid, Tanh), softmax cross-entropy loss, and backpropagation written by hand with NumPy only

\- Backprop verified with numerical gradient checking (relative error: \*\*1.2e-7\*\*)

\- Fair NumPy vs PyTorch comparison: same architecture, same starting weights, same learning rate and batch size

\- Six ablation experiments with plots



\## Results: NumPy vs PyTorch on MNIST



Architecture: 784 → 128 → 64 → 10, ReLU, SGD (lr = 0.1, batch size = 64, 10 epochs)



| Model   | Test Accuracy | Training Time |

|---------|---------------|---------------|

| NumPy   | 97.42%        | 18.3 s        |

| PyTorch | 97.58%        | 4.1 s         |



!\[Loss comparison](results/mnist\_loss.png)

!\[Validation accuracy](results/mnist\_val\_acc.png)



\*\*Takeaway:\*\* Both models achieve nearly identical final test accuracy because they implement the exact same underlying mathematics and optimization algorithms under identical hyperparameter conditions. PyTorch executes significantly faster than pure NumPy because it utilizes highly optimized C++ backends, vectorized matrix multiplications, and efficient memory management. While NumPy requires manual derivation and tracking of gradients, PyTorch seamlessly automates backpropagation via its dynamic computational graph.



\## Spiral dataset (non-linear decision boundary)



A 3-class spiral dataset cannot be separated by a linear model. The NumPy network learns the boundary below.



!\[Spiral decision boundary](results/spiral\_boundary.png)



\## Ablation Studies



| Experiment | Plot | Finding |

|------------|------|---------|

| Activation functions | !\[](results/abl\_activation.png) | ReLU converges fastest and achieves the highest test accuracy, whereas Sigmoid suffers heavily from vanishing gradients during early iterations. Tanh offers reasonable performance but remains slower than ReLU. |

| Learning rates | !\[](results/abl\_lr.png) | A rate of 0.1 strikes an optimal balance; setting lr = 1.0 causes explosive loss oscillations and instability, while setting lr = 0.001 results in prematurely stalled training. |

| Weight initialization | !\[](results/abl\_init.png) | Zero initialization fails entirely (\~10% accuracy) as all neurons compute identical updates due to symmetry. He (Kaiming) initialization avoids vanishing/exploding gradients and speeds up convergence. |

| Batch size | !\[](results/abl\_batch.png) | Smaller batch sizes (32, 64) train faster per epoch with helpful stochastic noise, whereas larger batches (512) offer smoother gradient estimates but require more epochs to reach equivalent accuracy. |

| Optimizers (PyTorch) | !\[](results/abl\_optim.png) | Adam and Momentum SGD converge much faster than standard SGD. Adam achieves top accuracy in fewer epochs by adaptively adjusting per-parameter learning rates. |

| Overfitting and L2 | !\[](results/abl\_l2.png) | Adding L2 weight decay ($\\lambda = 10^{-4}$) shrinks the training-validation accuracy gap on small training subsets, preventing the model from memorizing noise. |



\## Project Structure





```

nn-from-scratch/

├── README.md

├── requirements.txt

├── download\_mnist.py          # downloads MNIST into data/

├── src/

│   ├── numpy\_nn.py            # NumPy network: layers, losses, backprop

│   ├── torch\_nn.py            # equivalent PyTorch model

│   ├── train.py               # training loops for both frameworks

│   └── utils.py               # data loading and plotting helpers

├── experiments/

│   ├── exp1\_spiral.py         # gradient check + spiral dataset

│   ├── exp2\_mnist\_compare.py  # NumPy vs PyTorch on MNIST

│   └── exp3\_ablations.py      # ablation studies

└── results/                   # saved plots

```



\## How to Run



```bash

\# 1. Clone the repository

git clone https://github.com/YOUR\_USERNAME/nn-from-scratch.git

cd nn-from-scratch



\# 2. Create and activate a virtual environment

python -m venv venv

venv\\Scripts\\activate          # Windows

\# source venv/bin/activate     # Mac/Linux



\# 3. Install dependencies

pip install -r requirements.txt



\# 4. Run the experiments (MNIST downloads automatically on first run)

python experiments/exp1\_spiral.py

python experiments/exp2\_mnist\_compare.py

python experiments/exp3\_ablations.py

```



\## What I Learned



\- (How backpropagation applies the chain rule layer by layer)

\- (Why gradient checking is useful for catching bugs)

\- (Why initialization and activation choice affect training so much)

\- (What PyTorch automates compared with a manual implementation)



\## Tech Stack



Python, NumPy, PyTorch, torchvision, Matplotlib

