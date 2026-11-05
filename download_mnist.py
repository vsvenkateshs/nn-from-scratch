from torchvision import datasets

print("Downloading MNIST...")
train = datasets.MNIST("data", train=True, download=True)
test = datasets.MNIST("data", train=False, download=True)
print("Done!")
print("Train images:", train.data.shape)
print("Test images :", test.data.shape)