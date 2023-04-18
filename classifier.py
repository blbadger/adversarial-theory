# cifar10_generalization.py
# MLP-style model with GPU acceleration for latent space exploration.

# import standard libraries
import time
import pathlib
import os
import pandas as pd 
import random

# import third party libraries
import numpy as np 
import torch
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader, Dataset
import torchvision
import matplotlib
import matplotlib.pyplot as plt  
import torchvision.transforms as transforms 

# send model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print (f"Device: {device}") 

transform = transforms.Compose(
    [transforms.ToTensor()])

batch_size = 512
image_size = 28
channels = 1
# trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False)

# testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

class FCnet(nn.Module):

    def __init__(self, starting_size, image_size=28, channels=1):

        super().__init__()
        starting = starting_size
        self.input_transform = nn.Linear(image_size*image_size*channels, starting)
        self.d1 = nn.Linear(starting, starting//2)
        self.d2 = nn.Linear(starting//2, starting//4)
        self.d3 = nn.Linear(starting//4, starting//8)
        self.d4 = nn.Linear(starting//8, 100)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, input_tensor):
        input_tensor = torch.flatten(input_tensor, start_dim=1)
        out = self.input_transform(input_tensor)
        out = self.relu(out)

        out = self.d1(out)
        out = self.relu(out)

        out = self.d2(out)
        out = self.relu(out)

        out = self.d3(out)
        out = self.relu(out)

        out = self.d4(out)
        return out


class ConvNet(nn.Module):

    def __init__(self):

        super().__init__()
        self.conv1 = nn.Conv2d(3, 60, 5, padding=2)
        self.conv2 = nn.Conv2d(60, 80, 5, padding=2)
        self.conv3 = nn.Conv2d(80, 160, 3, padding=1)
        self.conv4 = nn.Conv2d(160, 320, 3, padding=1)
        self.dense = nn.Linear(20480, 10)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.softmax = nn.Softmax()

    def forward(self, input_tensor):
        out = self.conv1(input_tensor)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.conv4(out)
        out = self.relu(out)

        out = torch.flatten(out, start_dim=1)
        out = self.dense(out)
        out = self.softmax(out)
        return out


def loss_gradient(model, input_tensor, target_output, output_dim):
    """
     Computes the gradient of the input wrt. the objective function
     Args:
        input: torch.Tensor() object of input
        model: Transformer() class object, trained neural network
     Returns:
        gradientxinput: arr[float] of input attributions
    """

    # change output to float
    target_output = target_output.reshape(1)
    input_tensor.requires_grad = True
    output = model.forward(input_tensor)

    loss = loss_fn(output, target_output)

    # backpropegate output gradient to input
    loss.backward(retain_graph=True)
    gradient = input_tensor.grad

    return gradient


def show_batch(input_batch, count=0, grayscale=False):
    """
    Show a batch of images with gradientxinputs superimposed
    Args:
        input_batch: arr[torch.Tensor] of input images
        output_batch: arr[torch.Tensor] of classification labels
        gradxinput_batch: arr[torch.Tensor] of attributions per input image
    kwargs:
        individuals: Bool, if True then plots 1x3 image figs for each batch element
        count: int
    returns:
        None (saves .png img)
    """

    plt.figure(figsize=(15, 15))
    for n in range(8*8):
        ax = plt.subplot(8, 8, n+1)
        plt.axis('off')
        if grayscale:
            plt.imshow(input_batch[n], cmap='gray_r')
        else:
            plt.imshow(input_batch[n])
        plt.tight_layout()

    plt.tight_layout()
    plt.savefig('adversarial{0:04d}.png'.format(count), dpi=300)
    plt.close()
    return

def train_model(model, optimizer, loss_fn, epochs):
    """
    Train the model using gradient splitting.
    Args:
        model: torch.nn object
        optimizer: torch.nn optimizer
        loss_fn: torch.optim object
        epochs: int, number of desired training epochs
    kwargs:
        size: str, one of '2k', '5k', '10k', '20k', size of training
            ` and test data
    Returns:
        None (modifies model in-place, prints training curve data)
    """
    model.train()
    count = 0
    total_loss = 0
    start = time.time()
    train_array, test_array = [], []

    for e in range(epochs):
        total_loss = 0
        count = 0

        for pair in trainloader:
            train_x, train_y= pair[0], pair[1]
            count += 1
            trainx = train_x.to(device)
            output = model(trainx)
            loss = loss_fn(output.to(device), train_y.to(device))
            loss = loss.to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss

        ave_loss = float(total_loss) / count
        elapsed_time = time.time() - start
        print (f"Average Loss: {ave_loss:.04}")
        train_array.append(test_model(trainloader, model))
        test_array.append(test_model(testloader, model))
        start = time.time()

    print (train_array, test_array)

    return


def train_onehot_model(model, optimizer, loss_fn, epochs):
    """
    Train the model using gradient splitting.
    Args:
        model: torch.nn object
        optimizer: torch.nn optimizer
        loss_fn: torch.optim object
        epochs: int, number of desired training epochs
    kwargs:
        size: str, one of '2k', '5k', '10k', '20k', size of training
            ` and test data
    Returns:
        None (modifies model in-place, prints training curve data)
    """
    model.train()
    count = 0
    total_loss = 0
    start = time.time()
    train_array, test_array = [], []

    for e in range(epochs):
        total_loss = 0
        count = 0

        for pair in trainloader:
            train_x, train_y= pair[0], pair[1]
            count += 1
            trainx = train_x.to(device)
            output = model(trainx)
            trainy = torch.zeros(len(train_y), 10)
            for i in range(len(train_x)):
                trainy[i][train_y[i]] = 1.
            loss = loss_fn(output.to(device), trainy.to(device))
            loss = loss.to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss

        ave_loss = float(total_loss) / count
        elapsed_time = time.time() - start
        print (f"Epoch: {e + 1}")
        print (f"Average Loss: {ave_loss:.04}")
        train_array.append(test_model(trainloader, model))
        test_array.append(test_model(testloader, model))
        start = time.time()

    print (train_array, test_array)

    return

def test_model(test_dataloader, model):
    model.eval()
    correct, count = 0, 0
    batches = 0
    for batch, (x, y) in enumerate(test_dataloader):
        if batch > 20:
            break
        x = x.to(device)
        predictions = model(x)
        _, predicted = torch.max(predictions.data, 1)
        count += len(y)
        correct += (predicted == y.to(device)).sum().item()
        batches += 1

    print (f'Accuracy: {correct / count}')
    return correct / count


train_accuracies, test_accuracies = [], []
torch.cuda.empty_cache()
epochs = 50
loss_fn = nn.CrossEntropyLoss()
# loss_fn = nn.L1Loss()
model = FCnet(1000)
# model = ConvNet()

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print (total_params)

data_dir = 'fcnet_classifier_fmnist.pth'
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# train_model(model, optimizer, loss_fn, epochs)
# torch.save(model.state_dict(), data_dir)
# trainloader = trainloader
# testloader = testloader
# print ('end')
# train_accuracies.append(test_model(trainloader, model))
# test_accuracies.append(test_model(testloader, model))

# model.load_state_dict(torch.load(data_dir))
model.eval()

def test_injectivity():
    outputs = []
    inputs = []
    for x, y in trainloader:
        model_output = model(x.to(device))
        for t in model_output:
            outputs.append(t)
        for z in x:
            inputs.append(z)

    outputs = torch.stack(outputs)
    inputs = torch.stack(inputs)
    input_distances = []
    distances = []
    min_distance = np.infty

    for i in range(len(outputs)-1):
        point = outputs[i]
        rest_tensor = outputs[i+1:]
        difference = rest_tensor - point
        difference = torch.abs(difference)
        min_d = float(torch.min(torch.sum(difference, dim=1)))
        if min_d == 0:
            difference = inputs[i] - inputs[i+1:]
            difference = torch.abs(difference)
            min_d = float(torch.min(torch.sum(difference, dim=1)))
            if min_d == 0:
                min_d = np.infty
                continue
            break
        min_distance = min(min_distance, min_d)
    
    return min_distance
            
print ('Closest pairwise distance: ', test_injectivity())

def generate_adversarials(lr=0.0001):
    """
    Generate an adversarial example using the gradient of the output wrt the input

    """
    input_tensor, output_tensor = next(iter(trainloader))
    input_tensor, output_tensor = input_tensor.to(device), output_tensor.to(device)
    otensor = torch.zeros(batch_size, 10)

    input_tensor.requires_grad = True
    output = model(input_tensor)
    otensor = torch.zeros(len(output_tensor), 10).to(device)
    for i in range(len(output_tensor)):
        otensor[i][output_tensor[i]] = 1.
    output_tensor = otensor
    # loss = loss_fn(output, output_tensor)
    loss = torch.sum(output)
    loss.backward()
    gradient = torch.sign(input_tensor.grad)
    adversarial_inputs = input_tensor + lr * gradient
    input_tensor, output_tensor = input_tensor.to(device), otensor.to(device)

    random_input_l1 = torch.sum(torch.abs(lr*torch.randn(input_tensor.shape)))
    adversarial_input_l1 = torch.sum(torch.abs(input_tensor - adversarial_inputs))

    adversarial_output = model(adversarial_inputs)
    adversarial_output_l1 = torch.sum(torch.abs(output - adversarial_output))

    routput = model(input_tensor + lr * torch.randn(input_tensor.shape).to(device))
    random_output_l1 = torch.sum(torch.abs(routput - output))

    print (adversarial_output_l1, random_output_l1)
    ratio = (adversarial_output_l1 * random_input_l1) / (random_output_l1 * adversarial_input_l1) 
    print (f'r = {ratio}')
    adversarial_inputs = adversarial_inputs.cpu().permute(0, 2, 3, 1).detach().numpy()
    input_tensor = input_tensor.cpu().permute(0, 2, 3, 1).detach().numpy()
    # show_batch(adversarial_inputs, count=0)
    # show_batch(input_tensor, count=1)
    # print (torch.argmax(output, dim=1) == torch.argmax(adversarial_output, dim=1))

    return float(ratio)

matplotlib.rcParams.update({'font.size': 16})
arrs = []
for i in range(1):
    lr = 1
    arr = []
    for i in range(8):
        arr.append(generate_adversarials(lr))
        lr /= 10
    arrs.append(arr)

model.load_state_dict(torch.load(data_dir))


for i in range(1):
    lr = 1
    arr = []
    for i in range(8):
        arr.append(generate_adversarials(lr))
        lr /= 10
    arrs.append(arr)

print (arr)
for arr in arrs:
    plt.plot([i for i in range(8)], arr)
# plt.yscale('log')
# plt.ylim([0, 20])
plt.ylabel('Expansion ratio')
plt.xlabel('Learning rate, 1*10^-n')
plt.savefig(f'classifier', dpi=300, bbox_inches='tight')
plt.close()

# generate_adversarials()


