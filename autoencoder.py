import math, random, time, os
from inspect import isfunction
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
from pathlib import Path
import pathlib
import numpy as np

import torch
from torch import nn, einsum
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torchvision import transforms, utils
import torchvision
from torchvision.utils import save_image
from torch.optim import Adam
import matplotlib.pyplot as plt
import matplotlib
from prettytable import PrettyTable

device = "cuda" if torch.cuda.is_available() else "cpu"
print (device)

batch_size = 512 # global variable
image_size = 28
channels = 1

transform = transforms.Compose([
			# transforms.Resize((image_size, image_size)),
			transforms.ToTensor()
])

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)


class SmallFCEncoder(nn.Module):

	def __init__(self, starting_size, channels, image_size=28):
		super().__init__()
		starting = starting_size
		self.input_transform = nn.Linear(image_size*image_size*channels, starting)
		self.d1 = nn.Linear(starting, starting//4)
		self.d2 = nn.Linear(starting//4, starting//8)
		self.d3 = nn.Linear(starting//8, starting//4)
		self.d4 = nn.Linear(starting//4, starting)
		self.d5 = nn.Linear(starting, image_size*image_size*channels)
		self.gelu = nn.GELU()
		self.layernorm1 = nn.LayerNorm(starting)
		self.layernorm2 = nn.LayerNorm(starting//2)
		self.layernorm3 = nn.LayerNorm(starting//4)
		self.layernorm4 = nn.LayerNorm(starting//2)
		self.layernorm5 = nn.LayerNorm(starting)

	def forward(self, input_tensor):
		input_tensor = torch.flatten(input_tensor, start_dim=1)
		
		out = self.input_transform(input_tensor)
		out = self.layernorm1(self.gelu(out))

		out = self.d1(out)
		out = self.gelu(out)

		out = self.d2(out)
		out = self.gelu(out)

		out = self.d3(out)
		out = self.gelu(out)

		out = self.d4(out)
		out = self.gelu(out)

		out = self.d5(out)
		out = out.reshape(batch_size, channels, image_size, image_size)
		return out


class SmallDeepFCEncoder(nn.Module):

	def __init__(self, starting_size, channels, image_size=28):
		super().__init__()
		starting = starting_size
		self.input_transform = nn.Linear(image_size*image_size*channels, starting)
		self.d1 = nn.Linear(starting, starting//2)
		self.d2 = nn.Linear(starting//2, starting//4)
		self.d3 = nn.Linear(starting//4, starting//8)
		self.d4 = nn.Linear(starting//8, starting//12)
		self.d5 = nn.Linear(starting//12, starting//16)
		self.d6 = nn.Linear(starting//16, starting//12)
		self.d7 = nn.Linear(starting//12, starting//8)
		self.d8 = nn.Linear(starting//8, starting//4)
		self.d9 = nn.Linear(starting//4, starting//2)
		self.d10 = nn.Linear(starting//2, starting)
		self.d11 = nn.Linear(starting, image_size*image_size*channels)
		self.gelu = nn.GELU()
		self.layernorm1 = nn.LayerNorm(starting)
		self.layernorm2 = nn.LayerNorm(starting//2)
		self.layernorm3 = nn.LayerNorm(starting//4)
		self.layernorm4 = nn.LayerNorm(starting//2)
		self.layernorm5 = nn.LayerNorm(starting)

	def forward(self, input_tensor):
		input_tensor = torch.flatten(input_tensor, start_dim=1)
		
		out = self.input_transform(input_tensor)
		out = self.layernorm1(self.gelu(out))

		for i in range(1, 11):
			dense_layer = eval('self.d{}'.format(i))
			out = dense_layer(out)
			out = self.gelu(out)

		out = self.d11(out)
		out = out.reshape(batch_size, channels, image_size, image_size)
		return out


class FCEncoder(nn.Module):

	def __init__(self, starting_size, channels, image_size=28):
		super().__init__()
		starting = starting_size
		self.input_transform = nn.Linear(image_size*image_size*channels, starting)
		self.d0 = nn.Linear(starting, starting).to(device)
		self.d1 = nn.Linear(starting, starting).to(device)
		self.d2 = nn.Linear(starting, starting).to(device)
		self.d3 = nn.Linear(starting, starting).to(device)
		self.d4 = nn.Linear(starting, starting).to(device)

		self.gelu = nn.GELU()
		self.d6 = nn.Linear(starting, image_size*image_size*channels)

	def forward(self, input_tensor):
		input_tensor = torch.flatten(input_tensor, start_dim=1)
		out = self.input_transform(input_tensor)
		out = self.gelu(out)

		for i in range(5):
			layer = eval('self.d{}'.format(i))
			out = layer(out)
			out = self.gelu(out)

		out = self.d6(out)
		out = out.reshape(batch_size, channels, image_size, image_size)
		return out 


def show_batch(input_batch, count=0, grayscale=False, normalize=True):
	"""
	Show a batch of images with gradientxinputs superimposed

	Args:
		input_batch: arr[torch.Tensor] of input images
		output_batch: arr[torch.Tensor] of classification labels
		gradxinput_batch: arr[torch.Tensor] of atransformsributions per input image
	kwargs:
		individuals: Bool, if True then plots 1x3 image figs for each batch element
		count: int

	returns:
		None (saves .png img)

	"""

	plt.figure(figsize=(15, 15))
	length, width = 8, 8
	for n in range(length*width):
		ax = plt.subplot(length, width, n+1)
		plt.axis('off')
		if normalize: 
			# rescale to [0, 1]
			input_batch[n] = (input_batch[n] - np.min(input_batch[n])) / (np.max(input_batch[n]) - np.min(input_batch[n]))
		if grayscale:
			plt.imshow(input_batch[n], cmap='gray_r')
		else:
			plt.imshow(input_batch[n])
		plt.tight_layout()

	plt.tight_layout()
	plt.savefig('image{0:04d}.png'.format(count), dpi=300, transparent=True)
	print ('Image Saved')
	plt.close()
	return 


def count_parameters(model):
    """
    Display the tunable parameters in the model of interest

    Args:
        model: torch.nn object

    Returns:
        total_params: the number of model parameters

    """

    table = PrettyTable(['Module', 'Parameters'])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param 	 

    print (table)
    print (f'Total trainable parameters: {total_params}')
    return total_params


def train_autoencoder():
	epochs = 100
	for epoch in range(epochs):
		start_time = time.time()
		total_loss = 0
		for step, batch in enumerate(dataloader):
			if len(batch[0]) < batch_size:
				break 
			optimizer.zero_grad()
			batch = batch[0].to(device) # discard class labels
			output = model(batch)
			loss = loss_fn(output, batch) # + loss_fn(gen_res, real_res)
			total_loss += loss.item()
	 
			loss.backward()
			optimizer.step()

		print (f"Epoch {epoch} completed in {time.time() - start_time} seconds")
		print (f"Average Loss: {round(total_loss / step, 5)}")
		torch.save(model.state_dict(), 'fcnet_autoencoder_fmnist.pth')

		if epoch % 10 == 0:
			batch = next(iter(dataloader))[0].to(device)
			gen_images = model(batch).cpu().permute(0, 2, 3, 1).detach().numpy()
			show_batch(gen_images, count=epoch, grayscale=False, normalize=False)
	return

model = FCEncoder(image_size*image_size*channels, 1).to(device)
# model = SmallFCEncoder(2000, 1).to(device)
# model = SmallDeepFCEncoder(2000, 1).to(device)
count_parameters(model)
optimizer = Adam(model.parameters(), lr=1e-4) 
loss_fn = torch.nn.MSELoss()
train_autoencoder()
model.load_state_dict(torch.load('fcnet_autoencoder_fmnist.pth'))
model.eval()

def interpolate_latent():
	data = iter(dataloader)
	batch1 = next(data).to(device)
	batch2 = next(data).to(device)
	random = torch.normal(0.5, 0.2, batch1.shape).to(device) 
	for i in range(61):
		alpha = 1 - i / 30
		if i <= 30:
			beta = i / 30
		else:
			beta = abs(2 - i / 30)
		batch = alpha * batch1 + (1 - alpha)* batch2 + 2 * beta * random 
		gen_images = model(batch.to(device)).cpu().permute(0, 2, 3, 1).detach().numpy()
		show_batch(gen_images, count=i, grayscale=False, normalize=True)

	return

# interpolate_latent() 

def generate_adversarials(lr=0.01):
	"""
	Generate an adversarial example using the gradietn of the output wrt the input

	"""
	input_tensor = next(iter(dataloader))[0].to(device)
	input_tensor.requires_grad = True
	output = model(input_tensor)
	loss = loss_fn(output, input_tensor)
	loss.backward()
	random_input_l1 = torch.sum(torch.abs(lr*torch.randn(input_tensor.shape)))
	gradient = torch.sign(input_tensor.grad)
	adversarial_inputs = input_tensor + lr * gradient
	adversarial_input_l1 = torch.sum(torch.abs(input_tensor - adversarial_inputs))
	adversarial_output = model(adversarial_inputs)
	adversarial_output_l1 = torch.sum(torch.abs(output - adversarial_output))
	routput = model(input_tensor + lr * torch.randn(input_tensor.shape).to(device))
	random_output_l1 = torch.sum(torch.abs(routput - output))
	ratio = (adversarial_output_l1 * random_output_l1) / (random_output_l1 * adversarial_input_l1)
	print (f'r = {ratio}')
	adversarial_inputs = adversarial_inputs.cpu().permute(0, 2, 3, 1).detach().numpy()
	input_tensor = input_tensor.cpu().permute(0, 2, 3, 1).detach().numpy()
	# show_batch(adversarial_inputs, count=0)
	# show_batch(input_tensor, count=1)
	return float(ratio)

# matplotlib.rcParams.update({'font.size': 16})
# arrs = []
# for i in range(10):
# 	lr = 1
# 	arr = []
# 	for i in range(8):
# 		arr.append(generate_adversarials(lr))
# 		lr /= 10
# 	arrs.append(arr)

# print (arrs)
# for arr in arrs:
# 	plt.plot([i for i in range(8)], arr)
# # plt.yscale('log')
# plt.ylabel('Expansion ratio')
# plt.xlabel('Learning rate, 1*10^-n')
# plt.savefig(f'autoencoder', dpi=300, bbox_inches='tight')
# plt.close()


@torch.no_grad()
def random_manifold_walk():
	data = iter(dataloader)
	batch = next(data).to(device)[0]
	output, hidden = model(batch)
	gen_images = output.cpu().permute(0, 2, 3, 1).detach().numpy()
	show_batch(gen_images, count=0, grayscale=False, normalize=True)

	unet_decoder = UnetDecoder(model, batch)
	og_hidden = hidden
	for i in range(300):
		random = torch.normal(0, 0.2, hidden.shape).to(device)
		hidden += random 
		output = unet_decoder(hidden)
		gen_images = output.cpu().permute(0, 2, 3, 1).detach().numpy()
		show_batch(gen_images, count=i + 1, grayscale=False, normalize=True)
		out, hidden = model(output)
	return

@torch.no_grad()
def directed_manifold_walk():
	data = iter(dataloader)
	batch = next(data).to(device)
	batch2 = next(data).to(device)
	output, hidden_original = model(batch)
	target_output, target_hidden = model(batch2)
	gen_images = output.cpu().permute(0, 2, 3, 1).detach().numpy()
	show_batch(gen_images, count=0, grayscale=False, normalize=True)
	unet_decoder = UnetHiddenDecoder(model, batch)
	for i in range(60):
		alpha = i/60
		hidden = (1- alpha) * hidden_original + alpha * target_hidden
		output = unet_decoder(hidden)
		gen_images = output[0].cpu().permute(0, 2, 3, 1).detach().numpy()
		show_batch(gen_images, count=i + 1, grayscale=False, normalize=True)

	return

# model.eval()
# random_manifold_walk()

@torch.no_grad()
def observe_denoising(alpha=0.5, count=100):
	batch = next(iter(dataloader))[0]
	original_batch = batch
	show_batch(batch.cpu().permute(0, 2, 3, 1).detach().numpy(), count=101, grayscale=True, normalize=False)
	# alpha = 0.5
	# batch = alpha * batch + (1-alpha) * torch.normal(0.7, 0.2, batch.shape)

	# original = batch[0]
	# original_output = model(batch.to(device))[0][0]
	# batch = torchvision.transforms.GaussianBlur(19, 8)(batch)
	# transformed = batch[0]
	# transformed_output = model(batch.to(device))[0][0]

	# # shown = batch.cpu().permute(0, 2, 3, 1).detach().numpy()
	# # show_batch(shown, count=98, grayscale=False, normalize=False)
	# gen_images = model(batch.to(device))[0].cpu().permute(0, 2, 3, 1).detach().numpy()
	# show_batch(gen_images, count=99, grayscale=False, normalize=False)
	# input_distance = torch.sum((original - transformed)**2)**0.5
	# output_distance = torch.sum((original_output - transformed_output)**2)**0.5
	# print (f'L2 Distance on the Input after Blurring: {input_distance}')
	# print (f'L2 Distance on the Autoencoder Output after Blurring: {output_distance}')

	# alpha = 1
	batch = original_batch

	original = batch[0]
	original_output = model(batch.to(device))
	batch = alpha * batch + (1-alpha) * torch.normal(0.7, 0.2, batch.shape)
	transformed = batch
	transformed_output = model(batch.to(device))

	shown = batch.cpu().permute(0, 2, 3, 1).detach().numpy()
	show_batch(shown, count=count, grayscale=True, normalize=True)
	gen_images = model(batch.to(device)).cpu().permute(0, 2, 3, 1).detach().numpy()
	show_batch(gen_images, count=count, grayscale=True, normalize=True)
	input_distance = torch.sum((original - transformed)**2)**0.5
	output_distance = torch.sum((original_output - transformed_output)**2)**0.5
	print (f'L2 Distance on the Input after Gaussian Noise: {input_distance}')
	print (f'L2 Distance on the Autoencoder Output after Gaussian Noise: {output_distance}')

observe_denoising(alpha=1, count=100)
observe_denoising(alpha=0.5, count=200)

@torch.no_grad()
def generate_with_noise():
	batch = next(iter(dataloader))[0]
	alpha = 0
	batch = alpha * batch + (1-alpha) * torch.normal(0.7, 0.2, batch.shape) # random initial input
	for i in range(80):
		alpha = i / 80
		gen_images = model(batch.to(device))
		show_batch(gen_images.cpu().permute(0, 2, 3, 1).detach().numpy(), count=i, grayscale=False, normalize=False)
		batch = alpha * gen_images + (1-alpha) * torch.normal(0.6, 0.2, batch.shape).to(device) 

	return batch


def find_analogues(input):
	batch = next(iter(dataloader))
	images = []
	for step, batch in enumerate(dataloader):
		if step > 20:
			break
		batch = batch[0]
		for i in range(512):
			images.append(batch[i, :, :, :])

	min_distance = np.inf
	for image in images:
		if torch.sum((input - image.to(device))**2)**0.5 < min_distance:
			closest_image = image
			min_distance = torch.sum((input - image[0].to(device))**2)**0.5 

	closest_image = closest_image.cpu().permute(1, 2, 0).detach().numpy()
	# plt.figure(figsize=(15, 15))
	# plt.imshow(closest_image)
	# plt.tight_layout()
	# plt.savefig('closest_image.png', dpi=300, transparent=True)

	input_batch = input.cpu().permute(1, 2, 0).detach().numpy(), closest_image
	length, width = 1, 2
	for n in range(length*width):
		ax = plt.subplot(length, width, n+1)
		plt.axis('off')
		plt.imshow(input_batch[n])
		plt.tight_layout()

	plt.tight_layout()
	plt.savefig('closest_pair.png', dpi=300, transparent=True)
	print ('Image Saved')
	plt.close()
	return 
	plt.close()
	return


# batch = generate_with_noise()
# data = iter(dataloader)
# batch = next(data)[0].to(device)
# show_batch(batch.cpu().permute(0, 2, 3, 1).detach().numpy())
# find_analogues(batch[0])

