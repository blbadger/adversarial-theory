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

# specify batch size
minibatch_size = 256

transform = transforms.Compose([
			transforms.ToTensor(),
			# transforms.Normalize((0.5), (0.5))
			])

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_dataloader = DataLoader(trainset, batch_size=minibatch_size, shuffle=True)

class FCnet(nn.Module):

	def __init__(self, input_dim=28):

		super().__init__()
		self.input_transform = nn.Linear(input_dim*input_dim, 1024)
		self.d1 = nn.Linear(1024, 512)
		self.d2 = nn.Linear(512, 256)
		self.d3 = nn.Linear(256, 1)
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		self.dropout = nn.Dropout(0.1)

	def forward(self, input_tensor):
		out = self.input_transform(input_tensor)
		out = self.relu(out)
		# out = self.dropout(out)  

		out = self.d1(out)
		out = self.relu(out)
		# out = self.dropout(out)
 
		out = self.d2(out)
		out = self.relu(out)
		# out = self.dropout(out)

		out = self.d3(out)
		out = self.sigmoid(out)
		return out


class InvertedFC(nn.Module):

	def __init__(self, hidden_dim=2, input_dim=28):
		super().__init__()
		self.d1 = nn.Linear(hidden_dim, 256)
		self.d2 = nn.Linear(256, 512)
		self.d3 = nn.Linear(512, 1024)
		self.input_transform = nn.Linear(1024, input_dim*input_dim)
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		self.tanh= nn.Tanh()

	def forward(self, input_tensor):
		out = self.d1(input_tensor)
		out = self.relu(out)

		out = self.d2(out)
		out = self.relu(out)

		out = self.d3(out)
		out = self.relu(out)

		out = self.input_transform(out)
		out = self.tanh(out)
		return out

class BigIF(nn.Module):
	def __init__(self, hidden_dim=2, input_dim=28):
		super().__init__()
		self.d1 = nn.Linear(hidden_dim, 10000)
		self.d2 = nn.Linear(10000, 5000)
		self.d3 = nn.Linear(5000, 2000)
		self.input_transform = nn.Linear(2000, input_dim*input_dim)
		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		self.tanh= nn.Tanh()

	def forward(self, input_tensor):
		out = self.d1(input_tensor)
		out = self.relu(out)

		out = self.d2(out)
		out = self.relu(out)

		out = self.d3(out)
		out = self.relu(out)

		out = self.input_transform(out)
		out = self.tanh(out)
		return out

class MediumIF(nn.Module):
	def __init__(self, hidden_dim=2, input_dim=28):
		super().__init__()
		self.d1 = nn.Linear(hidden_dim, 50)
		self.d2 = nn.Linear(50, 500)
		self.d3 = nn.Linear(500, 1000)
		self.input_transform = nn.Linear(1000, 1000)
		self.relu = nn.ReLU()
		self.tanh= nn.Tanh()
		self.dropout = nn.Dropout()

	def forward(self, input_tensor):
		out = self.d1(input_tensor)
		out = self.relu(out)
		# out = self.dropout(out)

		out = self.d2(out)
		out = self.relu(out)

		out = self.d3(out)
		out = self.relu(out)

		out = self.input_transform(out)
		out = self.tanh(out)
		return out


def loss_gradient(model, input_tensor, true_output, output_dim):
	"""
	 Computes the gradient of the input wrt. the objective function

	 Args:
		input: torch.Tensor() object of input
		model: Transformer() class object, trained neural network

	 Returns:
		gradientxinput: arr[float] of input attributions

	"""

	input_tensor.requires_grad = True
	output = model.forward(input_tensor)
	loss = loss_fn(output, true_output)

	# backpropegate output gradient to input
	loss.backward()
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
	for n in range(100):
		ax = plt.subplot(10, 10, n+1)
		plt.axis('off')
		if grayscale:
			plt.imshow(input_batch[n], cmap='gray')
		else:
			plt.imshow(input_batch[n])
		plt.tight_layout()

	# plt.style.use('dark_background')
	plt.tight_layout()
	plt.savefig('gan_set{0:04d}.png'.format(count), dpi=300)
	plt.close()
	return

def train_fcgan(dataloader, discriminator, generator, loss_fn, epochs, embedding_size):
	"""
	Trains the generative adversarial network model.

	Args:
		dataloader: torch.utils.data.Dataloader object, iterable for loading training and test data
		discriminator: torch.nn.Module() object
		discriminator_optimizer: torch.optim object, optimizes discriminator params during gradient descent
		generator: torch.nn.Module() object
		generator_optimizer: torc.optim object, optimizes generator params during gradient descent
		loss_fn: arbitrary method to apply to models (default binary cross-entropy loss)
		epochs: int, number of training epochs desired

	Returns:
		None (modifies generator and discriminator in-place)
	"""
	discriminator.train()
	generator.train()
	count = 0
	total_loss = 0
	start = time.time()
	fixed_input = torch.randn(minibatch_size, embedding_size).to(device)

	for e in range(epochs):
		print (f"Epoch {e+1} \n" + '~'*100)
		total_d_loss, total_g_loss = 0, 0

		for batch, (x, y) in enumerate(dataloader):
			if len(x) < minibatch_size:
				break 

			x, y = x.to(device), y.to(device)
			x = torch.flatten(x, start_dim=1)
			count += 1
			random_output = torch.randn(minibatch_size, embedding_size).to(device)
			generated_samples = generator(random_output)
			input_dataset = torch.cat([x, generated_samples])
			output_labels = torch.cat([torch.ones(len(y)), torch.zeros(len(generated_samples))]).to(device)
			discriminator_prediction = discriminator(input_dataset).reshape(minibatch_size*2)
			discriminator_loss = loss_fn(discriminator_prediction, output_labels)

			discriminator_optimizer.zero_grad()
			discriminator_loss.backward()
			discriminator_optimizer.step()
			
			generated_outputs = generator(random_output)
			discriminator_outputs = discriminator(generated_outputs).reshape(minibatch_size)
			generator_loss = loss_fn(discriminator_outputs, torch.ones(len(y)).to(device)) # pretend that all generated inputs are in the dataset

			generator_optimizer.zero_grad()
			generator_loss.backward()
			generator_optimizer.step()

			total_d_loss += discriminator_loss.item()
			total_g_loss += generator_loss.item()

		# fixed_outputs = generator(fixed_input)
		# inputs = fixed_outputs.reshape(minibatch_size, 1, 28, 28).cpu().permute(0, 2, 3, 1).detach().numpy()
		# show_batch(inputs, e, grayscale=True)

		ave_loss = float(total_loss) / count
		elapsed_time = time.time() - start
		print (f"Discriminator Loss: {total_d_loss:.04}")
		print (f"Generator Loss: {total_g_loss:.04}")
		print (f"Completed in {int(elapsed_time)} seconds")
		start = time.time()

	return

def gsplit_fcgan(dataloader, discriminator, generator, loss_fn, epochs, embedding_size):
	"""
	Trains the generative adversarial network model.

	Args:
		dataloader: torch.utils.data.Dataloader object, iterable for loading training and test data
		discriminator: torch.nn.Module() object
		discriminator_optimizer: torch.optim object, optimizes discriminator params during gradient descent
		generator: torch.nn.Module() object
		generator_optimizer: torc.optim object, optimizes generator params during gradient descent
		loss_fn: arbitrary method to apply to models (default binary cross-entropy loss)
		epochs: int, number of training epochs desired

	Returns:
		None (modifies generator and discriminator in-place)
	"""
	discriminator.train()
	generator.train()
	count = 0
	total_loss = 0
	start = time.time()
	# fixed_input = torch.randn(minibatch_size, embedding_size).to(device)
	fixed_input = torch.randn(100, 100).to(device)


	for e in range(epochs):
		print (f"Epoch {e+1} \n" + '~'*100)
		total_d_loss, total_g_loss = 0, 0

		for batch, (x, y) in enumerate(dataloader):
			if batch > len(dataloader) // 2:
				break 

			x, y = x.to(device), y.to(device)
			x = torch.flatten(x, start_dim=1)
			count += 1
			random_output = torch.randn(minibatch_size, embedding_size).to(device)
			generated_samples = generator(random_output)
			input_dataset = torch.cat([x, generated_samples])
			output_labels = torch.cat([torch.ones(len(y)), torch.zeros(len(generated_samples))]).to(device) # flipped assignment
			discriminator_prediction = discriminator(input_dataset).reshape(minibatch_size*2)
			discriminator_loss = loss_fn(discriminator_prediction, output_labels)

			discriminator_optimizer.zero_grad()
			discriminator_loss.backward()
			discriminator_optimizer.step()
			
			generated_outputs = generator(random_output)
			discriminator_outputs = discriminator(generated_outputs).reshape(minibatch_size)
			generator_loss = loss_fn(discriminator_outputs, torch.ones(len(y)).to(device)) # pretend that all generated inputs are in the dataset

			generator_optimizer.zero_grad()
			generator_loss.backward()
			generator_optimizer.step()

			total_d_loss += discriminator_loss.item()
			total_g_loss += generator_loss.item()

		for batch, (x, y) in enumerate(dataloader):
			if batch < len(dataloader)// 2:
				continue
			if len(x) < minibatch_size:
				break 

			x, y = x.to(device), y.to(device)
			x = torch.flatten(x, start_dim=1)
			count += 1
			random_output = torch.randn(minibatch_size, embedding_size).to(device)
			generated_samples = generator(random_output)
			input_dataset = torch.cat([x, generated_samples])
			output_labels = torch.cat([torch.zeros(len(y)), torch.ones(len(generated_samples))]).to(device) # flipped assignment
			discriminator_prediction = discriminator(input_dataset).reshape(minibatch_size*2)
			discriminator_loss = loss_fn(discriminator_prediction, output_labels)

			discriminator_optimizer.zero_grad()
			discriminator_loss.backward()
			discriminator_optimizer.step()
			
			generated_outputs = generator(random_output)
			discriminator_outputs = discriminator(generated_outputs).reshape(minibatch_size)
			generator_loss = loss_fn(discriminator_outputs, torch.ones(len(y)).to(device)) # pretend that all generated inputs are in the dataset

			generator_optimizer.zero_grad()
			generator_loss.backward()
			generator_optimizer.step()

			total_d_loss += discriminator_loss.item()
			total_g_loss += generator_loss.item()

		fixed_outputs = generator(fixed_input)
		inputs = fixed_outputs.reshape(100, 1, 28, 28).cpu().permute(0, 2, 3, 1).detach().numpy()
		# show_batch(inputs, e, grayscale=True)
		explore_latentspace(generator, fixed_input, e)

		ave_loss = float(total_loss) / count
		elapsed_time = time.time() - start
		print (f"Discriminator Loss: {total_d_loss:.04}")
		print (f"Generator Loss: {total_g_loss:.04}")
		print (f"Completed in {int(elapsed_time)} seconds")
		start = time.time()

	return

def generate_input(model, input_tensors, output_tensors, index, count):
	"""
	Generates an input for a desired output class

	Args:
		input_tensor: torch.Tensor object, minibatch of inputs
		output_tensor: torch.Tensor object, minibatch of outputs
		index: int

	returns:
		None (saves .png image)
	"""

	target_input = input_tensors[index].reshape(1, 3, 256, 256)
	single_input = torch.rand(1, 3, 256, 256) # uniform distribution initialization
	output_tensors[index] = torch.Tensor([4])

	for i in range(1000):
		single_input = single_input.detach() # remove the gradient for the input (if present)
		input_grad = loss_gradient(model, single_input, output_tensors[index], 5) # compute input gradient
		last_input = single_input.clone()
		single_input = single_input - 10*input_grad # gradient descent step

	single_input = single_input.clone() / torch.max(single_input) * 10
	single_input = single_input.reshape(3, 256, 256).permute(1, 2, 0).detach().numpy()
	target_input = target_input.reshape(3, 256, 256).permute(1, 2, 0).detach().numpy()
	target_name = class_names[int(output_tensors[index])].title()

	plt.axis('off')
	plt.title(f'{target_name}')
	plt.savefig('adversarial_example{0:04d}.png'.format(count), dpi=410)
	plt.close()

	return


def explore_latentspace(generator, fixed_input=None, embedding=0):
	"""
	Plot a 10x10 grid of outputs from the latent space (assumes a 100x100 grid)

	Args:
		generator: torch.nn.Module() object of the generator

	Returns:
		None (saves png image in call to show_batch())
	"""
	if fixed_input is None:
		fixed_input = torch.randn(100, 100) # minibatch_size, embedding dim
	final_input = fixed_input.clone()
	for i in range(10):
		for j in range(10):
			new_input = fixed_input.clone()
			new_input[0][1:20] += 0.25 * (i+1)
			new_input[0][20:40] -= 0.25* (j+1)
			new_input[0][40:60] += 0.25 * (i+1)
			new_input[0][60:80] -= 0.25 * (j+1)
			new_input[0][80:100] += 0.25 * (j+1)
			final_input = torch.cat([final_input, new_input])

	images = generator(final_input[1:101]).cpu().reshape(len(fixed_input[0]), 28, 28).detach().numpy()
	show_batch(images, embedding, grayscale=True)
	return


def test_adversarials_generator(lr=0.01, embedding_size=100):
	"""
	Generate an adversarial example using the gradient of the output wrt the input

	"""
	input_tensor = torch.randn(minibatch_size, embedding_size).to(device)
	model = generator
	input_tensor.requires_grad = True
	output = model(input_tensor)
	# loss = loss_fn(discriminator(output), torch.ones(minibatch_size, 1).to(device))
	loss = torch.sum(output)
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
	# adversarial_inputs = adversarial_inputs.cpu().permute(0, 2, 3, 1).detach().numpy()
	# input_tensor = input_tensor.cpu().permute(0, 2, 3, 1).detach().numpy()
	# show_batch(adversarial_output, count=0)
	# show_batch(output, count=1)
	return float(ratio)


def test_adversarials_discriminator(lr=0.01, embedding_size=100):
	"""
	Generate an adversarial example using the gradient of the output wrt the input

	"""
	# input_tensor = torch.randn(minibatch_size, embedding_size).to(device)
	
	# output = model(input_tensor)
	# loss = torch.sum(output)
	input_tensor, y = next(iter(train_dataloader))
	input_tensor = torch.flatten(input_tensor, start_dim=1).to(device)
	input_tensor.requires_grad = True
	output = discriminator(input_tensor)
	loss = loss_fn(output, torch.zeros(minibatch_size, 1).to(device))
	# loss = loss_fn(discriminator(input_tensor), torch.ones(minibatch_size, 1).to(device))
	loss.backward()
	random_input_l1 = torch.sum(torch.abs(lr*torch.randn(input_tensor.shape)))
	gradient = torch.sign(input_tensor.grad)
	adversarial_inputs = input_tensor + lr * gradient
	adversarial_input_l1 = torch.sum(torch.abs(input_tensor - adversarial_inputs))
	adversarial_output = discriminator(adversarial_inputs)
	adversarial_output_l1 = torch.sum(torch.abs(output - adversarial_output))
	routput = discriminator(input_tensor + lr * torch.randn(input_tensor.shape).to(device))
	random_output_l1 = torch.sum(torch.abs(routput - output))
	ratio = (adversarial_output_l1 * random_output_l1) / (random_output_l1 * adversarial_input_l1)
	print (f'r = {ratio}')
	# adversarial_inputs = adversarial_inputs.cpu().permute(0, 2, 3, 1).detach().numpy()
	# input_tensor = input_tensor.cpu().permute(0, 2, 3, 1).detach().numpy()
	# show_batch(adversarial_inputs, count=0)
	# show_batch(input_tensor, count=1)
	# print (torch.argmax(output, dim=1) == torch.argmax(adversarial_output, dim=1))
	return float(ratio)


embedding_dim = 100
epochs = 50
discriminator = FCnet().to(device)
generator = InvertedFC(hidden_dim=embedding_dim).to(device)
loss_fn = nn.BCELoss()
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
generator.eval(), discriminator.eval()
# train_fcgan(train_dataloader, discriminator, generator, loss_fn, epochs, embedding_dim)
# torch.save(discriminator.state_dict(), f'fcgan_discriminator.pth')
# torch.save(generator.state_dict(), f'fcgan_generator.pth')
# print ('trained models saved')

matplotlib.rcParams.update({'font.size': 16})
# loss_fn = nn.BCELoss()

# discriminator = FCnet().to(device)
# generator = MediumIF(hidden_dim=10000).to(device)
generator.load_state_dict(torch.load(f'fcgan_generator.pth'))
discriminator.load_state_dict(torch.load(f'fcgan_discriminator.pth'))
# model = generator.to(device)
# discriminator = discriminator.to(device)

arrs = []
for i in range(10):
	lr = 1
	arr = []
	for i in range(8):
		arr.append(test_adversarials_generator(lr, embedding_size=embedding_dim))
		lr /= 10
	arrs.append(arr)

for arr in arrs:
	plt.plot([i for i in range(8)], arr)
# plt.yscale('log')
plt.ylabel('Expansion ratio')
plt.xlabel('Learning rate, 1*10^-n')
plt.savefig(f'figure_{embedding_dim}', dpi=300, bbox_inches='tight')
plt.close()


