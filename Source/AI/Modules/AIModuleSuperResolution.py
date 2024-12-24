import math

import torch
import torchvision

class ConvolutionalBlock(torch.nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=False, activation=None):
		super(ConvolutionalBlock, self).__init__()
		layers = []
		layers.append(torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2))

		if batch_norm is True:
			layers.append(torch.nn.BatchNorm2d(num_features=out_channels))
		if activation is not None:
			activation = activation.lower()
			assert activation in {'prelu', 'leakyrelu', 'tanh'}
		if activation == 'prelu':
			layers.append(torch.nn.PReLU())
		elif activation == 'leakyrelu':
			layers.append(torch.nn.LeakyReLU(0.2))
		elif activation == 'tanh':
			layers.append(torch.nn.Tanh())

		self.conv_block = torch.nn.Sequential(*layers)

	def forward(self, input):
		output = self.conv_block(input)

		return output

class SubPixelConvolutionalBlock(torch.nn.Module):
	def __init__(self, kernel_size=3, n_channels=64, scaling_factor=2):
		super(SubPixelConvolutionalBlock, self).__init__()
		self.conv = torch.nn.Conv2d(in_channels=n_channels, out_channels=n_channels * (scaling_factor ** 2), kernel_size=kernel_size, padding=kernel_size // 2)
		self.pixel_shuffle = torch.nn.PixelShuffle(upscale_factor=scaling_factor)
		self.prelu = torch.nn.PReLU()

	def forward(self, input):
		output = self.conv(input)
		output = self.pixel_shuffle(output)
		output = self.prelu(output)

		return output

class ResidualBlock(torch.nn.Module):
	def __init__(self, kernel_size=3, n_channels=64):
		super(ResidualBlock, self).__init__()
		self.conv_block1 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, batch_norm=True, activation='PReLu')
		self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, batch_norm=True, activation=None)

	def forward(self, input):
		residual = input 
		output = self.conv_block1(input) 
		output = self.conv_block2(output)  
		output = output + residual

		return output

class SRResNet(torch.nn.Module):
	def __init__(self, kernel_larger_size=9, kernel_smaller_size=3, channels_count=64, blocks_residual_count=16, scaling_factor : int=4):
		super(SRResNet, self).__init__()
		assert scaling_factor in {2, 4, 8}
		self.scaling_factor = scaling_factor
		self.conv_block1 = ConvolutionalBlock(3, channels_count, kernel_size=kernel_larger_size, batch_norm=False, activation='PReLu')
		
		layers_residual=[]
		for i in range(blocks_residual_count):
			layers_residual.append(ResidualBlock(kernel_smaller_size, channels_count))
		self.residual_blocks = torch.nn.Sequential(*layers_residual)
		self.conv_block2 = ConvolutionalBlock(channels_count, channels_count, kernel_smaller_size, batch_norm=True)
		
		layers_subpixel=[]
		for i in range(int(math.log2(scaling_factor))):
			layers_subpixel.append(SubPixelConvolutionalBlock(kernel_smaller_size, channels_count, scaling_factor=2) )
		self.subpixel_convolutional_blocks = torch.nn.Sequential(*layers_subpixel)
		self.conv_block3 = ConvolutionalBlock(channels_count, 3, kernel_size=kernel_larger_size, batch_norm=False, activation='Tanh')

	def forward(self, input):
		output = self.conv_block1(input)  
		residual = output  
		output = self.residual_blocks(output) 
		output = self.conv_block2(output)
		output = output + residual
		output = self.subpixel_convolutional_blocks(output) 
		output = self.conv_block3(output)

		return output

class SRResNetDiscriminator(torch.nn.Module):
	def __init__(self, kernel_size=3, channels_count=64, blocks_convolution_count=8, layer_fully_connected_size=1024):
		super(SRResNetDiscriminator, self).__init__()

		channels_count_in = 3
		conv_blocks = []
		for i in range(blocks_convolution_count):
			if i % 2==0:
				if i==0:
					channels_count_out=channels_count
				else:
					channels_count_out=channels_count_in*2
				stride=1
			else:
				channels_count_out=channels_count_in
				stride=2
			conv_blocks.append(ConvolutionalBlock(channels_count_in, channels_count_out, kernel_size, stride, activation='LeakyReLu'))
			channels_count_in = channels_count_out
		self.conv_blocks = torch.nn.Sequential(*conv_blocks)
		self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((6, 6))
		self.fc1 = torch.nn.Linear(channels_count_out*6*6, layer_fully_connected_size)
		self.leaky_relu = torch.nn.LeakyReLU(0.2)
		self.fc2 = torch.nn.Linear(1024, 1)

	def forward(self, input):
		batch_size = input.size(0)
		output = self.conv_blocks(input)
		output = self.adaptive_pool(output)
		output = self.fc1(output.view(batch_size, -1))
		output = self.leaky_relu(output)
		logit = self.fc2(output)

		return logit

class TruncatedVGG19(torch.nn.Module):
	def __init__(self, i, j):
		super(TruncatedVGG19, self).__init__()
		vgg19 = torchvision.models.vgg19(pretrained=True)
		maxpool_counter = 0
		conv_counter = 0
		modules_children=[]
		
		for layer in vgg19.features.children():
			if isinstance(layer, torch.nn.Conv2d):
				conv_counter += 1
			if isinstance(layer, torch.nn.MaxPool2d):
				maxpool_counter += 1
				conv_counter = 0
			modules_children.append(layer)
			if maxpool_counter == i-1 and conv_counter == j:
				modules_children.append(layer)	
				break

		self.vgg19_truncated = torch.nn.Sequential(*modules_children)

	def forward(self, input):
		output = self.vgg19_truncated(input)

		return output