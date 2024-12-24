import os
import copy
import json

import cv2 as cv
from PIL import Image

import torch
from torchvision import transforms

class SuperResolutionDataset(torch.utils.data.Dataset):
	def __init__(self, path_data, type, crop_size, scaling_factor):
		super(SuperResolutionDataset, self).__init__()
		self.data_folder = path_data
		self.type = type

		assert self.type in {'train', 'test'}
		if self.type == 'train':
			assert crop_size % scaling_factor == 0, "Crop dimensions are not perfectly divisible by scaling factor! This will lead to a mismatch in the dimensions of the original HR patches and their super-resolved (SR) versions!"
		path=f"{path_data}/{type}"
		self.image_files = []
		for file_name in os.listdir(path):
			file_name_full=f"{path}/{file_name}"
			if os.path.isfile(file_name_full):
				self.image_files.append(file_name_full)
		self.transform_crop=transforms.RandomCrop(crop_size)
		self.transform_resize=transforms.Resize(int(crop_size/scaling_factor), transforms.InterpolationMode.BICUBIC)
		self.transform_normilise=transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])

	def __getitem__(self, i):
		image=Image.open(self.image_files[i]).convert('RGB')
		image=self.transform_crop(image)
		image_downsampled=copy.deepcopy(image)
		image_downsampled=self.transform_crop(image_downsampled)
		image_downsampled=self.transform_resize(image_downsampled)
		
		return self.transform_normilise(image_downsampled), self.transform_normilise(image)

	def __len__(self):
		return len(self.image_files)