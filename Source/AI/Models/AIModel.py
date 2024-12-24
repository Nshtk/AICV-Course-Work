import os
import sys
import numpy as np

import torch

from Utilities.Logger import *

class AIModel():
	def __init__(self, device, logging_path=None):
		self.device=device
		self.logging_path=logging_path
		self.logger_training : Logger
		self.logger_testing : Logger
		if logging_path is not None:
			if not os.path.exists(logging_path):
				os.makedirs(logging_path)
			self.logger_training=Logger("training.log")
			self.logger_testing=Logger("testing.log")
		
	def train(self, epochs_count : int):
		pass
	
	def test(self, epochs_count : int):
		pass
	
	def load(self, path : str):
		return torch.load(path)
	
	def save(self, model, optimiser, epoch, path : str):
		torch.save({'model': model, 'optimizer': optimiser, 'epoch': epoch}, path+'srresnet.pth.tar')