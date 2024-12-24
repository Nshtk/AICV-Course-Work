import numpy as np
import cv2 as cv
from PIL import Image

import torch
from torchvision import transforms

from AI.Models.AIModel import *
from AI.Modules.AIModuleSuperResolution import *
from Utilities.Logger import *

class AIModelSuperResolution(AIModel):
	def __init__(self, device, logging_path=None):
		super(AIModelSuperResolution, self).__init__(device, logging_path)
		self.model_generator : torch.nn.Module=SRResNet().to(device)
		self.model_discriminator : torch.nn.Module=SRResNetDiscriminator().to(device)
		
	def train(self, dataloader : torch.utils.data.DataLoader, epochs_count : int, loss_layer, optimiser=None, metric=None):
		self.model_generator.train()
		if optimiser is None:
			optimiser=torch.optim.Adam(self.model_generator.parameters(), lr=0.0001)
		loss_layer=loss_layer.to(self.device)
		
		print("TRAINING PHASE 1 STARTED.")	#Training SRResNet
		for i in range(epochs_count):
			loss_epoch = 0
			metric_epoch=0
			j=0
			for x, y in dataloader:
				x, y = x.to(self.device), y.to(self.device)
				optimiser.zero_grad()
				prediction = self.model_generator(x)
				loss = loss_layer(prediction, y)
				loss.backward()
				optimiser.step()
				loss_current=loss.item()
				loss_epoch += loss_current
				print(f"Batch: {j}, loss {loss_current}", end="")
				if(metric is not None):
					metric.update(prediction, y)
					metric_batch=metric.compute()
					metric_epoch+=metric_batch
					print(f" accuracy {metric_batch}.", end="")
				print(end="\r")
				j+=1
			print(f"Epoch: {i}, average loss {loss_epoch/j}")
			if(metric is not None):
				print(f" average accuracy {metric_epoch/j}.")
				metric.reset()
				
		print("TRAINING PHASE 2 STARTED.")	#Training Discriminator
		self.model_discriminator.train()
		if optimiser is None:
			optimiser=torch.optim.Adam(self.model_generator.parameters(), lr=0.0001)
		
		for i in range(epochs_count):
			loss_epoch = 0
			metric_epoch=0
			j=0
			for x, y in dataloader:
				x, y = x.to(self.device), y.to(self.device)
				optimiser.zero_grad()
				prediction = self.model_generator(x)
				loss = loss_layer(prediction, y)
				loss.backward()
				optimiser.step()
				loss_current=loss.item()
				loss_epoch += loss_current
				print(f"Batch: {j}, loss {loss_current}", end="")
				if(metric is not None):
					metric.update(prediction, y)
					metric_batch=metric.compute()
					metric_epoch+=metric_batch
					print(f" accuracy {metric_batch}.", end="")
				print(end="\r")
				j+=1
			print(f"Epoch: {i}, average loss {loss_epoch/j}")
			if(metric is not None):
				print(f" average accuracy {metric_epoch/j}.")
				metric.reset()
	
	def test(self, dataloader : torch.utils.data.DataLoader, metric):
		metric_value_cummulative=0
		i=0
		
		self.model_generator.eval()
		for i in range(5):
			metric.reset()
			with torch.no_grad():
				for x, y in dataloader:
					x, y = x.to(self.device), y.to(self.device)
					prediction = self.model_generator(x)
					metric.update(prediction.argmax(1), y)
			metric_value_cummulative+=metric.compute().item()
			
		return metric_value_cummulative/i
	
	def predict(self, file_video : str, path_save : str, batch_size : int):
		if not os.path.exists(path_save):
			os.makedirs(path_save)
		
		video_capture = cv.VideoCapture(file_video)
		if video_capture.isOpened(): 
			width  = video_capture.get(cv.CAP_PROP_FRAME_WIDTH) 
			height = video_capture.get(cv.CAP_PROP_FRAME_HEIGHT)
			fps = video_capture.get(cv.CAP_PROP_FPS)
		video_writer = cv.VideoWriter(f"{path_save}/{os.path.splitext(os.path.basename(file_video))[0]}_super_resolution.mp4", cv.VideoWriter_fourcc(*'mp4v'), fps, (int(width*self.model_generator.scaling_factor), int(height*self.model_generator.scaling_factor)))
		transform_normilise=transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
		
		self.model_generator.eval()	
		with torch.no_grad():
			while video_capture.isOpened():
				x_batched=[]
				for i in range(batch_size):
					is_received, frame = video_capture.read()
					if not is_received:
						break
					frame_pil = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
					x_batched.append(transform_normilise(frame_pil))
				x_batched_as_tensor=torch.from_numpy(np.array(x_batched)).to(self.device)
				prediction=self.model_generator(x_batched_as_tensor)#.squeeze(0)
				
				for i in range(batch_size):
					frame = np.transpose(prediction[i].numpy(), (1, 2, 0))
					frame = (frame + 1.) / 2.
					frame = 255. * frame
					frame=frame.astype(np.uint8)
					video_writer.write(frame)
		
		video_capture.release()
		video_writer.release()
			

	def load(self, path: str):
		if not os.path.exists(path):
			raise FileNotFoundError(f"Файлы модели не найдены в {path}")
		self.model_generator.load_state_dict(torch.load(path, map_location=self.device))
		self.model_generator=self.model_generator.to(self.device)
		
	def save(self, base_path: str):
		if not os.path.exists(base_path):
			os.makedirs(base_path)
	
		subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
		run_numbers = [int(d) for d in subdirs if d.isdigit()]
		next_run_number = max(run_numbers, default=0) + 1
		run_path = os.path.join(base_path, str(next_run_number))
		os.makedirs(run_path)
		torch.save(self.model_generator.state_dict(), os.path.join(run_path, 'model_generator.pth'))
