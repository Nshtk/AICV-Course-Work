import os
import math
import argparse

import torch

import matplotlib.pyplot as plt

from AI.Models.AIModel import *
from AI.Models.AIModelSuperResolution import *
from AI.Data.SuperResolutionDataset import *
from Utilities.Logger import *

THREADS_COUNT = os.cpu_count()
DEVICE : torch.device
if not torch.cuda.is_available():
	DEVICE = torch.device("cuda")
	print("Cuda is available.")
else:
	DEVICE=torch.device("cpu")
torch.set_num_threads(THREADS_COUNT)

if __name__ == "__main__":
	argparser=argparse.ArgumentParser("lab3")
	argparser.add_argument('-data_path', 	required=True,  type=str, dest="path_data", help="Is loading model from file.")
	argparser.add_argument('-load', 		required=False, type=str, dest="path_model_load", help="Is loading model from file.")
	argparser.add_argument('-train', 		required=False, type=bool, help="Is training model with number of epochs.")
	argparser.add_argument('-test', 		required=False, type=bool, help="Is testing model with number of epochs.")
	argparser.add_argument('-epochs_count', required=True,  type=int, dest="epochs_count", help="Model batch size.")
	argparser.add_argument('-batch_size', 	required=True,  type=int, dest="batch_size", help="Model batch size.")
	path_logs="../Logs/Latest"
	path_models="../Data/Models"
	path_data="../Data/Datasets"
	path_model_load=""
	epochs_count=1
	batch_size=16
	args = argparser.parse_args()
	
	if args.path_data!="":
		path_data=args.path_data
	if args.path_model_load:
		path_model_load=args.path_model_load
	if args.train!=None:
		is_training_model=True
		epochs_count=args.epochs_count
	if args.test!=None:
		is_testing_model=True
	if args.batch_size:
		batch_size=args.batch_size
	if args.epochs_count:
		epochs_count=args.epochs_count
	if not os.path.exists(path_models):
		os.makedirs(path_models)
	if not os.path.exists(path_logs):
		os.makedirs(path_logs)
	
	model=AIModelSuperResolution(DEVICE)
	if path_model_load!="":
		model.load(path_model_load)
	path_data_raw=path_data+"/raw"		# Обработка видеофайлов, необходимо поместить нужные видео в папку raw
	for file in os.listdir(path_data_raw):
		if file.endswith(".mp4"):
			file_video=f"{path_data_raw}/{file}"
			model.predict(file_video, f"{path_data}/{os.path.splitext(os.path.basename(file_video))[0]}", batch_size)
	
	if is_training_model:
		dataset=SuperResolutionDataset(path_data, "train", 96, 4)
		dataloader=torch.utils.data.DataLoader(dataset, batch_size, True, num_workers=0, pin_memory=True)
		model.train(dataloader, epochs_count, torch.nn.MSELoss())
	if is_testing_model:
		dataset=SuperResolutionDataset(path_data, "test", 0, 4)
		dataloader=torch.utils.data.DataLoader(dataset, batch_size, False, num_workers=0, pin_memory=True)
		model.test(dataloader, torch.nn.MSELoss())