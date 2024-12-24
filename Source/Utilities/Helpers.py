import torch

def clipGradient(optimizer, grad_clip):
	for group in optimizer.param_groups:
		for param in group['params']:
			if param.grad is not None:
				param.grad.data.clamp_(-grad_clip, grad_clip)

def adjustLearningRate(optimizer, shrink_factor):
	for param_group in optimizer.param_groups:
		param_group['lr'] = param_group['lr'] * shrink_factor
	
def saveCheckpoint(state, filename):
	torch.save(state, filename)