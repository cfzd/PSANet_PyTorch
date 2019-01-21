import torch
from torch.autograd import Function,Variable
import sys
from .._ext import PSAWapper
import pdb

class PSANetCollectFunction(Function):
	@staticmethod
	def forward(ctx,x):
		if not x.is_cuda:
			print('PSAFunction.py: PSANet is not running on a CUDA device')

		out = torch.zeros(x.shape[0],x.shape[2]*x.shape[3],x.shape[2],x.shape[3]).cuda()

		PSAWapper.PSA_forward(x,out,1)

		return out

	@staticmethod
	def backward(ctx,mask_grad):

		if not mask_grad.is_cuda:
			print('PSAFunction.py: PSANet is not running on a CUDA device')
		b1_grad_n = mask_grad.shape[0]
		b1_grad_c = (2 * mask_grad.shape[2] - 1)*(2 * mask_grad.shape[3] - 1)
		b1_grad_h = mask_grad.shape[2]
		b1_grad_w = mask_grad.shape[3]


		bottom1_grad = torch.zeros(b1_grad_n,b1_grad_c,b1_grad_h,b1_grad_w).cuda()
	
		PSAWapper.PSA_backward(1,bottom1_grad, mask_grad.data)
		
		return Variable(bottom1_grad)

class PSANetDistributeFunction(Function):
	@staticmethod
	def forward(ctx,x):
		if not x.is_cuda:
			print('PSAFunction.py: PSANet is not running on a CUDA device')

		out = torch.zeros(x.shape[0],x.shape[2]*x.shape[3],x.shape[2],x.shape[3]).cuda()

		PSAWapper.PSA_forward(x,out,2)

		return out

	@staticmethod
	def backward(ctx,mask_grad):

		if not mask_grad.is_cuda:
			print('PSAFunction.py: PSANet is not running on a CUDA device')
		b1_grad_n = mask_grad.shape[0]
		b1_grad_c = (2 * mask_grad.shape[2] - 1)*(2 * mask_grad.shape[3] - 1)
		b1_grad_h = mask_grad.shape[2]
		b1_grad_w = mask_grad.shape[3]


		bottom1_grad = torch.zeros(b1_grad_n,b1_grad_c,b1_grad_h,b1_grad_w).cuda()
	
		PSAWapper.PSA_backward(2,bottom1_grad, mask_grad.data)
		
		return Variable(bottom1_grad)