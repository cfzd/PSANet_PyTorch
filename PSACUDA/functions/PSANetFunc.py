import torch
from torch.autograd import Function,Variable
import sys
from .._ext import PSAWapper
import pdb

class PSANetCollectFunction(Function):
	@staticmethod
	def forward(ctx,input1,input2):
		if not input1.is_cuda or not input2.is_cuda:
			print('PSAFunction.py: PSANet is not running on a CUDA device')

		mask = torch.zeros(input1.shape[0],input1.shape[2]*input1.shape[3],
			input1.shape[2],input1.shape[3]).cuda()
		out = torch.zeros(input1.shape).cuda()
		PSAWapper.PSA_forward(input1, input2, out,mask,1)

		ctx.save_for_backward(input1,input2,mask,out)
		# pdb.set_trace()
		return out,mask

	@staticmethod
	def backward(ctx,output_grad,mask_grad):
		# pdb.set_trace()
		
		if not output_grad.is_cuda:
			print('PSAFunction.py: PSANet is not running on a CUDA device')

		input1,input2,mask,out = ctx.saved_tensors

		input1_grad = torch.zeros(input1.shape).cuda()
		input2_grad = torch.zeros(input2.shape).cuda()
		# mask_grad   = torch.zeros(mask.shape).cuda()
		# pdb.set_trace()
		PSAWapper.PSA_backward( input1, input2, out ,mask ,1,
			input1_grad, input2_grad, output_grad.data,mask_grad.data)
		
		return Variable(input1_grad),Variable(input2_grad)
class PSANetDistributeFunction(Function):
	@staticmethod
	def forward(ctx,input1,input2):
		if not input1.is_cuda or not input2.is_cuda:
			print('PSAFunction.py: PSANet is not running on a CUDA device')

		mask = torch.zeros(input1.shape[0],input1.shape[2]*input1.shape[3],
			input1.shape[2],input1.shape[3]).cuda()
		out = torch.zeros(input1.shape).cuda()
		PSAWapper.PSA_forward(input1, input2, out,mask,2)

		ctx.save_for_backward(input1,input2,mask,out)
		return out,mask

	@staticmethod
	def backward(ctx,output_grad,mask_grad):
		# pdb.set_trace()
		if not output_grad.is_cuda:
			print('PSAFunction.py: PSANet is not running on a CUDA device')

		input1,input2,mask,out = ctx.saved_tensors

		input1_grad = torch.zeros(input1.shape).cuda()
		input2_grad = torch.zeros(input2.shape).cuda()
		# mask_grad   = torch.zeros(mask.shape).cuda()
		# pdb.set_trace()

		PSAWapper.PSA_backward( input1, input2, out ,mask ,2,
			input1_grad, input2_grad, output_grad.data,mask_grad.data)
		
		return Variable(input1_grad),Variable(input2_grad)