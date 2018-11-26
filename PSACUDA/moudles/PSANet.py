import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from ..functions.PSANetFunc import PSANetCollectFunction,PSANetDistributeFunction

def initialize_weights(*models):

    for model in models:
        real_init_weights(model)

def real_init_weights(m):

    if isinstance(m, nn.Conv2d):
        init.xavier_uniform(m.weight, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        n = m.weight.size(1)
        m.weight.data.normal_(0, math.sqrt(2. / n))
        m.bias.data.zero_()
    elif isinstance(m, nn.Module):
        for mm in m.children():
            real_init_weights(mm)
# in_channels is the number of input feature channel
# reduced_channels is the number of channels after first reduction, i.e. the C2 in the paper of PSANet
# fea_h and fea_w are the height and width of input feature respectively
# if keep_channel_size is True, the output of this module would be the same size as the input
class PSANetModule(nn.Module):
	def __init__(self,in_channels,reduced_channels,fea_h,fea_w,keep_channel_size = False):
		super(PSANetModule, self).__init__()
		self.reduced_channels = reduced_channels
		self.fea_h = fea_h
		self.fea_w = fea_w
		self.keep_channel_size = keep_channel_size
		self.reduction_c1 = nn.Sequential(
			nn.Conv2d(in_channels, reduced_channels, 1, padding = 0),
			nn.BatchNorm2d(reduced_channels),
			nn.ReLU()
			)

		self.reduction_d1 = nn.Sequential(

			nn.Conv2d(in_channels, reduced_channels, 1, padding = 0),
			nn.BatchNorm2d(reduced_channels),
			nn.ReLU()
			)

		self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

		self.reduction_c2 = nn.Sequential(
			nn.Conv2d(reduced_channels, reduced_channels, 1, padding = 0),
			nn.BatchNorm2d(reduced_channels),
			nn.ReLU(),
			nn.Conv2d(reduced_channels, (fea_h - 1) * (fea_w - 1) , 1, padding = 0)
			)
		self.reduction_d2 = nn.Sequential(
			nn.Conv2d(reduced_channels, reduced_channels, 1, padding = 0),
			nn.BatchNorm2d(reduced_channels),
			nn.ReLU(),
			nn.Conv2d(reduced_channels, (fea_h - 1) * (fea_w - 1) , 1, padding = 0)
			)
		self.enlarge = nn.Sequential(
			nn.Conv2d(reduced_channels*2,in_channels, 1, padding = 0),
			nn.BatchNorm2d(in_channels),
			nn.ReLU()
			)
		if keep_channel_size:
			self.last_conv = nn.Conv2d(in_channels*2,in_channels, 1, padding = 0)
			initialize_weights(self.last_conv)
		initialize_weights(self.reduction_c1,self.reduction_c2,self.reduction_d1,self.reduction_d2,self.enlarge)
	def forward(self,x):

		x_col = self.reduction_c1(x)
		x_dis = self.reduction_d1(x)
		x_col = self.pool(x_col)
		x_dis = self.pool(x_dis)
		x_col_over_complete = self.reduction_c2(x_col)
		x_dis_over_complete = self.reduction_d2(x_dis)

		x_col_over_complete = PSANetCollectFunction.apply(x_col_over_complete)
		x_dis_over_complete = PSANetDistributeFunction.apply(x_dis_over_complete)
		x_col_list = []
		x_dis_list = []
		for i in range(x_col.shape[0]):
			x_col_i = x_col[i].view(self.reduced_channels,-1)
			x_col_over_complete_i = x_col_over_complete[i].view(
				x_col_over_complete.shape[1],-1)
			x_col_list.append(
				torch.mm(x_col_i,x_col_over_complete_i).view(
					1,self.reduced_channels,self.fea_h / 2,self.fea_w / 2))

			x_dis_i = x_dis[i].view(self.reduced_channels,-1)
			x_dis_over_complete_i = x_dis_over_complete[i].view(
				x_dis_over_complete.shape[1],-1)
			x_dis_list.append(
				torch.mm(x_dis_i,x_dis_over_complete_i).view(
					1,self.reduced_channels,self.fea_h / 2,self.fea_w / 2))
		x_col = torch.cat(x_col_list)
		x_dis = torch.cat(x_dis_list)

		psa = torch.cat((x_col,x_dis),dim = 1)
		psa = self.enlarge(psa)
		psa = F.upsample(psa, scale_factor=2, mode='bilinear')
		if self.keep_channel_size:
			return self.last_conv(torch.cat((x,psa),dim = 1))
		return torch.cat((x,psa),dim = 1)
		
