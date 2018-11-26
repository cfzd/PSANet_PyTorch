# MTA_lane_detection

This is a non-official PyTorch implementation of **PSANet: Point-wise Spatial Attention Network for Scene Parsing**. 
The official implementation can be found in [here](https://github.com/hszhao/PSANet).
This code has been tested on Python 2.7, PyTorch 0.3.1.

##configuration

Change the corresponding architecture of your CUDA device in ./PSACUDA/make.sh with `-arch=sm_xx`.
The definition of architectures can be found in [here](http://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/).
##installation

`cd ./PSACUDA`

`./make.sh`

This will build the CUDA kernel code by NVCC and create a python extension.

##Usage

`from PSACUDA.moudles.PSANet import PSANetModule`

`net=PSANetModule(in_channels,reduced_channels,fea_h,fea_w,keep_channel_size = False)`

- in_channels - the number of input feature channel
- reduced_channels - the number of channels after first reduction, i.e. the C2 in the paper of PSANet
- fea_h - the hight of input
- fea_w - the width of input
- keep_channel_size(optional) - if keep_channel_size is True, the output of this module would be the same size as the input. 

