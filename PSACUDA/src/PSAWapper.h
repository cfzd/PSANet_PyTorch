void PSA_forward(THCudaTensor* bottom0, THCudaTensor* bottom1, THCudaTensor* output,THCudaTensor * mask,const int type);
void PSA_backward(THCudaTensor* bottom0, THCudaTensor* bottom1, THCudaTensor* output,THCudaTensor * mask,const int type,THCudaTensor* bottom0_grad, THCudaTensor* bottom1_grad, THCudaTensor* output_grad,THCudaTensor * mask_grad);
