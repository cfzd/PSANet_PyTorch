void PSA_forward(THCudaTensor* bottom1,THCudaTensor * mask,const int type);
void PSA_backward(const int type,THCudaTensor* bottom1_grad, THCudaTensor * mask_grad);