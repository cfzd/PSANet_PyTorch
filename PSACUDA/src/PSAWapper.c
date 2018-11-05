#include <THC/THC.h>
 
#include "PSAkernel.h"
 
// symbol to be automatically resolved by PyTorch libs
extern THCState *state;
 
void PSA_forward(THCudaTensor* bottom0, THCudaTensor* bottom1, THCudaTensor* output,THCudaTensor * mask,const int type) {
	int num        = bottom0->size[0];
	int nChannels  = bottom0->size[1];
	int feature_H_ = bottom0->size[2];
	int feature_W_ = bottom0->size[3];
	float * b0 = THCudaTensor_data(state,bottom0);
	float * b1 = THCudaTensor_data(state,bottom1);

	//THCudaTensor_fill(state, output, 0);
	float * top = THCudaTensor_data(state,output);
	float * mask_buffer_ = THCudaTensor_data(state,mask);


	cudaStream_t stream = THCState_getCurrentStream(state);
 

	PSAkernel_Forward_Launcher(b0,b1, num,
	nChannels,feature_H_,feature_W_,
	top,mask_buffer_,type,stream);
	 
}
void PSA_backward(THCudaTensor* bottom0, THCudaTensor* bottom1, THCudaTensor* output,THCudaTensor * mask,const int type,
	THCudaTensor* bottom0_grad, THCudaTensor* bottom1_grad, THCudaTensor* output_grad,THCudaTensor * mask_grad){
	int num        = bottom0->size[0];
	int nChannels  = bottom0->size[1];
	int feature_H_ = bottom0->size[2];
	int feature_W_ = bottom0->size[3];

	float * b0           = THCudaTensor_data(state,bottom0);
	float * b1           = THCudaTensor_data(state,bottom1);
	float * top          = THCudaTensor_data(state,output);
	float * mask_buffer_ = THCudaTensor_data(state,mask);

	float * b0_grad          = THCudaTensor_data(state,bottom0_grad);
	float * b1_grad          = THCudaTensor_data(state,bottom1_grad);
	float * top_grad         = THCudaTensor_data(state,output_grad);
	float * mask_buffer_grad = THCudaTensor_data(state,mask_grad);

	cudaStream_t stream = THCState_getCurrentStream(state);

	PSAkernel_Backward_Launcher(b0,b1,
	b0_grad,b1_grad, num,
	nChannels,feature_H_,feature_W_,
	top, top_grad, mask_buffer_, 
	mask_buffer_grad, type,stream);
}