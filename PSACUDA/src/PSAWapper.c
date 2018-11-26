#include <THC/THC.h>
 
#include "PSAkernel.h"
 
// symbol to be automatically resolved by PyTorch libs
extern THCState *state;
 
void PSA_forward(THCudaTensor* bottom1,THCudaTensor * mask,const int type) {
	int num        = bottom1->size[0];
	int feature_H_ = bottom1->size[2];
	int feature_W_ = bottom1->size[3];

	float * b1 = THCudaTensor_data(state,bottom1);
	float * mask_buffer_ = THCudaTensor_data(state,mask);


	cudaStream_t stream = THCState_getCurrentStream(state);
 

	void PSAkernel_Forward_Launcher(b1, num,
		feature_H_,feature_W_,
		mask_buffer_,type,stream);
	 
}
void PSA_backward(const int type,
	THCudaTensor* bottom1_grad, THCudaTensor * mask_grad){
	int num        = bottom1_grad->size[0];

	int feature_H_ = bottom1_grad->size[2];
	int feature_W_ = bottom1_grad->size[3];


	float * b1_grad          = THCudaTensor_data(state,bottom1_grad);
	float * mask_buffer_grad = THCudaTensor_data(state,mask_grad);

	cudaStream_t stream = THCState_getCurrentStream(state);

	void PSAkernel_Backward_Launcher(
		b1_grad, num,
		feature_H_,feature_W_,
		mask_buffer_grad, type,stream);
}