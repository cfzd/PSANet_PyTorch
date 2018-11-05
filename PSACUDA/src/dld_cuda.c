#include <THC/THC.h>
 
#include "dld_kernel.h"
 
// symbol to be automatically resolved by PyTorch libs
extern THCState *state;
 
int dld_forward(THCudaTensor* input, THCudaTensor* output,int y_sky,int max_width) {
    int nChannels = input->size[0];
    int height = input->size[1];
    int width = input->size[2];
 
    THCudaTensor_fill(state, output, 0);

 
    int success = 0;
    success = dld_kernel_launcher(THCudaTensor_data(state, input), 
        height, width, 
        THCudaTensor_data(state, output),y_sky,max_width);
     
    //Check for errors
    if ( !success ) {
        THError("aborting");
    }
    return 1;
}