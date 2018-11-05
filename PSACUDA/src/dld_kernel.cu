#include <stdlib.h>
#include <stdio.h>
#include "dld_kernel.h"

 __global__ void dld_kernel(const float * images, const int height, const int width, float * dld_output,const int y_sky, const int max_width) {
    int tid = blockIdx.x * blockDim.x +threadIdx.x + y_sky * width;

    int pos_x = tid % width;
    int pos_y = tid / width;
    int cur_width = (pos_y - y_sky) * max_width / (height - y_sky);
    if (pos_x >= cur_width && pos_x < width - cur_width){
        dld_output[pos_y * width + pos_x] = images[pos_y * width + pos_x] - images[pos_y * width + pos_x - cur_width];
        // printf("current id:%d,current pos_x : %d \n",tid, pos_x );
        dld_output[pos_y * width + pos_x + height * width - cur_width] = - dld_output[tid];
    }
}


int dld_kernel_launcher(const float * images, const int height, const int width, float * dld_output,const int y_sky, const int max_width){

    dld_kernel<<<(height-y_sky),width>>>(images,height,width,dld_output,y_sky,max_width);

    cudaError_t err;
    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}
// int main() {

//     float* input; float* output;
//     int height = 10; int width = 10;
//     int y_sky = 5; int max_width = 4;

//     input = (float *) malloc(sizeof(float)*width*height);
//     output = (float *) malloc(sizeof(float)*width*height * 2);

//     for(int i=0;i<height;i++){
//         for(int j=0;j<width;j++){
//             input[i*width+j] = j;
//             printf("%.3f ",input[i*width+j]);
//         }
//         printf("\n");
//     }
//     for(int i=0;i<height*2;i++){
//         for(int j=0;j<width;j++){
//             output[i*width+j] = 0;
//         }
//     }
//     // Memory addresses in GPU
//     float *d_input, * d_output;
//     cudaMalloc((void **) & d_input,sizeof(float)*width*height);
//     cudaMalloc((void **) & d_output,sizeof(float)*width*height*2);

//     cudaMemcpy(d_input,input,sizeof(float)*width*height, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_output,output,sizeof(float)*width*height*2, cudaMemcpyHostToDevice);

//     dld_kernel<<< (height-y_sky), width >>>(d_input,height, width, d_output, y_sky, max_width);

//     cudaMemcpy(output,d_output,sizeof(float)*width*height*2, cudaMemcpyDeviceToHost);

//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         printf("Error in dld_kernel: %s\n", cudaGetErrorString(err));
//         return 0;
//     }
//     printf("++++++++++++++++++++++++++++++++++++++++++\n");
//     for(int i=0;i<height*2;i++){
//         for(int j=0;j<width;j++){
//             printf("%.1f ",output[i*width+j]);
//         }
//         printf("\n");
//     }
//     // Free alloced memory in GPU
 
//     // Check for errors

//     return 1;
// }


 
