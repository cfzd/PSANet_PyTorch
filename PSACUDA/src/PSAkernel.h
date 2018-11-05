
#include <cublas_v2.h>
#ifdef __cplusplus
extern "C" {
#endif

#include <cblas.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define PSA_TYPE_COLLECT 1
#define PSA_TYPE_DISTRIBUTE 2
const int CAFFE_CUDA_NUM_THREADS = 512;
#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())
#define CUDA_CHECK(condition) \
   /* Code block avoids redefinition of cudaError_t error */ \
  do { \
     cudaError_t error = condition; \
     CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
        i < (n); \
        i += blockDim.x * gridDim.x)

inline int CAFFE_GET_BLOCKS(const int N) {
   return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
 }


void PSAkernel_Forward_Launcher(const float *  bottom0,const float *  bottom1, const int num_,
    const int channels_,const int feature_H_,const int feature_W_,
    float * top,float * mask_buffer_,const int forward_type,cudaStream_t stream);


void PSAkernel_Backward_Launcher(const float *  bottom0_data,const float *  bottom1_data,
    float *  bottom0_diff,float *  bottom1_diff, const int num_,
    const int channels_,const int feature_H_,const int feature_W_,
    const float * top, const float * top_diff, const float * mask_buffer_, 
    float * mask_buffer_diff, const int forward_type,cudaStream_t stream);


#ifdef __cplusplus
}

#endif


