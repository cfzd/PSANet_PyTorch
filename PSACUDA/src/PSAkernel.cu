#include <stdlib.h>
#include <stdio.h>
#include "PSAkernel.h"
#include <vector>
void caffe_gpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSgemm(handle, cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N);
}

template <typename Dtype>
__global__ void PSAForward_buffer_mask_collect_gpu(const int nthreads,
    const int feature_H_, const int feature_W_,
    const int mask_H_, const int mask_W_,
    const int half_mask_H_, const int half_mask_W_,
    const Dtype* mask_data, Dtype* buffer_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % feature_W_;
    const int h = (index / feature_W_) % feature_H_;
    const int n = index / feature_W_ / feature_H_;
    // effective mask region : [hstart, hend) x [wstart, wend) with mask-indexed
    const int hstart = max(0, half_mask_H_ - h);
    const int hend = min(mask_H_, feature_H_ + half_mask_H_ - h);
    const int wstart = max(0, half_mask_W_ - w);
    const int wend = min(mask_W_, feature_W_ + half_mask_W_ - w);
    // (hidx,                    widx                   ) with mask-indexed
    // (hidx + h - half_mask_H_, widx + w - half_mask_W_) with feature-indexed
    for (int hidx = hstart; hidx < hend; hidx++) {
      for (int widx = wstart; widx < wend; widx++) {
        buffer_data[(n * feature_H_ * feature_W_ + (hidx + h - half_mask_H_) * feature_W_ + (widx + w - half_mask_W_)) * feature_H_ * feature_W_ + h * feature_W_ + w] =
            mask_data[((n * mask_H_ * mask_W_ + hidx * mask_W_ + widx) * feature_H_ + h) * feature_W_ + w];
      }
    }
  }
}
template <typename Dtype>
__global__ void PSAForward_buffer_mask_distribute_gpu(const int nthreads,
    const int feature_H_, const int feature_W_,
    const int mask_H_, const int mask_W_,
    const int half_mask_H_, const int half_mask_W_,
    const Dtype* mask_data, Dtype* buffer_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % feature_W_;
    const int h = (index / feature_W_) % feature_H_;
    const int n = index / feature_W_ / feature_H_;
    // effective mask region : [hstart, hend) x [wstart, wend) with mask-indexed
    const int hstart = max(0, half_mask_H_ - h);
    const int hend = min(mask_H_, feature_H_ + half_mask_H_ - h);
    const int wstart = max(0, half_mask_W_ - w);
    const int wend = min(mask_W_, feature_W_ + half_mask_W_ - w);
    // (hidx,                    widx                   ) with mask-indexed
    // (hidx + h - half_mask_H_, widx + w - half_mask_W_) with feature-indexed
    // printf("hstart: %d  hend: %d  wstart: %d wend:%d\n",hstart,hend,wstart,wend);
    for (int hidx = hstart; hidx < hend; hidx++) {
      for (int widx = wstart; widx < wend; widx++) {
        buffer_data[(n * feature_H_ * feature_W_ + h * feature_W_ + w) * feature_H_ * feature_W_ + (hidx + h - half_mask_H_) * feature_W_ + (widx + w - half_mask_W_)] =
            mask_data[((n * mask_H_ * mask_W_ + hidx * mask_W_ + widx) * feature_H_ + h) * feature_W_ + w];
      }
    }
  }
}
template <typename Dtype>
__global__ void PSABackward_buffer_mask_collect_gpu(const int nthreads,
    const int feature_H_, const int feature_W_,
    const int mask_H_, const int mask_W_,
    const int half_mask_H_, const int half_mask_W_,
    const Dtype* buffer_diff, Dtype* mask_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % feature_W_;
    const int h = (index / feature_W_) % feature_H_;
    const int n = index / feature_W_ / feature_H_;
    // effective mask region : [hstart, hend) x [wstart, wend) with mask-indexed
    const int hstart = max(0, half_mask_H_ - h);
    const int hend = min(mask_H_, feature_H_ + half_mask_H_ - h);
    const int wstart = max(0, half_mask_W_ - w);
    const int wend = min(mask_W_, feature_W_ + half_mask_W_ - w);
    // (hidx,                    widx                   ) with mask-indexed
    // (hidx + h - half_mask_H_, widx + w - half_mask_W_) with feature-indexed
    for (int hidx = hstart; hidx < hend; hidx++) {
      for (int widx = wstart; widx < wend; widx++) {
        mask_diff[((n * mask_H_ * mask_W_ + hidx * mask_W_ + widx) * feature_H_ + h) * feature_W_ + w] =
            buffer_diff[(n * feature_H_ * feature_W_ + (hidx + h - half_mask_H_) * feature_W_ + (widx + w - half_mask_W_)) * feature_H_ * feature_W_ + h * feature_W_ + w];
      }
    }
  }
}

template <typename Dtype>
__global__ void PSABackward_buffer_mask_distribute_gpu(const int nthreads,
    const int feature_H_, const int feature_W_,
    const int mask_H_, const int mask_W_,
    const int half_mask_H_, const int half_mask_W_,
    const Dtype* buffer_diff, Dtype* mask_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % feature_W_;
    const int h = (index / feature_W_) % feature_H_;
    const int n = index / feature_W_ / feature_H_;
    // effective mask region : [hstart, hend) x [wstart, wend) with mask-indexed
    const int hstart = max(0, half_mask_H_ - h);
    const int hend = min(mask_H_, feature_H_ + half_mask_H_ - h);
    const int wstart = max(0, half_mask_W_ - w);
    const int wend = min(mask_W_, feature_W_ + half_mask_W_ - w);
    // (hidx,                    widx                   ) with mask-indexed
    // (hidx + h - half_mask_H_, widx + w - half_mask_W_) with feature-indexed
    for (int hidx = hstart; hidx < hend; hidx++) {
      for (int widx = wstart; widx < wend; widx++) {
        mask_diff[((n * mask_H_ * mask_W_ + hidx * mask_W_ + widx) * feature_H_ + h) * feature_W_ + w] =
            buffer_diff[(n * feature_H_ * feature_W_ + h * feature_W_ + w) * feature_H_ * feature_W_ + (hidx + h - half_mask_H_) * feature_W_ + (widx + w - half_mask_W_)];
      }
    }
  }
}



void PSAkernel_Forward_Launcher(const float *  bottom0,const float *  bottom1, const int num_,
    const int channels_,const int feature_H_,const int feature_W_,
    float * top,float * mask_buffer_,const int forward_type,cudaStream_t stream) {
  // set mask buffer

  const int mask_H_ = 2 * feature_H_ - 1;const int mask_W_ = 2 * feature_W_ - 1;
  const int half_mask_H_ = (mask_H_ - 1) / 2;const int half_mask_W_ = (mask_W_ - 1) / 2;
 
  int nthreads = num_ * feature_H_ * feature_W_;
  switch (forward_type) {
  case PSA_TYPE_COLLECT:
    PSAForward_buffer_mask_collect_gpu<float><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS,0,stream>>>(
        nthreads, feature_H_, feature_W_, mask_H_, mask_W_, half_mask_H_, half_mask_W_,
        bottom1, mask_buffer_);

    do {
    cudaError_t error = cudaPeekAtLastError();
    if (error != cudaSuccess){
      printf(cudaGetErrorString(error));}
    } while(0);
    break;
  case PSA_TYPE_DISTRIBUTE:
    PSAForward_buffer_mask_distribute_gpu<float><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS,0,stream>>>(
        nthreads, feature_H_, feature_W_, mask_H_, mask_W_, half_mask_H_, half_mask_W_,
        bottom1, mask_buffer_);
    do {
    cudaError_t error = cudaPeekAtLastError();
    if (error != cudaSuccess){
      printf(cudaGetErrorString(error));}
    } while(0);
    break;
  default:
    printf("Unknown PSA type.");
  }

  for(int n = 0; n < num_; n++) {
    const float* this_bottom_data = bottom0 + n*channels_*feature_H_*feature_W_;
    const float* this_mask_data = mask_buffer_ + n*feature_H_*feature_W_* feature_H_* feature_W_;
    float * this_top_data = top + n * channels_* feature_H_* feature_W_;

    caffe_gpu_gemm(CblasNoTrans, CblasNoTrans,
                   channels_, feature_H_ * feature_W_, feature_H_ * feature_W_,
                   float(1.0), this_bottom_data, this_mask_data, float(0), this_top_data);
  }
}

void debuger(const float * data){
  float * h_data = (float *) malloc(5*sizeof(float));
  cudaMemcpy((void *) h_data,(const void *)data,5*sizeof(float),cudaMemcpyDeviceToHost);
  for(int i=0;i<5;i++){
    printf("%f ", h_data[i]);
  }
  printf("\n");
}

void PSAkernel_Backward_Launcher(const float *  bottom0_data,const float *  bottom1_data,
    float *  bottom0_diff,float *  bottom1_diff, const int num_,
    const int channels_,const int feature_H_,const int feature_W_,
    const float * top, const float * top_diff, const float * mask_buffer_, 
    float * mask_buffer_diff, const int forward_type,cudaStream_t stream) {
  // BP to feature
  int debug = 0;
  if (debug)
  {
    printf("before bp to bottom0\n");
    debuger(bottom0_diff);
  }
  for(int n = 0; n < num_; n++) {
    const float* this_top_diff = top_diff + n*channels_* feature_H_* feature_W_;
    const float* this_mask_data = mask_buffer_ + n*feature_H_*feature_W_* feature_H_* feature_W_;
    float* this_bottom_diff = bottom0_diff + n*channels_*feature_H_*feature_W_;
    caffe_gpu_gemm(CblasNoTrans, CblasTrans,
                     channels_, feature_H_ * feature_W_, feature_H_ * feature_W_,
                     float(1.0), this_top_diff, this_mask_data, float(0), this_bottom_diff);
  }
  if (debug)
  {
    printf("after bp to bottom0\n");
    debuger(bottom0_diff);
  }
  if (debug)
  {
    printf("before bp to mask\n");
    debuger(mask_buffer_diff);
  }
  // BP to attention
  for(int n = 0; n < num_; n++) {
    const float* this_top_diff = top_diff + n*channels_* feature_H_* feature_W_;
    const float* this_bottom_data = bottom0_data + n*channels_*feature_H_*feature_W_;
    float * this_mask_diff = mask_buffer_diff + n*feature_H_*feature_W_* feature_H_* feature_W_;
    caffe_gpu_gemm(CblasTrans, CblasNoTrans,
                     feature_H_ * feature_W_, feature_H_ * feature_W_, channels_,
                     float(1.0), this_bottom_data, this_top_diff, float(0), this_mask_diff);
  }
  if (debug)
  {
    printf("after bp to mask\n");
    debuger(mask_buffer_diff);
  }
  int nthreads = num_ * feature_H_ * feature_W_;
  // int nthreads = 10;
  const int mask_H_ = 2 * feature_H_ - 1;const int mask_W_ = 2 * feature_W_ - 1;
  const int half_mask_H_ = (mask_H_ - 1) / 2;const int half_mask_W_ = (mask_W_ - 1) / 2;
  if (debug)
  {
    printf("feature_H_: %d, feature_W_:%d, mask_H_:%d, mask_W_:%d, half_mask_H_:%d, half_mask_W_:%d \n",
      feature_H_, feature_W_, mask_H_, mask_W_, half_mask_H_, half_mask_W_);
    printf("before bp to bottom1\n");
    debuger(bottom1_diff);
  }
  switch (forward_type) {
  case PSA_TYPE_COLLECT:
    PSABackward_buffer_mask_collect_gpu<float><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS,0,stream>>>(
          nthreads, feature_H_, feature_W_, mask_H_, mask_W_, half_mask_H_, half_mask_W_,
          mask_buffer_diff, bottom1_diff);
    do {
    cudaError_t error = cudaPeekAtLastError();
    if (error != cudaSuccess){
      printf(cudaGetErrorString(error));}
    } while(0);
    break;
  case PSA_TYPE_DISTRIBUTE:
    PSABackward_buffer_mask_distribute_gpu<float><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS,0,stream>>>(
          nthreads, feature_H_, feature_W_, mask_H_, mask_W_, half_mask_H_, half_mask_W_,
          mask_buffer_diff, bottom1_diff);
    do {
    cudaError_t error = cudaPeekAtLastError();
    if (error != cudaSuccess){
      printf(cudaGetErrorString(error));}
    } while(0);
    break;
  default:
    printf("Unknown PSA type.");
  }
  if (debug)
  {
    printf("after bp to bottom1\n");
    debuger(bottom1_diff);
  } 

}