#include <stdlib.h>
#include <stdio.h>
#include "PSAkernel.h"
#include <vector>


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



void PSAkernel_Forward_Launcher(const float *  bottom1, const int num_,
    const int feature_H_,const int feature_W_,
    float * mask_buffer_,const int forward_type,cudaStream_t stream) {
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

}

void PSAkernel_Backward_Launcher(
    float *  bottom1_diff, const int num_,
    const int feature_H_,const int feature_W_,
    float * mask_buffer_diff, const int forward_type,cudaStream_t stream) {


  int nthreads = num_ * feature_H_ * feature_W_;
  // int nthreads = 10;
  const int mask_H_ = 2 * feature_H_ - 1;const int mask_W_ = 2 * feature_W_ - 1;
  const int half_mask_H_ = (mask_H_ - 1) / 2;const int half_mask_W_ = (mask_W_ - 1) / 2;

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


}