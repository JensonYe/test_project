// Copyright 2014 BVLC and contributors.
// Gaussian neuron activation function layer.
// Adapted from ReLU layer code written by Yangqing Jia

#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void GaussianForward(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
	  Dtype x = in[index];
	  out[index] = exp(Dtype(-1)*x*x/Dtype(16));
    //Dtype exp2x = exp(2*in[index]);
   // out[index] = (exp2x - Dtype(1))/(exp2x + Dtype(1));
  }
}

template <typename Dtype>
Dtype GaussianLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  GaussianForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
  return Dtype(0);
}

template <typename Dtype>
__global__ void GaussianBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype inx = in_data[index];
	Dtype exp2x = exp(Dtype(-1)*inx*inx/Dtype(16));
	out_diff[index] = in_diff[index] * (exp2x*Dtype(-2)*inx/Dtype(16));
    //Dtype exp2x = exp(2*in_data[index]);
    //Dtype tanhx = (exp2x - Dtype(1))/(exp2x + Dtype(1));
    //out_diff[index] = in_diff[index] * (1 - tanhx*tanhx);
  }
}

template <typename Dtype>
void GaussianLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down) {
    const Dtype* bottom_data = (*bottom)[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
    const int count = (*bottom)[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    GaussianBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_CLASS(GaussianLayer);


}  // namespace caffe
