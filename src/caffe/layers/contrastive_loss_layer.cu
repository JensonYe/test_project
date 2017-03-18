#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
Dtype ContrastiveLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
 /* const int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),  // a
      bottom[1]->gpu_data(),  // b
      diff_.mutable_gpu_data());  // a_i-b_i
  caffe_gpu_powx(
      count,
      diff_.mutable_gpu_data(),  // a_i-b_i
      Dtype(2),
      diff_sq_.mutable_gpu_data());  // (a_i-b_i)^2
  caffe_gpu_gemv(
      CblasNoTrans,
      bottom[0]->num(),
      bottom[0]->channels(),
      Dtype(1.0),
      diff_sq_.gpu_data(),  // (a_i-b_i)^2
      summer_vec_.gpu_data(),
      Dtype(0.0),
      dist_sq_.mutable_gpu_data());  // \Sum (a_i-b_i)^2
  Dtype margin = this->margin_;
  Dtype loss(0.0);
  for (int i = 0; i < bottom[0]->num(); ++i) {
    if ((bottom[2]->cpu_data()[i] == bottom[3]->cpu_data()[i])) {  // similar pairs
      loss += dist_sq_.cpu_data()[i];
    } else {  // dissimilar pairs
      loss += std::max(margin-dist_sq_.cpu_data()[i], Dtype(0.0));
    }
  }
  loss = loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
  (*top)[0]->mutable_cpu_data()[0] = loss;
  //(*top)[0]->mutable_cpu_diff()[0] = 1;
  return loss;*/
  return Forward_cpu(bottom,top); 
}

template <typename Dtype>
__global__ void CLLForward(const int count, const int channels,
    const Dtype margin, const Dtype alpha,
    const Dtype* y, const Dtype* diff, const Dtype* dist_sq,
    Dtype *bottom_diff) {
  CUDA_KERNEL_LOOP(i, count) {
    int n = i / channels;  // the num index, to access y and dist_sq
    if (static_cast<int>(y[n])) {  // similar pairs
      bottom_diff[i] = alpha * diff[i];
    } else {  // dissimilar pairs
      if ((margin-dist_sq[n]) > 0.0) {
        bottom_diff[i] = -alpha * diff[i];
      } else {
        bottom_diff[i] = 0;
      }
    }
  }
}

template <typename Dtype>
void ContrastiveLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  Backward_cpu(top,propagate_down,bottom);
}

INSTANTIATE_CLASS(ContrastiveLossLayer);

}  // namespace caffe
