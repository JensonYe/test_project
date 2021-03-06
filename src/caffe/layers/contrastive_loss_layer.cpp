#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ContrastiveLossLayer<Dtype>::SetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  CHECK_EQ(bottom[2]->channels(), 1);
  CHECK_EQ(bottom[2]->height(), 1);
  CHECK_EQ(bottom[2]->width(), 1);
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  diff_sq_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
  dist_sq_.Reshape(bottom[0]->num(), 1, 1, 1);
  margin_ = 25;/**************************************************************************************************************************/
  // vector of ones used to sum along channels
  summer_vec_.Reshape(bottom[0]->channels(), 1, 1, 1);
  for (int i = 0; i < bottom[0]->channels(); ++i)
    summer_vec_.mutable_cpu_data()[i] = Dtype(1);
  (*top)[0]->Reshape(1, 1, 1, 1);
  //(*top)[0]->mutable_cpu_diff()[0] = 1;
}

template <typename Dtype>
Dtype ContrastiveLossLayer<Dtype>::Forward_cpu( const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  //Dtype const (*p)[64*128];
  int count = bottom[0]->count();
  caffe_sub(count,bottom[0]->cpu_data(),  // a
      bottom[1]->cpu_data(),  // b
      diff_.mutable_cpu_data());  // a_i-b_i
 // p = (Dtype const (*)[64*128])diff_.cpu_data();
  
  const int channels = bottom[0]->channels();
  Dtype margin = this->margin_;
  Dtype loss(0.0);
  const Dtype* label1 = bottom[2]->cpu_data();
  const Dtype* label2 = bottom[3]->cpu_data();
 const Dtype* pdist = dist_sq_.cpu_data();
  for (int i = 0; i < bottom[0]->num(); ++i) {
    dist_sq_.mutable_cpu_data()[i] = caffe_cpu_dot(channels,
        diff_.cpu_data() + (i*channels), diff_.cpu_data() + (i*channels));
    if ((label1[i] == label2[i])) {  // similar pairs
      loss += dist_sq_.cpu_data()[i];
    } else {  // dissimilar pairs
      loss += std::max(margin-dist_sq_.cpu_data()[i], Dtype(0.0));
    }
  }
  loss = loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
  (*top)[0]->mutable_cpu_data()[0] = loss;
  return 0;
}

template <typename Dtype>
void ContrastiveLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
 // Dtype const (*p)[64*128];
 // p = (Dtype const (*)[64*128])diff_.cpu_data();
  Dtype margin = this->margin_;
  const Dtype* label1 = (*bottom)[2]->cpu_data();
   const Dtype* label2 = (*bottom)[3]->cpu_data();
  for (int i = 0; i < 2; ++i) {
    if (propagate_down) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * 0.01 /            //****************************************************************************************************************************//
          static_cast<Dtype>((*bottom)[i]->num());
      int num = (*bottom)[i]->num();
      int channels = (*bottom)[i]->channels();
      for (int j = 0; j < num; ++j) {
        Dtype* bout = (*bottom)[i]->mutable_cpu_diff();
        if (label1[j] == label2[j]) {  // similar pairs
          caffe_cpu_axpby(
              channels,
              alpha,
              diff_.cpu_data() + (j*channels),
              Dtype(0.0),
              bout + (j*channels));
        } else {  // dissimilar pairs
          if ((margin-dist_sq_.cpu_data()[j]) > Dtype(0.0)) {
            caffe_cpu_axpby(
                channels,
                -alpha,
                diff_.cpu_data() + (j*channels),
                Dtype(0.0),
                bout + (j*channels));
          } else {
            caffe_set(channels, Dtype(0), bout + (j*channels));
          }
        }
      }
    }
  }
}

INSTANTIATE_CLASS(ContrastiveLossLayer);

}  // namespace caffe
