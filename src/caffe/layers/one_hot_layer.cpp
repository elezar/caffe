#include <cmath>
#include <vector>
#include "caffe/layers/one_hot_layer.hpp"

namespace caffe {

template <typename Dtype>
void OneHotLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(this->layer_param_.has_one_hot_param())
      << "one_hot_param must be defined";
  CHECK(this->layer_param_.one_hot_param().has_num_label())
      << "one_hot_param.num_label must be defined";
  num_label_ = this->layer_param_.one_hot_param().num_label();
  CHECK_GT(num_label_, 0) << "one_hot_param.num_label must be > 0";
}

template <typename Dtype>
void OneHotLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // output blob is the same shape
  vector<int> output_shape(bottom[0]->shape());

  // except that a new dimension is added with the number of labels
  output_shape.push_back(num_label_);

  top[0]->Reshape(output_shape);
}

template <typename Dtype>
void OneHotLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  // set the whole vector to zero
  caffe_set(top[0]->count(), Dtype(0), top_data);

  // now set the stuff to 1 in the right index
  for (int i = 0; i < bottom[0]->count(); ++i) {
    CHECK_EQ(bottom_data[i], round(bottom_data[i]))
        << "the bottom data must be integers";
    CHECK_GE(bottom_data[i], 0) << "the bottom data must be non negative";
    int idx = bottom_data[i];
    CHECK_LT(idx, num_label_)
        << "the bottom data must be less than the total number of labels";
    top_data[i * num_label_ + idx] = 1;
  }
}

template <typename Dtype>
void OneHotLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  CHECK(!propagate_down[0]) <<"errors can not be propagated for this layer";
}

#ifdef CPU_ONLY
STUB_GPU(OneHotLayer);
#endif

INSTANTIATE_CLASS(OneHotLayer);
REGISTER_LAYER_CLASS(OneHot);

}  // namespace caffe
