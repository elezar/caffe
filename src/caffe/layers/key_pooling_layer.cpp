#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/key_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void KeyPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0))
      << "The number of keys must be equal to the size of the batch";

  pooling_layer_.LayerSetUp(bottom, top);

}

template <typename Dtype>
void KeyPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  int num_keys = bottom[1]->shape(0);
  has_keys_.clear();
  key_start_.clear();
  key_end_.clear();

  vector<Blob<Dtype>*> pooling_top;
  pooling_top.push_back(top[0]);

  pooling_layer_.Reshape(bottom, pooling_top);

  if (num_keys > 0) {
    const Dtype* keys = bottom[1]->cpu_data();

    Dtype current_key = keys[0];
    has_keys_.push_back(current_key);
    key_start_.push_back(0);

    int j = 0;
    for (int i = 1; i < num_keys; ++i) {
      if (keys[i] != current_key) {
        j++;
        key_end_.push_back(i);

        largest_key_set_ = std::max(key_end_[j-1] - key_start_[j-1], largest_key_set_);

        current_key = keys[i];
        has_keys_.push_back(current_key);
        key_start_.push_back(i);

      }
    }
    key_end_.push_back(num_keys);


    largest_key_set_ = std::max(key_end_[num_keys-1] - key_start_[num_keys-1], largest_key_set_);
  }

  CHECK_LE(has_keys_.size(), num_keys);

  // Resize the tops to match the keys.
  vector<int> required_shape(top[0]->shape());
  required_shape[0] = has_keys_.size();
  top[0]->Reshape(required_shape);

  if (top.size() > 1) {
    top[1]->Reshape(has_keys_.size(), 1, 1, 1);
  }

}

template <typename Dtype>
void KeyPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  vector<Blob<Dtype>*> pooling_top;
  pooling_top.push_back(top[0]);

  for (int i = 0; i < has_keys_.size(); ++i) {
    // Create a local copy of the blobs for the top and the bottom
    Blob<Dtype> key_bottom;
    Blob<Dtype> key_top;





    pooling_layer_.Forward(bottom, pooling_top);

  }




  if (top.size() > 1) {
    caffe_copy(has_keys_.size(), &has_keys_[0], top[1]->mutable_cpu_data());
  }

}

template <typename Dtype>
void KeyPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  pooling_layer_.Backward(bottom, propagate_down, top);
}

INSTANTIATE_CLASS(KeyPoolingLayer);

}  // namespace caffe
