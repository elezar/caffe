#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/collection_spread_layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void CollectionSpreadLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void CollectionSpreadLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_   = bottom[0]->height();
  width_    = bottom[0]->width();

  //DLOG(INFO) << "bottom[0] is " << bottom[0]->num() <<", "<< bottom[0]->channels() <<", "<< bottom[0]->height() <<", "<< bottom[0]->width();
  //DLOG(INFO) << "top[0] is " << top[0]->num() <<", "<< top[0]->channels() <<", "<< top[0]->height() <<", "<< top[0]->width();

  top[0]->Reshape(bottom[2]->num(), channels_, height_, width_);
  //DLOG(INFO) << "top[0] is " << top[0]->num() <<", "<< top[0]->channels() <<", "<< top[0]->height() <<", "<< top[0]->width();
}

template <typename Dtype>
void CollectionSpreadLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                                const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_input  = bottom[0]->cpu_data();
  const Dtype* bottom_splits = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  const int size = channels_ * height_ * width_;

  int n = 0;
  for( int l=0; l<bottom[1]->num(); l++) {
      for( ; n<bottom_splits[l]; n++) {
          caffe_copy(size, bottom_input, top_data);
          top_data += size;
      }
      bottom_input += size;
  }
}




template <typename Dtype>
void CollectionSpreadLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                 const vector<bool>& propagate_down, 
                                                 const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }

  const int size = channels_ * height_ * width_;
  const Dtype* top_diff      = top[0]->cpu_diff();
        Dtype* bottom_diff   = bottom[0]->mutable_cpu_diff();
  const Dtype* bottom_splits = bottom[1]->cpu_data();

  int n = 0;
  caffe_set(bottom[0]->count(), (Dtype)0.0, bottom_diff);
  for( int l=0; l<bottom[1]->num(); l++) {
      for( ; n<bottom_splits[l]; n++) {
          caffe_cpu_axpby(size, (Dtype)1.0, top_diff, (Dtype)1.0, bottom_diff);
          top_diff += size;
      }
      bottom_diff += size;
  }

}



#ifdef CPU_ONLY
STUB_GPU(CollectionSpreadLayer);
#endif

INSTANTIATE_CLASS(CollectionSpreadLayer);
REGISTER_LAYER_CLASS(CollectionSpread);

}  // namespace caffe
