#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/collection_spread_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
void CollectionSpreadLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_input  = bottom[0]->gpu_data();
  const Dtype* bottom_splits = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  //DLOG(INFO) << "bottom_input is" << bottom[0]->num() <<", "<< bottom[0]->channels() <<", "<< bottom[0]->height() <<", "<< bottom[0]->width();
  //DLOG(INFO) << "top is" << top[0]->num() <<", "<< top[0]->channels() <<", "<< top[0]->height() <<", "<< top[0]->width();

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
void CollectionSpreadLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                                const vector<bool>& propagate_down,
                                               const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }

  const int size = channels_ * height_ * width_;
  const Dtype* top_diff      = top[0]->gpu_diff();
        Dtype* bottom_diff   = bottom[0]->mutable_gpu_diff();
  const Dtype* bottom_splits = bottom[1]->cpu_data();

  caffe_gpu_set(bottom[0]->count(), (Dtype)0.0, bottom_diff);
  int n = 0;
  for( int l=0; l<bottom[1]->num(); l++) {
      for( ; n<bottom_splits[l]; n++) {
          caffe_gpu_axpy(size, (Dtype)1.0, top_diff, bottom_diff);
          top_diff += size;
      }
      bottom_diff += size;
  }

}



INSTANTIATE_LAYER_GPU_FUNCS(CollectionSpreadLayer);

}  // namespace caffe
