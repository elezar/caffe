#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/collection_pooling_layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void CollectionPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                               const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void CollectionPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(bottom[1]->num(), channels_, height_, width_);
  if (top.size() > 1) {
    top[1]->ReshapeLike(*(top[0]));
  }
  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.collection_pooling_param().pool() ==
      CollectionPoolingParameter_PoolMethod_MAX && top.size() == 1) {
    max_idx_.Reshape(bottom[1]->num(), channels_, height_, width_);
  }
  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.collection_pooling_param().pool() ==
      CollectionPoolingParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(bottom[1]->num(), channels_, height_, width_);
  }
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void CollectionPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                                const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_splits = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;  // suppress warnings about uninitalized variables
  Dtype* top_mask = NULL;

  const int size = channels_ * height_ * width_;
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.collection_pooling_param().pool()) {
  case CollectionPoolingParameter_PoolMethod_MAX:
    // Initialize
    if (use_top_mask) {
      top_mask = top[1]->mutable_cpu_data();
      caffe_set(top_count, Dtype(-1), top_mask);
    } else {
      mask = max_idx_.mutable_cpu_data();
      caffe_set(top_count, -1, mask);
    }

    //set the top data to very negative values so we can do the max on them
    caffe_set(top_count, Dtype(-FLT_MAX), top_data);
    for(int l = 0; l < bottom[1]->num(); ++l) {
      for(int n = (l == 0 ? 0 : bottom_splits[l-1]); n < bottom_splits[l]; ++n) {
        for(int index = 0; index < size; ++index) {
          if(bottom_data[index] > top_data[index]) {
            top_data[index] = bottom_data[index];
            if (use_top_mask) {
              top_mask[index] = static_cast<Dtype>(n);
            } else {
              mask[index] = n;
            }
          }
        }
        bottom_data += bottom[0]->offset(1);
      }
      top_data += top[0]->offset(1);
      if (use_top_mask) {
        top_mask += top[0]->offset(1);
      } else {
        mask += top[0]->offset(1);
      }
    }
    break;
  case CollectionPoolingParameter_PoolMethod_AVE:
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = 0;
    }
    for (int l = 0; l < bottom[1]->num(); ++l) {
      for(int n = (l == 0 ? 0 : bottom_splits[l-1]); n < bottom_splits[l]; ++n) {
        for(int index = 0; index < size; ++index) {
          top_data[index] += bottom_data[index];
        }
        // compute offset
        bottom_data += bottom[0]->offset(1);
      }
      //divide by the number of images in the collection
      int collection_size = bottom_splits[l] - (l == 0 ? 0 : bottom_splits[l-1]);
      for(int i = 0; i < top[0]->offset(1); i++)
        top_data[i] /= collection_size;
      top_data += top[0]->offset(1);
    }
    break;
  case CollectionPoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}
//TODO: When I do the backward pass, will I even have the collection data to use,
//or is the bottom vector filled with garbage until we use it?
//It seems like I am safe on this one, when I take a good look at net.cpp
template <typename Dtype>
void CollectionPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                 const vector<bool>& propagate_down, 
                                                 const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const int size = channels_ * height_ * width_;
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* bottom_splits = bottom[1]->cpu_data();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;  // suppress warnings about uninitialized variables
  const Dtype* top_mask = NULL;
  const int bottom_step = bottom[0]->offset(1); //TODO isn't this the same size as size (defined above)?
  switch (this->layer_param_.collection_pooling_param().pool()) {
  case CollectionPoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->cpu_data();
    } else {
      mask = max_idx_.cpu_data();
    }
    // The main loop
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int index = 0; index < size; ++index) {
        const int bottom_offset = use_top_mask ? top_mask[index] : mask[index];
        bottom_diff[bottom_step * bottom_offset + index] += top_diff[index];
      }
      top_diff += top[0]->offset(1);
      if (use_top_mask) {
        top_mask += top[0]->offset(1);
      } else {
        mask += top[0]->offset(1);
      }
    }
    break;
  case CollectionPoolingParameter_PoolMethod_AVE:
    // The main loop
    for (int l = 0; l < bottom[1]->num(); ++l) {
      int collection_size = bottom_splits[l] - (l == 0 ? 0 : bottom_splits[l-1]);
      for(int n = (l == 0 ? 0 : bottom_splits[l-1]); n < bottom_splits[l]; ++n) {
        for (int index = 0; index < size; ++index) {
          // TODO can save divisions by copying the stuff for following n's
          bottom_diff[index] = top_diff[index] / collection_size;
        }
        // offset
        bottom_diff += bottom[0]->offset(1);
        //offset_b += size;
      }
      top_diff += top[0]->offset(1);
    }
    break;
  case CollectionPoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}


#ifdef CPU_ONLY
STUB_GPU(CollectionPoolingLayer);
#endif

INSTANTIATE_CLASS(CollectionPoolingLayer);
REGISTER_LAYER_CLASS(CollectionPooling);

}  // namespace caffe
