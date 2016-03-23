#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/collection_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MaxPoolForward(const int size, const Dtype* bottom_data, const Dtype* bottom_splits,
    const int num, const int num_collections, const int channels, const int height,
    const int width, Dtype* top_data, int* mask, Dtype* top_mask) {
    const int nthreads = size * num_collections;
  CUDA_KERNEL_LOOP(index, nthreads) {
    Dtype maxval = -FLT_MAX;
    int maxidx = -1; //keep maxidx management in cache
    //calculate which collection am I currently on?
    int l = index / size;
    bottom_data += index % size; //get us to the right pixle in the wrong image
    top_data += index % size;
    for(int n = (l == 0 ? 0 : bottom_splits[l-1]); n < bottom_splits[l]; ++n) {
      //n * size : get us to the correct image
      //index % size : correct position in the image
      if(bottom_data[n * size] > maxval) {
        maxidx = n;
        maxval = bottom_data[n * size];
      }
    }
    top_data[l * size] = maxval;
    if (mask) {
      mask += index % size;
      mask[l * size] = maxidx;
    } else {
      top_mask += index % size;
      top_mask[l * size] = maxidx;
    }
  }
}

template <typename Dtype>
__global__ void AvePoolForward(const int size, const Dtype* bottom_data, const Dtype* bottom_splits,
    const int num, const int num_collections, const int channels, const int height,
    const int width, Dtype* top_data) {
    const int nthreads = size * num_collections;
  CUDA_KERNEL_LOOP(index, nthreads) {
    //calculate which collection am I currently on?
    int l = index / size;
    bottom_data += index % size; //get us to the right pixle in the wrong image
    top_data += index % size;
    Dtype aveval = 0;
    for(int n = (l == 0 ? 0 : bottom_splits[l-1]); n < bottom_splits[l]; ++n) {
      aveval += bottom_data[n * size];
    }
    const int pool_size = bottom_splits[l] - (l == 0 ? 0 : bottom_splits[l-1]);
    top_data[l * size] = aveval / pool_size;
  }
}

template <typename Dtype>
__global__ void StoPoolForwardTrain(const int nthreads,
    const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, Dtype* rand_idx, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h;
    int hend = min(hstart + kernel_h, height);
    int wstart = pw * stride_w;
    int wend = min(wstart + kernel_w, width);
    Dtype cumsum = 0.;
    bottom_data += (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_data[h * width + w];
      }
    }
    float thres = rand_idx[index] * cumsum;
    // Second pass: get value, and set index.
    cumsum = 0;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_data[h * width + w];
        if (cumsum >= thres) {
          rand_idx[index] = ((n * channels + c) * height + h) * width + w;
          top_data[index] = bottom_data[h * width + w];
          return;
        }
      }
    }
  }
}


template <typename Dtype>
__global__ void StoPoolForwardTest(const int nthreads,
    const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h;
    int hend = min(hstart + kernel_h, height);
    int wstart = pw * stride_w;
    int wend = min(wstart + kernel_w, width);
    // We set cumsum to be 0 to avoid divide-by-zero problems
    Dtype cumsum = FLT_MIN;
    Dtype cumvalues = 0.;
    bottom_data += (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_data[h * width + w];
        cumvalues += bottom_data[h * width + w] * bottom_data[h * width + w];
      }
    }
    top_data[index] = cumvalues / cumsum;
  }
}


template <typename Dtype>
void CollectionPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                                const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_splits = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  
  //the number of things we will have to pool together
  const int size = height_ * width_ * channels_;
  
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;
  Dtype* top_mask = NULL;
  switch (this->layer_param_.collection_pooling_param().pool()) {
  case CollectionPoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->mutable_gpu_data();
    } else {
      mask = max_idx_.mutable_gpu_data();
    }
    MaxPoolForward<Dtype><<<CAFFE_GET_BLOCKS(size * bottom[1]->num()), CAFFE_CUDA_NUM_THREADS>>>(
        size, bottom_data, bottom_splits, bottom[0]->num(), bottom[1]->num(), channels_,
        height_, width_, top_data, mask, top_mask);
    break;
  case CollectionPoolingParameter_PoolMethod_AVE:
    AvePoolForward<Dtype><<<CAFFE_GET_BLOCKS(size * bottom[1]->num()), CAFFE_CUDA_NUM_THREADS>>>(
        size, bottom_data, bottom_splits, bottom[0]->num(), bottom[1]->num(), channels_,
        height_, width_, top_data);
    break;
  case CollectionPoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED; 
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void MaxPoolBackward(const int size, const Dtype* top_diff, 
    const int* mask, const Dtype* top_mask, const Dtype* collection_splits,
    const int num, const int num_collections, const int channels, const int height,
    const int width, Dtype* bottom_diff) {
    const int nthreads = size * num_collections;
  CUDA_KERNEL_LOOP(index, nthreads) {
    //calculate which collection am I currently on?
    int l = index / size;
    bottom_diff += index % size; //get us to the right pixle in the wrong image
    top_diff += index % size;
    //set the bottom diff vector to zero
    for(int n = (l == 0 ? 0 : collection_splits[l-1]); n < collection_splits[l]; ++n) {
      bottom_diff[n * size] = 0;
    }
    if (mask) {
      mask += index % size;
      bottom_diff[mask[l * size] * size] = top_diff[l * size];
    } else {
      top_mask += index % size;
      bottom_diff[static_cast<int>(top_mask[l * size] * size)] = top_diff[l * size];
    }
  }
}

template <typename Dtype>
__global__ void AvePoolBackward(const int size, const Dtype* top_diff, const Dtype* collection_splits,
    const int num, const int num_collections, const int channels, const int height,
    const int width, Dtype* bottom_diff) {
    const int nthreads = size * num_collections;
  CUDA_KERNEL_LOOP(index, nthreads) {
    //calculate which collection am I currently on?
    int l = index / size;
    const int pool_size = collection_splits[l] - (l == 0 ? 0 : collection_splits[l-1]);
    bottom_diff += index % size; //get us to the right pixle in the wrong image
    top_diff += index % size;
    for(int n = (l == 0 ? 0 : collection_splits[l-1]); n < collection_splits[l]; ++n) {
      bottom_diff[n * size] = top_diff[l * size] / pool_size;
    }
  }
}


template <typename Dtype>
__global__ void StoPoolBackward(const int nthreads,
    const Dtype* rand_idx, const Dtype* top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    int phend = min(h / stride_h + 1, pooled_height);
    int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    int pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    rand_idx += (n * channels + c) * pooled_height * pooled_width;
    top_diff += (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        gradient += top_diff[ph * pooled_width + pw] *
            (index == static_cast<int>(rand_idx[ph * pooled_width + pw]));
      }
    }
    bottom_diff[index] = gradient;
  }
}


template <typename Dtype>
void CollectionPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                                 const vector<bool>& propagate_down, 
                                                 const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const int size = height_ * width_ * channels_;
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;
  const Dtype* top_mask = NULL;
  switch (this->layer_param_.collection_pooling_param().pool()) {
  case CollectionPoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->gpu_data();
    } else {
      mask = max_idx_.gpu_data();
    }
     MaxPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(size * bottom[1]->num()), CAFFE_CUDA_NUM_THREADS>>>(
         size, top_diff, mask, top_mask, bottom[1]->gpu_data(), top[0]->num(), bottom[1]->num(), channels_,
         height_, width_, bottom_diff);
    break;
  case CollectionPoolingParameter_PoolMethod_AVE:
    AvePoolBackward<Dtype><<<CAFFE_GET_BLOCKS(size * bottom[1]->num()), CAFFE_CUDA_NUM_THREADS>>>(
        size, top_diff, bottom[1]->gpu_data(), top[0]->num(), bottom[1]->num(), channels_,
        height_, width_, bottom_diff);
    break;
  case CollectionPoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(CollectionPoolingLayer);

}  // namespace caffe
