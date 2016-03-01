#include <vector>
#include "caffe/layers/one_hot_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void OneHotKernel(const int n, const int num_label,
                             const Dtype* bottom_data, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, n) {
    int idx = bottom_data[index];
    top_data[index * num_label + idx] = 1.;
  }
}

template <typename Dtype>
void OneHotLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = bottom[0]->count();

  // set the whole vector to zero
  caffe_gpu_set(top[0]->count(), Dtype(0.), top_data);
  // NOLINT_NEXT_LINE(whitespace/operators)
  OneHotKernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    count, num_label_, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void OneHotLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  CHECK(!propagate_down[0]) <<"errors can not be propagated for this layer";
}

INSTANTIATE_LAYER_GPU_FUNCS(OneHotLayer);

}  // namespace caffe
