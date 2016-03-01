#ifndef CAFFE_ONE_HOT_LAYER_HPP_
#define CAFFE_ONE_HOT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/**
 * @brief Takes integer inputs, returns a tensor that is zero everywhere
 *        except at the index of the input where it is 1
 * 
 * example: if the input is [1, 3] the output is
 * [[0,1,0,0],[0,0,0,1]]
 */
template <typename Dtype>
class OneHotLayer : public Layer<Dtype> {
 public:
  explicit OneHotLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "OneHot"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int num_label_;        /// Maximum value that is allowed to come from bottom
};

}  // namespace caffe

#endif  // CAFFE_ONE_HOT_LAYER_HPP_
