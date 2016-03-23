#ifndef CAFFE_COLLECTION_SPREAD_LAYER_HPP_
#define CAFFE_COLLECTION_SPREAD_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Does the reverse of the CollectionPoolingLayer by spreading its input to its output over collections.
 * So that means that an input blob of dimensions l x c x h x w for l collections  will be spread to a
 * blob with dimensions n x c x h x w where each entry is replicated the numeber of immeges per collection times.
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class CollectionSpreadLayer : public Layer<Dtype> {
 public:
  explicit CollectionSpreadLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CollectionSpread"; }
  virtual inline int ExactNumBottomBlobs() const { return 3; } // input, collection, data (data solely to know the number of channels)
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  int channels_;
  int height_, width_;
  Blob<Dtype> rand_idx_;
  Blob<int> max_idx_;
};

}  // namespace caffe

#endif  // CAFFE_COLLECTION_SPREAD_LAYER_HPP_
