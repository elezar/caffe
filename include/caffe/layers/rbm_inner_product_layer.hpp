#ifndef CAFFE_RBM_INNER_PRODUCT_LAYER_HPP_
#define CAFFE_RBM_INNER_PRODUCT_LAYER_HPP_

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Fully connected RBM layer, with both hidden and visible biases
 *
 * The layer takes up to two bottom blobs.  The first blob just specifies the
 * input data during forward pass, update and forward sample.  During backward
 * sampling, the first bottom blob's diff gets the backward propigated
 * probabilites, and the blob's data gets samples from these probabilites.
 *
 * The top contains then one to three blobs describing the hidden state of the
 * RBM followed by an arbitrary number of blobs with different error values.
 *
 * The layer also contains two bias terms.  The hidden bias term is the same as
 * the bias term in the InnerProductLayer, and the visible bias term is new.
 * The backwards pass of this RBM is just the forward pass of the 
 * InnerProductLayer with the weight matrix transposed and the bias being the
 * visible bias
 *
 */
template <typename Dtype>
class RBMInnerProductLayer : public Layer<Dtype> {
 public:
  explicit RBMInnerProductLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "RBMInnerProduct"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline bool forward_is_update() const { return forward_is_update_;}
  virtual inline void forward_is_update(bool f) { forward_is_update_ = f;}

 protected:
  /**
   * Performing a forward pass either performs an unsupervised update if
   * forward_is_update is set to true in the prototxt (or set to true by the
   * setter or performs a normal forward pass if it is set to false.
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  virtual void update_diffs(const int k,
                            const vector<Blob<Dtype>*>& hidden_k,
                            const vector<Blob<Dtype>*>& visible_k);
  /// number of top blobs used for error reporting
  int num_error_;
  /// Forward pass is an update and not just a simple forward through connection
  bool forward_is_update_;
  int num_sample_steps_for_update_;
  bool visible_bias_term_;

  int visible_bias_index_;
  Blob<Dtype> bias_multiplier_;
  int batch_size_;
  int num_visible_;
  /// The size of the top and bottom at setup
  vector<int> setup_sizes_;

 private:
  void sample_h_given_v(const vector<Blob<Dtype>*>& bottom,
                        const vector<Blob<Dtype>*>& top);
  void sample_v_given_h(const vector<Blob<Dtype>*>& bottom,
                        const vector<Blob<Dtype>*>& top);
  void gibbs_hvh(const vector<Blob<Dtype>*>& bottom,
                 const vector<Blob<Dtype>*>& top);
  void update_diffs(Dtype alpha, const vector<Blob<Dtype>*>& hidden_k,
                    const vector<Blob<Dtype>*>& visible_k);
  /// The layer which is used to fist process the input.
  shared_ptr<Layer<Dtype> > connection_layer_;
  /// Layer used to squash the hidden units
  shared_ptr<Layer<Dtype> > hidden_activation_layer_;
  /// Layer used to squash the visible units
  shared_ptr<Layer<Dtype> > visible_activation_layer_;
  /// A layer used to sample the hidden activations.
  shared_ptr<Layer<Dtype> > hidden_sampling_layer_;
  /// A layer used to sample the visible activations.
  shared_ptr<Layer<Dtype> > visible_sampling_layer_;

  /// vectors that store the data after connection_layer_ has done forward
  /// or backward pass, but before the activation_layer_ has been called
  shared_ptr<Blob<Dtype> > pre_activation_h1_blob_;
  shared_ptr<Blob<Dtype> > pre_activation_v1_blob_;
  vector<Blob<Dtype>*> pre_activation_h1_vec_;
  vector<Blob<Dtype>*> pre_activation_v1_vec_;

  /// vectors that store the data after activation_layer_ has done forward
  /// or backward pass, but before the sampling_layer_ has been called
  shared_ptr<Blob<Dtype> > post_activation_h1_blob_;
  shared_ptr<Blob<Dtype> > post_activation_v1_blob_;
  vector<Blob<Dtype>*> post_activation_h1_vec_;
  vector<Blob<Dtype>*> post_activation_v1_vec_;

  shared_ptr<Blob<Dtype> > sample_h1_blob_;
  vector<Blob<Dtype>*> sample_h1_vec_;
  vector<Blob<Dtype>*> sample_v1_vec_;
  
  /// copy bottom data to this so sampling doesn't change actual bottom data
  shared_ptr<Blob<Dtype> > bottom_copy_;
};

}  // namespace caffe

#endif  // CAFFE_RBM_INNER_PRODUCT_LAYER_HPP_
