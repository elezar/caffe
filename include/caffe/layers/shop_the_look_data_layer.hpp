#ifndef CAFFE_SHOP_THE_LOOK_DATA_LAYER_HPP_
#define CAFFE_SHOP_THE_LOOK_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides shop the look and sku images through image files
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class ShopTheLookDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ShopTheLookDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~ShopTheLookDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ShopTheLookData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 4; }
  /**
   * Outputs 4 blobs
   * 
   * 1) batch_size shop the look images
   * 2) batch_size collections of sku images
   * 3) batch_size ints where each int tells us how many sku images
   *    were in all previous (and this) batch
   * 4) zero or one, where one means the image pair is a positive example
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    return Forward_cpu(bottom, top);
  }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);

  /// Shape of the collection images that are loaded
  vector<int> sku_shape_;
  /// Shape of the shop the look images
  vector<int> shop_the_look_shape_;
  /// Map from shop the look index to all skus that appear in this shop the look image
  std::map<int, std::set<int> > positive_examples_map_;
  /// Vector of filenames of shop the look images
  vector<std::string> shop_the_look_images_;
  /// for each shop the look image, we have a collection of skus
  vector<vector<std::string> > shop_the_look_skus_;
  /// convert from an sku string to and index
  map<std::string, int> sku_to_index_;
  /// for each sku, we have a collection of images
  vector<vector<std::string> > sku_collection_images_;
  vector<std::pair<int, int> > positive_examples_;
  int lines_id_;
  int num_shop_the_look_;
  int num_sku_;
};


}  // namespace caffe

#endif  // CAFFE_SHOP_THE_LOOK_DATA_LAYER_HPP_
