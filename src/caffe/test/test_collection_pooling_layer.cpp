#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/collection_pooling_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class CollectionPoolingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  CollectionPoolingLayerTest()
      : blob_bottom_data_(new Blob<Dtype>()),
        blob_bottom_collection_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()),
        blob_top_mask_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_data_->Reshape(6, 3, 2, 3);
    blob_bottom_collection_->Reshape(3, 1, 1, 1);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    filler.Fill(this->blob_bottom_collection_);
    blob_bottom_vec_.push_back(blob_bottom_collection_);
    blob_top_vec_.push_back(blob_top_);
    propagate_down_.push_back(true);
    propagate_down_.push_back(true);
  }
  virtual ~CollectionPoolingLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_collection_;
    delete blob_top_;
    delete blob_top_mask_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_collection_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_mask_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  vector<bool> propagate_down_;
  void CreateBottomSplits() {
    // define collection
    const int num_collection = 3;
    blob_bottom_collection_->Reshape(num_collection, 1, 1, 1);
    blob_bottom_collection_->mutable_cpu_data()[0] = 1;
    blob_bottom_collection_->mutable_cpu_data()[1] = 4;
    blob_bottom_collection_->mutable_cpu_data()[2] = 6;
  }
  void CreateBottomData() {
    CreateBottomSplits();
    // define image data
    const int num = 6;
    const int channels = 3;
    const int height = 2;
    const int width = 3;
    blob_bottom_data_->Reshape(num, channels, height, width);
    // consider the following 3 collections, 6 images of format 3x2
    // [1 0 1]  ||  [1 0 0]    [0 2 0]    [3 3 0]  ||  [1 0 2]    [ 2 1 5]      
    // [0 0 1]  ||  [0 0 1]    [0 2 0]    [3 0 0]  ||  [0 4 1]    [-1 2 1]
    // where the channel index is added to each pixel
    // in order to produce different arrays for different channels
    for (int i = 0; i < channels; ++i) {
      // image 0
      blob_bottom_data_->mutable_cpu_data()[(0 * channels  + i) * height * width  +  0] = 1 + i;
      blob_bottom_data_->mutable_cpu_data()[(0 * channels  + i) * height * width  +  1] = 0 + i;
      blob_bottom_data_->mutable_cpu_data()[(0 * channels  + i) * height * width  +  2] = 1 + i;
      blob_bottom_data_->mutable_cpu_data()[(0 * channels  + i) * height * width  +  3] = 0 + i;
      blob_bottom_data_->mutable_cpu_data()[(0 * channels  + i) * height * width  +  4] = 0 + i;
      blob_bottom_data_->mutable_cpu_data()[(0 * channels  + i) * height * width  +  5] = 1 + i;
      // image 1
      blob_bottom_data_->mutable_cpu_data()[(1 * channels  + i) * height * width  +  0] = 1 + i;
      blob_bottom_data_->mutable_cpu_data()[(1 * channels  + i) * height * width  +  1] = 0 + i;
      blob_bottom_data_->mutable_cpu_data()[(1 * channels  + i) * height * width  +  2] = 0 + i;
      blob_bottom_data_->mutable_cpu_data()[(1 * channels  + i) * height * width  +  3] = 0 + i;
      blob_bottom_data_->mutable_cpu_data()[(1 * channels  + i) * height * width  +  4] = 0 + i;
      blob_bottom_data_->mutable_cpu_data()[(1 * channels  + i) * height * width  +  5] = 1 + i;
      // image 2
      blob_bottom_data_->mutable_cpu_data()[(2 * channels  + i) * height * width  +  0] = 0 + i;
      blob_bottom_data_->mutable_cpu_data()[(2 * channels  + i) * height * width  +  1] = 2 + i;
      blob_bottom_data_->mutable_cpu_data()[(2 * channels  + i) * height * width  +  2] = 0 + i;
      blob_bottom_data_->mutable_cpu_data()[(2 * channels  + i) * height * width  +  3] = 0 + i;
      blob_bottom_data_->mutable_cpu_data()[(2 * channels  + i) * height * width  +  4] = 2 + i;
      blob_bottom_data_->mutable_cpu_data()[(2 * channels  + i) * height * width  +  5] = 0 + i;
      // image 3
      blob_bottom_data_->mutable_cpu_data()[(3 * channels  + i) * height * width  +  0] = 3 + i;
      blob_bottom_data_->mutable_cpu_data()[(3 * channels  + i) * height * width  +  1] = 3 + i;
      blob_bottom_data_->mutable_cpu_data()[(3 * channels  + i) * height * width  +  2] = 0 + i;
      blob_bottom_data_->mutable_cpu_data()[(3 * channels  + i) * height * width  +  3] = 3 + i;
      blob_bottom_data_->mutable_cpu_data()[(3 * channels  + i) * height * width  +  4] = 0 + i;
      blob_bottom_data_->mutable_cpu_data()[(3 * channels  + i) * height * width  +  5] = 0 + i;
      // image 4
      blob_bottom_data_->mutable_cpu_data()[(4 * channels  + i) * height * width  +  0] = 1 + i;
      blob_bottom_data_->mutable_cpu_data()[(4 * channels  + i) * height * width  +  1] = 0 + i;
      blob_bottom_data_->mutable_cpu_data()[(4 * channels  + i) * height * width  +  2] = 2 + i;
      blob_bottom_data_->mutable_cpu_data()[(4 * channels  + i) * height * width  +  3] = 0 + i;
      blob_bottom_data_->mutable_cpu_data()[(4 * channels  + i) * height * width  +  4] = 4 + i;
      blob_bottom_data_->mutable_cpu_data()[(4 * channels  + i) * height * width  +  5] = 1 + i;
      // image 5
      blob_bottom_data_->mutable_cpu_data()[(5 * channels  + i) * height * width  +  0] = 2 + i;
      blob_bottom_data_->mutable_cpu_data()[(5 * channels  + i) * height * width  +  1] = 1 + i;
      blob_bottom_data_->mutable_cpu_data()[(5 * channels  + i) * height * width  +  2] = 5 + i;
      blob_bottom_data_->mutable_cpu_data()[(5 * channels  + i) * height * width  +  3] =-1 + i;
      blob_bottom_data_->mutable_cpu_data()[(5 * channels  + i) * height * width  +  4] = 2 + i;
      blob_bottom_data_->mutable_cpu_data()[(5 * channels  + i) * height * width  +  5] = 1 + i;
    }
  }
  void CreateTopData() {
    const int num = 3;
    const int channels = 3;
    const int height = 2;
    const int width = 3;
    // define top, meaning, partial derivatives
    blob_top_->Reshape(num, channels, height, width);
    // consider the following 3 partial derivatives
    // [1 2 1]  ||  [3 2 6]  ||  [ 2 1 5]      
    // [1 0 1]  ||  [3 1 4]  ||  [-1 2 1]
    // where the channel index is added to each pixel
    // in order to produce different arrays for different channels
    for (int i = 0; i < channels; ++i) {
      // diff 0
      blob_top_->mutable_cpu_diff()[blob_top_->offset(0,i,0,0)] = 1 + i;
      blob_top_->mutable_cpu_diff()[blob_top_->offset(0,i,0,1)] = 2 + i;
      blob_top_->mutable_cpu_diff()[blob_top_->offset(0,i,0,2)] = 1 + i;
      blob_top_->mutable_cpu_diff()[blob_top_->offset(0,i,1,0)] = 1 + i;
      blob_top_->mutable_cpu_diff()[blob_top_->offset(0,i,1,1)] = 0 + i;
      blob_top_->mutable_cpu_diff()[blob_top_->offset(0,i,1,2)] = 1 + i;
      // diff 1
      blob_top_->mutable_cpu_diff()[blob_top_->offset(1,i,0,0)] = 3 + i;
      blob_top_->mutable_cpu_diff()[blob_top_->offset(1,i,0,1)] = 2 + i;
      blob_top_->mutable_cpu_diff()[blob_top_->offset(1,i,0,2)] = 6 + i;
      blob_top_->mutable_cpu_diff()[blob_top_->offset(1,i,1,0)] = 3 + i;
      blob_top_->mutable_cpu_diff()[blob_top_->offset(1,i,1,1)] = 1 + i;
      blob_top_->mutable_cpu_diff()[blob_top_->offset(1,i,1,2)] = 4 + i;
      // diff 2
      blob_top_->mutable_cpu_diff()[blob_top_->offset(2,i,0,0)] = 2 + i;
      blob_top_->mutable_cpu_diff()[blob_top_->offset(2,i,0,1)] = 1 + i;
      blob_top_->mutable_cpu_diff()[blob_top_->offset(2,i,0,2)] = 5 + i;
      blob_top_->mutable_cpu_diff()[blob_top_->offset(2,i,1,0)] =-1 + i;
      blob_top_->mutable_cpu_diff()[blob_top_->offset(2,i,1,1)] = 2 + i;
      blob_top_->mutable_cpu_diff()[blob_top_->offset(2,i,1,2)] = 1 + i;
    }
    // define top_mask, needed for MAX pooling tests
    blob_top_mask_->Reshape(num, channels, height, width);
    // channel 0,1
    // [0 0 0]  ||  [3 3 1]  ||  [5 5 5]      
    // [0 0 0]  ||  [3 2 1]  ||  [4 4 4]
    // channel 2
    // [0 0 0]  ||  [1 2 1]  ||  [5 4 5]      
    // [0 0 0]  ||  [3 1 3]  ||  [5 4 4]
    for (int i = 0; i < channels - 1; ++i) {
      // mask 0
      blob_top_mask_->mutable_cpu_data()[(0 * channels  + i) * height * width  +  0] = 0;
      blob_top_mask_->mutable_cpu_data()[(0 * channels  + i) * height * width  +  1] = 0;
      blob_top_mask_->mutable_cpu_data()[(0 * channels  + i) * height * width  +  2] = 0;
      blob_top_mask_->mutable_cpu_data()[(0 * channels  + i) * height * width  +  3] = 0;
      blob_top_mask_->mutable_cpu_data()[(0 * channels  + i) * height * width  +  4] = 0;
      blob_top_mask_->mutable_cpu_data()[(0 * channels  + i) * height * width  +  5] = 0;
      // mask 1
      blob_top_mask_->mutable_cpu_data()[(1 * channels  + i) * height * width  +  0] = 3;
      blob_top_mask_->mutable_cpu_data()[(1 * channels  + i) * height * width  +  1] = 3;
      blob_top_mask_->mutable_cpu_data()[(1 * channels  + i) * height * width  +  2] = 1;
      blob_top_mask_->mutable_cpu_data()[(1 * channels  + i) * height * width  +  3] = 3;
      blob_top_mask_->mutable_cpu_data()[(1 * channels  + i) * height * width  +  4] = 2;
      blob_top_mask_->mutable_cpu_data()[(1 * channels  + i) * height * width  +  5] = 1;      
      // mask 2
      blob_top_mask_->mutable_cpu_data()[(2 * channels  + i) * height * width  +  0] = 5;
      blob_top_mask_->mutable_cpu_data()[(2 * channels  + i) * height * width  +  1] = 5;
      blob_top_mask_->mutable_cpu_data()[(2 * channels  + i) * height * width  +  2] = 5;
      blob_top_mask_->mutable_cpu_data()[(2 * channels  + i) * height * width  +  3] = 4;
      blob_top_mask_->mutable_cpu_data()[(2 * channels  + i) * height * width  +  4] = 4;
      blob_top_mask_->mutable_cpu_data()[(2 * channels  + i) * height * width  +  5] = 4;      
    }
    // mask 0
    blob_top_mask_->mutable_cpu_data()[(0 * channels  + 2) * height * width  +  0] = 0;
    blob_top_mask_->mutable_cpu_data()[(0 * channels  + 2) * height * width  +  1] = 0;
    blob_top_mask_->mutable_cpu_data()[(0 * channels  + 2) * height * width  +  2] = 0;
    blob_top_mask_->mutable_cpu_data()[(0 * channels  + 2) * height * width  +  3] = 0;
    blob_top_mask_->mutable_cpu_data()[(0 * channels  + 2) * height * width  +  4] = 0;
    blob_top_mask_->mutable_cpu_data()[(0 * channels  + 2) * height * width  +  5] = 0;
    // mask 1
    blob_top_mask_->mutable_cpu_data()[(1 * channels  + 2) * height * width  +  0] = 1;
    blob_top_mask_->mutable_cpu_data()[(1 * channels  + 2) * height * width  +  1] = 2;
    blob_top_mask_->mutable_cpu_data()[(1 * channels  + 2) * height * width  +  2] = 1;
    blob_top_mask_->mutable_cpu_data()[(1 * channels  + 2) * height * width  +  3] = 3;
    blob_top_mask_->mutable_cpu_data()[(1 * channels  + 2) * height * width  +  4] = 1;
    blob_top_mask_->mutable_cpu_data()[(1 * channels  + 2) * height * width  +  5] = 3;      
    // mask 2
    blob_top_mask_->mutable_cpu_data()[(2 * channels  + 2) * height * width  +  0] = 5;
    blob_top_mask_->mutable_cpu_data()[(2 * channels  + 2) * height * width  +  1] = 4;
    blob_top_mask_->mutable_cpu_data()[(2 * channels  + 2) * height * width  +  2] = 5;
    blob_top_mask_->mutable_cpu_data()[(2 * channels  + 2) * height * width  +  3] = 5;
    blob_top_mask_->mutable_cpu_data()[(2 * channels  + 2) * height * width  +  4] = 4;
    blob_top_mask_->mutable_cpu_data()[(2 * channels  + 2) * height * width  +  5] = 4;      
  }
  void TestForwardMax() {
    const int num_collection = 3;
    const int channels = 3;
    const int height = 2;
    const int width = 3;
    LayerParameter layer_param;
    CollectionPoolingParameter* pooling_param = layer_param.mutable_collection_pooling_param();
    pooling_param->set_pool(CollectionPoolingParameter_PoolMethod_MAX);
    CollectionPoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num_collection);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), height);
    EXPECT_EQ(blob_top_->width(), width);
    if (blob_top_vec_.size() > 1) {
      EXPECT_EQ(blob_top_mask_->num(), num_collection);
      EXPECT_EQ(blob_top_mask_->channels(), channels);
      EXPECT_EQ(blob_top_mask_->height(), height);
      EXPECT_EQ(blob_top_mask_->width(), width);
    }
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // Expected output for channel = 0
    // [1 0 1]  ||  [3 3 0]  ||  [2 1 5]
    // [0 0 1]  ||  [3 2 1]  ||  [0 4 1]
    // adding channel index per pixel gives output for channel != 0
    for (int i = 0; i < channels; ++i) {
      // output 0
      EXPECT_EQ(blob_top_->mutable_cpu_data()[(0 * channels  + i) * height * width  +  0], 1 + i);
      EXPECT_EQ(blob_top_->mutable_cpu_data()[(0 * channels  + i) * height * width  +  1], 0 + i);
      EXPECT_EQ(blob_top_->mutable_cpu_data()[(0 * channels  + i) * height * width  +  2], 1 + i);
      EXPECT_EQ(blob_top_->mutable_cpu_data()[(0 * channels  + i) * height * width  +  3], 0 + i);
      EXPECT_EQ(blob_top_->mutable_cpu_data()[(0 * channels  + i) * height * width  +  4], 0 + i);
      EXPECT_EQ(blob_top_->mutable_cpu_data()[(0 * channels  + i) * height * width  +  5], 1 + i);
      // output 1
      EXPECT_EQ(blob_top_->mutable_cpu_data()[(1 * channels  + i) * height * width  +  0], 3 + i);
      EXPECT_EQ(blob_top_->mutable_cpu_data()[(1 * channels  + i) * height * width  +  1], 3 + i);
      EXPECT_EQ(blob_top_->mutable_cpu_data()[(1 * channels  + i) * height * width  +  2], 0 + i);
      EXPECT_EQ(blob_top_->mutable_cpu_data()[(1 * channels  + i) * height * width  +  3], 3 + i);
      EXPECT_EQ(blob_top_->mutable_cpu_data()[(1 * channels  + i) * height * width  +  4], 2 + i);
      EXPECT_EQ(blob_top_->mutable_cpu_data()[(1 * channels  + i) * height * width  +  5], 1 + i);
      // output 2
      EXPECT_EQ(blob_top_->mutable_cpu_data()[(2 * channels  + i) * height * width  +  0], 2 + i);
      EXPECT_EQ(blob_top_->mutable_cpu_data()[(2 * channels  + i) * height * width  +  1], 1 + i);
      EXPECT_EQ(blob_top_->mutable_cpu_data()[(2 * channels  + i) * height * width  +  2], 5 + i);
      EXPECT_EQ(blob_top_->mutable_cpu_data()[(2 * channels  + i) * height * width  +  3], 0 + i);
      EXPECT_EQ(blob_top_->mutable_cpu_data()[(2 * channels  + i) * height * width  +  4], 4 + i);
      EXPECT_EQ(blob_top_->mutable_cpu_data()[(2 * channels  + i) * height * width  +  5], 1 + i);
    }
    if (blob_top_vec_.size() > 1) {
      // test the mask
      // Expected output for every channel
      // [0 0 0]  ||  [3 3 1]  ||  [5 5 5]
      // [0 0 0]  ||  [3 2 1]  ||  [4 4 4]
      for (int i = 0; i < channels; ++i) {
        // output 0
        EXPECT_EQ(blob_top_mask_->mutable_cpu_data()[(0 * channels  + i) * height * width  +  0], 0);
        EXPECT_EQ(blob_top_mask_->mutable_cpu_data()[(0 * channels  + i) * height * width  +  1], 0);
        EXPECT_EQ(blob_top_mask_->mutable_cpu_data()[(0 * channels  + i) * height * width  +  2], 0);
        EXPECT_EQ(blob_top_mask_->mutable_cpu_data()[(0 * channels  + i) * height * width  +  3], 0);
        EXPECT_EQ(blob_top_mask_->mutable_cpu_data()[(0 * channels  + i) * height * width  +  4], 0);
        EXPECT_EQ(blob_top_mask_->mutable_cpu_data()[(0 * channels  + i) * height * width  +  5], 0);
        // output 1
        EXPECT_EQ(blob_top_mask_->mutable_cpu_data()[(1 * channels  + i) * height * width  +  0], 3);
        EXPECT_EQ(blob_top_mask_->mutable_cpu_data()[(1 * channels  + i) * height * width  +  1], 3);
        EXPECT_EQ(blob_top_mask_->mutable_cpu_data()[(1 * channels  + i) * height * width  +  2], 1);
        EXPECT_EQ(blob_top_mask_->mutable_cpu_data()[(1 * channels  + i) * height * width  +  3], 3);
        EXPECT_EQ(blob_top_mask_->mutable_cpu_data()[(1 * channels  + i) * height * width  +  4], 2);
        EXPECT_EQ(blob_top_mask_->mutable_cpu_data()[(1 * channels  + i) * height * width  +  5], 1);
        // output 2
        EXPECT_EQ(blob_top_mask_->mutable_cpu_data()[(2 * channels  + i) * height * width  +  0], 5);
        EXPECT_EQ(blob_top_mask_->mutable_cpu_data()[(2 * channels  + i) * height * width  +  1], 5);
        EXPECT_EQ(blob_top_mask_->mutable_cpu_data()[(2 * channels  + i) * height * width  +  2], 5);
        EXPECT_EQ(blob_top_mask_->mutable_cpu_data()[(2 * channels  + i) * height * width  +  3], 4);
        EXPECT_EQ(blob_top_mask_->mutable_cpu_data()[(2 * channels  + i) * height * width  +  4], 4);
        EXPECT_EQ(blob_top_mask_->mutable_cpu_data()[(2 * channels  + i) * height * width  +  5], 4);      
      }
    }
  }
  void TestForwardAve() {
    const int num_collection = 3;
    const int channels = 3;
    const int height = 2;
    const int width = 3;
    LayerParameter layer_param;
    CollectionPoolingParameter* pooling_param = layer_param.mutable_collection_pooling_param();
    pooling_param->set_pool(CollectionPoolingParameter_PoolMethod_AVE);
    CollectionPoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num_collection);
    EXPECT_EQ(blob_top_->channels(), channels);
    EXPECT_EQ(blob_top_->height(), height);
    EXPECT_EQ(blob_top_->width(), width);
    layer.Forward(blob_bottom_vec_, blob_top_vec_);
    // Expected output for channel = 0
    // [1 0 1]  ||  [3/3 5/3 0/3]  ||  [3/2 1/2 7/2]
    // [0 0 1]  ||  [3/3 2/3 1/3]  || [-1/2 6/2 2/2]
    // adding channel index per pixel gives output for channel != 0
    Dtype epsilon = 1e-6;
    for (int i = 0; i < channels; ++i) {
      // output 0
      EXPECT_NEAR(blob_top_->mutable_cpu_data()[(0 * channels  + i) * height * width  +  0], 1./1. + i, epsilon);
      EXPECT_NEAR(blob_top_->mutable_cpu_data()[(0 * channels  + i) * height * width  +  1], 0./1. + i, epsilon);
      EXPECT_NEAR(blob_top_->mutable_cpu_data()[(0 * channels  + i) * height * width  +  2], 1./1. + i, epsilon);
      EXPECT_NEAR(blob_top_->mutable_cpu_data()[(0 * channels  + i) * height * width  +  3], 0./1. + i, epsilon);
      EXPECT_NEAR(blob_top_->mutable_cpu_data()[(0 * channels  + i) * height * width  +  4], 0./1. + i, epsilon);
      EXPECT_NEAR(blob_top_->mutable_cpu_data()[(0 * channels  + i) * height * width  +  5], 1./1. + i, epsilon);
      // output 1
      EXPECT_NEAR(blob_top_->mutable_cpu_data()[(1 * channels  + i) * height * width  +  0], 4./3. + i, epsilon);
      EXPECT_NEAR(blob_top_->mutable_cpu_data()[(1 * channels  + i) * height * width  +  1], 5./3. + i, epsilon);
      EXPECT_NEAR(blob_top_->mutable_cpu_data()[(1 * channels  + i) * height * width  +  2], 0./3. + i, epsilon);
      EXPECT_NEAR(blob_top_->mutable_cpu_data()[(1 * channels  + i) * height * width  +  3], 3./3. + i, epsilon);
      EXPECT_NEAR(blob_top_->mutable_cpu_data()[(1 * channels  + i) * height * width  +  4], 2./3. + i, epsilon);
      EXPECT_NEAR(blob_top_->mutable_cpu_data()[(1 * channels  + i) * height * width  +  5], 1./3. + i, epsilon);
      // output 2
      EXPECT_NEAR(blob_top_->mutable_cpu_data()[(2 * channels  + i) * height * width  +  0], 3./2. + i, epsilon);
      EXPECT_NEAR(blob_top_->mutable_cpu_data()[(2 * channels  + i) * height * width  +  1], 1./2. + i, epsilon);
      EXPECT_NEAR(blob_top_->mutable_cpu_data()[(2 * channels  + i) * height * width  +  2], 7./2. + i, epsilon);
      EXPECT_NEAR(blob_top_->mutable_cpu_data()[(2 * channels  + i) * height * width  +  3],-1./2. + i, epsilon);
      EXPECT_NEAR(blob_top_->mutable_cpu_data()[(2 * channels  + i) * height * width  +  4], 6./2. + i, epsilon);
      EXPECT_NEAR(blob_top_->mutable_cpu_data()[(2 * channels  + i) * height * width  +  5], 2./2. + i, epsilon);
    }
  }
  void TestBackwardAve() {
    //const int num_collection = 3;
    const int channels = 3;
    const int height = 2;
    const int width = 3;
    LayerParameter layer_param;
    CollectionPoolingParameter* pooling_param = layer_param.mutable_collection_pooling_param();
    pooling_param->set_pool(CollectionPoolingParameter_PoolMethod_AVE);
    CollectionPoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    
    layer.Backward(blob_top_vec_, propagate_down_, blob_bottom_vec_);
    // Expected output for partial derivatives in cpu_diff, channel 0
    // [1/1 2/1 1/1]  ||  [3/3 2/3 6/3] [3/3 2/3 6/3] [3/3 2/3 6/3]  ||  [2/2 1/2 5/2]  [2/2 1/2 5/2]
    // [1/1 0/1 1/1]  ||  [3/3 1/3 4/3] [3/3 1/3 4/3] [3/3 1/3 4/3]  || [-1/2 2/2 1/2] [-1/2 2/2 1/2]
    // channel 1
    // [2/1 3/1 2/1]  ||  [4/3 3/3 7/3] [4/3 3/3 7/3] [4/3 3/3 7/3]  ||  [3/2 2/2 6/2] [3/2 2/2 6/2]
    // [2/1 1/1 2/1]  ||  [4/3 2/3 5/3] [4/3 2/3 5/3] [4/3 2/3 5/3]  ||  [0/2 3/2 2/2] [0/2 3/2 2/2]
    // channel 2
    // [3/1 4/1 3/1]  ||  [5/3 4/3 8/3] [5/3 4/3 8/3] [5/3 4/3 8/3]  ||  [4/2 3/2 7/2] [4/2 3/2 7/2]
    // [3/1 2/1 3/1]  ||  [5/3 3/3 6/3] [5/3 3/3 6/3] [5/3 3/3 6/3]  ||  [0/2 4/2 3/2] [1/2 4/2 3/2]
    ASSERT_EQ(blob_bottom_data_->num(), 6);
    ASSERT_EQ(blob_bottom_data_->channels(), channels);
    ASSERT_EQ(blob_bottom_data_->height(), height);
    ASSERT_EQ(blob_bottom_data_->width(), width);
    Dtype epsilon = 1e-6;
    
    for(int c = 0; c < channels; c++) {
      // output 0
      EXPECT_NEAR(blob_bottom_data_->mutable_cpu_diff()[blob_bottom_data_->offset(0,c,0,0)], (c+1.)/1., epsilon);
      EXPECT_NEAR(blob_bottom_data_->mutable_cpu_diff()[blob_bottom_data_->offset(0,c,0,1)], (c+2.)/1., epsilon);
      EXPECT_NEAR(blob_bottom_data_->mutable_cpu_diff()[blob_bottom_data_->offset(0,c,0,2)], (c+1.)/1., epsilon);
      EXPECT_NEAR(blob_bottom_data_->mutable_cpu_diff()[blob_bottom_data_->offset(0,c,1,0)], (c+1.)/1., epsilon);
      EXPECT_NEAR(blob_bottom_data_->mutable_cpu_diff()[blob_bottom_data_->offset(0,c,1,1)], (c+0.)/1., epsilon);
      EXPECT_NEAR(blob_bottom_data_->mutable_cpu_diff()[blob_bottom_data_->offset(0,c,1,2)], (c+1.)/1., epsilon);

      // output 1,2,3
      for(int n = 1; n < 4; n++) {
        EXPECT_NEAR(blob_bottom_data_->mutable_cpu_diff()[blob_bottom_data_->offset(n,c,0,0)], (c+3.)/3., epsilon);
        EXPECT_NEAR(blob_bottom_data_->mutable_cpu_diff()[blob_bottom_data_->offset(n,c,0,1)], (c+2.)/3., epsilon);
        EXPECT_NEAR(blob_bottom_data_->mutable_cpu_diff()[blob_bottom_data_->offset(n,c,0,2)], (c+6.)/3., epsilon);
        EXPECT_NEAR(blob_bottom_data_->mutable_cpu_diff()[blob_bottom_data_->offset(n,c,1,0)], (c+3.)/3., epsilon);
        EXPECT_NEAR(blob_bottom_data_->mutable_cpu_diff()[blob_bottom_data_->offset(n,c,1,1)], (c+1.)/3., epsilon);
        EXPECT_NEAR(blob_bottom_data_->mutable_cpu_diff()[blob_bottom_data_->offset(n,c,1,2)], (c+4.)/3., epsilon);
      } 
      // output 4,5
      for(int n = 4; n < 6; n++) {
        EXPECT_NEAR(blob_bottom_data_->mutable_cpu_diff()[blob_bottom_data_->offset(n,c,0,0)], (c+2.)/2., epsilon);
        EXPECT_NEAR(blob_bottom_data_->mutable_cpu_diff()[blob_bottom_data_->offset(n,c,0,1)], (c+1.)/2., epsilon);
        EXPECT_NEAR(blob_bottom_data_->mutable_cpu_diff()[blob_bottom_data_->offset(n,c,0,2)], (c+5.)/2., epsilon);
        EXPECT_NEAR(blob_bottom_data_->mutable_cpu_diff()[blob_bottom_data_->offset(n,c,1,0)], (c-1.)/2., epsilon);
        EXPECT_NEAR(blob_bottom_data_->mutable_cpu_diff()[blob_bottom_data_->offset(n,c,1,1)], (c+2.)/2., epsilon);
        EXPECT_NEAR(blob_bottom_data_->mutable_cpu_diff()[blob_bottom_data_->offset(n,c,1,2)], (c+1.)/2., epsilon);
      } 
    }
  }
  void TestBackwardMax() {
    LayerParameter layer_param;
    CollectionPoolingParameter* pooling_param = layer_param.mutable_collection_pooling_param();
    pooling_param->set_pool(CollectionPoolingParameter_PoolMethod_MAX);
    CollectionPoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, blob_top_vec_);
    layer.Backward(blob_top_vec_, propagate_down_, blob_bottom_vec_);
    CheckBackwardMax();
  }
  
  void CheckBackwardMax() {
    // Expected output for partial derivatives in cpu_diff, channel 0
    // [1 2 1]  ||  [0 0 6] [0 0 0] [3 2 0]  ||  [0 0 0] [2 1 5]
    // [1 0 1]  ||  [0 0 4] [0 1 0] [3 0 0]  || [-1 2 1] [0 0 0]
    // channel 1
    // [2 3 2]  ||  [0 0 7] [0 0 0] [4 3 0]  ||  [0 0 0] [3 2 6]
    // [2 1 2]  ||  [0 0 5] [0 2 0] [4 0 0]  ||  [0 3 2] [0 0 0]
    // channel 2
    // [3 4 3]  ||  [5 0 8] [0 4 0] [0 0 0]  ||  [0 3 0] [4 0 7]
    // [3 2 3]  ||  [0 3 0] [0 0 0] [5 0 6]  ||  [0 4 3] [1 0 0]
    const int num = 6;
    const int num_collection = 3;
    const int channels = 3;
    const int height = 2;
    const int width = 3;
    ASSERT_EQ(blob_bottom_vec_[1]->num(), num_collection);
    ASSERT_EQ(blob_bottom_data_->num(), num);
    ASSERT_EQ(blob_bottom_data_->channels(), channels);
    ASSERT_EQ(blob_bottom_data_->height(), height);
    ASSERT_EQ(blob_bottom_data_->width(), width);
    EXPECT_EQ(blob_bottom_data_->cpu_diff()[0], 1);
    for (int i = 0; i < channels; ++i) {
      // output 0
      EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[blob_bottom_data_->offset(0,i,0,0)], 1 + i);
      EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[blob_bottom_data_->offset(0,i,0,1)], 2 + i);
      EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[blob_bottom_data_->offset(0,i,0,2)], 1 + i);
      EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[blob_bottom_data_->offset(0,i,1,0)], 1 + i);
      EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[blob_bottom_data_->offset(0,i,1,1)], 0 + i);
      EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[blob_bottom_data_->offset(0,i,1,2)], 1 + i);
    }
    // channel 0, output 1
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(1 * channels + 0) * height * width + 0], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(1 * channels + 0) * height * width + 1], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(1 * channels + 0) * height * width + 2], 6);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(1 * channels + 0) * height * width + 3], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(1 * channels + 0) * height * width + 4], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(1 * channels + 0) * height * width + 5], 4);
    // channel 0, output 2
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(2 * channels + 0) * height * width + 0], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(2 * channels + 0) * height * width + 1], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(2 * channels + 0) * height * width + 2], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(2 * channels + 0) * height * width + 3], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(2 * channels + 0) * height * width + 4], 1);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(2 * channels + 0) * height * width + 5], 0);    
    // channel 0, output 3
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(3 * channels + 0) * height * width + 0], 3);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(3 * channels + 0) * height * width + 1], 2);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(3 * channels + 0) * height * width + 2], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(3 * channels + 0) * height * width + 3], 3);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(3 * channels + 0) * height * width + 4], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(3 * channels + 0) * height * width + 5], 0);
    // channel 0, output 4
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(4 * channels + 0) * height * width + 0], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(4 * channels + 0) * height * width + 1], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(4 * channels + 0) * height * width + 2], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(4 * channels + 0) * height * width + 3],-1);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(4 * channels + 0) * height * width + 4], 2);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(4 * channels + 0) * height * width + 5], 1);    
    // channel 0, output 5
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(5 * channels + 0) * height * width + 0], 2);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(5 * channels + 0) * height * width + 1], 1);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(5 * channels + 0) * height * width + 2], 5);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(5 * channels + 0) * height * width + 3], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(5 * channels + 0) * height * width + 4], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(5 * channels + 0) * height * width + 5], 0);
    // channel 1, output 1
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(1 * channels + 1) * height * width + 0], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(1 * channels + 1) * height * width + 1], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(1 * channels + 1) * height * width + 2], 7);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(1 * channels + 1) * height * width + 3], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(1 * channels + 1) * height * width + 4], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(1 * channels + 1) * height * width + 5], 5);    
    // channel 1, output 2
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(2 * channels + 1) * height * width + 0], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(2 * channels + 1) * height * width + 1], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(2 * channels + 1) * height * width + 2], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(2 * channels + 1) * height * width + 3], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(2 * channels + 1) * height * width + 4], 2);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(2 * channels + 1) * height * width + 5], 0);    
    // channel 1, output 3
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(3 * channels + 1) * height * width + 0], 4);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(3 * channels + 1) * height * width + 1], 3);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(3 * channels + 1) * height * width + 2], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(3 * channels + 1) * height * width + 3], 4);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(3 * channels + 1) * height * width + 4], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(3 * channels + 1) * height * width + 5], 0);    
    // channel 1, output 4
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(4 * channels + 1) * height * width + 0], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(4 * channels + 1) * height * width + 1], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(4 * channels + 1) * height * width + 2], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(4 * channels + 1) * height * width + 3], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(4 * channels + 1) * height * width + 4], 3);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(4 * channels + 1) * height * width + 5], 2);    
    // channel 1, output 5
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(5 * channels + 1) * height * width + 0], 3);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(5 * channels + 1) * height * width + 1], 2);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(5 * channels + 1) * height * width + 2], 6);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(5 * channels + 1) * height * width + 3], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(5 * channels + 1) * height * width + 4], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(5 * channels + 1) * height * width + 5], 0);
    // channel 2, output 1
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(1 * channels + 2) * height * width + 0], 5);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(1 * channels + 2) * height * width + 1], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(1 * channels + 2) * height * width + 2], 8);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(1 * channels + 2) * height * width + 3], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(1 * channels + 2) * height * width + 4], 3);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(1 * channels + 2) * height * width + 5], 0);    
    // channel 2, output 2
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(2 * channels + 2) * height * width + 0], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(2 * channels + 2) * height * width + 1], 4);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(2 * channels + 2) * height * width + 2], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(2 * channels + 2) * height * width + 3], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(2 * channels + 2) * height * width + 4], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(2 * channels + 2) * height * width + 5], 0);    
    // channel 2, output 3
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(3 * channels + 2) * height * width + 0], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(3 * channels + 2) * height * width + 1], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(3 * channels + 2) * height * width + 2], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(3 * channels + 2) * height * width + 3], 5);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(3 * channels + 2) * height * width + 4], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(3 * channels + 2) * height * width + 5], 6);    
    // channel 2, output 4
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(4 * channels + 2) * height * width + 0], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(4 * channels + 2) * height * width + 1], 3);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(4 * channels + 2) * height * width + 2], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(4 * channels + 2) * height * width + 3], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(4 * channels + 2) * height * width + 4], 4);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(4 * channels + 2) * height * width + 5], 3);    
    // channel 2, output 5
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(5 * channels + 2) * height * width + 0], 4);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(5 * channels + 2) * height * width + 1], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(5 * channels + 2) * height * width + 2], 7);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(5 * channels + 2) * height * width + 3], 1);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(5 * channels + 2) * height * width + 4], 0);
    EXPECT_EQ(blob_bottom_data_->mutable_cpu_diff()[(5 * channels + 2) * height * width + 5], 0);
  }
};

TYPED_TEST_CASE(CollectionPoolingLayerTest, TestDtypesAndDevices);

TYPED_TEST(CollectionPoolingLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CollectionPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_collection_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_data_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_data_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_data_->width());
}

TYPED_TEST(CollectionPoolingLayerTest, TestForwardMax) {
  this->CreateBottomData();
  this->TestForwardMax();
}

TYPED_TEST(CollectionPoolingLayerTest, TestForwardMaxTopMask) {
  this->CreateBottomData();
  this->blob_top_vec_.push_back(this->blob_top_mask_);
  this->TestForwardMax();  
}

TYPED_TEST(CollectionPoolingLayerTest, TestBackwardMax) {
  this->CreateTopData();
  this->blob_top_vec_.push_back(this->blob_top_mask_);
  this->TestBackwardMax();  
}

TYPED_TEST(CollectionPoolingLayerTest, TestForwardAve) {
  this->CreateBottomData();
  this->TestForwardAve();
}

TYPED_TEST(CollectionPoolingLayerTest, TestBackwardAve) {
  this->CreateBottomSplits();
  this->CreateTopData();
  this->TestBackwardAve();
}

// Here we test if doing a forwand then backwards pass with mask is the same as without
TYPED_TEST(CollectionPoolingLayerTest, TestMaxNoMask) {
  typedef typename TypeParam::Dtype Dtype;
  const int num_collections = 3;
  const int num_images= 6;
  const int num = 3;
  const int channels = 3;
  const int height = 2;
  const int width = 3;
  this->CreateBottomData();
  this->CreateTopData();
  //this->blob_top_->Reshape(num, channels, height, width);
  this->blob_top_mask_->Reshape(num, channels, height, width);
  this->blob_top_vec_.push_back(this->blob_top_mask_);
  ASSERT_EQ(this->blob_top_vec_.size(), 2);

  LayerParameter layer_param;
  CollectionPoolingParameter* pooling_param = layer_param.mutable_collection_pooling_param();
  pooling_param->set_pool(CollectionPoolingParameter_PoolMethod_MAX);
  CollectionPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);

  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Backward(this->blob_top_vec_, this->propagate_down_, this->blob_bottom_vec_);
  Blob<Dtype> correctblob, correcttopblob;
  correctblob.CopyFrom(*(this->blob_bottom_data_), true, true);
  correcttopblob.CopyFrom(*(this->blob_top_), true, true);
  

  this->CreateBottomData();
  //make sue that there is indeed no mask here
  this->blob_top_vec_.pop_back();
  this->CreateTopData();
  CollectionPoolingLayer<Dtype> layer2(layer_param);
  layer2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer2.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  layer2.Backward(this->blob_top_vec_, this->propagate_down_, this->blob_bottom_vec_);
  ASSERT_EQ(this->blob_bottom_data_->num(), num_images);
  ASSERT_EQ(correctblob.num(), num_images);
  ASSERT_EQ(this->blob_top_->num(), num_collections);
  ASSERT_EQ(correcttopblob.num(), num_collections);
  ASSERT_EQ(correctblob.channels(), this->blob_bottom_data_->channels());
  ASSERT_EQ(correctblob.height(), this->blob_bottom_data_->height());
  ASSERT_EQ(correctblob.width(), this->blob_bottom_data_->width());
  for(int i = 0; i < correctblob.count(); ++i)
    EXPECT_EQ(correctblob.cpu_diff()[i], this->blob_bottom_data_->cpu_diff()[i]);
}

}  // namespace caffe
