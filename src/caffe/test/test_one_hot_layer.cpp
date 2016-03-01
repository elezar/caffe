#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/one_hot_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class OneHotLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  OneHotLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()),
        num_label_(4) {
    vector<int> bottom_shape(2);
    bottom_shape[0] = 2;
    bottom_shape[1] = 3;
    blob_bottom_->Reshape(bottom_shape);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    OneHotParameter* one_hot_param =
      layer_param_.mutable_one_hot_param();
    one_hot_param->set_num_label(num_label_);
  }
  virtual ~OneHotLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  LayerParameter layer_param_;
  const int num_label_;
};

TYPED_TEST_CASE(OneHotLayerTest, TestDtypesAndDevices);

TYPED_TEST(OneHotLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  shared_ptr<OneHotLayer<Dtype> > layer(
      new OneHotLayer<Dtype>(this->layer_param_));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  ASSERT_EQ(this->blob_top_->shape().size(), 3);
  EXPECT_EQ(this->blob_top_->shape()[0], 2);
  EXPECT_EQ(this->blob_top_->shape()[1], 3);
  EXPECT_EQ(this->blob_top_->shape()[2], 4);
}

TYPED_TEST(OneHotLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  shared_ptr<OneHotLayer<Dtype> > layer(
      new OneHotLayer<Dtype>(this->layer_param_));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype* bottom_data = this->blob_bottom_->mutable_cpu_data();
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    bottom_data[i] = i % this->num_label_;
  }
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  const Dtype* data = this->blob_top_->cpu_data();

  ASSERT_EQ(this->blob_top_->shape().size(), 3);
  EXPECT_EQ(this->blob_top_->shape()[0], 2);
  EXPECT_EQ(this->blob_top_->shape()[1], 3);
  EXPECT_EQ(this->blob_top_->shape()[2], this->num_label_);

  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    for (int j = 0; j < this->num_label_; j++) {
      if (j == bottom_data[i]) {
        EXPECT_FLOAT_EQ(data[this->num_label_ * i + j], 1);
      } else {
        EXPECT_EQ(data[this->num_label_ * i + j], 0);
      }
    }
  }
}

}  // namespace caffe
