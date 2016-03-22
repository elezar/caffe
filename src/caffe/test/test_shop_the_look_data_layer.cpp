#ifdef USE_OPENCV
#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/shop_the_look_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class ShopTheLookDataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  ShopTheLookDataLayerTest()
      : seed_(1701),
        blob_top_shop_the_look_(new Blob<Dtype>()),
        blob_top_collections_(new Blob<Dtype>()),
        blob_top_splits_(new Blob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    blob_top_vec_.push_back(blob_top_shop_the_look_);
    blob_top_vec_.push_back(blob_top_collections_);
    blob_top_vec_.push_back(blob_top_splits_);
    blob_top_vec_.push_back(blob_top_label_);
    Caffe::set_random_seed(seed_);
    // Create test input files.
    MakeTempFilename(&shop_the_look_filename_);
    std::ofstream outfile(shop_the_look_filename_.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << shop_the_look_filename_;
    outfile << EXAMPLES_SOURCE_DIR "images/cat.jpg sku1 sku2";
    outfile.close();
    
    MakeTempFilename(&collection_filename_);
    std::ofstream outfile2(collection_filename_.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << collection_filename_;
    outfile2 << "sku1 " EXAMPLES_SOURCE_DIR "images/fish-bike.jpg " EXAMPLES_SOURCE_DIR "images/fish-bike.jpg\n";
    outfile2 << "sku2 " EXAMPLES_SOURCE_DIR "images/fish-bike.jpg\n";
    outfile2.close();
  }
  
  virtual void InitLayerFromProtoString(const string& proto) {
    LayerParameter layer_param;
    CHECK(google::protobuf::TextFormat::ParseFromString(proto, &layer_param));
    layer_.reset(new ShopTheLookDataLayer<Dtype>(layer_param));
  }

  virtual ~ShopTheLookDataLayerTest() {
    delete blob_top_shop_the_look_;
    delete blob_top_collections_;
    delete blob_top_splits_;
    delete blob_top_label_;
  }

  int seed_;
  string shop_the_look_filename_;
  string collection_filename_;
  Blob<Dtype>* const blob_top_shop_the_look_;
  Blob<Dtype>* const blob_top_collections_;
  Blob<Dtype>* const blob_top_splits_;
  Blob<Dtype>* const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  shared_ptr<ShopTheLookDataLayer<Dtype> > layer_;
};

TYPED_TEST_CASE(ShopTheLookDataLayerTest, TestDtypesAndDevices);

TYPED_TEST(ShopTheLookDataLayerTest, TestSetup) {
  string proto =
    "name: 'shop_the_look_data_layer' "
    "type: 'ShopTheLookData' "
    "shop_the_look_data_param { "
    "  shop_the_look_delimiter: ' ' "
    "  collection_delimiter: ' ' "
    "  shop_the_look_source: ";
    proto += "'" + this->shop_the_look_filename_ + "'";
    proto += "  collection_source: '" + this->collection_filename_ + "'";
    proto +=
    "  batch_size: 5 "
    "  positive_probability: 1 "
    "} ";
  this->InitLayerFromProtoString(proto);
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_shop_the_look_->num(), 5);
  EXPECT_EQ(this->blob_top_shop_the_look_->channels(), 3);
  EXPECT_EQ(this->blob_top_shop_the_look_->height(), 360);
  EXPECT_EQ(this->blob_top_shop_the_look_->width(), 480);
  EXPECT_EQ(this->blob_top_collections_->num(), 5);
  EXPECT_EQ(this->blob_top_collections_->channels(), 3);
  EXPECT_EQ(this->blob_top_collections_->height(), 323);
  EXPECT_EQ(this->blob_top_collections_->width(), 481);
  EXPECT_EQ(this->blob_top_splits_->num(), 5);
  EXPECT_EQ(this->blob_top_splits_->channels(), 1);
  EXPECT_EQ(this->blob_top_splits_->height(), 1);
  EXPECT_EQ(this->blob_top_splits_->width(), 1);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);
}

TYPED_TEST(ShopTheLookDataLayerTest, TestRead) {
  string proto =
    "name: 'shop_the_look_data_layer' "
    "type: 'ShopTheLookData' "
    "shop_the_look_data_param { "
    "  shop_the_look_delimiter: ' ' "
    "  collection_delimiter: ' ' "
    "  shop_the_look_source: ";
    proto += "'" + this->shop_the_look_filename_ + "'";
    proto += "  collection_source: '" + this->collection_filename_ + "'";
    proto +=
    "  batch_size: 5 "
    "  positive_probability: 1 "
    "} ";
  this->InitLayerFromProtoString(proto);
  this->layer_->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  
  // Go through the data twice
  for (int iter = 0; iter < 1; ++iter) {
    int split_sum = 0;
    this->layer_->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(this->blob_top_splits_->cpu_data()[i], (split_sum += ((iter + i) % 2 == 0 ? 2 : 1)));
    }
    for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(1, this->blob_top_label_->cpu_data()[i]);
    }
    EXPECT_EQ(this->blob_top_shop_the_look_->num(), 5);
    EXPECT_EQ(this->blob_top_shop_the_look_->channels(), 3);
    EXPECT_EQ(this->blob_top_shop_the_look_->height(), 360);
    EXPECT_EQ(this->blob_top_shop_the_look_->width(), 480);
    
    EXPECT_EQ(this->blob_top_collections_->num(), this->blob_top_splits_->cpu_data()[4]);
    EXPECT_EQ(this->blob_top_collections_->channels(), 3);
    EXPECT_EQ(this->blob_top_collections_->height(), 323);
    EXPECT_EQ(this->blob_top_collections_->width(), 481);
  }
}

}  // namespace caffe
#endif  // USE_OPENCV
