#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/shop_the_look_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
ShopTheLookDataLayer<Dtype>::~ShopTheLookDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void ShopTheLookDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  string collection_root_folder = this->layer_param_.shop_the_look_data_param().collection_image_data_param().root_folder();
  string stl_root_folder = this->layer_param_.shop_the_look_data_param().shop_the_look_image_data_param().root_folder();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
      
  // Read the file with the skus and the images of those SKUs
  string delim_tmp = this->layer_param_.shop_the_look_data_param().collection_delimiter();
  CHECK_EQ(delim_tmp.size(), 1) << "delimiter string can only have one character";
  char delimiter = delim_tmp[0];
  
  const string& source = this->layer_param_.shop_the_look_data_param().collection_source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string sku_string;
  string line;
  bool has_next;
  while (getline(infile, line)) {
    has_next = true;
    for (int i = 0; has_next; ++i) {
      size_t end = line.find(delimiter);
      if (i < 1) {
        CHECK_NE(end, string::npos)
          << "line number " << sku_collection_images_.size() << " has too few columns";
      }
      if (i == 0) {
        // the first entry is always the sku string
        sku_string = line.substr(0, end);
        sku_to_index_[sku_string] = sku_collection_images_.size();
        sku_collection_images_.push_back(vector<string>(0));
      } else {
        // the other entries are the file names
        // If there is no more delimiter, append last string and finish this line
        if (end == string::npos) {
          sku_collection_images_[sku_to_index_[sku_string]].push_back(line.substr(0, line.size() - 1)); // we need to take the substring because the newline character is somehow there
          has_next = false;
        } else {
          sku_collection_images_[sku_to_index_[sku_string]].push_back(line.substr(0, end));
        }
      }
      if (end != string::npos) {
        line = line.substr(end+1);
      }
    }
  }
  
  CHECK_GE(sku_collection_images_.size(), 1) << "you must specify at least one sku collection";
  
  delim_tmp = this->layer_param_.shop_the_look_data_param().shop_the_look_delimiter();
  CHECK_EQ(delim_tmp.size(), 1) << "delimiter string can only have one character";
  delimiter = delim_tmp[0];
  
  // Read the file with shop the look images and the SKUs contained in those images
  const string& source2 = this->layer_param_.shop_the_look_data_param().shop_the_look_source();
  LOG(INFO) << "Opening file " << source2;
  std::ifstream infile2(source2.c_str());
  
  std::set<string> stl_set;
  while (getline(infile2, line)) {
    has_next = true;
    for (int i = 0; has_next; ++i) {
      size_t end = line.find(delimiter);
      if (i < 2) {
        CHECK_NE(end, string::npos)
          << "line number " << shop_the_look_skus_.size() << " has too few columns";
      }
      // push back an empty vector
      shop_the_look_skus_.push_back(vector<string>(0));
      if (i == 0) {
        //CHECK_EQ(stl_set.find(line.substr(0, end)), stl_set.end()) << "already saw shop the look image ";// << line.substr(0, end);
        stl_set.insert(line.substr(0, end));
        shop_the_look_images_.push_back(line.substr(0, end));
      } else {
        // If there is no more delimiter, append last string and finish this line
        string sku_str;
        if (end == string::npos) {
          sku_str = line;
          has_next = false;
        } else {
          sku_str = line.substr(0, end);
        }
        //CHECK_NE(sku_to_index_.find(sku_str), sku_to_index_.end()) << "don't know where pictures for "<< sku_str << "are";
        shop_the_look_skus_[shop_the_look_skus_.size() - 1].push_back(sku_str);
        
        int stl_index = shop_the_look_images_.size()-1;
        int sku_index = sku_to_index_[sku_str];
        positive_examples_.push_back(std::pair<int, int>(stl_index, sku_index));
        if (positive_examples_map_.find(stl_index) == positive_examples_map_.end()) {
          positive_examples_map_[stl_index] = std::set<int>();
        }
        positive_examples_map_[stl_index].insert(sku_index);
      }
      if (end != string::npos) {
        line = line.substr(end+1);
      }
    }
  }
  CHECK_GE(shop_the_look_images_.size(), 1) << "you must specify at least one shop the look image";
  
  if (this->layer_param_.shop_the_look_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  num_shop_the_look_ = shop_the_look_images_.size();
  LOG(INFO) << "A total of " << num_shop_the_look_ << " shop the look images.";
  num_sku_ = sku_collection_images_.size();
  LOG(INFO) << "A total of " << num_sku_ << " image collections.";

  lines_id_ = 0;
  
  //figure out shape of sku images
  cv::Mat cv_img = ReadImageToCVMat(collection_root_folder + sku_collection_images_[0][0],
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << sku_collection_images_[0][0];
  // Use data_transformer to infer the expected blob shape from a cv_img.
  sku_shape_ = this->data_transformer_->InferBlobShape(cv_img);
  
  //figure out shape of shop the look
  cv_img = ReadImageToCVMat(stl_root_folder + shop_the_look_images_[0],
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << shop_the_look_images_[0];
  // Use data_transformer to infer the expected blob shape from a cv_img.
  shop_the_look_shape_ = this->data_transformer_->InferBlobShape(cv_img);
  
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.shop_the_look_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  sku_shape_[0] = batch_size;
  shop_the_look_shape_[0] = batch_size;
  top[0]->Reshape(shop_the_look_shape_);

  LOG(INFO) << "output shop the look data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  
  // Note that the first dimensions of this will get changed for collections
  top[1]->Reshape(sku_shape_);
  
  LOG(INFO) << "output collection sku data size: " << top[1]->num() << ","
      << top[1]->channels() << "," << top[1]->height() << ","
      << top[1]->width();
  
  // label
  vector<int> label_shape(1, batch_size);
  top[2]->Reshape(label_shape);
  top[3]->Reshape(label_shape);
  label_shape[0] *= 2;

  // this->prefetch_[i].data_ gets reshaped again during forward pass
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(sku_shape_);
  }
  
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void ShopTheLookDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(positive_examples_.begin(), positive_examples_.end(), prefetch_rng);
}

/**
 * Load an image and save it to transformed data
 */
template <typename Dtype>
void fetch_image(const ImageDataParameter& image_data_param, string file_name, shared_ptr<DataTransformer<Dtype> >& data_transformer, Blob<Dtype>& transformed_data) {
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  const string& root_folder = image_data_param.root_folder();
  cv::Mat cv_img = ReadImageToCVMat(root_folder + file_name,
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << root_folder + file_name;
  // Apply transformations (mirror, crop...) to the image
  data_transformer->Transform(cv_img, &transformed_data);
}

/*
 * This function is called on prefetch thread. It saves both shop the look and
 * sku collection images to the batch->data_ blob. The shape of this blob is
 * meaningless, it just stores the data which is then split between different
 * top blobs, and put in correct size.
 */
template <typename Dtype>
void ShopTheLookDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CHECK(batch->data_.count());
  const ShopTheLookDataParam& stl_param =
      this->layer_param_.shop_the_look_data_param();
  const int batch_size = stl_param.batch_size();
  CHECK_GE(stl_param.positive_probability(), 0);
  CHECK_LE(stl_param.positive_probability(), 1);
  
  // figure out the size of the images that will be fetched, reshape batch->data_
  int stl_data_size = 1;
  for (int i = 1; i < shop_the_look_shape_.size(); ++i)
    stl_data_size *= shop_the_look_shape_[i];
  int sku_data_size = 1;
  for (int i = 1; i < sku_shape_.size(); ++i)
    sku_data_size *= sku_shape_[i];
  
  Blob<Dtype> transformed_data;
    
  vector<float> uniform_random(batch_size);
  caffe_rng_uniform(batch_size, float(0), float(1), uniform_random.data());
  int num_sku_images(0);
  vector<int> shop_the_look_index(batch_size), sku_index(batch_size);
  
  
  // start off by determining which images will be loaded so that everything can be resized
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // either fetch a positive or a negative example
    if (uniform_random[item_id] < stl_param.positive_probability()) {
      // positive example case
      shop_the_look_index[item_id] = positive_examples_[lines_id_].first;
      sku_index[item_id] = positive_examples_[lines_id_++].second;
      batch->label_.mutable_cpu_data()[item_id] = 1;
      if (lines_id_ >= positive_examples_.size()) {
        // We have reached the end. Restart from the first.
        DLOG(INFO) << "Restarting data prefetching from start.";
        lines_id_ = 0;
        if (this->layer_param_.shop_the_look_data_param().shuffle()) {
          ShuffleImages();
        }
      }
    } else {
      // negative example case
      while(true) {
        shop_the_look_index[item_id] = caffe_rng_rand() % num_shop_the_look_;
        sku_index[item_id]           = caffe_rng_rand() % num_sku_;
        // check to see if this is a positive example
        if (positive_examples_map_.find(shop_the_look_index[item_id]) == positive_examples_map_.end() || 
            positive_examples_map_[shop_the_look_index[item_id]].find(sku_index[item_id]) == positive_examples_map_[shop_the_look_index[item_id]].end()) {
            // ok, it is not a positive example
            break;
        }
      }
      batch->label_.mutable_cpu_data()[item_id] = 0;
    }
    num_sku_images += sku_collection_images_[sku_index[item_id]].size();
  }
  // reshape the output data so it's large enough for the next collection images
  vector<int> data_shape(1, batch_size * stl_data_size + num_sku_images * sku_data_size);
  batch->data_.Reshape(data_shape);
  Dtype* data = batch->data_.mutable_cpu_data();
  
  num_sku_images = 0;
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // fetch the shop the look image
    vector<int> shop_the_look_shape(shop_the_look_shape_);
    shop_the_look_shape[0] = 1;
    transformed_data.Reshape(shop_the_look_shape);
    transformed_data.set_cpu_data(data + item_id * stl_data_size);
    fetch_image(stl_param.shop_the_look_image_data_param(), shop_the_look_images_[shop_the_look_index[item_id]], this->data_transformer_, transformed_data);
    
    // fetch the collection images
    const vector<string>& collection = sku_collection_images_[sku_index[item_id]];
    
    
    vector<int> sku_shape(sku_shape_);
    sku_shape[0] = 1;
    transformed_data.Reshape(sku_shape);
    for (int i = 0; i < collection.size(); ++i) {
      transformed_data.set_cpu_data(data + batch_size * stl_data_size + num_sku_images * sku_data_size);
      fetch_image(stl_param.collection_image_data_param(), collection[i], this->data_transformer_, transformed_data);
      num_sku_images++;
    }
    batch->label_.mutable_cpu_data()[batch_size + item_id] = num_sku_images;
  }
}

/*
 * Reshape the top blobs and take the next batch and copy the data into the top blobs.
 */
template <typename Dtype>
void ShopTheLookDataLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = this->prefetch_full_.pop("Data layer prefetch queue empty");
  
  // Reshape to a batch of shop the look images
  top[0]->Reshape(shop_the_look_shape_);
  // Copy the data
  caffe_copy(top[0]->count(), batch->data_.cpu_data(),
             top[0]->mutable_cpu_data());
  
  // do the collection images
  int collection_image_size = 1;
  for (int i = 1; i < sku_shape_.size(); ++i)
    collection_image_size *= sku_shape_[i];
  int num_sku_images = (batch->data_.count() - top[0]->count()) / collection_image_size;
  sku_shape_[0] = num_sku_images;
  top[1]->Reshape(sku_shape_);
  
  // Copy the data
  caffe_copy(top[1]->count(), batch->data_.cpu_data() + top[0]->count(),
             top[1]->mutable_cpu_data());
  
  vector<int> label_shape(batch->label_.shape());
  label_shape[0] /= 2;
  
  // copy the top splits
  top[2]->Reshape(label_shape);
  caffe_copy(top[2]->count(), batch->label_.cpu_data() + label_shape[0],
             top[2]->mutable_cpu_data());
  
  // copy the labels
  top[3]->Reshape(label_shape);
  caffe_copy(top[3]->count(), batch->label_.cpu_data(),
             top[3]->mutable_cpu_data());
  
  DLOG(INFO) << "Prefetch copied";
  this->prefetch_free_.push(batch);
}

INSTANTIATE_CLASS(ShopTheLookDataLayer);
REGISTER_LAYER_CLASS(ShopTheLookData);

}  // namespace caffe
#endif  // USE_OPENCV