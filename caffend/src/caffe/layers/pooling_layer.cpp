#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void PoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  PoolingParameter pool_param = this->layer_param_.pooling_param();
  if (pool_param.global_pooling()) {
    CHECK(!(pool_param.has_kernel_size() ||
      pool_param.has_kernel_h() || pool_param.has_kernel_w()))
      << "With Global_pooling: true Filter size cannot specified";
  } else {
    CHECK(!pool_param.has_kernel_size() !=
      !(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
    CHECK(pool_param.has_kernel_size() ||
      (pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  }
  CHECK((!pool_param.has_pad() && pool_param.has_pad_h()
      && pool_param.has_pad_w())
      || (!pool_param.has_pad_h() && !pool_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!pool_param.has_stride() && pool_param.has_stride_h()
      && pool_param.has_stride_w())
      || (!pool_param.has_stride_h() && !pool_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";

  /* for ND support */
  num_spatial_axes_ = bottom[0]->num_axes() - 2;
  kernel_shape_ = std::vector<int>(num_spatial_axes_, 0);
  /**/

  global_pooling_ = pool_param.global_pooling();
  if (global_pooling_) {
  	/* for ND support
    kernel_h_ = bottom[0]->height();
    kernel_w_ = bottom[0]->width();
    */
  	for (int i = 0; i < num_spatial_axes_; ++i)
  	  kernel_shape_[i] = bottom[0]->shape(i + 2);
  	/**/
  } else {
    if (pool_param.has_kernel_size()) {
    	/* for ND support
      kernel_h_ = kernel_w_ = pool_param.kernel_size();
      */
    	for (int i = 0; i < num_spatial_axes_; ++i)
    	  kernel_shape_[i] = pool_param.kernel_size();
    	/**/
    } else {
    	/* for ND support
      kernel_h_ = pool_param.kernel_h();
      kernel_w_ = pool_param.kernel_w();
      */
    	kernel_shape_[0] = pool_param.kernel_h();
    	kernel_shape_[1] = pool_param.kernel_w();
    	/**/
    }
  }
  /* for ND support
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  */
  for (int i = 0; i < num_spatial_axes_; ++i)
  	CHECK_GT(kernel_shape_[i], 0) << "Filter dimensions cannot be zero.";
  /**/
  /* for ND support */
  pad_ = std::vector<int>(num_spatial_axes_, 0);
  /**/
  if (!pool_param.has_pad_h()) {
  	/* for ND support
    pad_h_ = pad_w_ = pool_param.pad();
    */
  	for (int i = 0; i < num_spatial_axes_; ++i)
  		pad_[i] = pool_param.pad();
  	/**/
  } else {
  	/* for ND support
    pad_h_ = pool_param.pad_h();
    pad_w_ = pool_param.pad_w();
    */
  	pad_[0] = pool_param.pad_h();
  	pad_[1] = pool_param.pad_w();
  	/**/
  }
  /* for ND support */
  stride_ = std::vector<int>(num_spatial_axes_, 0);
  /**/
  if (!pool_param.has_stride_h()) {
  	/* for ND support
    stride_h_ = stride_w_ = pool_param.stride();
    */
  	for (int i = 0; i < num_spatial_axes_; ++i)
  		stride_[i] = pool_param.stride();
  	/**/
  } else {
  	/* for ND support
    stride_h_ = pool_param.stride_h();
    stride_w_ = pool_param.stride_w();
    */
  	stride_[0] = pool_param.stride_h();
  	stride_[1] = pool_param.stride_w();
  	/**/
  }
  /* for ND support
  if (global_pooling_) {
    CHECK(pad_h_ == 0 && pad_w_ == 0 && stride_h_ == 1 && stride_w_ == 1)
      << "With Global_pooling: true; only pad = 0 and stride = 1";
  }
  if (pad_h_ != 0 || pad_w_ != 0) {
    CHECK(this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_AVE
        || this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_MAX)
        << "Padding implemented only for average and max pooling.";
    CHECK_LT(pad_h_, kernel_h_);
    CHECK_LT(pad_w_, kernel_w_);
  }
  */
  for (int i = 0; i < num_spatial_axes_; ++i) {
    if (global_pooling_) {
      CHECK(pad_[i] == 0 && stride_[i] == 1)
        << "With Global_pooling: true; only pad = 0 and stride = 1";
    }
    if (pad_[i] != 0) {
      CHECK(this->layer_param_.pooling_param().pool()
          == PoolingParameter_PoolMethod_AVE
          || this->layer_param_.pooling_param().pool()
          == PoolingParameter_PoolMethod_MAX)
          << "Padding implemented only for average and max pooling.";
      CHECK_LT(pad_[i], kernel_shape_[i]);
    }
  }
  /**/
}

/* for ND support
template <typename Dtype>
void PoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  if (global_pooling_) {
    kernel_h_ = bottom[0]->height();
    kernel_w_ = bottom[0]->width();
  }
  pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
  pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
  if (pad_h_ || pad_w_) {
    // If we have padding, ensure that the last pooling starts strictly
    // inside the image (instead of at the padding); otherwise clip the last.
    if ((pooled_height_ - 1) * stride_h_ >= height_ + pad_h_) {
      --pooled_height_;
    }
    if ((pooled_width_ - 1) * stride_w_ >= width_ + pad_w_) {
      --pooled_width_;
    }
    CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_);
    CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_);
  }
  top[0]->Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  if (top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }
  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_MAX && top.size() == 1) {
    max_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
        pooled_width_);
  }
  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  }
}
*/
template <typename Dtype>
void PoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  shape_ = bottom[0]->shape();
  CHECK_EQ(shape_.size(), bottom[0]->num_axes()) << "The axes of the pooling layer and the bottom blob should be the same.";
  if (global_pooling_) {
    for (int i = 0; i < num_spatial_axes_; ++i)
      kernel_shape_[i] = shape_[i + 2];
  }
  pooled_shape_ = std::vector<int>(shape_.size());
  pooled_shape_[0] = shape_[0];
  pooled_shape_[1] = shape_[1];
  for (unsigned int i = 0; i < num_spatial_axes_; ++i) {
    pooled_shape_[i + 2] = static_cast<int>(std::ceil(static_cast<float>(
        shape_[i + 2] + 2 * pad_[i] - kernel_shape_[i]) /
        stride_[i])) + 1;
  }
  for (int i = 0; i < num_spatial_axes_; ++i) {
		if (pad_[i]) {
			// If we have padding, ensure that the last pooling starts strictly
			// inside the image (instead of at the padding); otherwise clip the last.
			if ((pooled_shape_[i + 2] - 1) * stride_[i] >=
					shape_[i + 2] + pad_[i]) {
				--pooled_shape_[i + 2];
			}
			CHECK_LT((pooled_shape_[i + 2] - 1) * stride_[i],
					shape_[i + 2] + pad_[i]);
		}
  }
  // reshape outputs
  top[0]->Reshape(pooled_shape_);

  if (top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }
  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_MAX && top.size() == 1) {
    max_idx_.Reshape(pooled_shape_);
  }
  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(pooled_shape_);
  }
}
/**/

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void PoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;  // suppress warnings about uninitalized variables
  Dtype* top_mask = NULL;
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // Initialize
    if (use_top_mask) {
      top_mask = top[1]->mutable_cpu_data();
      caffe_set(top_count, Dtype(-1), top_mask);
    } else {
      mask = max_idx_.mutable_cpu_data();
      caffe_set(top_count, -1, mask);
    }
    caffe_set(top_count, Dtype(-FLT_MAX), top_data);
    // The main loop
    /* for ND support
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_);
            int wend = min(wstart + kernel_w_, width_);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            const int pool_index = ph * pooled_width_ + pw;
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                const int index = h * width_ + w;
                if (bottom_data[index] > top_data[pool_index]) {
                  top_data[pool_index] = bottom_data[index];
                  if (use_top_mask) {
                    top_mask[pool_index] = static_cast<Dtype>(index);
                  } else {
                    mask[pool_index] = index;
                  }
                }
              }
            }
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
    */
    {
      int pool_size = 1;
      for(int dim = 0; dim < num_spatial_axes_; ++dim) pool_size *= pooled_shape_[dim + 2];

      int input_size = 1;
      for(int dim = 0; dim < num_spatial_axes_; ++dim) input_size *= shape_[dim + 2];

      vector<int> one_channel(2);
      one_channel[0] = 0;
      one_channel[1] = 1;

      for (int n = 0; n < shape_[0]; ++n) {
        for (int c = 0; c < shape_[1]; ++c) {

          std::vector<int> pool_axis(num_spatial_axes_);
          for (int pool_index = 0; pool_index < top[0]->count() / (pooled_shape_[0] * pooled_shape_[1]); ++pool_index) {
            for(int dim = 0, volume = pool_size, residual = pool_index; dim < num_spatial_axes_; ++dim) {
              volume /= pooled_shape_[dim + 2];
              pool_axis[dim] = residual / volume;
              residual %= volume;
            }

            std::vector<int> start(num_spatial_axes_);
            std::vector<int> end(num_spatial_axes_);

            int slice_size = 1;
            for (int dim = 0; dim < num_spatial_axes_; ++dim) {
              start[dim] = pool_axis[dim] * stride_[dim] - pad_[dim];
              end[dim] = min(start[dim] + kernel_shape_[dim], shape_[dim + 2]);
              start[dim] = max(start[dim], 0);
              slice_size *= end[dim] - start[dim];
            }

            std::vector<int> slice_axis(num_spatial_axes_);
            for (int slice_index = 0; slice_index < slice_size; ++slice_index) {
              for(int dim = 0, volume = slice_size, residual = slice_index; dim < num_spatial_axes_; ++dim) {
                volume /= shape_[dim + 2];
                slice_axis[dim] = residual / volume;
                residual %= volume;
              }

              int index = 0;

              for (int dim = 0, volume = input_size; dim < num_spatial_axes_; ++dim) {
                volume /= shape_[dim + 2];
                index += (slice_axis[dim] + start[dim]) * volume;
              }

              if (bottom_data[index] > top_data[pool_index]) {
                top_data[pool_index] = bottom_data[index];
                if (use_top_mask) {
                  top_mask[pool_index] = static_cast<Dtype>(index);
                } else {
                  mask[pool_index] = index;
                }
              }
            }
          }

          // compute offset

          bottom_data += bottom[0]->offset(one_channel);
          top_data += top[0]->offset(one_channel);
          if (use_top_mask) {
            top_mask += top[0]->offset(one_channel);
          } else {
            mask += top[0]->offset(one_channel);
          }
        }
      }
    }

    /**/
    break;
  case PoolingParameter_PoolMethod_AVE:
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = 0;
    }
    // The main loop
    /* for ND support
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                top_data[ph * pooled_width_ + pw] +=
                    bottom_data[h * width_ + w];
              }
            }
            top_data[ph * pooled_width_ + pw] /= pool_size;
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
      }
    }
    */
    {
      int pool_size = 1;
      for(int dim = 0; dim < num_spatial_axes_; ++dim) pool_size *= pooled_shape_[dim + 2];

      int input_size = 1;
      for(int dim = 0; dim < num_spatial_axes_; ++dim) input_size *= shape_[dim + 2];

      std::vector<int> one_channel(2);
      one_channel[0] = 0;
      one_channel[1] = 1;

      for (int n = 0; n < shape_[0]; ++n) {
        for (int c = 0; c < shape_[1]; ++c) {
          std::vector<int> pool_axis(num_spatial_axes_);
          for (int pool_index = 0; pool_index < top[0]->count() / (shape_[0] * shape_[1]); ++pool_index) {
            for(int dim = 0, volume = pool_size, residual = pool_index; dim < num_spatial_axes_; ++dim) {
              volume /= pooled_shape_[dim + 2];
              pool_axis[dim] = residual / volume;
              residual %= volume;
            }

            std::vector<int> start(num_spatial_axes_);
            std::vector<int> end(num_spatial_axes_);

            int slice_size = 1;
            for (int dim = 0; dim < num_spatial_axes_; ++dim) {
              start[dim] = pool_axis[dim] * stride_[dim] - pad_[dim];
              end[dim] = min(start[dim] + kernel_shape_[dim], shape_[dim + 2]);
              start[dim] = max(start[dim], 0);
              slice_size *= end[dim] - start[dim];
            }

            std::vector<int> slice_axis(num_spatial_axes_);
            for (int slice_index = 0; slice_index < slice_size; ++slice_index) {
              for(int dim = 0, volume = slice_size, residual = slice_index; dim < num_spatial_axes_; ++dim) {
                volume /= shape_[dim + 2];
                slice_axis[dim] = residual / volume;
                residual %= volume;
              }

              int index = 0;

              for (int dim = 0, volume = input_size; dim < num_spatial_axes_; ++dim) {
                volume /= shape_[dim + 2];
                index += (slice_axis[dim] + start[dim]) * volume;
              }

              top_data[pool_index] += bottom_data[index];
            }

            top_data[pool_index] /= slice_size;
          }

          // compute offset
          bottom_data += bottom[0]->offset(one_channel);
          top_data += top[0]->offset(one_channel);
        }
      }
    }
    /**/
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;  // suppress warnings about uninitialized variables
  const Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // The main loop
  	/* for ND support
    if (use_top_mask) {
      top_mask = top[1]->cpu_data();
    } else {
      mask = max_idx_.cpu_data();
    }
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            const int index = ph * pooled_width_ + pw;
            const int bottom_index =
                use_top_mask ? top_mask[index] : mask[index];
            bottom_diff[bottom_index] += top_diff[index];
          }
        }
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
    */
    {
      if (use_top_mask) {
        top_mask = top[1]->cpu_data();
      } else {
        mask = max_idx_.cpu_data();
      }
      std::vector<int> one_channel(2);
      one_channel[0] = 0;
      one_channel[1] = 1;
      for (int n = 0; n < shape_[0]; ++n) {
        for (int c = 0; c < shape_[1]; ++c) {
          for (int index = 0; index < top[0]->count() / (shape_[0] * shape_[1]); ++index) {
            const int bottom_index = use_top_mask ? top_mask[index] : mask[index];
            bottom_diff[bottom_index] += top_diff[index];
          }
          bottom_diff += bottom[0]->offset(one_channel);
          top_diff += top[0]->offset(one_channel);
          if (use_top_mask) {
            top_mask += top[0]->offset(one_channel);
          } else {
            mask += top[0]->offset(one_channel);
          }
        }
      }
    }

  	/**/
    break;
  case PoolingParameter_PoolMethod_AVE:
    // The main loop
  	/* for ND support
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                bottom_diff[h * width_ + w] +=
                  top_diff[ph * pooled_width_ + pw] / pool_size;
              }
            }
          }
        }
        // offset
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
    */
    {
      int pool_size = 1;
      for(int dim = 0; dim < num_spatial_axes_; ++dim) pool_size *= pooled_shape_[dim + 2];

      int input_size = 1;
      for(int dim = 0; dim < num_spatial_axes_; ++dim) input_size *= shape_[dim + 2];

      vector<int> one_channel(2);
      one_channel[0] = 0;
      one_channel[1] = 1;

      for (int n = 0; n < shape_[0]; ++n) {
        for (int c = 0; c < shape_[1]; ++c) {
          std::vector<int> pool_axis(num_spatial_axes_);
          for (int pool_index = 0; pool_index < top[0]->count() / (shape_[0] * shape_[1]); ++pool_index) {
            for(int dim = 0, volume = pool_size, residual = pool_index; dim < num_spatial_axes_; ++dim) {
              volume /= pooled_shape_[dim + 2];
              pool_axis[dim] = residual / volume;
              residual %= volume;
            }

            std::vector<int> start(num_spatial_axes_);
            std::vector<int> end(num_spatial_axes_);

            int slice_size = 1;
            for (int dim = 0; dim < num_spatial_axes_; ++dim) {
              start[dim] = pool_axis[dim] * stride_[dim] - pad_[dim];
              end[dim] = min(start[dim] + kernel_shape_[dim], shape_[dim + 2]);
              start[dim] = max(start[dim], 0);
              slice_size *= end[dim] - start[dim];
            }

            std::vector<int> slice_axis(num_spatial_axes_);
            for (int slice_index = 0; slice_index < slice_size; ++slice_index) {
              for(int dim = 0, volume = slice_size, residual = slice_index; dim < num_spatial_axes_; ++dim) {
                volume /= shape_[dim + 2];
                slice_axis[dim] = residual / volume;
                residual %= volume;
              }

              int index = 0;

              for (int dim = 0, volume = input_size; dim < num_spatial_axes_; ++dim) {
                volume /= shape_[dim + 2];
                index += (slice_axis[dim] + start[dim]) * volume;
              }

              bottom_diff[index] += top_diff[pool_index] / slice_size;
            }
          }

          // offset
          bottom_diff += bottom[0]->offset(one_channel);
          top_diff += top[0]->offset(one_channel);
        }
      }
    }

  	/**/
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}


#ifdef CPU_ONLY
STUB_GPU(PoolingLayer);
#endif

INSTANTIATE_CLASS(PoolingLayer);

}  // namespace caffe
