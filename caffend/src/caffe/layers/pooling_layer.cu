#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MaxPoolForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    Dtype* const top_data, int* mask, Dtype* top_mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_slice[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_slice[maxidx];
        }
      }
    }
    top_data[index] = maxval;
    if (mask) {
      mask[index] = maxidx;
    } else {
      top_mask[index] = maxidx;
    }
  }
}

/* for ND support */
template <typename Dtype>
__global__ void MaxPoolForward_ND(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int num_spatial_axes,
    const int* const input_shape,
    const int* const pool_shape,
    const int* const kernel_shape,
    const int* const stride_shape,
    const int* const pad_shape,
    Dtype* const top_data, int* mask, Dtype* top_mask) {

  __shared__ int share[CAFFE_CUDA_MAX_SHARED_MEM/sizeof(int)];

  const int shared_block_size = 2 + 4*num_spatial_axes;
  int* const start      = &share[threadIdx.x*shared_block_size];
  int* const end        = &share[threadIdx.x*shared_block_size + num_spatial_axes];
  int* const slice_axis = &share[threadIdx.x*shared_block_size + 2*num_spatial_axes];
  int* const pool_axis  = &share[threadIdx.x*shared_block_size + 3*num_spatial_axes];

  const int num_all_axes = num_spatial_axes + 2;

  int pool_size = 1;
  int input_size = 1;
  for (int dim = 0; dim < num_all_axes; ++dim) {
	  pool_size *= pool_shape[dim];
	  input_size *= input_shape[dim];
  }

  CUDA_KERNEL_LOOP(pool_index, nthreads) {

    for (int dim = 0, volume = pool_size, residual = pool_index; dim < num_all_axes; ++dim) {
      volume /= pool_shape[dim];
      pool_axis[dim] = residual / volume;
      residual %= volume;
    }

    int slice_size = 1;
    for (int dim = 0; dim < num_spatial_axes; ++dim) {
      start[dim] = pool_axis[dim + 2] * stride_shape[dim] - pad_shape[dim];
      end[dim] = min(start[dim] + kernel_shape[dim], input_shape[dim + 2]);
      start[dim] = max(start[dim], 0);
      slice_size *= end[dim] - start[dim];
    }

    Dtype maxval = -FLT_MAX;
    int maxidx = -1;

    int slice_offset = (pool_axis[0] * channels + pool_axis[1]);
    for (int dim = 0; dim < num_spatial_axes; ++dim) slice_offset *= input_shape[dim + 2];
    const Dtype* const bottom_slice = bottom_data + slice_offset;

    for(int slice_index = 0; slice_index < slice_size; ++slice_index) {
      for(int dim = 0, volume = slice_size, residual = slice_index; dim < num_spatial_axes; ++dim) {
        volume /= end[dim] - start[dim];
        slice_axis[dim] = residual/volume;
        residual %= volume;
      }

      int index = 0;

      for (int dim = 0, volume = input_size/(input_shape[0]*input_shape[1]); dim < num_spatial_axes; ++dim) {
        volume /= input_shape[dim + 2];
        index += (start[dim] + slice_axis[dim]) * volume;
      }

      if (bottom_slice[index] > maxval) {
        maxidx = index;
        maxval = bottom_slice[maxidx];
      }
    }

    top_data[pool_index] = maxval;
    if (mask) {
      mask[pool_index] = maxidx;
    } else {
      top_mask[pool_index] = maxidx;
    }

  }
}
/**/

template <typename Dtype>
__global__ void AvePoolForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    const int pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    Dtype aveval = 0;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_slice[h * width + w];
      }
    }
    top_data[index] = aveval / pool_size;
  }
}

/* for ND support */
template <typename Dtype>
__global__ void AvePoolForward_ND(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int num_spatial_axes,
    const int* const input_shape,
    const int* const pool_shape,
    const int* const kernel_shape,
    const int* const stride_shape,
    const int* const pad_shape,
    Dtype* const top_data) {

  __shared__ int share[CAFFE_CUDA_MAX_SHARED_MEM/sizeof(int)];

  const int shared_block_size = 2 + 4*num_spatial_axes;
  int* const start      = &share[threadIdx.x*shared_block_size];
  int* const end        = &share[threadIdx.x*shared_block_size + num_spatial_axes];
  int* const slice_axis = &share[threadIdx.x*shared_block_size + 2*num_spatial_axes];
  int* const pool_axis  = &share[threadIdx.x*shared_block_size + 3*num_spatial_axes];
  const int num_all_axes = num_spatial_axes + 2;

  int pool_size = 1;
  int input_size = 1;
  for (int dim = 0; dim < num_all_axes; ++dim) {
	  pool_size *= pool_shape[dim];
	  input_size *= input_shape[dim];
  }

  CUDA_KERNEL_LOOP(pool_index, nthreads) {
    for (int dim = 0, volume = pool_size, residual = pool_index; dim < num_all_axes; ++dim) {
      volume /= pool_shape[dim];
      pool_axis[dim] = residual / volume;
      residual %= volume;
    }

    int slice_size = 1;
    for (int dim = 0; dim < num_spatial_axes; ++dim) {
      start[dim] = pool_axis[dim + 2] * stride_shape[dim] - pad_shape[dim];
      end[dim] = min(start[dim] + kernel_shape[dim], input_shape[dim + 2]);
      start[dim] = max(start[dim], 0);
      slice_size *= end[dim] - start[dim];
    }

    int slice_offset = pool_axis[0] * channels + pool_axis[1];
    for (int dim = 0; dim < num_spatial_axes; ++dim) slice_offset *= input_shape[dim+2];
    const Dtype* const bottom_slice = bottom_data + slice_offset;

    Dtype aveval = 0;

    for(int slice_index = 0; slice_index < slice_size; ++slice_index) {
      for(int dim = 0, volume = slice_size, residual = slice_index; dim < num_spatial_axes; ++dim) {
        volume /= end[dim] - start[dim];
        slice_axis[dim] = residual/volume;
        residual %= volume;
      }

      int index = 0;

      for (int dim = 0, volume = input_size/(input_shape[0]*input_shape[1]); dim < num_spatial_axes; ++dim) {
        volume /= input_shape[dim + 2];
        index += (start[dim] + slice_axis[dim]) * volume;
      }

      aveval += bottom_slice[index];
    }

    top_data[pool_index] = aveval / slice_size;
  }
}
/**/

template <typename Dtype>
__global__ void StoPoolForwardTrain(const int nthreads,
    const Dtype* const bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, Dtype* const rand_idx, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    const int hstart = ph * stride_h;
    const int hend = min(hstart + kernel_h, height);
    const int wstart = pw * stride_w;
    const int wend = min(wstart + kernel_w, width);
    Dtype cumsum = 0.;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
      }
    }
    const float thres = rand_idx[index] * cumsum;
    // Second pass: get value, and set index.
    cumsum = 0;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
        if (cumsum >= thres) {
          rand_idx[index] = ((n * channels + c) * height + h) * width + w;
          top_data[index] = bottom_slice[h * width + w];
          return;
        }
      }
    }
  }
}


template <typename Dtype>
__global__ void StoPoolForwardTest(const int nthreads,
    const Dtype* const bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    const int hstart = ph * stride_h;
    const int hend = min(hstart + kernel_h, height);
    const int wstart = pw * stride_w;
    const int wend = min(wstart + kernel_w, width);
    // We set cumsum to be 0 to avoid divide-by-zero problems
    Dtype cumsum = 0.;
    Dtype cumvalues = 0.;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
        cumvalues += bottom_slice[h * width + w] * bottom_slice[h * width + w];
      }
    }
    top_data[index] = (cumsum > 0.) ? cumvalues / cumsum : 0.;
  }
}


template <typename Dtype>
void PoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;
  Dtype* top_mask = NULL;

  /* for ND support */
  int* input_shape_d;
  CUDA_CHECK(cudaMalloc((void**)&input_shape_d, shape_.size() * sizeof(int)));
  caffe_gpu_memcpy(shape_.size() * sizeof(int), &shape_[0], input_shape_d);
  int* pool_shape_d;
  CUDA_CHECK(cudaMalloc((void**)&pool_shape_d, shape_.size() * sizeof(int)));
  caffe_gpu_memcpy(shape_.size() * sizeof(int), &pooled_shape_[0], pool_shape_d);
  int* kernel_shape_d;
  CUDA_CHECK(cudaMalloc((void**)&kernel_shape_d, num_spatial_axes_ * sizeof(int)));
  caffe_gpu_memcpy(num_spatial_axes_ * sizeof(int), &kernel_shape_[0], kernel_shape_d);
  int* stride_shape_d;
  CUDA_CHECK(cudaMalloc((void**)&stride_shape_d, num_spatial_axes_ * sizeof(int)));
  caffe_gpu_memcpy(num_spatial_axes_ * sizeof(int), &stride_[0], stride_shape_d);
  int* pad_shape_d;
  CUDA_CHECK(cudaMalloc((void**)&pad_shape_d, num_spatial_axes_ * sizeof(int)));
  caffe_gpu_memcpy(num_spatial_axes_ * sizeof(int), &pad_[0], pad_shape_d);
  /**/

  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->mutable_gpu_data();
    } else {
      mask = max_idx_.mutable_gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    /* for ND support
    MaxPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,
        mask, top_mask);
    */
    MaxPoolForward_ND<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->shape(0), bottom[0]->shape(1),
				num_spatial_axes_,
        input_shape_d, pool_shape_d, kernel_shape_d, stride_shape_d, pad_shape_d,
        top_data, mask, top_mask);
    /**/
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
  	/* for ND support
    AvePoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data);
    */
    AvePoolForward_ND<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->shape(0), bottom[0]->shape(1),
				num_spatial_axes_,
        input_shape_d, pool_shape_d, kernel_shape_d, stride_shape_d, pad_shape_d,
        top_data);
  	/**/
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
  	/* for ND support
    if (this->phase_ == TRAIN) {
      // We need to create the random index as well.
      caffe_gpu_rng_uniform(count, Dtype(0), Dtype(1),
                            rand_idx_.mutable_gpu_data());
      // NOLINT_NEXT_LINE(whitespace/operators)
      StoPoolForwardTrain<Dtype><<<CAFFE_GET_BLOCKS(count),
                                   CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, bottom[0]->num(), channels_,
          height_, width_, pooled_height_, pooled_width_, kernel_h_,
          kernel_w_, stride_h_, stride_w_,
          rand_idx_.mutable_gpu_data(), top_data);
    } else {
      // NOLINT_NEXT_LINE(whitespace/operators)
      StoPoolForwardTest<Dtype><<<CAFFE_GET_BLOCKS(count),
                                  CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, bottom[0]->num(), channels_,
          height_, width_, pooled_height_, pooled_width_, kernel_h_,
          kernel_w_, stride_h_, stride_w_, top_data);
    }
    */
  	LOG(FATAL) << "PoolingParameter_PoolMethod_STOCHASTIC is not implemented for ND yet.";
  	/**/
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;

  CUDA_CHECK(cudaFree(input_shape_d));
  CUDA_CHECK(cudaFree(pool_shape_d));
  CUDA_CHECK(cudaFree(kernel_shape_d));
  CUDA_CHECK(cudaFree(stride_shape_d));
  CUDA_CHECK(cudaFree(pad_shape_d));
}


template <typename Dtype>
__global__ void MaxPoolBackward(const int nthreads, const Dtype* const top_diff,
    const int* const mask, const Dtype* const top_mask, const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart =
         (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    const int pwstart =
         (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const int offset = (n * channels + c) * pooled_height * pooled_width;
    const Dtype* const top_diff_slice = top_diff + offset;
    if (mask) {
      const int* const mask_slice = mask + offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (mask_slice[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff_slice[ph * pooled_width + pw];
          }
        }
      }
    } else {
      const Dtype* const top_mask_slice = top_mask + offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (top_mask_slice[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff_slice[ph * pooled_width + pw];
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

/* for ND support */
template <typename Dtype>
__global__ void MaxPoolBackward_ND(const int nthreads, const Dtype* const top_diff,
    const int* const mask, const Dtype* const top_mask, const int num,
    const int channels,
    const int num_spatial_axes,
    const int* const input_shape,
    const int* const pool_shape,
    const int* const kernel_shape,
    const int* const stride_shape,
    const int* const pad_shape,
    Dtype* const bottom_diff) {

  __shared__ int share[CAFFE_CUDA_MAX_SHARED_MEM/sizeof(int)];

  const int shared_block_size = 2 + 4*num_spatial_axes;
  int* const pstart     = &share[threadIdx.x*shared_block_size];
  int* const pend       = &share[threadIdx.x*shared_block_size + num_spatial_axes];
  int* const slice_axis = &share[threadIdx.x*shared_block_size + 2*num_spatial_axes];
  int* const input_axis = &share[threadIdx.x*shared_block_size + 3*num_spatial_axes];
  const int num_all_axes = num_spatial_axes + 2;

  int pool_size = 1;
  int input_size = 1;
  for (int dim = 0; dim < num_all_axes; ++dim) {
	  pool_size *= pool_shape[dim];
	  input_size *= input_shape[dim];
  }

  CUDA_KERNEL_LOOP(input_index, nthreads) {
    // find out the local index
    // find out the local offset

    for (int dim = 0, volume = input_size, residual = input_index; dim < num_all_axes; ++dim) {
      volume /= input_shape[dim];
      input_axis[dim] = residual / volume;
      residual %= volume;
    }

    int slice_size = 1;
    for (int dim = 0; dim < num_spatial_axes; ++dim) {
      pstart[dim] = (input_axis[dim + 2] + pad_shape[dim] < kernel_shape[dim]) ? 0 : (input_axis[dim + 2] + pad_shape[dim] - kernel_shape[dim]) / stride_shape[dim] + 1;
      pend[dim] = min((input_axis[dim + 2] + pad_shape[dim]) / stride_shape[dim] + 1, pool_shape[dim + 2]);
      slice_size *= pend[dim] - pstart[dim];
    }

    Dtype gradient = 0;

    int offset = (input_axis[0] * channels + input_axis[1]);
    for (int dim = 0; dim < num_spatial_axes; ++dim) offset *= pool_shape[dim+2];

    const Dtype* const top_diff_slice = top_diff + offset;
    if (mask) {
      const int* const mask_slice = mask + offset;

      for(int slice_index = 0; slice_index < slice_size; ++slice_index) {
        for(int dim = 0, volume = slice_size, residual = slice_index; dim < num_spatial_axes; ++dim) {
          volume /= pend[dim] - pstart[dim];
          slice_axis[dim] = residual / volume;
          residual %= volume;
        }

      }

      int index = 0;

      for (int dim = 0, volume = pool_size / (pool_shape[0] * pool_shape[1]); dim < num_spatial_axes; ++dim) {
        volume /= pool_shape[dim + 2];
        index += (slice_axis[dim] + pstart[dim]) * volume;
      }

      if (mask_slice[index] == input_index) {
        gradient += top_diff_slice[index];
      }
    } else {
      const Dtype* const top_mask_slice = top_mask + offset;

      int index = 0;

      for (int dim = 0, volume = pool_size / (pool_shape[0] * pool_shape[1]); dim < num_spatial_axes; ++dim) {
        volume /= pool_shape[dim + 2];
        index += (slice_axis[dim] + pstart[dim]) * volume;
      }

      if (top_mask_slice[index] == input_index) {
        gradient += top_diff_slice[index];
      }
    }
    bottom_diff[input_index] = gradient;
  }
}
/**/

template <typename Dtype>
__global__ void AvePoolBackward(const int nthreads, const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width + pad_w;
    const int h = (index / width) % height + pad_h;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const Dtype* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);
        int pool_size = (hend - hstart) * (wend - wstart);
        gradient += top_diff_slice[ph * pooled_width + pw] / pool_size;
      }
    }
    bottom_diff[index] = gradient;
  }
}

/* for ND support */
template <typename Dtype>
__global__ void AvePoolBackward_ND(const int nthreads, const Dtype* const top_diff,
    const int num, const int channels,
    const int num_spatial_axes,
    const int* const input_shape,
    const int* const pool_shape,
    const int* const kernel_shape,
    const int* const stride_shape,
    const int* const pad_shape,
    Dtype* const bottom_diff) {

  __shared__ int share[CAFFE_CUDA_MAX_SHARED_MEM/sizeof(int)];

  const int shared_block_size = 2 + 4*num_spatial_axes;
  int* const pstart     = &share[threadIdx.x*shared_block_size];
  int* const pend       = &share[threadIdx.x*shared_block_size + num_spatial_axes];
  int* const slice_axis = &share[threadIdx.x*shared_block_size + 2*num_spatial_axes];
  int* const input_axis = &share[threadIdx.x*shared_block_size + 3*num_spatial_axes];
  const int num_all_axes = num_spatial_axes + 2;

  int pool_size = 1;
  int input_size = 1;
  for (int dim = 0; dim < num_spatial_axes+2; ++dim) {
	  pool_size *= pool_shape[dim];
	  input_size *= input_shape[dim];
  }

  CUDA_KERNEL_LOOP(input_index, nthreads) {
    // find out the local index
    // find out the local offset

    for (int dim = 0, volume = input_size, residual = input_index; dim < num_all_axes; ++dim) {
      volume /= input_shape[dim];
      input_axis[dim] = residual / volume;
      residual %= volume;
    }

    int slice_size = 1;
    for (int dim = 0; dim < num_spatial_axes; ++dim) {
      pstart[dim] = (input_axis[dim + 2] + pad_shape[dim] < kernel_shape[dim]) ? 0 : (input_axis[dim + 2] + pad_shape[dim] - kernel_shape[dim]) / stride_shape[dim] + 1;
      pend[dim] = min((input_axis[dim + 2] + pad_shape[dim]) / stride_shape[dim] + 1, pool_shape[dim + 2]);
      slice_size *= pend[dim] - pstart[dim];
    }

    Dtype gradient = 0;

    int offset = (input_axis[0] * channels + input_axis[1]);
    for (int dim = 0; dim < num_spatial_axes; ++dim) offset *= pool_shape[dim];

    const Dtype* const top_diff_slice = top_diff + offset;

    for(int slice_index = 0; slice_index < slice_size; ++slice_index) {
      for(int dim = 0, volume = slice_size, residual = slice_index; dim < num_spatial_axes; ++dim) {
        volume /= pend[dim]-pstart[dim];
        slice_axis[dim] = residual/volume;
        residual %= volume;
      }

      int index = 0;

      for (int dim = 0, volume = pool_size; dim < num_spatial_axes; ++dim) {
    	volume /= pool_shape[dim + 2];
        index += (slice_axis[dim] + pstart[dim]) * volume;
      }

      gradient += top_diff_slice[index] / slice_size;

    }
    bottom_diff[input_index] = gradient;

  }
}
/**/

template <typename Dtype>
__global__ void StoPoolBackward(const int nthreads,
    const Dtype* const rand_idx, const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const Dtype* const rand_idx_slice =
        rand_idx + (n * channels + c) * pooled_height * pooled_width;
    const Dtype* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        gradient += top_diff_slice[ph * pooled_width + pw] *
            (index == static_cast<int>(rand_idx_slice[ph * pooled_width + pw]));
      }
    }
    bottom_diff[index] = gradient;
  }
}


template <typename Dtype>
void PoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;
  const Dtype* top_mask = NULL;

  /* for ND support */
  int* input_shape_d;
  CUDA_CHECK(cudaMalloc(&input_shape_d, shape_.size() * sizeof(int)));
  caffe_gpu_memcpy(shape_.size() * sizeof(int), &shape_[0], input_shape_d);
  int* pool_shape_d;
  CUDA_CHECK(cudaMalloc(&pool_shape_d, shape_.size() * sizeof(int)));
  caffe_gpu_memcpy(shape_.size() * sizeof(int), &pooled_shape_[0], pool_shape_d);
  int* kernel_shape_d;
  CUDA_CHECK(cudaMalloc(&kernel_shape_d, num_spatial_axes_ * sizeof(int)));
  caffe_gpu_memcpy(num_spatial_axes_ * sizeof(int), &kernel_shape_[0], kernel_shape_d);
  int* stride_shape_d;
  CUDA_CHECK(cudaMalloc(&stride_shape_d, num_spatial_axes_ * sizeof(int)));
  caffe_gpu_memcpy(num_spatial_axes_ * sizeof(int), &stride_[0], stride_shape_d);
  int* pad_shape_d;
  CUDA_CHECK(cudaMalloc(&pad_shape_d, num_spatial_axes_ * sizeof(int)));
  caffe_gpu_memcpy(num_spatial_axes_ * sizeof(int), &pad_[0], pad_shape_d);
  /**/

  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->gpu_data();
    } else {
      mask = max_idx_.gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    /* for ND support
    MaxPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, top_mask, top[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_,
        kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
        bottom_diff);
    */

    MaxPoolBackward_ND<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, mask, top_mask, top[0]->shape(0), top[0]->shape(1),
						num_spatial_axes_,
            input_shape_d, pool_shape_d, kernel_shape_d, stride_shape_d, pad_shape_d,
            bottom_diff);
    /**/
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
  	/* for ND support
    AvePoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, bottom_diff);
    */
    AvePoolBackward_ND<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_diff, top[0]->shape(0), top[0]->shape(1),
						num_spatial_axes_,
            input_shape_d, pool_shape_d, kernel_shape_d, stride_shape_d, pad_shape_d, bottom_diff);
  	/**/
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    // NOLINT_NEXT_LINE(whitespace/operators)
  	/* for ND support
    StoPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, rand_idx_.gpu_data(), top_diff,
        top[0]->num(), channels_, height_, width_, pooled_height_,
        pooled_width_, kernel_h_, kernel_w_, stride_h_, stride_w_,
        bottom_diff);
    */
  	LOG(FATAL) << "PoolingParameter_PoolMethod_STOCHASTIC is not implemented for ND yet.";
  	/**/
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;

/* for ND support */
  CUDA_CHECK(cudaFree(input_shape_d));
  CUDA_CHECK(cudaFree(pool_shape_d));
  CUDA_CHECK(cudaFree(kernel_shape_d));
  CUDA_CHECK(cudaFree(stride_shape_d));
  CUDA_CHECK(cudaFree(pad_shape_d));
/**/
}


INSTANTIATE_LAYER_GPU_FUNCS(PoolingLayer);


}  // namespace caffe
