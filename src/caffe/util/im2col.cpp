#include <vector>

#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// Function uses casting from int to unsigned to compare if value of
// parameter a is greater or equal to zero and lower than value of
// parameter b. The b parameter is of type signed and is always positive,
// therefore its value is always lower than 0x800... where casting
// negative value of a parameter converts it to value higher than 0x800...
// The casting allows to use one condition instead of two.
inline bool is_a_ge_zero_and_a_lt_b(int_tp a, int_tp b) {
  return static_cast<uint_tp>(a) < static_cast<uint_tp>(b);
}

template<typename Dtype>
void im2col_cpu(const Dtype* data_im, const int_tp channels,
                const int_tp height, const int_tp width, const int_tp kernel_h,
                const int_tp kernel_w, const int_tp pad_h, const int_tp pad_w,
                const int_tp stride_h, const int_tp stride_w,
                const int_tp dilation_h, const int_tp dilation_w,
                Dtype* data_col, const QuantizerValues* const data_quant) {
  const Dtype zero = data_quant ? data_quant->template get_zero<Dtype>()
                                : Dtype(0);
  const int_tp output_h = (height + 2 * pad_h
      - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int_tp output_w =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int_tp channel_size = height * width;
  for (int_tp channel = channels; channel--; data_im += channel_size) {
    for (int_tp kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int_tp kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int_tp input_row = -pad_h + kernel_row * dilation_h;
        for (int_tp output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (int_tp output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = zero;
            }
          } else {
            int_tp input_col = -pad_w + kernel_col * dilation_w;
            for (int_tp output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                *(data_col++) = zero;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

// Explicit instantiation
INSTANTIATE_FUNC_1T_GUARDED(im2col_cpu, PROTO_TYPES);

template<typename Dtype>
inline void im2col_nd_core_cpu(const Dtype* data_input, const bool im2col,
                               const int_tp num_spatial_axes,
                               const int_tp* im_shape, const int_tp* col_shape,
                               const int_tp* kernel_shape, const int_tp* pad,
                               const int_tp* stride, const int_tp* dilation,
                               Dtype* data_output,
                               const QuantizerValues* const data_quant) {
  const Dtype zero = data_quant ? data_quant->template get_zero<Dtype>()
                                : Dtype(0);
  if (!im2col) {
    int_tp im_size = im_shape[0];
    for (int_tp i = 0; i < num_spatial_axes; ++i) {
      im_size *= im_shape[1 + i];
    }
    caffe_set(im_size, zero, data_output);
  }
  int_tp kernel_size = 1;
  for (int_tp i = 0; i < num_spatial_axes; ++i) {
    kernel_size *= kernel_shape[i];
  }
  const int_tp channels_col = col_shape[0];
  vector<int_tp> d_offset(num_spatial_axes, 0);
  vector<int_tp> d_iter(num_spatial_axes, 0);
  for (int_tp c_col = 0; c_col < channels_col; ++c_col) {
    // Loop over spatial axes in reverse order to compute a per-axis offset.
    int_tp offset = c_col;
    for (int_tp d_i = num_spatial_axes - 1; d_i >= 0; --d_i) {
      if (d_i < num_spatial_axes - 1) {
        offset /= kernel_shape[d_i + 1];
      }
      d_offset[d_i] = offset % kernel_shape[d_i];
    }
    for (bool incremented = true; incremented;) {
      // Loop over spatial axes in forward order to compute the indices in the
      // image and column, and whether the index lies in the padding.
      int_tp index_col = c_col;
      int_tp index_im = c_col / kernel_size;
      bool is_padding = false;
      for (int_tp d_i = 0; d_i < num_spatial_axes; ++d_i) {
        const int_tp d = d_iter[d_i];
        const int_tp d_im = d * stride[d_i] - pad[d_i]
            + d_offset[d_i] * dilation[d_i];
        is_padding |= d_im < 0 || d_im >= im_shape[d_i + 1];
        index_col *= col_shape[d_i + 1];
        index_col += d;
        index_im *= im_shape[d_i + 1];
        index_im += d_im;
      }
      if (im2col) {
        if (is_padding) {
          data_output[index_col] = zero;
        } else {
          data_output[index_col] = data_input[index_im];
        }
      } else if (!is_padding) {  // col2im
        data_output[index_im] += data_input[index_col];
      }
      // Loop over spatial axes in reverse order to choose an index,
      // like counting.
      incremented = false;
      for (int_tp d_i = num_spatial_axes - 1; d_i >= 0; --d_i) {
        const int_tp d_max = col_shape[d_i + 1];
        DCHECK_LT(d_iter[d_i], d_max);
        if (d_iter[d_i] == d_max - 1) {
          d_iter[d_i] = 0;
        } else {  // d_iter[d_i] < d_max - 1
          ++d_iter[d_i];
          incremented = true;
          break;
        }
      }
    }  // while(incremented) {
  }  // for (int_tp c = 0; c < channels_col; ++c) {
}

template<typename Dtype>
void im2col_nd_cpu(const Dtype* data_im, const int_tp num_spatial_axes,
                   const int_tp* im_shape, const int_tp* col_shape,
                   const int_tp* kernel_shape, const int_tp* pad,
                   const int_tp* stride, const int_tp* dilation,
                   Dtype* data_col, const QuantizerValues* const data_quant) {
  const bool kIm2Col = true;
  im2col_nd_core_cpu(data_im, kIm2Col, num_spatial_axes, im_shape, col_shape,
                     kernel_shape, pad, stride, dilation, data_col, data_quant);
}

INSTANTIATE_FUNC_1T_GUARDED(im2col_nd_cpu, PROTO_TYPES);

template<typename Dtype>
void col2im_cpu(const Dtype* data_col, const int_tp channels,
                const int_tp height, const int_tp width, const int_tp kernel_h,
                const int_tp kernel_w, const int_tp pad_h, const int_tp pad_w,
                const int_tp stride_h, const int_tp stride_w,
                const int_tp dilation_h, const int_tp dilation_w,
                Dtype* data_im) {
  caffe_set(height * width * channels, Dtype(0), data_im);
  const int_tp output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int_tp output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int_tp channel_size = height * width;
  for (int_tp channel = channels; channel--; data_im += channel_size) {
    for (int_tp kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int_tp kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int_tp input_row = -pad_h + kernel_row * dilation_h;
        for (int_tp output_rows = output_h; output_rows; output_rows--) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            data_col += output_w;
          } else {
            int_tp input_col = -pad_w + kernel_col * dilation_w;
            for (int_tp output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                data_im[input_row * width + input_col] += *data_col;
              }
              data_col++;
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

// Explicit instantiation
INSTANTIATE_FUNC_1T_GUARDED(col2im_cpu, PROTO_TYPES);


template<typename Dtype>
void col2im_nd_cpu(const Dtype* data_col, const int_tp num_spatial_axes,
                   const int_tp* im_shape, const int_tp* col_shape,
                   const int_tp* kernel_shape, const int_tp* pad,
                   const int_tp* stride, const int_tp* dilation,
                   Dtype* data_im) {
  const bool kIm2Col = false;
  im2col_nd_core_cpu(data_col, kIm2Col, num_spatial_axes, im_shape, col_shape,
                     kernel_shape, pad, stride, dilation, data_im, nullptr);
}

// Explicit instantiation
INSTANTIATE_FUNC_1T_GUARDED(col2im_nd_cpu, PROTO_TYPES);


}  // namespace caffe
