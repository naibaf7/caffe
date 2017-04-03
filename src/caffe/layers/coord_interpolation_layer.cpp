#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/coord_interpolation_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CoordInterpolationLayer<Dtype>::LayerSetUp(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  interp_mode_ = this->layer_param_.coord_param().interpol();
  scale_mode_ = this->layer_param_.coord_param().scale();
  src_grid_type_ = this->layer_param_.coord_param().srcgrid();
  dst_grid_type_ = this->layer_param_.coord_param().dstgrid();
}

template <typename Dtype>
void CoordInterpolationLayer<Dtype>::Reshape(
      const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  std::vector<int_tp> shape = bottom[0]->shape();
  if (this->layer_param_.coord_param().outdim_size() > 0) {
    if (this->layer_param_.coord_param().outdim_size() > 2) {
      shape[1] = this->layer_param_.coord_param().outdim(std::max(0,
                 this->layer_param_.coord_param().outdim_size()-3));
    }
    shape[2] = this->layer_param_.coord_param().outdim(std::max(0,
               this->layer_param_.coord_param().outdim_size()-2));
    shape[3] = this->layer_param_.coord_param().outdim(std::max(0,
               this->layer_param_.coord_param().outdim_size()-1));
  } else {
    if (this->layer_param_.coord_param().num_output() > 0) {
      shape[1] = this->layer_param_.coord_param().num_output();
    }
    shape[2] = this->layer_param_.coord_param().outheight();
    shape[3] = this->layer_param_.coord_param().outwidth();
  }
  top[0]->Reshape(shape);
  if (interp_mode_ != CoordParameter_InterpolMode_OMMATIDIA) {
    CHECK_EQ(shape[1], bottom[0]->shape(1))
        << "Input and output feature map count needs to be equal.";
  } else {
    CHECK_EQ(src_grid_type_, GridType::CARTESIAN) << "Ommatidia sampling"
        << "only supported with cartesian source.";
    CHECK_EQ(dst_grid_type_, GridType::HEXTILES) << "Ommatidia sampling"
        << "only supported with hextiles destination.";
    CHECK_EQ(bottom[0]->shape(1), 3) << "Ommatidia sampling only supported with"
        << "3 input feature maps";
    CHECK_EQ(shape[1], 8) << "Ommatidia sampling only supported with 8 output"
        << "feature maps";
  }
}

template <typename Dtype>
void CoordInterpolationLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void CoordInterpolationLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(EltwiseLayer);
#endif

INSTANTIATE_CLASS(CoordInterpolationLayer);
REGISTER_LAYER_CLASS(CoordInterpolation);

}  // namespace caffe
