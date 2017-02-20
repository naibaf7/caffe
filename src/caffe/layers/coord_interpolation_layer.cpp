#include <cfloat>
#include <vector>

#include "caffe/layers/coord_interpolation_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CoordInterpolationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  interp_mode_ = this->layer_param_.coord_param().interpol();
  scale_mode_ = this->layer_param_.coord_param().scale();
  src_grid_type_ = this->layer_param_.coord_param().srcgrid();
  dst_grid_type_ = this->layer_param_.coord_param().dstgrid();
}

template <typename Dtype>
void CoordInterpolationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  std::vector<int_tp> shape = bottom[0]->shape();
  if (this->layer_param_.coord_param().outdim_size() > 0) {
    shape[2] = this->layer_param_.coord_param().outdim(std::max(0,
               this->layer_param_.coord_param().outdim_size()-2));
    shape[3] = this->layer_param_.coord_param().outdim(std::max(0,
               this->layer_param_.coord_param().outdim_size()-1));
  } else {
    shape[2] = this->layer_param_.coord_param().outheight();
    shape[3] = this->layer_param_.coord_param().outwidth();
  }
  top[0]->Reshape(shape);
}

template <typename Dtype>
void CoordInterpolationLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void CoordInterpolationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(EltwiseLayer);
#endif

INSTANTIATE_CLASS(CoordInterpolationLayer);
REGISTER_LAYER_CLASS(CoordInterpolation);

}  // namespace caffe
