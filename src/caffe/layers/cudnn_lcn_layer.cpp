#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_lcn_layer.hpp"

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
void CuDNNLCNLayer<Dtype, MItype, MOtype>::LayerSetUp(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  LRNLayer<Dtype, MItype, MOtype>::LayerSetUp(bottom, top);

  CUDNN_CHECK(cudnnCreate(&handle_));
  CUDNN_CHECK(cudnnCreateLRNDescriptor(&norm_desc_));
  cudnn::createTensorNdDesc<Dtype>(&bottom_desc_);
  cudnn::createTensorNdDesc<Dtype>(&top_desc_);

  // create a LRN handle
  handles_setup_ = true;

  size_ = this->layer_param().lrn_param().local_size();
  pre_pad_ = (size_ - 1) / 2;
  alpha_ = this->layer_param().lrn_param().alpha();
  beta_ = this->layer_param().lrn_param().beta();
  k_ = this->layer_param().lrn_param().k();
}

template<typename Dtype, typename MItype, typename MOtype>
void CuDNNLCNLayer<Dtype, MItype, MOtype>::Reshape(
    const vector<Blob<MItype>*>& bottom,
    const vector<Blob<MOtype>*>& top) {
  LRNLayer<Dtype, MItype, MOtype>::Reshape(bottom, top);
  vector<int_tp> shape;

  shape.push_back(bottom[0]->num());
  shape.push_back(this->channels_);
  shape.push_back(this->height_);
  shape.push_back(this->width_);


  const int_tp* shape_ptr = &shape[0];

  cudnn::setTensorNdDesc<Dtype>(&bottom_desc_, 4, shape_ptr);
  cudnn::setTensorNdDesc<Dtype>(&top_desc_, 4, shape_ptr);
  CUDNN_CHECK(cudnnSetLRNDescriptor(norm_desc_, size_, alpha_, beta_, k_));

  // allocate / reallocate tempData buffers
  uint_tp totalSizeInBytes = sizeof(Dtype)*bottom[0]->num()* \
                            this->channels_*this->height_*this->width_;

  if (totalSizeInBytes > tempDataSize) {
    tempDataSize = totalSizeInBytes;

    cudaFree(tempData1);
    cudaFree(tempData2);

    // allocate new buffers
    CUDA_CHECK(cudaMalloc(&tempData1, totalSizeInBytes));
    CUDA_CHECK(cudaMalloc(&tempData2, totalSizeInBytes));
  }
}

template<typename Dtype, typename MItype, typename MOtype>
CuDNNLCNLayer<Dtype, MItype, MOtype>::~CuDNNLCNLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  cudnnDestroyTensorDescriptor(bottom_desc_);
  cudnnDestroyTensorDescriptor(top_desc_);

  // destroy LRN handle
  cudnnDestroy(handle_);

  // free temp buffers
  cudaFree(tempData1);
  cudaFree(tempData2);
}

INSTANTIATE_CLASS_3T_GUARDED(CuDNNLCNLayer, (float), (float), (float));
INSTANTIATE_CLASS_3T_GUARDED(CuDNNLCNLayer, (double), (double), (double));

}   // namespace caffe
#endif
