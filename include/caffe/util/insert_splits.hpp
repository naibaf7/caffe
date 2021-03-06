#ifndef CAFFE_UTIL_INSERT_SPLITS_HPP_
#define CAFFE_UTIL_INSERT_SPLITS_HPP_

#include <string>

#include "caffe/proto/caffe.pb.h"

namespace caffe {

// Copy NetParameters with SplitLayers added to replace any shared bottom
// blobs with unique bottom blobs provided by the SplitLayer.
void InsertSplits(const NetParameter& param, NetParameter* param_split);

void ConfigureSplitLayer(const string& layer_name, const string& blob_name,
       const int_tp blob_idx, const int_tp split_count, const float loss_weight,
       LayerParameter* split_layer_param, const DataType bottom_data_type,
       const DataType top_data_type, const QuantizerParameter* ref_quant_param);

string SplitLayerName(const string& layer_name, const string& blob_name,
    const int_tp blob_idx);

string SplitBlobName(const string& layer_name, const string& blob_name,
    const int_tp blob_idx, const int_tp split_idx);

}  // namespace caffe

#endif  // CAFFE_UTIL_INSERT_SPLITS_HPP_
