#ifndef CAFFE_SRPRNN_LAYER_HPP_
#define CAFFE_SRPRNN_LAYER_HPP_

#include <map>
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/recurrent_layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/greentea/libdnn.hpp"

namespace caffe {

typedef enum {
  SRPRNN_GRID_CARTESIAN                = 0,
  SRPRNN_GRID_HEXTILES                 = 1
} srprnnGrid_t;

typedef enum {
  SRPRNN_ACTIVATION_NONE               = 0,
  SRPRNN_ACTIVATION_RELU               = 1,
  SRPRNN_ACTIVATION_TANH               = 2,
  SRPRNN_ACTIVATION_SIGMOID            = 3
} srprnnActivation_t;

typedef enum {
  SRPRNN_OP_ADD                        = 0,
  SRPRNN_OP_MUL                        = 1
} srprnnOp_t;


typedef enum {
  SRPRNN_RESTRICT_FREE                 = 0,
  SRPRNN_RESTRICT_FIXED_SIGN           = 1,
  SRPRNN_RESTRICT_FIXED_VALUE          = 2
} srprnnRestrict_t;

struct SRPRNNIOConfig {
  SRPRNNIOConfig() {}
  std::string name;
  int_tp temp_offset;
  std::vector<int_tp> offset;
  float weight;
  srprnnRestrict_t weight_restrict;
  int_tp weight_mem_off;
};

struct SRPRNNNeuronConfig {
  SRPRNNNeuronConfig() {}
  std::string name;
  srprnnOp_t operation;
  srprnnActivation_t activation;
  std::vector<SRPRNNIOConfig> inputs;
  std::vector<SRPRNNIOConfig> outputs;
  float bias;
  srprnnRestrict_t bias_restrict;
  std::vector<int_tp> periodicity;
  int_tp neuron_off;
};

struct SRPRNNConfig {
  SRPRNNConfig() {}
  int_tp fw_phases;
  int_tp bw_phases;
  int_tp backprop_steps;
  int_tp batch_size;
  int_tp temp_steps;
  int_tp weight_count;
  device* dev_ptr = nullptr;
  srprnnGrid_t grid;
  std::vector<int_tp> shape;
  bool bias_term = false;
  bool fast_unsafe_math = false;
  bool weights_backward = true;
  bool bias_backward = true;
  std::vector<SRPRNNNeuronConfig> neurons;
  std::vector<std::string> input_neurons;
  std::vector<std::string> output_neurons;
  std::vector<std::string> export_neurons;
  std::map<std::string, int_tp> neuron_offsets;
  std::vector<int_tp> in_shape;
  std::vector<int_tp> out_shape;
  std::vector<int_tp> export_shape;
};


template <typename Dtype>
class SRPRNN : public LibDNN<Dtype> {
 public:
  explicit SRPRNN(SRPRNNConfig config);
  void InitWeight(Dtype* cpu_weight_data,
                  Dtype* cpu_weight_restrict_data);
  void InitBias(Dtype* cpu_bias_data,
                Dtype* cpu_bias_restrict_data);
  void Forward(Dtype* flag_data,
               const Dtype* bottom_data, const Dtype* weight,
               const Dtype* bias, Dtype* top_data,
               Dtype* export_data);
  void Backward(Dtype* flag_data,
                const Dtype* top_data, const Dtype* top_diff,
                const Dtype* weight, Dtype* weight_diff,
                const Dtype* bias, Dtype* bias_diff,
                const Dtype* bottom_data, Dtype* bottom_diff);
  void ResetTime();
  void GenerateKernels();
  std::string string_identifier();
  std::string generate_defs();
  void export_config();
  std::string generate_fw_defs();
  std::string generate_bw_defs();
  std::string generate_fw_kernels(std::string name);
  std::string generate_bw_kernels(std::string name);
  std::string generate_wgc_kernels(std::string name);
  const SRPRNNConfig get_config();

 private:
  std::string relu_fw(std::string data_in,
                      std::string data_out,
                      Dtype neg_slope);
  std::string relu_bw(std::string data_in,
                      std::string diff_in,
                      std::string diff_out,
                      Dtype neg_slope);
  std::string tanh_fw(std::string data_in,
                      std::string data_out);
  std::string tanh_bw(std::string data_out,
                      std::string diff_in,
                      std::string diff_out);
  std::string sigmoid_fw(std::string data_in,
                      std::string data_out);
  std::string sigmoid_bw(std::string data_out,
                      std::string diff_in,
                      std::string diff_out);
  // SRPRNN Configuration
  SRPRNNConfig config_;
  int_tp base_;
  bool check_weights_;

  // Required memory blobs
  std::shared_ptr<Blob<Dtype>> buf_;
  std::shared_ptr<Blob<Dtype>> wg_ref_;
  std::shared_ptr<Blob<Dtype>> bias_ref_;
  std::shared_ptr<Blob<Dtype>> wg_restrict_;
  std::shared_ptr<Blob<Dtype>> bias_restrict_;
};


template <typename Dtype>
class SRPRNNLayer : public Layer<Dtype>{
 public:
  explicit SRPRNNLayer(const LayerParameter& param)
  : Layer<Dtype>(param) {}

  virtual inline const char* type() const { return "SRPRNN"; }

 protected:
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

  // Input is the bottom data blob plus the function flag blob (optional)
  virtual inline int_tp MinBottomBlobs() const { return 1; }
  virtual inline int_tp MaxBottomBlobs() const { return 2; }
  // Output is the main output neurons plus the export neurons (optional)
  virtual inline int_tp MinNumTopBlobs() const { return 1; }
  virtual inline int_tp MaxNumTopBlobs() const { return 2; }
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

 private:
  shared_ptr<SRPRNN<Dtype>> srprnn_;
};

}  // namespace caffe

#endif  // CAFFE_RNN_LAYER_HPP_
