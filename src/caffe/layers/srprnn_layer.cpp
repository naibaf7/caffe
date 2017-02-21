// Sparse repeated pattern recurrent neural network layer

#include <algorithm>
#include <string>
#include <vector>
#include "caffe/layers/srprnn_layer.hpp"

namespace caffe {

template<typename Dtype>
SRPRNN<Dtype>::SRPRNN(SRPRNNConfig config) {
  config_ = config;
  LibDNN<Dtype>::dev_ptr_ = config.dev_ptr;
  LibDNN<Dtype>::fast_unsafe_math_ = config.fast_unsafe_math;
  int_tp dims = config.shape.size();
  int_tp spatial_dims = dims - 2;

  int_tp k = 0;
  for (int_tp i = 0; i < config_.neurons.size(); ++i) {
    config_.neurons[i].neuron_off = i;
    for (int_tp j = 0; j < config.neurons[i].inputs.size(); ++j) {
      config_.neurons[i].inputs[j].input_off = k;
      ++k;
    }
  }

  GenerateKernels();
  LibDNN<Dtype>::CompileKernels();
}

template<typename Dtype>
const LibDNNConvConfig LibDNNConv<Dtype>::get_config() {
  return config_;
}

template<typename Dtype>
std::string SRPRNN<Dtype>::string_identifier() {
  return "SRPRNN";
}

template<typename Dtype>
void SRPRNN<Dtype>::Forward(const Dtype* flag_data,
                            const Dtype* bottom_data,
                            const Dtype* weight,
                            const Dtype* bias, Dtype* top_data,
                            Dtype* export_data) {
  for (int_tp phase = 0; phase < phases_; ++phase) {
    // TODO
  }
}

template<typename Dtype>
void SRPRNN<Dtype>::Backward(const Dtype* flag_data,
                             const Dtype* top_data, const Dtype* top_diff,
                             const Dtype* weight, Dtype* weight_diff,
                             const Dtype* bias, Dtype* bias_diff,
                             const Dtype* bottom_data, Dtype* bottom_diff) {
  for (int_tp phase = 0; phase < phases_; ++phase) {
    // TODO
  }
}

template<typename Dtype>
void SRPRNN<Dtype>::ResetTime() {
}

template<typename Dtype>
std::string SRPRNN<Dtype>::generate_fw_defs() {
  std::stringstream ss;
  // LibDNN<Dtype>::add_def(ss, "v_imsi", );
  // LibDNN<Dtype>::add_def(ss, "v_imso", );
  // LibDNN<Dtype>::add_Def(ss, "v_imse", );
  return ss.str();
}

template<typename Dtype>
std::string SRPRNN<Dtype>::generate_bw_defs() {
  std::stringstream ss;
  // LibDNN<Dtype>::add_def(ss, "v_imsi", );
  // LibDNN<Dtype>::add_def(ss, "v_imso", );
  // LibDNN<Dtype>::add_Def(ss, "v_imse", );
  return ss.str();
}

template<typename Dtype>
std::string SRPRNN<Dtype>::generate_wg_defs() {
  std::stringstream ss;
  // TODO
  return ss.str();
}

template<typename Dtype>
std::string SRPRNN<Dtype>::generate_fw_kernels(std::string name) {
  std::stringstream ss;
  std::vector<std::string>::iterator it;

  std::vector<std::string> processed_neurons;
  int_tp phase = 0;
  int_tp last_size = 0;
  do {
    ss << "__kernel" << std::endl;
    ss << "void " << name << "_" << phase << "(";
    ss << "__global const Dtype* __restrict im_in, ";
    ss << "__global Dtype* __restrict im_out, ";
    ss << "__global Dtype* __restrict im_exp, ";
    ss << "__global Dtype* __restrict buf, ";
    ss << "__global const Dtype* __restrict wg, ";
    ss << "__global const Dtype* __restrict bias, ";
    ss << "const int_tp batch";
    ss << ") {" << std::endl;
    ss << "__global const Dtype* im_in_ptr = im_in + batch * v_imsi;"
       << std::endl;
    ss << "__global Dtype* im_out_ptr = im_out + batch * v_imso;"
       << std::endl;
    ss << "__global Dtype* im_exp_ptr = im_exp + batch * v_imse;"
       << std::endl;
    std::vector<std::string> current_neurons;
    for (int_tp i = 0; i < config_.neurons.size(); ++i) {
      // Check if neuron hasn't been processed yet
      if (std::find(current_neurons.begin(), current_neurons.end(),
                    config_.neurons[i].name) == current_neurons.end() &&
          std::find(processed_neurons.begin(), processed_neurons.end(),
                    config_.neurons[i].name) == current_neurons.end()) {
        // Check if all inputs to the neuron are already processed
        bool all_inputs_processed = true;
        std::stringstream ss_sub;
        ss_sub << "Dtype " << config_.neurons[i].name
               << "_reg = ";
        // Check if the neuron is an input neuron
        if (std::find(config_.input_neurons.begin(),
                      config_.input_neurons.end(),
                      config_.neurons[i].name) ==
                          config_.input_neurons.end()) {
          ss_sub << "im_in[]" << std::endl;
        } else {
          ss_sub << "0.0;" << std::endl;
        }
        for (int_tp j = 0; j < config_.neurons[i].inputs.size(); ++j) {
          if (std::find(processed_neurons.begin(), processed_neurons.end(),
                        config_.neurons[i].inputs[j].name) ==
                            processed_neurons.end()) {
            ss_sub << "" << std::endl;
            all_inputs_processed = false;
          }
        }
        // Store result to internal buffer memory
        ss_sub << "buf_ptr[] = " << config_.neurons[i].name
               << "_reg;" << std::endl;
        // Check if the neuron is an output neuron
        it = std::find(config_.output_neurons.begin(),
                       config_.output_neurons.end(),
                       config_.neurons[i].name);
        if (it == config_.output_neurons.end()) {
          ss_sub << "im_out_ptr[" << (it - config_.output_neurons.begin())
                 << "] = " << config_.neurons[i].name
                 << "_reg;" << std::endl;
        }
        // Check if the neuron is an export neuron
        if (std::find(config_.export_neurons.begin(),
                      config_.export_neurons.end(),
                      config_.neurons[i].name) ==
                          config_.export_neurons.end()) {
          ss_sub << "im_exp_ptr[] = " << config_.neurons[i].name
                 << "_reg;" << std::endl;
        }
        if (all_inputs_processed) {
          // Add the neuron
          ss << ss_sub.str();
          current_neurons.push_back(config_.neurons[i].name);
          std::cout << config_.neurons[i].name << std::endl;
        }
      }
    }
    ss << "}" << std::endl;
    last_size = processed_neurons.size();
    processed_neurons.insert(processed_neurons.end(),
                             current_neurons.begin(),
                             current_neurons.end());
    ++phase;
  } while (processed_neurons.size() < config_.neurons.size() ||
      last_size != processed_neurons.size());

  phases_ = phase;

  return ss.str();
}

template<typename Dtype>
std::string SRPRNN<Dtype>::generate_bw_kernels(std::string name) {
  std::stringstream ss;
  // TODO
  return ss.str();
}

template<typename Dtype>
std::string SRPRNN<Dtype>::generate_wg_kernels(std::string name) {
  std::stringstream ss;
  // TODO
  return ss.str();
}

template<typename Dtype>
void SRPRNN<Dtype>::GenerateKernels() {
  std::stringstream ss;

  ss << LibDNN<Dtype>::generate_header();
  ss << generate_fw_defs();
  ss << generate_fw_kernels("srprnn_forward");
  ss << generate_bw_defs();
  ss << generate_bw_kernels("srprnn_backward");
  ss << generate_wg_defs();
  ss << generate_wg_kernels("srprnn_weights");

  // Write complete kernel string
  LibDNN<Dtype>::kernel_ = ss.str();
}

template<typename Dtype>
void SRPRNN<Dtype>::InitWeight(Dtype* cpu_weight_data) {
  int_tp off = 0;
  for (int_tp i = 0; i < config_.neurons.size(); ++i) {
    for (int_tp j = 0; j < config_.neurons[i].inputs.size(); ++j) {
      // Initial weight
      cpu_weight_data[off] = config_.neurons[i].inputs[j].weight;
      ++off;
    }
  }
}

template<typename Dtype>
void SRPRNN<Dtype>::InitBias(Dtype* cpu_bias_data) {
  for (int_tp i = 0; i < config_.neurons.size(); ++i) {
    // Initial bias
    cpu_bias_data[i] = config_.neurons[i].bias;
  }
}


INSTANTIATE_CLASS(SRPRNN);


template <typename Dtype>
void SRPRNNLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  Reshape(bottom, top);
}

template <typename Dtype>
void SRPRNNLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  bool shapes_changed = false;


  if (srprnn_.get() == nullptr || shapes_changed) {
    SRPRNNConfig config;
    SRPRNNParameter param = this->layer_param().srprnn_param();

    config.dev_ptr = this->device_;
    config.fast_unsafe_math = true;
    config.bias_backward = true;
    config.weights_backward = true;

    switch (param.grid()) {
      case SRPRNN_GRID_CARTESIAN:
        config.grid = SRPRNN_GRID_CARTESIAN;
        break;
      case SRPRNN_GRID_HEXTILES:
        config.grid = SRPRNN_GRID_HEXTILES;
        break;
    }

    int_tp max_temp_offset = 0;
    int_tp weight_count = 0;

    std::vector<SRPRNNNeuronConfig> neurons;
    for (int_tp i = 0; i < param.hidden_neuron_size(); ++i) {
      SRPRNNNeuron neuron_param = param.hidden_neuron(i);
      SRPRNNNeuronConfig neuron;
      neuron.name = neuron_param.name();
      neuron.bias = neuron_param.bias();
      std::vector<int_tp> periodicity;
      for (int_tp j = 0; j < neuron_param.periodicity_size(); ++j) {
        periodicity.push_back(neuron_param.periodicity(j));
        neuron.periodicity = periodicity;
      }
      switch (neuron_param.operation()) {
        case SRPRNNNeuron_SRPRNNOperation_ADD:
          neuron.operation = SRPRNN_OP_ADD;
          break;
        case SRPRNNNeuron_SRPRNNOperation_MUL:
          neuron.operation = SRPRNN_OP_MUL;
          break;
      }
      switch (neuron_param.activation()) {
        case SRPRNNNeuron_SRPRNNActivation_ReLU:
          neuron.activation = SRPRNN_ACTIVATION_RELU;
          break;
        case SRPRNNNeuron_SRPRNNActivation_TanH:
          neuron.activation = SRPRNN_ACTIVATION_TANH;
          break;
        case SRPRNNNeuron_SRPRNNActivation_Sigmoid:
          neuron.activation = SRPRNN_ACTIVATION_SIGMOID;
          break;
      }
      std::vector<SRPRNNInputConfig> inputs;
      for (int_tp j = 0; j < neuron_param.input_size(); ++j) {
        SRPRNNInput input_param = neuron_param.input(j);
        SRPRNNInputConfig input;
        std::vector<int_tp> offsets;
        for (int k = 0; k < input_param.offset_size(); ++k) {
          offsets.push_back(input_param.offset(k));
        }
        input.temp_offset = input_param.temp_offset();
        max_temp_offset = std::max(max_temp_offset, input.temp_offset);
        input.weight = input_param.weight();
        input.name = input_param.name();
        inputs.push_back(input);
        // Count number of synaptic connections (= weights)
        ++weight_count;
      }
      neurons.push_back(neuron);
    }
    config.neurons = neurons;

    std::vector<std::string> input_neurons;
    for (int_tp i = 0; i < param.input_neuron_size(); ++i) {
      input_neurons.push_back(param.input_neuron(i));
    }
    config.input_neurons = input_neurons;

    std::vector<std::string> output_neurons;
    for (int_tp i = 0; i < param.output_neuron_size(); ++i) {
      output_neurons.push_back(param.output_neuron(i));
    }
    config.output_neurons = output_neurons;

    std::vector<std::string> export_neurons;
    for (int_tp i = 0; i < param.export_neuron_size(); ++i) {
      export_neurons.push_back(param.export_neuron(i));
    }
    config.export_neurons = export_neurons;


    // Need to have as many backprop through time steps as specified,
    // and at least as many as time offsets of input neurons
    config.backprop_steps = std::max((int_tp)param.backprop_steps(),
                                     (int_tp)max_temp_offset);

    // Temporal dimension buffer size needs to be the number
    // of backpropagation steps plus the batch size.
    config.temp_steps = config.backprop_steps + bottom[0]->shape(0);

    // Number of input neurons must match the feature maps
    CHECK_EQ(config.input_neurons.size(), bottom[0]->shape(1));

    // Weights for all synaptic connections
    std::vector<int_tp> weight_shape(1);
    weight_shape[0] = weight_count;

    // Every neuron has a bias value
    std::vector<int_tp> bias_shape(1);
    bias_shape[0] = config.neurons.size();

    if (this->blobs_.size() < 2) {
      this->blobs_.resize(2);
    }

    SRPRNN<Dtype>* srprnn = new SRPRNN<Dtype>(config);


    // Initialize the weight blob
    if (this->blobs_[0] == nullptr) {
      this->blobs_[0].reset(new Blob<Dtype>(weight_shape, this->device_));
      Dtype* cpu_data = this->blobs_[0].get()->mutable_cpu_data();
      srprnn->InitWeight(cpu_data);
    }

    // Initialize the bias blob
    if (this->blobs_[1] == nullptr) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape, this->device_));
      Dtype* cpu_data = this->blobs_[1].get()->mutable_cpu_data();
      srprnn->InitBias(cpu_data);
    }


    srprnn_.reset(srprnn);
  }
}



template<typename Dtype>
void SRPRNNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template<typename Dtype>
void SRPRNNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                      const vector<bool>& propagate_down,
                                      const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(SRPRNNLayer);
#endif

template<typename Dtype>
void SRPRNNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  const Dtype* bias = this->blobs_[1]->gpu_data();

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();

  const Dtype* flag_data = nullptr;
  if (bottom.size() > 1) {
    flag_data = bottom[1]->cpu_data();
  }

  Dtype* export_data = nullptr;

  if (top.size() > 1) {
    export_data = top[1]->mutable_gpu_data();
  }

  srprnn_.get()->Forward(flag_data, bottom_data, weight, bias, top_data,
                         export_data);
}

template<typename Dtype>
void SRPRNNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                      const vector<bool>& propagate_down,
                                      const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  const Dtype* bias = this->blobs_[1]->gpu_data();
  Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();

  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  const Dtype* flag_data = nullptr;
  if (bottom.size() > 1) {
    flag_data = bottom[1]->cpu_data();
  }

  srprnn_.get()->Backward(flag_data,
                          top_data, top_diff,
                          weight, weight_diff,
                          bias, bias_diff,
                          bottom_data, bottom_diff);
}

INSTANTIATE_CLASS(SRPRNNLayer);
REGISTER_LAYER_CLASS(SRPRNN);

}  // namespace caffe
