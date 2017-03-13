// Sparse repeated pattern recurrent neural network layer

#include <algorithm>
#include <map>
#include <string>
#include <tuple>
#include <vector>
#include "caffe/layers/srprnn_layer.hpp"

namespace caffe {

template<typename Dtype>
SRPRNN<Dtype>::SRPRNN(SRPRNNConfig config) {
  config_ = config;
  check_weights_ = false;
  LibDNN<Dtype>::dev_ptr_ = config.dev_ptr;
  LibDNN<Dtype>::fast_unsafe_math_ = config.fast_unsafe_math;
  GenerateKernels();
  LibDNN<Dtype>::CompileKernels();

  // Prepare internal memory
  std::vector<int_tp> buf_shape;
  buf_shape.push_back(config_.temp_steps);       // Batch size + BPTT steps
  buf_shape.push_back(config_.neurons.size());   // Number of neurons
  buf_shape.push_back(config_.in_shape[2]);      // Height
  buf_shape.push_back(config_.in_shape[3]);      // Width

  buf_.reset(new Blob<Dtype>(buf_shape, this->dev_ptr_));
  this->SetMemory(buf_.get()->mutable_gpu_data(),
                  buf_.get()->count(), 0, 0.0);
  this->SetMemory(buf_.get()->mutable_gpu_diff(),
                  buf_.get()->count(), 0, 0.0);

  std::vector<int_tp> bias_shape;
  bias_shape.push_back(config_.neurons.size());  // Number of neurons
  bias_ref_.reset(new Blob<Dtype>(bias_shape, this->dev_ptr_));
  bias_restrict_.reset(new Blob<Dtype>(bias_shape, this->dev_ptr_));

  std::vector<int_tp> wg_shape;
  wg_shape.push_back(config_.weight_count);      // Number of inputs to neurons
  wg_ref_.reset(new Blob<Dtype>(wg_shape, this->dev_ptr_));
  wg_restrict_.reset(new Blob<Dtype>(wg_shape, this->dev_ptr_));

  // Initiate original weight and bias plus restrictions as reference
  InitWeight(wg_ref_.get()->mutable_cpu_data(),
             wg_restrict_.get()->mutable_cpu_data());
  InitBias(bias_ref_.get()->mutable_cpu_data(),
           bias_restrict_.get()->mutable_cpu_data());
}

template<typename Dtype>
const SRPRNNConfig SRPRNN<Dtype> ::get_config() {
  return config_;
}

template<typename Dtype>
std::string SRPRNN<Dtype>::string_identifier() {
  std::stringstream ss;
  ss << "SRPRNN_";
  if (std::is_same<Dtype, double>::value) {
    ss << "double_";
  } else {
    ss << "float_";
  }
  // Device name
  ss << LibDNN<Dtype>::dev_ptr_->name();
  return ss.str();
}

template<typename Dtype>
void SRPRNN<Dtype>::Forward(Dtype* flag_data,
                            const Dtype* bottom_data,
                            const Dtype* weight,
                            const Dtype* bias,
                            Dtype* top_data,
                            Dtype* export_data) {
  int_tp lws_x = 4;
  int_tp lws_y = 4;

  // Check if flags are available
  if (flag_data != nullptr) {
    // Test the reset flag
    if ((int_tp)flag_data[0] == 1) {
      // Reset current base and batch
      base_ = 0;
      // Reset the flag
      flag_data[0] = 0;
      this->SetMemory(buf_.get()->mutable_gpu_data(),
                      buf_.get()->count(), 0, 0.0);
    }
  }

  if (check_weights_) {
    const Dtype* wg_ref_data = wg_ref_.get()->gpu_data();
    const Dtype* wg_restrict_data = wg_restrict_.get()->gpu_data();
    const Dtype* bias_ref_data = bias_ref_.get()->gpu_data();
    const Dtype* bias_restrict_data = bias_restrict_.get()->gpu_data();

#ifdef USE_CUDA
    if (LibDNN<Dtype>::dev_ptr_->backend() == BACKEND_CUDA) {
      CUfunction kernel;
      cuModuleGetFunction(&kernel, LibDNN<Dtype>::cuda_module_,
                          "srprnn_weights_check");

      void *args[] = {&wg_ref_data, &weight,
                      &bias_ref_data, &bias,
                      &wg_restrict_data,
                      &bias_restrict_data};
      cuLaunchKernel(kernel,
                     (this->config_.neurons.size() + config_.weight_count - 1)
                                                  / lws_x + 1,  // Grid X
                     1,                                         // Grid Y
                     1,                                         // Grid Z
                     lws_x, 1, 1,                               // Local
                     0, NULL, args, 0);                         // Arguments
    }
#endif
#ifdef USE_GREENTEA
    if (LibDNN<Dtype>::dev_ptr_->backend() == BACKEND_OpenCL) {
      viennacl::ocl::kernel &kernel =
          LibDNN<Dtype>::ocl_program_.get_kernel("srprnn_weights_check");

      viennacl::ocl::context &ctx =
          viennacl::ocl::get_context(LibDNN<Dtype>::dev_ptr_->id());

      kernel.local_work_size(0, lws_x);
      kernel.local_work_size(1, 1);
      kernel.local_work_size(2, 1);

      kernel.global_work_size(0,
                ((this->config_.neurons.size() + config_.weight_count - 1)
                      / lws_x + 1) * lws_x);
      kernel.global_work_size(1, 1);
      kernel.global_work_size(2, 1);

      viennacl::ocl::enqueue(
          kernel(WrapHandle((cl_mem) wg_ref_data, &ctx),
                 WrapHandle((cl_mem) weight, &ctx),
                 WrapHandle((cl_mem) bias_ref_data, &ctx),
                 WrapHandle((cl_mem) bias, &ctx),
                 WrapHandle((cl_mem) wg_restrict_data, &ctx),
                 WrapHandle((cl_mem) bias_restrict_data, &ctx)),
          ctx.get_queue());
    }
#endif
    check_weights_ = false;
  }

  Dtype* buf_data = buf_.get()->mutable_gpu_data();

  for (int_tp batch = 0; batch < config_.batch_size; ++batch) {
    for (int_tp phase = 0; phase < config_.fw_phases; ++phase) {
  #ifdef USE_CUDA
      if (LibDNN<Dtype>::dev_ptr_->backend() == BACKEND_CUDA) {
        CUfunction kernel;
        cuModuleGetFunction(&kernel, LibDNN<Dtype>::cuda_module_,
                         ("srprnn_forward_" + std::to_string(phase)).c_str());

        void *args[] = {&bottom_data, &top_data, &export_data, &buf_data,
                        &weight, &bias, &base_, &batch};
        cuLaunchKernel(kernel,
                       (this->config_.shape[1] - 1) / lws_x + 1,  // Grid X
                       (this->config_.shape[0] - 1) / lws_y + 1,  // Grid Y
                       1,                                         // Grid Z
                       lws_x, lws_y, 1,                           // Local
                       0, NULL, args, 0);                         // Arguments
      }
  #endif
  #ifdef USE_GREENTEA
      if (LibDNN<Dtype>::dev_ptr_->backend() == BACKEND_OpenCL) {
        viennacl::ocl::kernel &kernel =
            LibDNN<Dtype>::ocl_program_.get_kernel(
                "srprnn_forward_" + std::to_string(phase));

        viennacl::ocl::context &ctx =
            viennacl::ocl::get_context(LibDNN<Dtype>::dev_ptr_->id());

        kernel.local_work_size(0, lws_x);
        kernel.local_work_size(1, lws_y);
        kernel.local_work_size(2, 1);

        kernel.global_work_size(0,
                          ((this->config_.shape[1] - 1) / lws_x + 1) * lws_x);
        kernel.global_work_size(1,
                          ((this->config_.shape[0] - 1) / lws_y + 1) * lws_y);

        kernel.global_work_size(2, 1);

        viennacl::ocl::enqueue(
            kernel(WrapHandle((cl_mem) bottom_data, &ctx),
                   WrapHandle((cl_mem) top_data, &ctx),
                   WrapHandle((cl_mem) export_data, &ctx),
                   WrapHandle((cl_mem) buf_data, &ctx),
                   WrapHandle((cl_mem) weight, &ctx),
                   WrapHandle((cl_mem) bias, &ctx),
                   base_, batch),
            ctx.get_queue());
      }
  #endif
    }
  }
  // Advance base
  base_ = (base_ + config_.batch_size) % config_.temp_steps;
}

template<typename Dtype>
void SRPRNN<Dtype>::Backward(Dtype* flag_data,
                             const Dtype* top_data, const Dtype* top_diff,
                             const Dtype* weight, Dtype* weight_diff,
                             const Dtype* bias, Dtype* bias_diff,
                             const Dtype* bottom_data, Dtype* bottom_diff) {
  int_tp lws_x = 4;
  int_tp lws_y = 4;

  int_tp bw_base = (config_.temp_steps + base_ - config_.batch_size)
                                  % config_.temp_steps;

  const Dtype* buf_data = buf_.get()->gpu_data();
  Dtype* buf_diff = buf_.get()->mutable_gpu_diff();

  this->SetMemory(buf_.get()->mutable_gpu_diff(),
                  buf_.get()->count(), 0, 0.0);

  // Compute backward pass from batch_size backwards to batch_size-temp_steps,
  // covering a total of temp_steps temporal backprop steps.
  for (int_tp batch = config_.batch_size - 1;
       batch >= config_.batch_size - config_.temp_steps; --batch) {
    for (int_tp phase = 0; phase < config_.bw_phases; ++phase) {
  #ifdef USE_CUDA
      if (LibDNN<Dtype>::dev_ptr_->backend() == BACKEND_CUDA) {
        CUfunction kernel;
        cuModuleGetFunction(&kernel, LibDNN<Dtype>::cuda_module_,
                         ("srprnn_backward_" + std::to_string(phase)).c_str());

        void *args[] = {&bottom_data, &bottom_diff, &top_data, &top_diff,
                        &buf_data, &buf_diff,
                        &weight, &weight_diff, &bias, &bias_diff,
                        &bw_base, &batch};
        cuLaunchKernel(kernel,
                       (this->config_.shape[1] - 1) / lws_x + 1,  // Grid X
                       (this->config_.shape[0] - 1) / lws_y + 1,  // Grid Y
                       1,                                         // Grid Z
                       lws_x, lws_y, 1,                           // Local
                       0, NULL, args, 0);                         // Arguments
      }
  #endif
  #ifdef USE_GREENTEA
      if (LibDNN<Dtype>::dev_ptr_->backend() == BACKEND_OpenCL) {
        viennacl::ocl::kernel &kernel =
            LibDNN<Dtype>::ocl_program_.get_kernel(
                "srprnn_backward_" + std::to_string(phase));

        viennacl::ocl::context &ctx =
            viennacl::ocl::get_context(LibDNN<Dtype>::dev_ptr_->id());

        kernel.local_work_size(0, lws_x);
        kernel.local_work_size(1, lws_y);
        kernel.local_work_size(2, 1);

        kernel.global_work_size(0,
                          ((this->config_.shape[1] - 1) / lws_x + 1) * lws_x);
        kernel.global_work_size(1,
                          ((this->config_.shape[0] - 1) / lws_y + 1) * lws_y);

        kernel.global_work_size(2, 1);

        viennacl::ocl::enqueue(
            kernel(WrapHandle((cl_mem) bottom_data, &ctx),
                   WrapHandle((cl_mem) bottom_diff, &ctx),
                   WrapHandle((cl_mem) top_data, &ctx),
                   WrapHandle((cl_mem) top_diff, &ctx),
                   WrapHandle((cl_mem) buf_data, &ctx),
                   WrapHandle((cl_mem) buf_diff, &ctx),
                   WrapHandle((cl_mem) weight, &ctx),
                   WrapHandle((cl_mem) weight_diff, &ctx),
                   WrapHandle((cl_mem) bias, &ctx),
                   WrapHandle((cl_mem) bias_diff, &ctx),
                   base_, batch),
            ctx.get_queue());
      }
  #endif
    }
  }

  // After a backward pass, the next forward pass (after weight update) must
  // check if the weights fulfill the requirements.
  check_weights_ = true;
}

template<typename Dtype>
void SRPRNN<Dtype>::ResetTime() {
}

template<typename Dtype>
std::string SRPRNN<Dtype>::generate_defs() {
  std::stringstream ss;
  int_tp size = config_.shape[0] * config_.shape[1];
  LibDNN<Dtype>::add_def(ss, "SRPRNN_RESTRICT_FREE", 0);
  LibDNN<Dtype>::add_def(ss, "SRPRNN_RESTRICT_FIXED_SIGN", 1);
  LibDNN<Dtype>::add_def(ss, "SRPRNN_RESTRICT_FIXED_VALUE", 2);

  LibDNN<Dtype>::add_def(ss, "v_bas", config_.batch_size);
  LibDNN<Dtype>::add_def(ss, "v_bps", config_.backprop_steps);
  LibDNN<Dtype>::add_def(ss, "v_tes", config_.temp_steps);
  LibDNN<Dtype>::add_def(ss, "v_ns", config_.neurons.size());
  LibDNN<Dtype>::add_def(ss, "v_ws", config_.weight_count);
  LibDNN<Dtype>::add_def(ss, "v_is", config_.input_neurons.size());
  LibDNN<Dtype>::add_def(ss, "v_os", config_.output_neurons.size());
  LibDNN<Dtype>::add_def(ss, "v_es", config_.export_neurons.size());
  LibDNN<Dtype>::add_def(ss, "v_ims", size);
  LibDNN<Dtype>::add_def(ss, "v_height", config_.shape[0]);
  LibDNN<Dtype>::add_def(ss, "v_width", config_.shape[1]);
  return ss.str();
}


template<typename Dtype>
std::string SRPRNN<Dtype>::generate_fw_defs() {
  std::stringstream ss;
  return ss.str();
}

template<typename Dtype>
std::string SRPRNN<Dtype>::generate_bw_defs() {
  std::stringstream ss;
  return ss.str();
}


// Helper functions
template<typename Dtype>
std::string SRPRNN<Dtype>::relu_fw(std::string data_in,
                                   std::string data_out,
                                   Dtype neg_slope) {
  std::stringstream ss;
  ss << data_out << " = " << data_in << " > 0 ? " << data_in
     << " : " << data_in << " * (Dtype)" << std::to_string(neg_slope) << ";"
     << std::endl;
  return ss.str();
}

template<typename Dtype>
std::string SRPRNN<Dtype>::relu_bw(std::string data_in,
                                   std::string diff_in,
                                   std::string diff_out,
                                   Dtype neg_slope) {
  std::stringstream ss;
  ss << diff_out << " = " << diff_in << " * "
     << "((" << data_in << " > 0?1.0:0.0) + "
     << "(" << data_in << " <= 0?1.0:0.0) * " << std::to_string(neg_slope)
     << ");"
     << std::endl;
  return ss.str();
}

std::string tuple_string(std::tuple<int_tp, int_tp, std::string> tup) {
  return std::to_string(std::get<0>(tup)) + "," +
         std::to_string(std::get<1>(tup));
}

std::string tuple_string(std::tuple<int_tp, int_tp, int_tp, std::string> tup) {
  return std::to_string(std::get<0>(tup)) + "," +
         std::to_string(std::get<1>(tup)) + "," +
         std::to_string(std::get<2>(tup));
}

template <typename Dtype>
std::string SRPRNN<Dtype>::generate_wgc_kernels(std::string name) {
  std::stringstream ss;
  ss << "__kernel" << std::endl;
  ss << "void " << name << "(";
  ss << "__global const Dtype* wg_ref,";
  ss << "__global Dtype* wg,";
  ss << "__global const Dtype* bias_ref,";
  ss << "__global Dtype* bias,";
  ss << "__global const Dtype* wg_restrict,";
  ss << "__global const Dtype* bias_restrict";
  ss << ") {" << std::endl;
  ss << "int_tp i = get_global_id(0);";
  ss << "if (i < v_ns) {" << std::endl;
  ss << "if (bias_restrict[i] == SRPRNN_RESTRICT_FIXED_SIGN) {" << std::endl;
  ss << "bias[i] = bias_ref[i] >= 0 ? fabs(bias[i]) : -fabs(bias[i]);"
     << std::endl;
  ss << "}" << std::endl;
  ss << "if (bias_restrict[i] == SRPRNN_RESTRICT_FIXED_VALUE) {" << std::endl;
  ss << "bias[i] = bias_ref[i];" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  ss << "if (i >= v_ns && i < v_ns + v_is) {" << std::endl;
  ss << "i -= v_ns;" << std::endl;
  ss << "if (wg_restrict[i] == SRPRNN_RESTRICT_FIXED_SIGN) {" << std::endl;
  ss << "wg[i] = wg_ref[i] >= 0 ? fabs(wg[i]) : -fabs(wg[i]);"
     << std::endl;
  ss << "}" << std::endl;
  ss << "if (wg_restrict[i] == SRPRNN_RESTRICT_FIXED_VALUE) {" << std::endl;
  ss << "wg[i] = wg_ref[i];" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  ss << "}" << std::endl;
  return ss.str();
}


template<typename Dtype>
std::string SRPRNN<Dtype>::generate_fw_kernels(std::string name) {
  std::stringstream ss;

  std::vector<std::string> processed_neurons;
  int_tp phase = 0;
  int_tp last_size = 0;
  // Generate one kernel per phase
  // A phase consists of computing the output of all neurons which have
  // all inputs available (computed). This entails neurons with t_off < 0 and
  // neurons that have been computed in a previous phase.
  do {
    std::stringstream ss_phase;
    ss_phase << "__kernel" << std::endl;
    ss_phase << "void " << name << "_" << phase << "(";
    ss_phase << "__global const Dtype* __restrict im_in, ";
    ss_phase << "__global Dtype* __restrict im_out, ";
    ss_phase << "__global Dtype* __restrict im_exp, ";
    ss_phase << "__global Dtype* __restrict buf, ";
    ss_phase << "__global const Dtype* __restrict wg, ";
    ss_phase << "__global const Dtype* __restrict bias, ";
    ss_phase << "const int_tp base,";
    ss_phase << "const int_tp batch";
    ss_phase << ") {" << std::endl;
    ss_phase << "int_tp y = get_global_id(1);" << std::endl;
    ss_phase << "int_tp x = get_global_id(0);" << std::endl;
    ss_phase << "if (y >= v_height || x >= v_width) {" << std::endl;
    ss_phase << "return;" << std::endl;
    ss_phase << "}" << std::endl;
    std::map<std::string, std::tuple<int_tp, int_tp, std::string>>
        offset_guard_map;
    std::vector<std::string> current_neurons;
    std::stringstream ss_sub;
    std::stringstream ss_act;
    std::stringstream ss_out;
    for (int_tp i = 0; i < config_.neurons.size(); ++i) {
      // Check if neuron hasn't been processed yet
      if (std::find(current_neurons.begin(), current_neurons.end(),
                    config_.neurons[i].name) == current_neurons.end() &&
          std::find(processed_neurons.begin(), processed_neurons.end(),
                    config_.neurons[i].name) == processed_neurons.end()) {
        std::stringstream ss_loc;
        // Check if all inputs to the neuron are already processed
        bool all_inputs_processed = true;
        ss_loc << "Dtype " << config_.neurons[i].name
               << "_reg = ";
        // Check if the neuron is an input neuron
        std::vector<std::string>::iterator it_input_neurons
                = std::find(config_.input_neurons.begin(),
                            config_.input_neurons.end(),
                            config_.neurons[i].name);
        if (it_input_neurons != config_.input_neurons.end()) {
          ss_loc << "im_in[((batch * v_is + "
                 << std::to_string(it_input_neurons
                           - config_.input_neurons.begin()) << ") "
                 << "* v_ims) + y * v_width + x];"
                 << std::endl;
        } else {
          ss_loc << "0.0;" << std::endl;
        }
        // Test if all the inputs from the current temporal offset are available
        for (int_tp j = 0; j < config_.neurons[i].inputs.size(); ++j) {
          if (config_.neurons[i].inputs[j].temp_offset >= 0 &&
              std::find(processed_neurons.begin(),
                        processed_neurons.end(),
                        config_.neurons[i].inputs[j].name) ==
                        processed_neurons.end()) {
            all_inputs_processed = false;
          }
        }
        if (all_inputs_processed) {
          // Transfer ss_loc to ss_sub
          ss_sub << ss_loc.str();
          for (int_tp j = 0; j < config_.neurons[i].inputs.size(); ++j) {
            std::stringstream ss_ld;
            std::tuple<int_tp, int_tp, std::string> guard_tuple(
                config_.neurons[i].inputs[j].offset[0],
                config_.neurons[i].inputs[j].offset[1],
                "");
            std::map<std::string,
              std::tuple<int_tp, int_tp, std::string>>::iterator
                offset_guard_map_it = offset_guard_map.find(
                                                     tuple_string(guard_tuple));
            if (offset_guard_map_it != offset_guard_map.end()) {
              guard_tuple = offset_guard_map_it->second;
              ss_ld << std::get<2>(guard_tuple);
            }
            // Test if periodicity of this input is higer than (1,1)
            // Add guards as necessary
            bool period_guard = false;
            for (int_tp k = 0; k < config_.neurons.size(); ++k) {
              if (config_.neurons[i].inputs[j].name ==
                  config_.neurons[k].name) {
                int_tp p0 = config_.neurons[k].periodicity[0];
                int_tp p1 = config_.neurons[k].periodicity[1];
                if (p0 > 1 || p1 > 1) {
                  period_guard = true;
                  ss_ld << "if (";
                }
                if (p0 > 1) {
                  ss_ld << "(yoff % " << std::to_string(p0) << " == 0) && ";
                }
                if (p1 > 1) {
                  ss_ld << "(xoff % " << std::to_string(p1) << " == 0) && ";
                }
                if (period_guard) {
                  ss_ld << "true) {" << std::endl;
                }
              }
            }

            ss_ld << config_.neurons[i].name << "_reg";
            switch (config_.neurons[i].operation) {
              case SRPRNN_OP_MUL:
                ss_ld << " *= ";
                break;
              case SRPRNN_OP_ADD:
              default:
                ss_ld << " += ";
                break;
            }
            ss_ld << "(wg[" << config_.neurons[i].inputs[j].weight_mem_off
                   << "] * buf[(((v_tes + base + batch + ("
                   << std::to_string(config_.neurons[i].inputs[j].temp_offset)
                   << ")) % v_tes) * v_ns + "
                   << config_.neuron_offsets[
                                config_.neurons[i].inputs[j].name]
                   << ") * v_ims + yoff * v_width + xoff]);" << std::endl;
            if (period_guard) {
              ss_ld << "}" << std::endl;
            }
            std::get<2>(guard_tuple) = ss_ld.str();
            offset_guard_map[tuple_string(guard_tuple)] = guard_tuple;
          }
          // Calculate activation function
          switch (config_.neurons[i].activation) {
            case SRPRNN_ACTIVATION_RELU:
              ss_act << relu_fw(config_.neurons[i].name+"_reg",
                                config_.neurons[i].name+"_reg",
                                0.0001);
              break;
            case SRPRNN_ACTIVATION_SIGMOID:
              // TODO
              break;
            case SRPRNN_ACTIVATION_TANH:
              // TODO
              break;
          }
          // Test if periodicity of this input is higer than (1,1)
          // Add guards as necessary
          bool period_guard = false;
          int_tp p0 = config_.neurons[i].periodicity[0];
          int_tp p1 = config_.neurons[i].periodicity[1];
          if (p0 > 1 || p1 > 1) {
            period_guard = true;
            ss_out << "if (";
          }
          if (p0 > 1) {
            ss_out << "(y % " << std::to_string(p0) << " == 0) && ";
          }
          if (p1 > 1) {
            ss_out << "(x % " << std::to_string(p1) << " == 0) && ";
          }
          if (period_guard) {
            ss_out << "true) {" << std::endl;
          }
          // Store result to internal buffer memory
          ss_out << "buf[(((v_tes + base + batch) % v_tes) * v_ns + "
                 << config_.neurons[i].neuron_off
                 << ") * v_ims + y * v_width + x] = "
                 << config_.neurons[i].name
                 << "_reg;" << std::endl;
          // Check if the neuron is an output neuron
          std::vector<std::string>::iterator it_output_neurons
               = std::find(config_.output_neurons.begin(),
                           config_.output_neurons.end(),
                           config_.neurons[i].name);
          if (it_output_neurons != config_.output_neurons.end()) {
            ss_out << "im_out[((batch * v_os + "
                   << std::to_string(it_output_neurons
                                     - config_.output_neurons.begin())
                   << ") * v_ims) + y * v_width + x] = "
                   << config_.neurons[i].name << "_reg;" << std::endl;
          }
          // Check if the neuron is an export neuron
          std::vector<std::string>::iterator it_export_neurons
                = std::find(config_.export_neurons.begin(),
                            config_.export_neurons.end(),
                            config_.neurons[i].name);
          if (it_export_neurons != config_.export_neurons.end()) {
            ss_out << "im_exp[((batch * v_es + "
                   << std::to_string(it_export_neurons
                                     - config_.export_neurons.begin())
                   << ") * v_ims) + y * v_width + x] = "
                   << config_.neurons[i].name << "_reg;" << std::endl;
          }
          if (period_guard) {
            ss_out << "}" << std::endl;
          }
          current_neurons.push_back(config_.neurons[i].name);
        }
      }
    }
    // Preparing neurons
    ss_phase << ss_sub.str();
    // Load and apply operator on weight and inputs of each neuron
    std::map<std::string, std::tuple<int_tp,
          int_tp, std::string>>::iterator offset_guard_map_it;
    for (offset_guard_map_it = offset_guard_map.begin();
         offset_guard_map_it != offset_guard_map.end();
         ++offset_guard_map_it) {
      std::tuple<int_tp, int_tp, std::string> guard_off =
          offset_guard_map_it->second;
      int_tp yoff = std::get<0>(guard_off);
      int_tp xoff = std::get<1>(guard_off);
      ss_phase << "{" << std::endl;
      ss_phase << "int_tp yoff = y + (" << yoff << ");" << std::endl;
      ss_phase << "int_tp xoff = x + (" << xoff << ");" << std::endl;
      ss_phase << "if ((xoff >= 0) && (xoff < v_width) &&"
               << " (yoff >= -v_height) && (yoff < 2*v_height) &&"
          << "!((2*y+x < 2 * v_height && 2*yoff+xoff >= 2 * v_height) || "
          << "(2*y+x >= 2 * v_height && 2*yoff+xoff < 2 * v_height))"
               << ") {" << std::endl;
      ss_phase << "yoff = yoff < 0 ? yoff + v_height : yoff;"
               << std::endl;
      ss_phase << "yoff = yoff >= v_height ? yoff - v_height : yoff;"
               << std::endl;
      ss_phase << std::get<2>(offset_guard_map_it->second);
      ss_phase << "}" << std::endl;
      ss_phase << "}" << std::endl;
    }
    // Calculate activations
    ss_phase << ss_act.str();
    // Write-out output and export data
    ss_phase << ss_out.str();
    ss_phase << "}" << std::endl;
    last_size = processed_neurons.size();
    processed_neurons.insert(processed_neurons.end(),
                             current_neurons.begin(),
                             current_neurons.end());
    if (last_size != processed_neurons.size()) {
      ss << ss_phase.str();
      ++phase;
    }
  } while (processed_neurons.size() < config_.neurons.size() &&
      last_size != processed_neurons.size());

  config_.fw_phases = phase;

  return ss.str();
}

template<typename Dtype>
std::string SRPRNN<Dtype>::generate_bw_kernels(std::string name) {
  std::stringstream ss;

  std::vector<std::string> processed_neurons;
  int_tp phase = 0;
  int_tp last_size = 0;
  // Generate one kernel per phase
  // A phase consists of computing the output of all neurons which have
  // all inputs available (computed). This entails neurons with t_off < 0 and
  // neurons that have been computed in a previous phase.
  do {
    int_tp temp_id = 0;
    std::stringstream ss_phase;
    ss_phase << "__kernel" << std::endl;
    ss_phase << "void " << name << "_" << phase << "(";
    ss_phase << "__global const Dtype* __restrict im_in, ";
    ss_phase << "__global Dtype* __restrict im_in_diff, ";
    ss_phase << "__global const Dtype* __restrict im_out, ";
    ss_phase << "__global const Dtype* __restrict im_out_diff, ";
    ss_phase << "__global const Dtype* __restrict buf, ";
    ss_phase << "__global Dtype* __restrict buf_diff, ";
    ss_phase << "__global const Dtype* __restrict wg, ";
    ss_phase << "__global Dtype* __restrict wg_diff, ";
    ss_phase << "__global const Dtype* __restrict bias, ";
    ss_phase << "__global Dtype* __restrict bias_diff, ";
    ss_phase << "const int_tp base,";
    ss_phase << "const int_tp batch";
    ss_phase << ") {" << std::endl;
    ss_phase << "int_tp y = get_global_id(1);" << std::endl;
    ss_phase << "int_tp x = get_global_id(0);" << std::endl;
    ss_phase << "if (y >= v_height || x >= v_width) {" << std::endl;
    ss_phase << "return;" << std::endl;
    ss_phase << "}" << std::endl;
    std::map<std::string, std::tuple<int_tp, int_tp, int_tp, std::string>>
        offset_guard_map;
    std::vector<std::string> current_neurons;
    std::stringstream ss_sub;
    std::stringstream ss_act;
    std::stringstream ss_out;
    for (int_tp i = 0; i < config_.neurons.size(); ++i) {
      // Check if neuron hasn't been processed yet
      if (std::find(current_neurons.begin(), current_neurons.end(),
                    config_.neurons[i].name) == current_neurons.end() &&
          std::find(processed_neurons.begin(), processed_neurons.end(),
                    config_.neurons[i].name) == processed_neurons.end()) {
        std::stringstream ss_loc;
        // Check if all outputs to the neuron are already processed
        bool all_outputs_processed = true;
        ss_loc << "Dtype " << config_.neurons[i].name << "_reg = ";
        ss_loc << "buf[(((v_tes + base + batch) % v_tes) * v_ns + "
               << config_.neurons[i].neuron_off
               << ") * v_ims + y * v_width + x];" << std::endl;
        ss_loc << "Dtype " << config_.neurons[i].name << "_diff_reg = ";
        // Check if the neuron is an output neuron
        std::vector<std::string>::iterator it_output_neurons
                = std::find(config_.output_neurons.begin(),
                            config_.output_neurons.end(),
                            config_.neurons[i].name);
        if (it_output_neurons != config_.output_neurons.end()) {
          ss_loc << "im_out_diff[((batch * v_os + "
                 << std::to_string(it_output_neurons
                           - config_.output_neurons.begin()) << ") "
                 << "* v_ims) + y * v_width + x];"
                 << std::endl;
        } else {
          ss_loc << "0.0;" << std::endl;
        }
        // Test if all the outputs from the current temporal offset
        // are available
        for (int_tp j = 0; j < config_.neurons[i].outputs.size(); ++j) {
          if (config_.neurons[i].outputs[j].temp_offset <= 0 &&
              std::find(processed_neurons.begin(),
                        processed_neurons.end(),
                        config_.neurons[i].outputs[j].name) ==
                        processed_neurons.end()) {
            all_outputs_processed = false;
          }
        }
        if (all_outputs_processed) {
          // Transfer ss_loc to ss_sub
          ss_sub << ss_loc.str();
          for (int_tp j = 0; j < config_.neurons[i].outputs.size(); ++j) {
            std::stringstream ss_ld;
            std::tuple<int_tp, int_tp, int_tp, std::string> guard_tuple(
                config_.neurons[i].outputs[j].offset[0],
                config_.neurons[i].outputs[j].offset[1],
                config_.neurons[i].outputs[j].temp_offset,
                "");
            std::map<std::string,
              std::tuple<int_tp, int_tp, int_tp, std::string>>::iterator
                offset_guard_map_it = offset_guard_map.find(
                                                     tuple_string(guard_tuple));
            if (offset_guard_map_it != offset_guard_map.end()) {
              guard_tuple = offset_guard_map_it->second;
              ss_ld << std::get<3>(guard_tuple);
            }
            // Test if periodicity of this input is higer than (1,1)
            // Add guards as necessary
            bool period_guard = false;
            for (int_tp k = 0; k < config_.neurons.size(); ++k) {
              if (config_.neurons[i].outputs[j].name ==
                  config_.neurons[k].name) {
                int_tp p0 = config_.neurons[k].periodicity[0];
                int_tp p1 = config_.neurons[k].periodicity[1];
                if (p0 > 1 || p1 > 1) {
                  period_guard = true;
                  ss_ld << "if (";
                }
                if (p0 > 1) {
                  ss_ld << "(yoff % " << std::to_string(p0) << " == 0) && ";
                }
                if (p1 > 1) {
                  ss_ld << "(xoff % " << std::to_string(p1) << " == 0) && ";
                }
                if (period_guard) {
                  ss_ld << "true) {" << std::endl;
                }
              }
            }
            // Load top diff
            ss_ld << "Dtype tmp_" << temp_id
                  << " = buf_diff[(((v_tes + base + batch + ("
                  << std::to_string(config_.neurons[i].outputs[j].temp_offset)
                  << ")) % v_tes) * v_ns + "
                  << config_.neuron_offsets[
                                config_.neurons[i].outputs[j].name]
                  << ") * v_ims + yoff * v_width + xoff];" << std::endl;
            // Add up bottom diff
            ss_ld << config_.neurons[i].name << "_diff_reg += ";
            ss_ld << "wg[" << config_.neurons[i].outputs[j].weight_mem_off
                  << "] * tmp_" << temp_id << ";" << std::endl;
            // Add up bias diff
            ss_ld << "atomicAdd(&"
                  << "(bias_diff[" << config_.neuron_offsets[
                                         config_.neurons[i].outputs[j].name]
                  << "]), tmp_" << temp_id << ");" << std::endl;
            // Add up weight diff
            ss_ld << "atomicAdd(&"
                  << "(wg_diff["
                  << config_.neurons[i].outputs[j].weight_mem_off
                  << "]), tmp_" << temp_id << " * " << config_.neurons[i].name
                  << "_reg);" << std::endl;
            ++temp_id;
            if (period_guard) {
              ss_ld << "}" << std::endl;
            }
            std::get<3>(guard_tuple) = ss_ld.str();
            offset_guard_map[tuple_string(guard_tuple)] = guard_tuple;
          }
          // Calculate activation function
          switch (config_.neurons[i].activation) {
            case SRPRNN_ACTIVATION_RELU:
              ss_act << relu_bw(config_.neurons[i].name+"_reg",
                                config_.neurons[i].name+"_diff_reg",
                                config_.neurons[i].name+"_diff_reg",
                                0.0001);
              break;
            case SRPRNN_ACTIVATION_SIGMOID:
              // TODO
              break;
            case SRPRNN_ACTIVATION_TANH:
              // TODO
              break;
          }
          // Test if periodicity of this input is higer than (1,1)
          // Add guards as necessary
          bool period_guard = false;
          int_tp p0 = config_.neurons[i].periodicity[0];
          int_tp p1 = config_.neurons[i].periodicity[1];
          if (p0 > 1 || p1 > 1) {
            period_guard = true;
            ss_out << "if (";
          }
          if (p0 > 1) {
            ss_out << "(y % " << std::to_string(p0) << " == 0) && ";
          }
          if (p1 > 1) {
            ss_out << "(x % " << std::to_string(p1) << " == 0) && ";
          }
          if (period_guard) {
            ss_out << "true) {" << std::endl;
          }
          // Only input diff of current batch is backpropagated
          ss_out << "if (batch >= 0) {" << std::endl;
          // Store result to internal buffer memory
          ss_out << "buf_diff[(((v_tes + base + batch) % v_tes) * v_ns + "
                 << config_.neurons[i].neuron_off
                 << ") * v_ims + y * v_width + x] = "
                 << config_.neurons[i].name
                 << "_diff_reg;" << std::endl;
          // Check if the neuron is an input neuron
          std::vector<std::string>::iterator it_input_neurons
               = std::find(config_.input_neurons.begin(),
                           config_.input_neurons.end(),
                           config_.neurons[i].name);
          if (it_input_neurons != config_.input_neurons.end()) {
            ss_out << "im_in_diff[((batch * v_os + "
                   << std::to_string(it_input_neurons
                                     - config_.input_neurons.begin())
                   << ") * v_ims) + y * v_width + x] = "
                   << config_.neurons[i].name << "_diff_reg;" << std::endl;
          }
          ss_out << "}" << std::endl;
          if (period_guard) {
            ss_out << "}" << std::endl;
          }
          current_neurons.push_back(config_.neurons[i].name);
        }
      }
    }
    // Preparing neurons
    ss_phase << ss_sub.str();
    // Load and apply operator on weight and inputs of each neuron
    std::map<std::string, std::tuple<int_tp, int_tp,
          int_tp, std::string>>::iterator offset_guard_map_it;
    for (offset_guard_map_it = offset_guard_map.begin();
         offset_guard_map_it != offset_guard_map.end();
         ++offset_guard_map_it) {
      std::tuple<int_tp, int_tp, int_tp, std::string> guard_off =
          offset_guard_map_it->second;
      int_tp yoff = std::get<0>(guard_off);
      int_tp xoff = std::get<1>(guard_off);
      int_tp toff = std::get<2>(guard_off);
      ss_phase << "{" << std::endl;
      ss_phase << "int_tp yoff = y + (" << yoff << ");" << std::endl;
      ss_phase << "int_tp xoff = x + (" << xoff << ");" << std::endl;
      if (toff > 0) {
        ss_phase << "int_tp toff = batch + (" << toff << ");" << std::endl;
      }
      ss_phase << "if ((xoff >= 0) && (xoff < v_width) &&"
               << " (yoff >= -v_height) && (yoff < 2*v_height) &&"
          << "!((2*y+x < 2 * v_height && 2*yoff+xoff >= 2 * v_height) || "
          << "(2*y+x >= 2 * v_height && 2*yoff+xoff < 2 * v_height))";
      if (toff > 0) {
        ss_phase << " && toff < v_bas){" << std::endl;
      } else {
      ss_phase << ") {" << std::endl;
      }
      ss_phase << "yoff = yoff < 0 ? yoff + v_height : yoff;"
               << std::endl;
      ss_phase << "yoff = yoff >= v_height ? yoff - v_height : yoff;"
               << std::endl;
      ss_phase << std::get<3>(offset_guard_map_it->second);
      ss_phase << "}" << std::endl;
      ss_phase << "}" << std::endl;
    }
    // Calculate activations
    ss_phase << ss_act.str();
    // Write-out output and export data
    ss_phase << ss_out.str();
    ss_phase << "}" << std::endl;
    last_size = processed_neurons.size();
    processed_neurons.insert(processed_neurons.end(),
                             current_neurons.begin(),
                             current_neurons.end());
    if (last_size != processed_neurons.size()) {
      ss << ss_phase.str();
      ++phase;
    }
  } while (processed_neurons.size() < config_.neurons.size() &&
      last_size != processed_neurons.size());

  config_.bw_phases = phase;

  return ss.str();
}


template<typename Dtype>
void SRPRNN<Dtype>::GenerateKernels() {
  std::stringstream ss;

  ss << LibDNN<Dtype>::generate_header();
  ss << generate_defs();
  ss << generate_fw_defs();
  ss << generate_fw_kernels("srprnn_forward");
  ss << generate_bw_defs();
  ss << generate_bw_kernels("srprnn_backward");
  ss << generate_wgc_kernels("srprnn_weights_check");

  // Write complete kernel string
  LibDNN<Dtype>::kernel_ = ss.str();
}

template<typename Dtype>
void SRPRNN<Dtype>::InitWeight(Dtype* cpu_weight_data,
                               Dtype* cpu_weight_restrict_data) {
  for (int_tp i = 0; i < config_.neurons.size(); ++i) {
    for (int_tp j = 0; j < config_.neurons[i].inputs.size(); ++j) {
      // Initial weight
      if (cpu_weight_data != nullptr) {
        cpu_weight_data[config_.neurons[i].inputs[j].weight_mem_off]
                    = config_.neurons[i].inputs[j].weight;
      }
      if (cpu_weight_restrict_data != nullptr) {
        cpu_weight_restrict_data[config_.neurons[i].inputs[j].weight_mem_off]
                      = (int_tp)config_.neurons[i].inputs[j].weight_restrict;
      }
    }
  }
}

template<typename Dtype>
void SRPRNN<Dtype>::InitBias(Dtype* cpu_bias_data,
                             Dtype* cpu_bias_restrict_data) {
  for (int_tp i = 0; i < config_.neurons.size(); ++i) {
    // Initial bias
    if (cpu_bias_data != nullptr) {
      cpu_bias_data[config_.neurons[i].neuron_off]
                  = config_.neurons[i].bias;
    }
    if (cpu_bias_restrict_data != nullptr) {
      cpu_bias_restrict_data[config_.neurons[i].neuron_off]
                  = (int_tp)config_.neurons[i].bias_restrict;
    }
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

  if (srprnn_.get() != nullptr) {
    shapes_changed = shapes_changed || (srprnn_.get()->get_config().in_shape
        != bottom[0]->shape());
    shapes_changed = shapes_changed || (srprnn_.get()->get_config().out_shape
        != top[0]->shape());
  }

  if (srprnn_.get() == nullptr || shapes_changed) {
    SRPRNNConfig config;

    SRPRNNParameter param = this->layer_param().srprnn_param();

    config.dev_ptr = this->device_;
    config.fast_unsafe_math = true;
    config.bias_backward = true;
    config.weights_backward = true;

    // Spatial dimensions of the SRPRNN layer
    config.shape.push_back(bottom[0]->shape(2));
    config.shape.push_back(bottom[0]->shape(3));

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
      std::vector<SRPRNNIOConfig> inputs;
      for (int_tp j = 0; j < neuron_param.input_size(); ++j) {
        SRPRNNInput input_param = neuron_param.input(j);
        SRPRNNIOConfig input;
        std::vector<int_tp> offset;
        for (int k = 0; k < input_param.offset_size(); ++k) {
          offset.push_back(input_param.offset(k));
        }
        input.offset = offset;
        input.temp_offset = input_param.temp_offset();
        max_temp_offset = std::max(max_temp_offset,
                                   std::abs(input.temp_offset));
        input.weight = input_param.weight();
        input.name = input_param.name();
        inputs.push_back(input);
        // Count number of synaptic connections (= weights)
        ++weight_count;
      }
      neuron.inputs = inputs;
      neurons.push_back(neuron);
    }

    config.neurons = neurons;
    config.weight_count = weight_count;

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

    if (top.size() > 1) {
      std::vector<std::string> export_neurons;
      for (int_tp i = 0; i < param.export_neuron_size(); ++i) {
        export_neurons.push_back(param.export_neuron(i));
      }
      config.export_neurons = export_neurons;
    }

    // Generate ordering for the neurons
    int_tp neuron_offset = 0;
    int_tp weight_mem_offset = 0;
    int_tp last_size = 0;
    std::vector<std::string> processed_neurons;
    std::map<std::string, int_tp> neuron_offsets;
    do {
      std::vector<std::string> current_neurons;
      for (int_tp i = 0; i < config.neurons.size(); ++i) {
        // Check if neuron hasn't been processed yet
        if (std::find(current_neurons.begin(), current_neurons.end(),
                      config.neurons[i].name) == current_neurons.end() &&
            std::find(processed_neurons.begin(), processed_neurons.end(),
                      config.neurons[i].name) == processed_neurons.end()) {
          bool all_inputs_processed = true;
          for (int_tp j = 0; j < config.neurons[i].inputs.size(); ++j) {
            if (config.neurons[i].inputs[j].temp_offset >= 0 &&
                std::find(processed_neurons.begin(),
                          processed_neurons.end(),
                          config.neurons[i].inputs[j].name) ==
                          processed_neurons.end()) {
              all_inputs_processed = false;
            }
          }
          if (all_inputs_processed) {
            config.neurons[i].neuron_off = neuron_offset;
            neuron_offsets[config.neurons[i].name] = neuron_offset;
            ++neuron_offset;
            for (int_tp j = 0; j < config.neurons[i].inputs.size(); ++j) {
              config.neurons[i].inputs[j].weight_mem_off = weight_mem_offset;
              ++weight_mem_offset;
            }
            current_neurons.push_back(config.neurons[i].name);
          }
        }
      }
      last_size = processed_neurons.size();
      processed_neurons.insert(processed_neurons.end(),
                               current_neurons.begin(),
                               current_neurons.end());
    } while (processed_neurons.size() < config.neurons.size() &&
        last_size != processed_neurons.size());

    // Figure out reverse connections (for backward pass)
    for (int_tp i = 0; i < config.neurons.size(); ++i) {
      for (int_tp j = 0; j < config.neurons[i].inputs.size(); ++j) {
        SRPRNNIOConfig input = config.neurons[i].inputs[j];
        for (int_tp k = 0; k < config.neurons.size(); ++k) {
          if (config.neurons[k].name == config.neurons[i].inputs[j].name) {
            SRPRNNIOConfig output;
            std::vector<int_tp> offset;
            for (int l = 0; l < input.offset.size(); ++l) {
              // Flip offset sign
              offset.push_back(-input.offset[l]);
            }
            output.offset = offset;
            // Flip temp offset sign
            output.temp_offset = -input.temp_offset;
            output.weight = input.weight;
            output.weight_mem_off = input.weight_mem_off;
            output.name = neurons[i].name;
            config.neurons[k].outputs.push_back(output);
          }
        }
      }
    }

    // Order the neurons to an offset in the blobs
    config.neuron_offsets = neuron_offsets;

    // Need to have as many backprop through time steps as specified,
    // and at least as many as time offsets of input neurons
    config.backprop_steps = std::max((int_tp)param.backprop_steps(),
                                     (int_tp)max_temp_offset);

    config.batch_size = bottom[0]->shape(0);

    // Temporal dimension buffer size needs to be the number
    // of backpropagation steps plus the batch size.
    config.temp_steps = config.backprop_steps + config.batch_size;

    // Number of input neurons must match the feature maps
    CHECK_EQ(config.input_neurons.size(), bottom[0]->shape(1));

    // Weights for all synaptic connections
    std::vector<int_tp> weight_shape;
    weight_shape.push_back(config.weight_count);

    // Every neuron has a bias value
    std::vector<int_tp> bias_shape;
    bias_shape.push_back(config.neurons.size());

    config.in_shape = bottom[0]->shape();

    config.out_shape.push_back(config.batch_size);
    config.out_shape.push_back(config.output_neurons.size());
    config.out_shape.push_back(config.in_shape[2]);
    config.out_shape.push_back(config.in_shape[3]);

    config.export_shape.push_back(config.batch_size);
    config.export_shape.push_back(config.export_neurons.size());
    config.export_shape.push_back(config.in_shape[2]);
    config.export_shape.push_back(config.in_shape[3]);

    top[0]->Reshape(config.out_shape);
    if (top.size() > 1) {
      top[1]->Reshape(config.export_shape);
    }

    if (this->blobs_.size() < 2) {
      this->blobs_.resize(2);
    }

    SRPRNN<Dtype>* srprnn = new SRPRNN<Dtype>(config);

    // Initialize the weight blob
    if (this->blobs_[0].get() == nullptr ||
        weight_shape != this->blobs_[0].get()->shape()) {
      this->blobs_[0].reset(new Blob<Dtype>(weight_shape, this->device_));
      Dtype* cpu_data = this->blobs_[0].get()->mutable_cpu_data();
      srprnn->InitWeight(cpu_data, nullptr);
    }

    // Initialize the bias blob
    if (this->blobs_[1].get() == nullptr ||
        bias_shape != this->blobs_[1].get()->shape()) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape, this->device_));
      Dtype* cpu_data = this->blobs_[1].get()->mutable_cpu_data();
      srprnn->InitBias(cpu_data, nullptr);
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

  Dtype* flag_data = nullptr;
  if (bottom.size() > 1) {
    flag_data = bottom[1]->mutable_cpu_data();
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

  Dtype* flag_data = nullptr;
  if (bottom.size() > 1) {
    flag_data = bottom[1]->mutable_cpu_data();
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
