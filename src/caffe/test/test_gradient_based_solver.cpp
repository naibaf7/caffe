#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/text_format.h"

#include "gtest/gtest.h"

#include "caffe/common.hpp"
#include "caffe/parallel.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/sgd_solvers.hpp"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

using std::ostringstream;

namespace caffe {

template<typename TypeParam>
class GradientBasedSolverTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  GradientBasedSolverTest() :
      seed_(1701), num_(4), channels_(3), height_(10), width_(10),
      share_(false) {
        input_file_ = new string(
        ABS_TEST_DATA_DIR "/solver_data_list.txt");
      }
  ~GradientBasedSolverTest() {
    delete input_file_;
  }

  string snapshot_prefix_;
  shared_ptr<SGDSolver<Dtype> > solver_;
#ifdef USE_NCCL
  shared_ptr<NCCL<Dtype> > nccl_;
#endif
  int seed_;
  // Dimensions are determined by generate_sample_data.py
  // TODO this is brittle and the hdf5 file should be checked instead.
  int num_, channels_, height_, width_;
  bool share_;
  Dtype delta_;  // Stability constant for RMSProp, AdaGrad, AdaDelta and Adam

  // Test data: check out generate_sample_data.py in the same directory.
  string* input_file_;

  virtual void InitSolver(const SolverParameter& param) = 0;

  virtual void InitSolverFromProtoString(const string& proto) {
    SolverParameter param;
    CHECK(google::protobuf::TextFormat::ParseFromString(proto, &param));
    // Set the solver_mode according to current Caffe::mode.
    switch (Caffe::mode()) {
      case Caffe::CPU:
        param.set_solver_mode(SolverParameter_SolverMode_CPU);
        break;
      case Caffe::GPU:
        param.set_solver_mode(SolverParameter_SolverMode_GPU);
        break;
      default:
        LOG(FATAL)<< "Unknown Caffe mode: " << Caffe::mode();
      }
    InitSolver(param);
    delta_ = param.delta();
  }

  string RunLeastSquaresSolver(const Dtype learning_rate,
                               const Dtype weight_decay, const Dtype momentum,
                               const int num_iters, const int iter_size = 1,
                               const int devices = 1, const bool snapshot =
                                   false,
                               const char* from_snapshot = NULL) {
    string bottom_type = std::is_same<Dtype, half_fp>::value ?
        "bottom_data_type: CAFFE_FLOAT" : "";
    string compute_type = std::is_same<Dtype, half_fp>::value ?
        "compute_data_type: CAFFE_FLOAT" : "";

    ostringstream proto;
    int device = 0;
#ifndef CPU_ONLY
#ifdef USE_CUDA
    if (Caffe::mode() == Caffe::GPU
        && Caffe::GetDefaultDevice()->backend() == BACKEND_CUDA) {
      CUDA_CHECK(cudaGetDevice(&device));
    }
#endif  // USE_CUDA
#endif  // !CPU_ONLY
    proto <<
       "snapshot_after_train: " << snapshot << " "
       "max_iter: " << num_iters << " "
       "base_lr: " << learning_rate << " "
       "lr_policy: 'fixed' "
       "iter_size: " << iter_size << " "
       "device: " << device << " "
       "layer_wise_reduce: " << (!share_) << " "
       "net_param { "
       "  name: 'TestNetwork' "
       "  layer { "
       "    name: 'data' "
       "    type: 'HDF5Data' "
       << bottom_type << " "
       << compute_type << " "
       "    hdf5_data_param { "
       "      source: '" << *(this->input_file_) << "' "
       "      batch_size: " << num_ / iter_size << " "
       "    } "
       "    top: 'data' "
       "    top: 'targets' "
       "  } ";
    if (share_) {
      proto << "  layer { "
            "    name: 'slice' "
            "    type: 'Slice' "
            "    bottom: 'data' "
            "    top: 'data1' "
            "    top: 'data2' "
            "    slice_param { "
            "      axis: 0 "
            "    } "
            "  } ";
    }
    proto << "  layer { "
          "    name: 'innerprod' "
          "    type: 'InnerProduct' "
          "    param { name: 'weights' } "
          "    param { name: 'bias' } "
          "    inner_product_param { "
          "      num_output: 1 "
          "      weight_filler { "
          "        type: 'gaussian' "
          "        std: 1.0 "
          "      } "
          "      bias_filler { "
          "        type: 'gaussian' "
          "        std: 1.0 "
          "      } "
          "    } "
          "    bottom: '"
          << string(share_ ? "data1" : "data") << "' "
          "    top: '"
          << string(share_ ? "innerprod1" : "innerprod") << "' "
          "  } ";
    if (share_) {
      proto << "  layer { "
            "    name: 'innerprod2' "
            "    type: 'InnerProduct' "
            "    param { name: 'weights' } "
            "    param { name: 'bias' } "
            "    inner_product_param { "
            "      num_output: 1 "
            "      weight_filler { "
            "        type: 'gaussian' "
            "        std: 1.0 "
            "      } "
            "      bias_filler { "
            "        type: 'gaussian' "
            "        std: 1.0 "
            "      } "
            "    } "
            "    bottom: 'data2' "
            "    top: 'innerprod2' "
            "  } "
            "  layer { "
            "    name: 'concat' "
            "    type: 'Concat' "
            "    bottom: 'innerprod1' "
            "    bottom: 'innerprod2' "
            "    top: 'innerprod' "
            "    concat_param { "
            "      axis: 0 "
            "    } "
            "  } ";
    }
    proto << "  layer { "
          "    name: 'loss' "
          "    type: 'EuclideanLoss' "
          "    bottom: 'innerprod' "
          "    bottom: 'targets' "
          "  } "
          "} ";
    if (weight_decay != 0) {
      proto << "weight_decay: " << weight_decay << " ";
    }
    if (momentum != 0) {
      proto << "momentum: " << momentum << " ";
    }
    MakeTempDir(&snapshot_prefix_);
#if defined(_MSC_VER)
    std::replace(snapshot_prefix_.begin(), snapshot_prefix_.end(), '\\', '/');
#endif
    proto << "snapshot_prefix: '" << snapshot_prefix_ << "/' ";
    if (snapshot) {
      proto << "snapshot: " << num_iters << " ";
    }
    Caffe::set_random_seed(this->seed_, Caffe::GetDefaultDevice());
    this->InitSolverFromProtoString(proto.str());
    if (from_snapshot) {
      this->solver_->Restore(from_snapshot);
      for (int i = 0; i < this->solver_->iter(); ++i) {
        this->solver_->net()->Forward();
      }
    }
    if (devices == 1) {
      this->solver_->Solve();
    } else {
      LOG(INFO)<< "Multi-GPU test on " << devices << " devices";
      vector<Device*> gpus;
      // put current device at the beginning
      Device* dc = Caffe::GetDevice(solver_->param().device(), true);
      gpus.push_back(dc);
      for (int i = 0; gpus.size() < devices; ++i) {
        if (i != device)
          gpus.push_back(Caffe::GetDevice(i, true));
      }
      Caffe::set_solver_count(gpus.size());
#ifdef USE_NCCL
      this->nccl_.reset(new NCCL<Dtype>(this->solver_));
      this->nccl_->Run(gpus, from_snapshot);
#endif
      Caffe::set_solver_count(1);
    }
    if (snapshot) {
      ostringstream resume_file;
      resume_file << snapshot_prefix_ << "/_iter_" << num_iters
                  << ".solverstate";
      string resume_filename = resume_file.str();
      return resume_filename;
    }
    return string();
  }

  // Compute an update value given the current state of the train net,
  // using the analytical formula for the least squares gradient.
  // updated_params will store the updated weight and bias results,
  // using the blobs' diffs to hold the update values themselves.
  void ComputeLeastSquaresUpdate(
      const Dtype learning_rate, const Dtype weight_decay, const Dtype momentum,
      const int num_iters, vector<shared_ptr<Blob<Dtype> > >* updated_params) {
    const int n = num_;
    const int D = channels_ * height_ * width_;

    // Run a forward pass, and manually compute the update values from the
    // result.
    Net<Dtype>& net = *this->solver_->net();
    net.Forward();
    ASSERT_TRUE(net.has_blob("data"));
    const Blob<Dtype>& data =
        *(static_pointer_cast<Blob<Dtype> >(net.blob_by_name("data")));
    ASSERT_TRUE(net.has_blob("targets"));
    const Blob<Dtype>& targets =
        *(static_pointer_cast<Blob<Dtype> >(net.blob_by_name("targets")));
    ASSERT_TRUE(net.has_layer("innerprod"));
    const vector<shared_ptr<BlobBase > >& param_blobs = net.layer_by_name(
        "innerprod")->blob_bases();
    const int num_param_blobs = 2;
    ASSERT_EQ(num_param_blobs, param_blobs.size());
    const Blob<Dtype>& weights =
        *(static_pointer_cast<Blob<Dtype> >(param_blobs[0]));
    const Blob<Dtype>& bias =
        *(static_pointer_cast<Blob<Dtype> >(param_blobs[1]));
    ASSERT_EQ(D * n, data.count());
    ASSERT_EQ(n, targets.count());
    ASSERT_EQ(D, weights.count());
    ASSERT_EQ(1, bias.count());

    updated_params->clear();
    updated_params->resize(num_param_blobs);
    for (int i = 0; i < num_param_blobs; ++i) {
      (*updated_params)[i].reset(new Blob<Dtype>());
    }
    Blob<Dtype>& updated_weights = *(*updated_params)[0];
    updated_weights.ReshapeLike(weights);
    Blob<Dtype>& updated_bias = *(*updated_params)[1];
    updated_bias.ReshapeLike(bias);

    for (int i = 0; i <= D; ++i) {
      // Compute the derivative with respect to the ith weight (i.e., the ith
      // element of the gradient).
      Dtype grad = 0;
      for (int j = 0; j <= D; ++j) {
        // Compute element (i, j) of X^T * X.
        Dtype element = 0;
        for (int k = 0; k < n; ++k) {
          // (i, k) in X^T (== (k, i) in X) times (k, j) in X.
          const Dtype element_i = (i == D) ?
                                  Dtype(1) : data.cpu_data()[k * D + i];
          const Dtype element_j = (j == D) ?
                                  Dtype(1) : data.cpu_data()[k * D + j];
          element += element_i * element_j;
        }
        if (j == D) {
          grad += element * bias.cpu_data()[0];
        } else {
          grad += element * weights.cpu_data()[j];
        }
      }
      for (int k = 0; k < n; ++k) {
        const Dtype element_i = (i == D) ?
                                Dtype(1) : data.cpu_data()[k * D + i];
        grad -= element_i * targets.cpu_data()[k];
      }
      // Scale the gradient over the n samples.
      grad /= n;
      // Add the weight decay to the gradient.
      grad += weight_decay
          * ((i == D) ? bias.cpu_data()[0] : weights.cpu_data()[i]);
      // Finally, compute update.
      const vector<shared_ptr<Blob<Dtype> > >& history = solver_->history();
      if (solver_->type() != string("AdaDelta")
          && solver_->type() != string("Adam")) {
        ASSERT_EQ(2, history.size());  // 1 blob for weights, 1 for bias
      } else {
        ASSERT_EQ(4, history.size());  // additional blobs for update history
      }
      Dtype update_value = learning_rate * grad;
      const Dtype history_value =
          (i == D) ? history[1]->cpu_data()[0] : history[0]->cpu_data()[i];
      const Dtype temp = momentum * history_value;
      if (solver_->type() == string("SGD")) {
        update_value += temp;
      } else if (solver_->type() == string("Nesterov")) {
        update_value += temp;
        // step back then over-step
        update_value = (1 + momentum) * update_value - temp;
      } else if (solver_->type() == string("AdaGrad")) {
        update_value /= std::sqrt(history_value + grad * grad) + delta_;
      } else if (solver_->type() == string("RMSProp")) {
        const Dtype rms_decay = 0.95;
        update_value /= std::sqrt(
            rms_decay * history_value + grad * grad * (1 - rms_decay)) + delta_;
      } else if (solver_->type() == string("AdaDelta")) {
        const Dtype update_history_value =
            (i == D) ?
                history[1 + num_param_blobs]->cpu_data()[0] :
                history[0 + num_param_blobs]->cpu_data()[i];
        const Dtype weighted_gradient_average = momentum * history_value
            + (1 - momentum) * (grad * grad);
        update_value = grad
            * std::sqrt(
                (update_history_value + delta_)
                    / (weighted_gradient_average + delta_)) * learning_rate;
        // not actually needed, just here for illustrative purposes
        // const Dtype weighted_update_average =
        //   momentum * update_history_value + (1 - momentum) * (update_value);
      } else if (solver_->type() == string("Adam")) {
        const Dtype momentum2 = 0.999;
        const Dtype m = history_value;
        const Dtype v =
            (i == D) ?
                history[1 + num_param_blobs]->cpu_data()[0] :
                history[0 + num_param_blobs]->cpu_data()[i];
        const Dtype val_m = (1 - momentum) * grad + momentum * m;
        const Dtype val_v = (1 - momentum2) * grad * grad + momentum2 * v;
        Dtype alpha_t = learning_rate
            * std::sqrt(Dtype(1) - pow(momentum2, Dtype(num_iters)))
            / (Dtype(1.) - pow(momentum, Dtype(num_iters)));
        update_value = alpha_t * val_m / (std::sqrt(val_v) + delta_);
      } else {
        LOG(FATAL)<< "Unknown solver type: " << solver_->type();
      }
      if (i == D) {
        updated_bias.mutable_cpu_diff()[0] = update_value;
        updated_bias.mutable_cpu_data()[0] = bias.cpu_data()[0] - update_value;
      } else {
        updated_weights.mutable_cpu_diff()[i] = update_value;
        updated_weights.mutable_cpu_data()[i] = weights.cpu_data()[i]
            - update_value;
      }
    }
  }

  void CheckLeastSquaresUpdate(
      const vector<shared_ptr<Blob<Dtype> > >& updated_params) {
    const int D = channels_ * height_ * width_;

    const Blob<Dtype>& updated_weights = *updated_params[0];
    const Blob<Dtype>& updated_bias = *updated_params[1];

    Net<Dtype>& net = *this->solver_->net();
    ASSERT_TRUE(net.has_layer("innerprod"));
    const vector<shared_ptr<BlobBase > >& param_blobs = net.layer_by_name(
        "innerprod")->blob_bases();
    ASSERT_EQ(2, param_blobs.size());
    const Blob<Dtype>& solver_updated_weights =
        *(static_pointer_cast<Blob<Dtype> >(param_blobs[0]));
    ASSERT_EQ(D, solver_updated_weights.count());
    const double kPrecision = std::is_same<Dtype, half_fp>::value ?
        1e0 : 1e-2;
    const double kMinPrecision = std::is_same<Dtype, half_fp>::value ?
        3e-2 : 1e-7;
    for (int i = 0; i < D; ++i) {
      const Dtype expected_updated_weight = updated_weights.cpu_data()[i];
      const Dtype solver_updated_weight = solver_updated_weights.cpu_data()[i];
      const Dtype error_margin = std::max(
          kMinPrecision,
          kPrecision
              * std::min(fabs(expected_updated_weight),
                         fabs(solver_updated_weight)));
      EXPECT_NEAR(expected_updated_weight, solver_updated_weight, error_margin);
    }
    const Blob<Dtype>& solver_updated_bias_blob =
        *(static_pointer_cast<Blob<Dtype> >(param_blobs[1]));
    ASSERT_EQ(1, solver_updated_bias_blob.count());
    const Dtype expected_updated_bias = updated_bias.cpu_data()[0];
    const Dtype solver_updated_bias = solver_updated_bias_blob.cpu_data()[0];
    const Dtype error_margin = std::max(
        kMinPrecision,
        kPrecision
            * std::min(fabs(expected_updated_bias), fabs(solver_updated_bias)));
    EXPECT_NEAR(expected_updated_bias, solver_updated_bias, error_margin);

    // Check the solver's history -- should contain the previous update value.
    if (solver_->type() == string("SGD")) {
      const vector<shared_ptr<Blob<Dtype> > >& history = solver_->history();
      ASSERT_EQ(2, history.size());
      for (int i = 0; i < D; ++i) {
        const Dtype expected_history = updated_weights.cpu_diff()[i];
        const Dtype solver_history = history[0]->cpu_data()[i];
        const Dtype error_margin_hist = std::max(
            kMinPrecision,
            kPrecision
                * std::min(fabs(expected_history), fabs(solver_history)));
        EXPECT_NEAR(expected_history, solver_history, error_margin_hist);
      }
      const Dtype expected_history = updated_bias.cpu_diff()[0];
      const Dtype solver_history = history[1]->cpu_data()[0];
      const Dtype error_margin_hist = std::max(
          kMinPrecision,
          kPrecision * std::min(fabs(expected_history), fabs(solver_history)));
      EXPECT_NEAR(expected_history, solver_history, error_margin_hist);
    }
  }

  void CheckAccumulation(const Dtype kLearningRate, const Dtype kWeightDecay,
                         const Dtype kMomentum, const int kNumIters,
                         const int kIterSize) {
    const double kPrecision = std::is_same<Dtype, half_fp>::value ?
        1e0 : 1e-2;
    const double kMinPrecision = std::is_same<Dtype, half_fp>::value ?
        3e-2 : 1e-7;
    // Solve without accumulation and save parameters.
    this->RunLeastSquaresSolver(kLearningRate, kWeightDecay, kMomentum,
                                kNumIters);
    // Save parameters for comparison.
    Net<Dtype>& net = *this->solver_->net();
    const vector<shared_ptr<BlobBase> >& param_blobs = net.layer_by_name(
        "innerprod")->blob_bases();
    vector<shared_ptr<Blob<Dtype> > > noaccum_params(param_blobs.size());
    for (int i = 0; i < param_blobs.size(); ++i) {
      noaccum_params[i].reset(new Blob<Dtype>());
      noaccum_params[i]->CopyFrom(
          *(static_pointer_cast<Blob<Dtype> >(param_blobs[i])),
              false, true);
    }
    // Solve by equivalent accumulation of gradients over divided batches.
    this->RunLeastSquaresSolver(kLearningRate, kWeightDecay, kMomentum,
                                kNumIters, kIterSize);
    Net<Dtype>& net_accum = *this->solver_->net();
    const vector<shared_ptr<BlobBase > >& accum_params = net_accum
        .layer_by_name("innerprod")->blob_bases();
    // Compare accumulated parameters against no accumulation standard.
    const int D = this->channels_ * this->height_ * this->width_;
    for (int i = 0; i < D; ++i) {
      const Dtype expected_param = noaccum_params[0]->cpu_data()[i];
      const Dtype accum_param =
         static_pointer_cast<Blob<Dtype> >(accum_params[0])->cpu_data()[i];
      const Dtype error_margin = std::max(
          kMinPrecision,
          kPrecision * std::min(fabs(expected_param), fabs(accum_param)));
      EXPECT_NEAR(expected_param, accum_param, error_margin);
    }
    ASSERT_EQ(1, accum_params[1]->count());
    const Dtype expected_bias = noaccum_params[1]->cpu_data()[0];
    const Dtype accum_bias =
         static_pointer_cast<Blob<Dtype> >(accum_params[1])->cpu_data()[0];
    const Dtype error_margin = std::max(
        kMinPrecision,
        kPrecision * std::min(fabs(expected_bias), fabs(accum_bias)));
    EXPECT_NEAR(expected_bias, accum_bias, error_margin);
  }

  // Test that the correct update is computed for a regularized least squares
  // problem:
  //
  //            E = (1/(2n)) || X w - Y ||^2 + (lambda / 2) || w ||^2
  //   \nabla_w E = (1/n) (X^T X w - X^T Y) + lambda * w
  //
  // X \in R^{n X (d+1)} (each example is a row, (d+1)th element is always 1)
  // w \in R^{(d+1) X 1} ((d+1)th element is the bias)
  // Y \in R^{n X 1}
  // lambda is weight_decay
  //
  // TestLeastSquaresUpdate works "inductively", assuming that the solver
  // correctly updates the net k (= iter_to_check) times, then given the history
  // from the Kth update, we compute the (k+1)th update and check that it
  // matches the solver's (k+1)th update.
  void TestLeastSquaresUpdate(const Dtype learning_rate = 1.0,
                              const Dtype weight_decay = 0.0,
                              const Dtype momentum = 0.0,
                              const int iter_to_check = 0) {
    const int kNum = num_;
    const int kIterSize = 1;
    // Test over all numbers of devices.
    int available_devices = 1;
#ifdef USE_CUDA
#ifdef USE_NCCL
    if (Caffe::mode() == Caffe::GPU) {
      CUDA_CHECK(cudaGetDeviceCount(&available_devices));
    }
#endif  // USE_NCCL
#endif  // USE_CUDA
    // Takes a while to test all sizes for each test so sparse
    vector<int> sizes;
    sizes.push_back(1);
    if (available_devices >= 2) {
      sizes.push_back(2);
    }
    if (available_devices >= 3) {
      sizes.push_back(3);
    }
    if (available_devices >= 8) {
      sizes.push_back(8);
    }
    if (available_devices >= 16) {
      sizes.push_back(16);
    }
    for (int i = 0; i < sizes.size(); ++i) {
      int devices = sizes[i];
      // Configure batch size for single / multi device equivalence.
      // Constant data is needed for multi device as for accumulation.
      num_ = kNum * devices;

      // Initialize the solver and run k (= iter_to_check) solver iterations
      // (on single device).
      RunLeastSquaresSolver(learning_rate, weight_decay, momentum,
                            iter_to_check, kIterSize, 1);

      // Compute the (k+1)th update using the analytic least squares gradient.
      vector<shared_ptr<Blob<Dtype> > > updated_params;
      ComputeLeastSquaresUpdate(learning_rate, weight_decay, momentum,
                                iter_to_check + 1, &updated_params);

      // Reinitialize the solver and run k+1 solver iterations.
      num_ = kNum;
      RunLeastSquaresSolver(learning_rate, weight_decay, momentum,
                            iter_to_check + 1, kIterSize, devices);

      // Check that the solver's solution matches ours.
      CheckLeastSquaresUpdate(updated_params);
    }
  }

  void TestSnapshot(const Dtype learning_rate = 1.0, const Dtype weight_decay =
                        0.0,
                    const Dtype momentum = 0.0, const int num_iters = 1) {
    // Run the solver for num_iters * 2 iterations.
    const int total_num_iters = num_iters * 2;
    bool snapshot = false;
    const int kIterSize = 1;
    const int kDevices = 1;
    RunLeastSquaresSolver(learning_rate, weight_decay, momentum,
                          total_num_iters, kIterSize, kDevices, snapshot);

    // Save the resulting param values.
    vector<shared_ptr<Blob<Dtype> > > param_copies;
    const vector<BlobBase*>& orig_params =
        solver_->net()->learnable_params();
    param_copies.resize(orig_params.size());
    for (int i = 0; i < orig_params.size(); ++i) {
      param_copies[i].reset(new Blob<Dtype>());
      const bool kReshape = true;
      param_copies[i]->CopyFrom(
         *(static_cast<Blob<Dtype>*>(orig_params[i])),
         false/*copy data*/, kReshape);
      param_copies[i]->CopyFrom(
         *(static_cast<Blob<Dtype>*>(orig_params[i])),
         true/*copy diff*/, kReshape);
    }

    // Save the solver history
    vector<shared_ptr<Blob<Dtype> > > history_copies;
    const vector<shared_ptr<Blob<Dtype> > >& orig_history = solver_->history();
    history_copies.resize(orig_history.size());
    for (int i = 0; i < orig_history.size(); ++i) {
      history_copies[i].reset(new Blob<Dtype>());
      const bool kReshape = true;
      history_copies[i]->CopyFrom(*orig_history[i],
            false/*copy data*/, kReshape);
      history_copies[i]->CopyFrom(*orig_history[i],
            true/*copy diff*/, kReshape);
    }

    // Run the solver for num_iters iterations and snapshot.
    snapshot = true;
    string snapshot_name = RunLeastSquaresSolver(learning_rate, weight_decay,
                                                 momentum, num_iters, kIterSize,
                                                 kDevices, snapshot);

    // Reinitialize the solver and run for num_iters more iterations.
    snapshot = false;
    RunLeastSquaresSolver(learning_rate, weight_decay, momentum,
                          total_num_iters, kIterSize, kDevices, snapshot,
                          snapshot_name.c_str());

    // Check that params now match.
    const vector<BlobBase*>& params = solver_->net()->learnable_params();
    for (int i = 0; i < params.size(); ++i) {
      for (int j = 0; j < params[i]->count(); ++j) {
        EXPECT_FLOAT_EQ(param_copies[i]->cpu_data()[j],
            static_cast<Blob<Dtype>*>(params[i])->cpu_data()[j])
            << "param " << i << " data differed at dim " << j;
        EXPECT_FLOAT_EQ(param_copies[i]->cpu_diff()[j],
            static_cast<Blob<Dtype>*>(params[i])->cpu_diff()[j])
            << "param " << i << " diff differed at dim " << j;
      }
    }

    // Check that history now matches.
    const vector<shared_ptr<Blob<Dtype> > >& history = solver_->history();
    for (int i = 0; i < history.size(); ++i) {
      for (int j = 0; j < history[i]->count(); ++j) {
        EXPECT_FLOAT_EQ(history_copies[i]->cpu_data()[j],
            history[i]->cpu_data()[j])
            << "history blob " << i << " data differed at dim " << j;
        EXPECT_FLOAT_EQ(history_copies[i]->cpu_diff()[j],
            history[i]->cpu_diff()[j])
            << "history blob " << i << " diff differed at dim " << j;
      }
    }
  }
};

template<typename TypeParam>
class SGDSolverTest : public GradientBasedSolverTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  virtual void InitSolver(const SolverParameter& param) {
    this->solver_.reset(new SGDSolver<Dtype>(param,
                                             Caffe::GetDefaultDevice()));
  }
};

TYPED_TEST_CASE(SGDSolverTest, TestDtypesFloatAndDevices);

TYPED_TEST(SGDSolverTest, TestLeastSquaresUpdate) {
  this->TestLeastSquaresUpdate();
}

TYPED_TEST(SGDSolverTest, TestLeastSquaresUpdateLROneHundredth) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  this->TestLeastSquaresUpdate(kLearningRate);
}

TYPED_TEST(SGDSolverTest, TestLeastSquaresUpdateWithWeightDecay) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0;
  const int kNumIters = 1;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(SGDSolverTest, TestLeastSquaresUpdateWithWeightDecayMultiIter) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0;
  const int kNumIters = 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(SGDSolverTest, TestLeastSquaresUpdateWithMomentum) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0;
  const Dtype kMomentum = 0.5;
  const int kNumIters = 1;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(SGDSolverTest, TestLeastSquaresUpdateWithMomentumMultiIter) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0;
  const Dtype kMomentum = 0.5;
  const int kNumIters = 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(SGDSolverTest, TestLeastSquaresUpdateWithEverything) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0.5;
  const int kNumIters = 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(SGDSolverTest, TestLeastSquaresUpdateWithEverythingShare) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0.5;
  const int kNumIters = 4;
  this->share_ = true;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(SGDSolverTest, TestLeastSquaresUpdateWithEverythingAccum) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0.9;
  const int kNumIters = 4;
  const int kIterSize = 2;
  this->CheckAccumulation(kLearningRate, kWeightDecay, kMomentum, kNumIters,
      kIterSize);
}

TYPED_TEST(SGDSolverTest, TestLeastSquaresUpdateWithEverythingAccumShare) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0.9;
  const int kNumIters = 4;
  const int kIterSize = 2;
  this->share_ = true;
  this->CheckAccumulation(kLearningRate, kWeightDecay, kMomentum, kNumIters,
      kIterSize);
}

TYPED_TEST(SGDSolverTest, TestSnapshot) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0.9;
  const int kNumIters = 4;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSnapshot(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(SGDSolverTest, TestSnapshotShare) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0.9;
  const int kNumIters = 4;
  this->share_ = true;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSnapshot(kLearningRate, kWeightDecay, kMomentum, i);
  }
}


template <typename TypeParam>
class AdaGradSolverTest : public GradientBasedSolverTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  virtual void InitSolver(const SolverParameter& param) {
    this->solver_.reset(new AdaGradSolver<Dtype>(param,
                                                 Caffe::GetDefaultDevice()));
  }
};

TYPED_TEST_CASE(AdaGradSolverTest, TestDtypesFloatAndDevices);

TYPED_TEST(AdaGradSolverTest, TestAdaGradLeastSquaresUpdate) {
  this->TestLeastSquaresUpdate();
}

TYPED_TEST(AdaGradSolverTest, TestAdaGradLeastSquaresUpdateLROneHundredth) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  this->TestLeastSquaresUpdate(kLearningRate);
}

TYPED_TEST(AdaGradSolverTest, TestAdaGradLeastSquaresUpdateWithWeightDecay) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay);
}

TYPED_TEST(AdaGradSolverTest, TestAdaGradLeastSquaresUpdateWithEverything) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0;
  const int kNumIters = 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(AdaGradSolverTest,
    TestAdaGradLeastSquaresUpdateWithEverythingShare) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0;
  const int kNumIters = 4;
  this->share_ = true;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(AdaGradSolverTest, TestLeastSquaresUpdateWithEverythingAccum) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0;
  const int kNumIters = 4;
  const int kIterSize = 2;
  this->CheckAccumulation(kLearningRate, kWeightDecay, kMomentum, kNumIters,
      kIterSize);
}

TYPED_TEST(AdaGradSolverTest, TestLeastSquaresUpdateWithEverythingAccumShare) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0;
  const int kNumIters = 4;
  const int kIterSize = 2;
  this->share_ = true;
  this->CheckAccumulation(kLearningRate, kWeightDecay, kMomentum, kNumIters,
      kIterSize);
}

TYPED_TEST(AdaGradSolverTest, TestSnapshot) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0;
  const int kNumIters = 4;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSnapshot(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(AdaGradSolverTest, TestSnapshotShare) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0;
  const int kNumIters = 4;
  this->share_ = true;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSnapshot(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

template<typename TypeParam>
class NesterovSolverTest : public GradientBasedSolverTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  virtual void InitSolver(const SolverParameter& param) {
    this->solver_.reset(new NesterovSolver<Dtype>(param,
                                                  Caffe::GetDefaultDevice()));
  }
};

TYPED_TEST_CASE(NesterovSolverTest, TestDtypesFloatAndDevices);

TYPED_TEST(NesterovSolverTest, TestNesterovLeastSquaresUpdate) {
  this->TestLeastSquaresUpdate();
}

TYPED_TEST(NesterovSolverTest, TestNesterovLeastSquaresUpdateLROneHundredth) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  this->TestLeastSquaresUpdate(kLearningRate);
}

TYPED_TEST(NesterovSolverTest, TestNesterovLeastSquaresUpdateWithWeightDecay) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay);
}

TYPED_TEST(NesterovSolverTest,
    TestNesterovLeastSquaresUpdateWithWeightDecayMultiIter) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0;
  const int kNumIters = 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(NesterovSolverTest, TestNesterovLeastSquaresUpdateWithMomentum) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0;
  const Dtype kMomentum = 0.5;
  const int kNumIters = 1;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(NesterovSolverTest, TestLeastSquaresUpdateWithMomentumMultiIter) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0;
  const Dtype kMomentum = 0.5;
  const int kNumIters = 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(NesterovSolverTest, TestNesterovLeastSquaresUpdateWithEverything) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0.9;
  const int kNumIters = 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(NesterovSolverTest,
    TestNesterovLeastSquaresUpdateWithEverythingShare) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0.9;
  const int kNumIters = 4;
  this->share_ = true;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(NesterovSolverTest, TestLeastSquaresUpdateWithEverythingAccum) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0.9;
  const int kNumIters = 4;
  const int kIterSize = 2;
  this->CheckAccumulation(kLearningRate, kWeightDecay, kMomentum, kNumIters,
      kIterSize);
}

TYPED_TEST(NesterovSolverTest, TestLeastSquaresUpdateWithEverythingAccumShare) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0.9;
  const int kNumIters = 4;
  const int kIterSize = 2;
  this->share_ = true;
  this->CheckAccumulation(kLearningRate, kWeightDecay, kMomentum, kNumIters,
      kIterSize);
}

TYPED_TEST(NesterovSolverTest, TestSnapshot) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0.9;
  const int kNumIters = 4;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSnapshot(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(NesterovSolverTest, TestSnapshotShare) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0.9;
  const int kNumIters = 4;
  this->share_ = true;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSnapshot(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

template<typename TypeParam>
class AdaDeltaSolverTest : public GradientBasedSolverTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  virtual void InitSolver(const SolverParameter& param) {
    this->solver_.reset(new AdaDeltaSolver<Dtype>(param,
                                                  Caffe::GetDefaultDevice()));
  }
};

TYPED_TEST_CASE(AdaDeltaSolverTest, TestDtypesFloatAndDevices);

TYPED_TEST(AdaDeltaSolverTest, TestAdaDeltaLeastSquaresUpdate) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.1;
  this->TestLeastSquaresUpdate(kLearningRate);
}

TYPED_TEST(AdaDeltaSolverTest, TestAdaDeltaLeastSquaresUpdateWithWeightDecay) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.1;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0.95;
  this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum);
}

TYPED_TEST(AdaDeltaSolverTest, TestAdaDeltaLeastSquaresUpdateWithHalfMomentum) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.1;
  const Dtype kWeightDecay = 0.0;
  const Dtype kMomentum = 0.5;
  const int kNumIters = 1;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum);
  }
}

TYPED_TEST(AdaDeltaSolverTest, TestAdaDeltaLeastSquaresUpdateWithMomentum) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.1;
  const Dtype kWeightDecay = 0.0;
  const Dtype kMomentum = 0.95;
  const int kNumIters = 1;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum);
  }
}

TYPED_TEST(AdaDeltaSolverTest, TestLeastSquaresUpdateWithMomentumMultiIter) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.1;
  const Dtype kWeightDecay = 0.0;
  const Dtype kMomentum = 0.95;
  const int kNumIters = 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(AdaDeltaSolverTest, TestAdaDeltaLeastSquaresUpdateWithEverything) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.1;
  const Dtype kWeightDecay = 0.1;
  const Dtype kMomentum = 0.95;
  const int kNumIters = 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(AdaDeltaSolverTest,
    TestAdaDeltaLeastSquaresUpdateWithEverythingShare) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.1;
  const Dtype kWeightDecay = 0.1;
  const Dtype kMomentum = 0.95;
  const int kNumIters = 4;
  this->share_ = true;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(AdaDeltaSolverTest, TestLeastSquaresUpdateWithEverythingAccum) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.1;
  const Dtype kWeightDecay = 0.1;
  const Dtype kMomentum = 0.95;
  const int kNumIters = 4;
  const int kIterSize = 2;
  this->CheckAccumulation(kLearningRate, kWeightDecay, kMomentum, kNumIters,
      kIterSize);
}

TYPED_TEST(AdaDeltaSolverTest, TestLeastSquaresUpdateWithEverythingAccumShare) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.1;
  const Dtype kWeightDecay = 0.1;
  const Dtype kMomentum = 0.95;
  const int kNumIters = 4;
  const int kIterSize = 2;
  this->share_ = true;
  this->CheckAccumulation(kLearningRate, kWeightDecay, kMomentum, kNumIters,
      kIterSize);
}

TYPED_TEST(AdaDeltaSolverTest, TestSnapshot) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.1;
  const Dtype kWeightDecay = 0.1;
  const Dtype kMomentum = 0.95;
  const int kNumIters = 4;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSnapshot(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(AdaDeltaSolverTest, TestSnapshotShare) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.1;
  const Dtype kWeightDecay = 0.1;
  const Dtype kMomentum = 0.95;
  const int kNumIters = 4;
  this->share_ = true;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSnapshot(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

template<typename TypeParam>
class AdamSolverTest : public GradientBasedSolverTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  virtual void InitSolver(const SolverParameter& param) {
    SolverParameter new_param = param;
    const Dtype momentum = 0.9;
    new_param.set_momentum(momentum);
    const Dtype momentum2 = 0.999;
    new_param.set_momentum2(momentum2);
    this->solver_.reset(new AdamSolver<Dtype>(new_param,
                                              Caffe::GetDefaultDevice()));
  }
};

TYPED_TEST_CASE(AdamSolverTest, TestDtypesFloatAndDevices);

TYPED_TEST(AdamSolverTest, TestAdamLeastSquaresUpdate) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0;
  const Dtype kMomentum = 0.9;
  this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum);
}

TYPED_TEST(AdamSolverTest, TestAdamLeastSquaresUpdateWithWeightDecay) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0.9;
  this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum);
}

TYPED_TEST(AdamSolverTest, TestAdamLeastSquaresUpdateWithEverything) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0.9;
  const int kNumIters = 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(AdamSolverTest, TestAdamLeastSquaresUpdateWithEverythingShare) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0.9;
  const int kNumIters = 4;
  this->share_ = true;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(AdamSolverTest, TestLeastSquaresUpdateWithEverythingAccum) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0.9;
  const int kNumIters = 4;
  const int kIterSize = 2;
  this->CheckAccumulation(kLearningRate, kWeightDecay, kMomentum, kNumIters,
      kIterSize);
}

TYPED_TEST(AdamSolverTest, TestLeastSquaresUpdateWithEverythingAccumShare) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0.9;
  const int kNumIters = 4;
  const int kIterSize = 2;
  this->share_ = true;
  this->CheckAccumulation(kLearningRate, kWeightDecay, kMomentum, kNumIters,
      kIterSize);
}

TYPED_TEST(AdamSolverTest, TestSnapshot) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0.9;
  const int kNumIters = 4;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSnapshot(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(AdamSolverTest, TestSnapshotShare) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0.9;
  const int kNumIters = 4;
  this->share_ = true;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSnapshot(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

template<typename TypeParam>
class RMSPropSolverTest : public GradientBasedSolverTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  virtual void InitSolver(const SolverParameter& param) {
    const Dtype rms_decay = 0.95;
    SolverParameter new_param = param;
    new_param.set_rms_decay(rms_decay);
    this->solver_.reset(new RMSPropSolver<Dtype>(new_param,
                                                 Caffe::GetDefaultDevice()));
  }
};

TYPED_TEST_CASE(RMSPropSolverTest, TestDtypesFloatAndDevices);

TYPED_TEST(RMSPropSolverTest, TestRMSPropLeastSquaresUpdateWithWeightDecay) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 1.0;
  const Dtype kWeightDecay = 0.5;
  this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay);
}

TYPED_TEST(RMSPropSolverTest, TestRMSPropLeastSquaresUpdateWithRmsDecay) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.0;
  const Dtype kMomentum = 0.0;
  const int kNumIters = 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(RMSPropSolverTest, TestRMSPropLeastSquaresUpdateWithEverything) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0.0;
  const int kNumIters = 4;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(RMSPropSolverTest,
    TestRMSPropLeastSquaresUpdateWithEverythingShare) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0.0;
  const int kNumIters = 4;
  this->share_ = true;
  for (int i = 0; i <= kNumIters; ++i) {
    this->TestLeastSquaresUpdate(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(RMSPropSolverTest, TestLeastSquaresUpdateWithEverythingAccum) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0.0;
  const int kNumIters = 4;
  const int kIterSize = 2;
  this->CheckAccumulation(kLearningRate, kWeightDecay, kMomentum, kNumIters,
      kIterSize);
}

TYPED_TEST(RMSPropSolverTest, TestLeastSquaresUpdateWithEverythingAccumShare) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0.0;
  const int kNumIters = 4;
  const int kIterSize = 2;
  this->share_ = true;
  this->CheckAccumulation(kLearningRate, kWeightDecay, kMomentum, kNumIters,
      kIterSize);
}

TYPED_TEST(RMSPropSolverTest, TestSnapshot) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0;
  const int kNumIters = 4;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSnapshot(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

TYPED_TEST(RMSPropSolverTest, TestSnapshotShare) {
  typedef typename TypeParam::Dtype Dtype;
  const Dtype kLearningRate = 0.01;
  const Dtype kWeightDecay = 0.5;
  const Dtype kMomentum = 0;
  const int kNumIters = 4;
  this->share_ = true;
  for (int i = 1; i <= kNumIters; ++i) {
    this->TestSnapshot(kLearningRate, kWeightDecay, kMomentum, i);
  }
}

}  // namespace caffe
