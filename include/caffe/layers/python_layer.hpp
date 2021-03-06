#ifndef CAFFE_PYTHON_LAYER_HPP_
#define CAFFE_PYTHON_LAYER_HPP_

#include <boost/python.hpp>
#include <vector>

#include "caffe/layer.hpp"

namespace bp = boost::python;

namespace caffe {

template<typename Dtype, typename MItype, typename MOtype>
class PythonLayer : public Layer<Dtype, MItype, MOtype> {
 public:
  PythonLayer(PyObject* self, const LayerParameter& param)
      : Layer<Dtype, MItype, MOtype>(param),
        self_(bp::handle<>(bp::borrowed(self))) { }

  virtual void LayerSetUp(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top) {
    // Disallow PythonLayer in MultiGPU training stage, due to GIL issues
    // Details: https://github.com/BVLC/caffe/issues/2936
    if (this->phase_ == TRAIN && Caffe::solver_count() > 1
        && !Caffe::multiprocess()) {
      LOG(FATAL) << "PythonLayer does not support CLI Multi-GPU, use train.py";
    }
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    self_.attr("param_str") = bp::str(
        this->layer_param_.python_param().param_str());
    self_.attr("phase") = static_cast<int>(this->phase_);
    self_.attr("setup")(bottom, top);
    PyGILState_Release(gstate);
  }
  virtual void Reshape(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top) {
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    self_.attr("reshape")(bottom, top);
    PyGILState_Release(gstate);
  }

  virtual inline const char* type() const { return "Python"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<MItype>*>& bottom,
      const vector<Blob<MOtype>*>& top) {
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    self_.attr("forward")(bottom, top);
    PyGILState_Release(gstate);
  }
  virtual void Backward_cpu(const vector<Blob<MOtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<MItype>*>& bottom) {
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    self_.attr("backward")(top, propagate_down, bottom);
    PyGILState_Release(gstate);
  }

 private:
  bp::object self_;
};

}  // namespace caffe

#endif  // CAFFE_PYTHON_LAYER_HPP_
