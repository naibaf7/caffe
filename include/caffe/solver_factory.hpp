/**
 * @brief a solver factory that allows one to register solvers, similar to
 * layer factory. During runtime, registered solvers could be called by passing
 * a SolverParameter protobuffer to the CreateSolver function:
 *
 *     SolverRegistry<Dtype>::CreateSolver(param);
 *
 * There are two ways to register a solver. Assuming that we have a solver like:
 *
 *   template <typename Dtype>
 *   class MyAwesomeSolver : public Solver<Dtype> {
 *     // your implementations
 *   };
 *
 * and its type is its c++ class name, but without the "Solver" at the end
 * ("MyAwesomeSolver" -> "MyAwesome").
 *
 * If the solver is going to be created simply by its constructor, in your c++
 * file, add the following line:
 *
 *    REGISTER_SOLVER_CLASS(MyAwesome);
 *
 * Or, if the solver is going to be created by another creator function, in the
 * format of:
 *
 *    template <typename Dtype>
 *    Solver<Dtype*> GetMyAwesomeSolver(const SolverParameter& param) {
 *      // your implementation
 *    }
 *
 * then you can register the creator function instead, like
 *
 * REGISTER_SOLVER_CREATOR(MyAwesome, GetMyAwesomeSolver)
 *
 * Note that each solver type should only be registered once.
 */

#ifndef CAFFE_SOLVER_FACTORY_H_
#define CAFFE_SOLVER_FACTORY_H_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class Solver;

template <typename Dtype>
class SolverRegistry {
 public:
  typedef Solver<Dtype>* (*Creator)(const SolverParameter&, Device* dev);
  typedef std::map<string, Creator> CreatorRegistry;

  static CreatorRegistry& Registry();

  // Adds a creator.
  static void AddCreator(const string& type, Creator creator);

  // Get a solver using a SolverParameter.
  static Solver<Dtype>* CreateSolver(const SolverParameter& param, Device* dev);

  static vector<string> SolverTypeList();

 private:
  // Solver registry should never be instantiated - everything is done with its
  // static variables.
  SolverRegistry();  // {}

  static string SolverTypeListString();
};

template <typename Dtype>
class SolverRegisterer {
 public:
  SolverRegisterer(const string& type,
                   Solver<Dtype>* (*creator)(const SolverParameter&,
                   Device* dev));
};

}  // namespace caffe

#endif  // CAFFE_SOLVER_FACTORY_H_
