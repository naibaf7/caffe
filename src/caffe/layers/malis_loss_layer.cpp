#include <boost/pending/disjoint_sets.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <map>
#include <numeric>
#include <queue>
#include <utility>
#include <vector>


#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

//#define CAFFE_MALIS_DEBUG

namespace caffe {

template<class Dtype>
class MalisAffinityGraphCompare {
 private:
  const Dtype * mEdgeWeightArray;
 public:
  explicit MalisAffinityGraphCompare(const Dtype * EdgeWeightArray) {
    mEdgeWeightArray = EdgeWeightArray;
  }
  bool operator()(const int64_t& ind1, const int64_t& ind2) const {
    return (mEdgeWeightArray[ind1] > mEdgeWeightArray[ind2]);
  }
};

// Derived from https://github.com/srinituraga/malis/blob/master/matlab/malis_loss_mex.cpp
// conn_data:   4d connectivity graph [y * x * z * #edges]
// nhood_data:  graph neighborhood descriptor [3 * #edges]
// seg_data:    true target segmentation [y * x * z]
// pos:         is this a positive example pass [true] or
//              a negative example pass [false] ?
// margin:      sq-sq loss margin [0.3]
template<typename Dtype>
void MalisLossLayer<Dtype>::Malis(Dtype* conn_data, int conn_num_dims,
                                  int* conn_dims, int* nhood_data,
                                  int* nhood_dims, int* seg_data, bool pos,
                                  Dtype* dloss_data, Dtype* loss_out,
                                  Dtype *classerr_out, Dtype *rand_index_out,
                                  Dtype margin) {
  Dtype threshold = 0.5;

  if ((nhood_dims[1] != (conn_num_dims - 1))
      || (nhood_dims[0] != conn_dims[conn_num_dims - 1])) {
    LOG(FATAL) << "nhood and conn dimensions don't match";
  }

  /* Cache for speed to access neighbors */
  // nVert stores (x * y * z)
  int64_t nVert = 1;
  for (int64_t i = 0; i < conn_num_dims - 1; ++i) {
    nVert = nVert * conn_dims[i];
  }

  // prodDims stores x, x*y, x*y*z offsets
  std::vector<int64_t> prodDims(conn_num_dims - 1);
  prodDims[0] = 1;
  for (int64_t i = 1; i < conn_num_dims - 1; ++i) {
    prodDims[i] = prodDims[i - 1] * conn_dims[i - 1];
  }

  /* convert n-d offset vectors into linear array offset scalars */
  // nHood is a vector of size #edges
  std::vector<int32_t> nHood(nhood_dims[0]);
  for (int64_t i = 0; i < nhood_dims[0]; ++i) {
    nHood[i] = 0;
    for (int64_t j = 0; j < nhood_dims[1]; ++j) {
      nHood[i] += (int32_t) nhood_data[i + j * nhood_dims[0]] * prodDims[j];
    }
  }

  /* Disjoint sets and sparse overlap vectors */
  std::vector<std::map<int64_t, int64_t> > overlap(nVert);
  std::vector<int64_t> rank(nVert);
  std::vector<int64_t> parent(nVert);
  std::map<int64_t, int64_t> segSizes;
  int64_t nLabeledVert = 0;
  int64_t nPairPos = 0;
  boost::disjoint_sets<int64_t*, int64_t*> dsets(&rank[0], &parent[0]);
  // Loop over all seg data items
  for (int64_t i = 0; i < nVert; ++i) {
    dsets.make_set(i);
    if (0 != seg_data[i]) {
      overlap[i].insert(std::pair<int64_t, int64_t>(seg_data[i], 1));
      ++nLabeledVert;
      ++segSizes[seg_data[i]];
      nPairPos += (segSizes[seg_data[i]] - 1);
    }
  }
  int64_t nPairTot = (nLabeledVert * (nLabeledVert - 1)) / 2;
  int64_t nPairNeg = nPairTot - nPairPos;
  int64_t nPairNorm;
  if (pos) {
    nPairNorm = nPairPos;
  } else {
    nPairNorm = nPairNeg;
  }

  /* Sort all the edges in increasing order of weight */
  std::vector<int64_t> pqueue(
      conn_dims[3] * std::max((conn_dims[0] - 1), 1)
                   * std::max((conn_dims[1] - 1), 1)
                   * std::max((conn_dims[2] - 1), 1));
  int64_t j = 0;
  // Loop over #edges
  for (int64_t d = 0, i = 0; d < conn_dims[3]; ++d) {
    // Loop over Z
    for (int64_t z = 0; z < conn_dims[2]; ++z) {
      // Loop over Y
      for (int64_t y = 0; y < conn_dims[1]; ++y) {
        // Loop over X
        for (int64_t x = 0; x < conn_dims[0]; ++x, ++i) {
          if (x < std::max(conn_dims[0] - 1, 1) &&
              y < std::max(conn_dims[1] - 1, 1) &&
              z < std::max(conn_dims[2] - 1, 1)) {
            pqueue[j++] = i;
          }
        }
      }
    }
  }

  pqueue.resize(j);

  std::sort(pqueue.begin(), pqueue.end(),
       MalisAffinityGraphCompare<Dtype>(conn_data));

  /* Start MST */
  int64_t minEdge;
  int64_t e, v1, v2;
  int64_t set1, set2;
  int64_t nPair = 0;
  double loss = 0, dl = 0;
  int64_t nPairIncorrect = 0;
  std::map<int64_t, int64_t>::iterator it1, it2;

  /* Start Kruskal's */
  for (int64_t i = 0; i < pqueue.size(); ++i) {
    minEdge = pqueue[i];
    // nVert = x * y * z, minEdge in [0, x * y * z * #edges]

    // e: edge dimension (0: X, 1: Y, 2: Z)
    e = minEdge / nVert;

    // v1: node at edge beginning
    v1 = minEdge % nVert;

    // v2: neighborhood node at edge e
    v2 = v1 + nHood[e];

    set1 = dsets.find_set(v1);
    set2 = dsets.find_set(v2);


    if (set1 != set2) {
      dsets.link(set1, set2);

      /* compute the dloss for this MST edge */
      for (it1 = overlap[set1].begin(); it1 != overlap[set1].end(); ++it1) {
        for (it2 = overlap[set2].begin(); it2 != overlap[set2].end(); ++it2) {
          nPair = it1->second * it2->second;

          if (pos && (it1->first == it2->first)) {
            // +ve example pairs
            dl = std::max(Dtype(0.0), threshold + margin - conn_data[minEdge]);
            loss += dl * nPair;
            dloss_data[minEdge] -= (dl > 0) * nPair;
            if (conn_data[minEdge] <= threshold) {  // an error
              nPairIncorrect += nPair;
            }

          } else if ((!pos) && (it1->first != it2->first)) {
            // -ve example pairs
            dl = std::max(Dtype(0.0), conn_data[minEdge] - threshold + margin);
            loss += dl * nPair;
            dloss_data[minEdge] += (dl > 0) * nPair;
            if (conn_data[minEdge] > threshold) {  // an error
              nPairIncorrect += nPair;
            }
          }
        }
      }

      dloss_data[minEdge] /= nPairNorm;

      if (dsets.find_set(set1) == set2) {
        std::swap(set1, set2);
      }

      for (it2 = overlap[set2].begin();
          it2 != overlap[set2].end(); ++it2) {
        it1 = overlap[set1].find(it2->first);
        if (it1 == overlap[set1].end()) {
          overlap[set1].insert(pair<int64_t, int64_t>
            (it2->first, it2->second));
        } else {
          it1->second += it2->second;
        }
      }
      overlap[set2].clear();
    }  // end link
  }  // end while

  /* Return items */
  double classerr, randIndex;
  loss /= nPairNorm;
  *loss_out = loss;
  classerr = static_cast<double>(nPairIncorrect)
      / static_cast<double>(nPairNorm);
  *classerr_out = classerr;
  randIndex = 1.0 - static_cast<double>(nPairIncorrect)
      / static_cast<double>(nPairNorm);
  *rand_index_out = randIndex;
}

// Derived from
// http://nghiaho.com/uploads/code/opencv_connected_component/blob.cpp
template<typename Dtype>
cv::Mat MalisLossLayer<Dtype>::FindBlobs(const cv::Mat &input) {
  // Fill the label_image with the blobs
  cv::Mat label_image;
  input.convertTo(label_image, CV_32SC1);

  // Segment into label numbers higher than the original label numbers
  int label_count = prob_.channels();

  for (int y = 0; y < label_image.rows; y++) {
    int *row = reinterpret_cast<int*>(label_image.ptr(y));
    for (int x = 0; x < label_image.cols; x++) {
      // Skip background and already labeled areas
      if (row[x] >= prob_.channels() || row[x] == 0) {
        continue;
      }
      cv::Rect rect;
      cv::floodFill(label_image, cv::Point(x, y), label_count, &rect, 0, 0, 4);
      label_count++;
    }
  }
  return label_image;
}

template<typename Dtype>
void MalisLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  // Set up the softmax layer
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
}

template<typename Dtype>
void MalisLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
  << "Number of labels must match number of predictions; "
  << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
  << "label count (number of labels) must be N*H*W, "
  << "with integer values in {0, 1, ..., C-1}.";
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }

  conn_dims_.clear();
  nhood_dims_.clear();
  nhood_data_.clear();

  conn_num_dims_ = 4;
  conn_dims_.push_back(bottom[0]->width());       // X-axis
  conn_dims_.push_back(bottom[0]->height());      // Y-axis
  conn_dims_.push_back(1);                        // Z-axis
  conn_dims_.push_back(2);                        // #edges

  nhood_dims_.push_back(2);                       // #edges
  nhood_dims_.push_back(3);                       // 3 dimensional

  nhood_data_.push_back(1);                       // Edge 1, X
  nhood_data_.push_back(0);                       // Edge 2, X

  nhood_data_.push_back(0);                       // Edge 1, Y
  nhood_data_.push_back(1);                       // Edge 2, Y

  nhood_data_.push_back(0);                       // Edge 1, Z
  nhood_data_.push_back(0);                       // Edge 2, Z
}

template<typename Dtype>
void MalisLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int dim = prob_.count() / outer_num_;
  int count = 0;
  Dtype loss = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.shape(softmax_axis_));
      loss -= log(
          std::max(prob_data[i * dim + label_value * inner_num_ + j],
                   Dtype(FLT_MIN)));
      ++count;
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / count;
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template<typename Dtype>
void MalisLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                         const vector<bool>& propagate_down,
                                         const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL)<< this->type()
    << " Layer cannot backpropagate to label inputs.";
  }

  if (propagate_down[0]) {
    // Diff to propagate to (size w * h * c)
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    // The predictions (size w * h * c)
    const Dtype* prob_data = prob_.cpu_data();

    // Labels (size w * h, c values)
    const Dtype* label = bottom[1]->cpu_data();

#ifdef CAFFE_MALIS_DEBUG
    cv::namedWindow("labelled");
    cv::namedWindow("cdn");
    cv::namedWindow("cdp");
    cv::namedWindow("prob");
    cv::namedWindow("diff");
#endif

    cv::Mat img(bottom[1]->height(), bottom[1]->width(), CV_8SC1);
#pragma omp parallel for
    for (int y = 0; y < bottom[1]->height(); ++y) {
      for (int x = 0; x < bottom[1]->width(); ++x) {
        img.at<unsigned char>(y, x) = label[y * bottom[1]->width() + x];
      }
    }

    cv::Mat seg = FindBlobs(img);

#ifdef CAFFE_MALIS_DEBUG
    // This is for debugging only:
    {
      std::vector<int> labels;

      for (int i = 0; i < bottom[1]->height() *bottom[1]->width(); ++i) {
        int val = reinterpret_cast<int*>(seg.ptr(0))[i];
        bool found = false;
        for (int j = 0; j < labels.size(); ++j) {
          if (val == labels[j]) {
            found = true;
          }
        }
        if (found == false) {
          labels.push_back(val);
        }
      }

      std::vector<cv::Vec3b> colors;

      for (int i = 0; i < labels.size(); ++i) {
        unsigned char r = 255 * (rand() / (1.0 + RAND_MAX));  // NOLINT
        unsigned char g = 255 * (rand() / (1.0 + RAND_MAX));  // NOLINT
        unsigned char b = 255 * (rand() / (1.0 + RAND_MAX));  // NOLINT

        cv::Vec3b color(r, g, b);
        colors.push_back(color);
      }

      cv::Mat output = cv::Mat::zeros(img.size(), CV_8UC3);

      for (int i = 0; i < bottom[1]->height() *bottom[1]->width(); ++i) {
        int val = reinterpret_cast<int*>(seg.ptr(0))[i];
        if (val == 0) {
          output.at<cv::Vec3b>(i) = cv::Vec3b(0, 0, 0);
          continue;
        }
        for (int j = 0; j < labels.size(); ++j) {
          if (val == labels[j]) {
            output.at<cv::Vec3b>(i) = colors[j];
          }
        }
      }

      cv::imshow("labelled", output);
    }
#endif

    Dtype loss_out = 0;
    Dtype classerr_out = 0;
    Dtype rand_index_out = 0;

    std::vector<Dtype> conn_data_pos(
        2 * bottom[0]->height() * bottom[0]->width());
    std::vector<Dtype> conn_data_neg(
        2 * bottom[0]->height() * bottom[0]->width());
    std::vector<Dtype> dloss_pos(
        2 * bottom[0]->height() * bottom[0]->width());
    std::vector<Dtype> dloss_neg(
        2 * bottom[0]->height() * bottom[0]->width());

    // Construct positive and negative affinity graph
#pragma omp parallel for
    for (int i = 0; i < bottom[0]->height() - 1; ++i) {
      for (int j = 0; j < bottom[0]->width() - 1; ++j) {
        // Center
        Dtype p0 = prob_data[bottom[0]->width()
                             * bottom[0]->height()
                             + i * bottom[0]->width() + j];
        // Right
        Dtype p1 = prob_data[bottom[0]->width()
                             * bottom[0]->height()
                             + i * bottom[0]->width() + (j + 1)];
        // Bottom
        Dtype p2 = prob_data[bottom[0]->width()
                             * bottom[0]->height()
                             + (i + 1) * bottom[0]->width() + j];

        // Center
        Dtype g0 = label[i * bottom[0]->width() + j];
        // Right
        Dtype g1 = label[i * bottom[0]->width() + (j + 1)];
        // Bottom
        Dtype g2 = label[(i + 1) * bottom[0]->width() + j];

        // X positive
        conn_data_pos[i * bottom[0]->width() + j] = std::min(
           std::min(1.0 - std::fabs(p0 - p1), (p0 + p1) / 2.0),
           std::min(1.0 - std::fabs(g0 - g1), (g0 + g1) / 2.0));

        // X negative
        conn_data_neg[i * bottom[0]->width() + j] = std::max(
           std::min(1.0 - std::fabs(p0 - p1), (p0 + p1) / 2.0),
           std::min(1.0 - std::fabs(g0 - g1), (g0 + g1) / 2.0));

        // Y positive
        conn_data_pos[bottom[0]->width() * bottom[0]->height()
            + i * bottom[0]->width() + j] = std::min(
           std::min(1.0 - std::fabs(p0 - p2), (p0 + p2) / 2.0),
           std::min(1.0 - std::fabs(g0 - g2), (g0 + g2) / 2.0));

        // Y negative
        conn_data_neg[bottom[0]->width() * bottom[0]->height()
            + i * bottom[0]->width() + j] = std::max(
           std::min(1.0 - std::fabs(p0 - p2), (p0 + p2) / 2.0),
           std::min(1.0 - std::fabs(g0 - g2), (g0 + g2) / 2.0));
      }
    }

#ifdef CAFFE_MALIS_DEBUG
    auto minmax = std::minmax_element(conn_data_neg.begin(),
                                      conn_data_neg.end());

    std::cout << "Conndata neg min/max: " <<
        conn_data_neg[minmax.first - conn_data_neg.begin()] << " " <<
        conn_data_neg[minmax.second - conn_data_neg.begin()]  << std::endl;

    minmax = std::minmax_element(conn_data_pos.begin(),
                                 conn_data_pos.end());

    std::cout << "Conndata pos min/max: " <<
        conn_data_pos[minmax.first - conn_data_pos.begin()] << " " <<
        conn_data_pos[minmax.second - conn_data_pos.begin()]  << std::endl;
#endif

    Malis(&conn_data_neg[0], conn_num_dims_, &conn_dims_[0], &nhood_data_[0],
          &nhood_dims_[0], reinterpret_cast<int*>(seg.ptr(0)),
          false, &dloss_neg[0],
          &loss_out, &classerr_out, &rand_index_out);

#ifdef CAFFE_MALIS_DEBUG
    std::cout << "Loss: " << loss_out << std::endl;
    std::cout << "Class: " << classerr_out << std::endl;
    std::cout << "Rand: " << rand_index_out << std::endl;
#endif

    Malis(&conn_data_pos[0], conn_num_dims_, &conn_dims_[0], &nhood_data_[0],
          &nhood_dims_[0], reinterpret_cast<int*>(seg.ptr(0)),
          true, &dloss_pos[0],
          &loss_out, &classerr_out, &rand_index_out);

#ifdef MALIS_DEBUG
    std::cout << "Loss: " << loss_out << std::endl;
    std::cout << "Class: " << classerr_out << std::endl;
    std::cout << "Rand: " << rand_index_out << std::endl;

    minmax = std::minmax_element(dloss_neg.begin(), dloss_neg.end());

    std::cout << "DLoss_neg min/max: " <<
        dloss_neg[minmax.first - dloss_neg.begin()] << " " <<
        dloss_neg[minmax.second - dloss_neg.begin()]  << std::endl;

    minmax = std::minmax_element(dloss_pos.begin(), dloss_pos.end());

    std::cout << "DLoss_pos min/max: " <<
        dloss_pos[minmax.first - dloss_pos.begin()] << " " <<
        dloss_pos[minmax.second - dloss_pos.begin()]  << std::endl;

    cv::Mat cd_pos(bottom[0]->height(), bottom[0]->width(),
                   cv::DataType<Dtype>::type,
                   &dloss_pos[0], sizeof(Dtype) * bottom[0]->width());

    double minVal, maxVal;
    cv::Mat tmp;

    cv::minMaxLoc(cd_pos, &minVal, &maxVal);
    cd_pos.convertTo(tmp, CV_32FC1, 1.0 / (maxVal - minVal),
        -minVal * 1.0 / (maxVal - minVal));

    cv::imshow("cdp", tmp);

    cv::Mat cd_neg(bottom[0]->height(), bottom[0]->width(),
                   cv::DataType<Dtype>::type,
                   &dloss_neg[0], sizeof(Dtype) * bottom[0]->width());

    cv::minMaxLoc(cd_neg, &minVal, &maxVal);

    cd_neg.convertTo(tmp, CV_32FC1, 1.0 / (maxVal - minVal),
        -minVal * 1.0 / (maxVal - minVal));
    cv::imshow("cdn", tmp);
#endif

    // Clear the diff
    caffe_set(bottom[0]->count(), Dtype(0.0), bottom_diff);

    // Spread out the losses to pixels
    for (int i = 0; i < bottom[0]->height() - 1; ++i) {
      for (int j = 0; j < bottom[0]->width() - 1; ++j) {
        Dtype lxp = dloss_pos[i * bottom[0]->width() + j];
        Dtype lxn = dloss_neg[i * bottom[0]->width() + j];

        Dtype lyp = dloss_pos[bottom[0]->width()
            * bottom[0]->height() + i * bottom[0]->width() + j];
        Dtype lyn = dloss_neg[bottom[0]->width()
            * bottom[0]->height() + i * bottom[0]->width() + j];

        // Pick label scalings
        /*const int l0 = static_cast<int>
          (label[i * bottom[0]->width() + j]) * 2 - 1;
        const int l1 = static_cast<int>
          (label[i * bottom[0]->width() + (j + 1)]) * 2 - 1;
        const int l2 = static_cast<int>
          (label[(i + 1) * bottom[0]->width() + j]) * 2 - 1;*/

        // Center
        bottom_diff[0 * inner_num_ + i * bottom[0]->width() + j] -= 0.5
             * (lxp + lxn + lyp + lyn);

        // Right
        bottom_diff[0 * inner_num_ + i * bottom[0]->width() + (j + 1)] -= 0.5
             * (lxp + lxn);

        // Bottom
        bottom_diff[0 * inner_num_ + (i + 1) * bottom[0]->width() + j] -= 0.5
             * (lyp + lyn);


        // Center
        bottom_diff[1 * inner_num_ + i * bottom[0]->width() + j] += 0.5
             * (lxp + lxn + lyp + lyn);

        // Right
        bottom_diff[1 * inner_num_ + i * bottom[0]->width() + (j + 1)] += 0.5
             * (lxp + lxn);

        // Bottom
        bottom_diff[1 * inner_num_ + (i + 1) * bottom[0]->width() + j] += 0.5
             * (lyp + lyn);
      }
    }

#ifdef CAFFE_MALIS_DEBUG
    Dtype* prob_rd = prob_.mutable_cpu_data();

    cv::Mat wrapped_prob(bottom[0]->height(), bottom[0]->width(),
                      cv::DataType<Dtype>::type,
                    prob_rd, sizeof(Dtype) * bottom[0]->width());
    cv::imshow("prob", wrapped_prob);

    cv::Mat wrapped_diff(bottom[0]->height(), bottom[0]->width(),
                      cv::DataType<Dtype>::type,
                    bottom_diff, sizeof(Dtype) * bottom[0]->width());

    cv::minMaxLoc(wrapped_diff, &minVal, &maxVal);

    std::cout << "Max loss: " << maxVal << std::endl;
    std::cout << "Min loss: " << minVal << std::endl;


    Dtype sum = std::accumulate(bottom_diff,
                                bottom_diff
                                + bottom[0]->height() * bottom[0]->width(),
                                0.0);

    Dtype mean = sum / (bottom[0]->width()*bottom[0]->height());

    std::vector<Dtype> msd(bottom[0]->height()*bottom[0]->width());
    std::transform(bottom_diff,
                   bottom_diff + (bottom[0]->height()*bottom[0]->width()),
                   msd.begin(), std::bind2nd(std::minus<Dtype>(), mean));

    Dtype sqsum = std::inner_product(msd.begin(), msd.end(), msd.begin(), 0.0);
    Dtype stdev = std::sqrt(sqsum/(bottom[0]->width()*bottom[0]->height()));


    wrapped_diff.convertTo(tmp, CV_32FC1, 1.0 / (2.0 * stdev),
        (stdev - mean) * 1.0 / (2.0 * stdev));

    cv::imshow("diff", tmp);
    cv::waitKey(2);
#endif
  }
}

INSTANTIATE_CLASS(MalisLossLayer);
REGISTER_LAYER_CLASS(MalisLoss);

}  // namespace caffe
