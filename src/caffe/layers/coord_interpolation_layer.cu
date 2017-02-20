#include <cfloat>
#include <vector>

#include "caffe/layers/coord_interpolation_layer.hpp"
#include "caffe/util/math_functions.hpp"

#ifdef USE_GREENTEA
#include "caffe/greentea/greentea.hpp"
#include "caffe/greentea/greentea_math_functions.hpp"
#endif

namespace caffe {

#define INTERPOL_NEAREST    0
#define INTERPOL_BILINEAR   1

#define TYPE_CARTESIAN      0
#define TYPE_HEXGRID        1

#ifdef USE_CUDA
template<typename Dtype>
__global__ void Interpolation(const int_tp nthreads,
                           const Dtype* bottom_data,
                           const int_tp srcheight,
                           const int_tp srcwidth,
                           const int_tp srctype,
                           const int_tp dstheight,
                           const int_tp dstwidth,
                           const int_tp dsttype,
                           const int_tp interpol,
                           const int_tp scale,
                           Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // Integer destination indexing
    const int_tp w = index % dstwidth;
    const int_tp h = (index / dstwidth) % dstheight;

    // Channels and batch are preserved
    const int_tp c = (index / dstwidth / dstheight);

    Dtype value = (Dtype)0.0;

    if (srctype == TYPE_CARTESIAN && dsttype == TYPE_HEXGRID) {
      Dtype cdstwidth = (Dtype)(dstwidth - 1)*sqrt((Dtype)3.0)/((Dtype)2.0) + (Dtype)2.0*sqrt((Dtype)3.0)/((Dtype)3.0);
      Dtype cdstheight = (Dtype)dstheight + (Dtype)0.5;

      Dtype fw = ((Dtype)srcwidth) / cdstwidth;
      Dtype fh = ((Dtype)srcheight) / cdstheight;
      Dtype fs = scale == 1 ? min(fw, fh) : max(fw, fh);

      // (v,u) cartesian coordinates of hextile (h,w) centerpoint
      Dtype u = (Dtype)w * (sqrt((Dtype)3.0)/((Dtype)2.0));
      Dtype v = (Dtype)0.5 * (Dtype)w + (Dtype)h;
      if (v >= dstheight - 0.25) {
        v = v - dstheight;
      }
      u = u + sqrt((Dtype)3.0)/((Dtype)3.0);
      v = v + (Dtype)0.5;

      if (interpol == INTERPOL_NEAREST) {
        int_tp x = (int_tp)round(u*fs);
        int_tp y = (int_tp)round(v*fs);
        // printf("(%d,%d)\n",y,x);
        if (x < 0 || y < 0 || x >= srcwidth || y >= srcheight) {
          value = (Dtype)0.0;
        } else {
          value = bottom_data[x+(y+c*srcheight)*srcwidth];
        }
      }

      if (interpol == INTERPOL_BILINEAR) {
        if (fs > (Dtype)2.0) {
          int_tp count = 0;
          value = (Dtype)0.0;
          int_tp x = (int_tp)round(u*fs);
          int_tp y = (int_tp)round(v*fs);
          if (x < 0 || y < 0 || x >= srcwidth || y >= srcheight) {
            value = (Dtype)0.0;
          } else {
            for (int_tp iy = (int_tp)floor((v-sqrt((Dtype)3.0)
                                      /((Dtype)3.0))*fs);
                        iy < (int_tp)ceil((v+sqrt((Dtype)3.0)
                                      /((Dtype)3.0))*fs);
                      ++iy) {
              for (int_tp ix = (int_tp)floor((u-sqrt((Dtype)3.0)
                                      /((Dtype)3.0))*fs);
                          ix < (int_tp)ceil((u+sqrt((Dtype)3.0)
                                      /((Dtype)3.0))*fs);
                        ++ix) {
                if (!(ix < 0 || iy < 0 || ix >= srcwidth || iy >= srcheight) &&
                    (((Dtype)1.0)/((Dtype)3.0)*pow(fs,(Dtype)2.0)
                    >= pow((Dtype)ix-u*fs,(Dtype)2.0)+pow((Dtype)iy-v*fs,(Dtype)2.0))) {
                  value += bottom_data[ix+(iy+c*srcheight)*srcwidth];
                  ++count;
                }
              }
            }
          }
          if (count > 0) {
            value /= (Dtype)count;
          }
        }
        if (fs <= (Dtype)2.0) {
          int_tp x0 = (int_tp)floor(u*fs);
          int_tp y0 = (int_tp)floor(v*fs);
          int_tp x1 = (int_tp)ceil(u*fs);
          int_tp y1 = (int_tp)ceil(v*fs);
          Dtype v0 = (Dtype)0.0;
          Dtype v1 = (Dtype)0.0;
          Dtype v2 = (Dtype)0.0;
          Dtype v3 = (Dtype)0.0;
          if (y0 >= 0 && y0 < srcheight) {
            if (x0 >= 0 && x0 < srcwidth) {
              v0 = bottom_data[x0+(y0+c*srcheight)*srcwidth];
            }
            if (x1 >= 0 && x1 < srcwidth) {
              v1 = bottom_data[x1+(y0+c*srcheight)*srcwidth];
            }
          }
          if (y1 >= 0 && y1 < srcheight) {
            if (x0 >= 0 && x0 < srcwidth) {
              v2 = bottom_data[x0+(y1+c*srcheight)*srcwidth];
            }
            if (x1 >= 0 && x1 < srcwidth) {
              v3 = bottom_data[x1+(y1+c*srcheight)*srcwidth];
            }
          }
          Dtype w0 = x0 == x1 ? (Dtype)0.5 : abs((Dtype)x0-u*fs)/(Dtype)abs(x1-x0);
          Dtype w1 = x0 == x1 ? (Dtype)0.5 : abs((Dtype)x1-u*fs)/(Dtype)abs(x1-x0);
          Dtype w2 = y0 == y1 ? (Dtype)0.5 : abs((Dtype)y0-v*fs)/(Dtype)abs(y1-y0);
          Dtype w3 = y0 == y1 ? (Dtype)0.5 : abs((Dtype)y1-v*fs)/(Dtype)abs(y1-y0);
          value = (v0*w1+v1*w0)*w3+(v2*w1+v3*w0)*w2;
        }
      }
    }

    if (srctype == TYPE_HEXGRID && dsttype == TYPE_CARTESIAN) {
      Dtype csrcwidth = (Dtype)(srcwidth - 1)*sqrt((Dtype)3.0)/((Dtype)2.0) + (Dtype)2.0*sqrt((Dtype)3.0)/((Dtype)3.0);
      Dtype csrcheight = (Dtype)srcheight + (Dtype)0.5;

      Dtype fw = csrcwidth / ((Dtype)dstwidth);
      Dtype fh = csrcheight / ((Dtype)dstheight);
      Dtype fs = scale == 1 ? min(fw, fh) : max(fw, fh);

      // (v,u) hextile coordinates of cartesian (h,w) centerpoint
      Dtype u = ((Dtype)w * fs - sqrt((Dtype)3.0)/((Dtype)3.0)) / (sqrt((Dtype)3.0)/((Dtype)2.0));
      Dtype v = ((Dtype)h * fs - (Dtype)0.5) - (Dtype)0.5 * u;

      if (interpol == INTERPOL_NEAREST) {
        int_tp x0 = (int_tp)floor(u);
        int_tp y0 = (int_tp)floor(v);
        int_tp x1 = (int_tp)ceil(u);
        int_tp y1 = (int_tp)ceil(v);
        Dtype w0 = pow((u-(Dtype)x0)*sqrt((Dtype)3.0)/((Dtype)2.0),(Dtype)2.0) +
                   pow((v+(Dtype)0.5*u)-((Dtype)y0+(Dtype)0.5*(Dtype)x0),(Dtype)2.0);
        Dtype w1 = pow((u-(Dtype)x1)*sqrt((Dtype)3.0)/((Dtype)2.0),(Dtype)2.0) +
                   pow((v+(Dtype)0.5*u)-((Dtype)y0+(Dtype)0.5*(Dtype)x1),(Dtype)2.0);
        Dtype w2 = pow((u-(Dtype)x0)*sqrt((Dtype)3.0)/((Dtype)2.0),(Dtype)2.0) +
                   pow((v+(Dtype)0.5*u)-((Dtype)y1+(Dtype)0.5*(Dtype)x0),(Dtype)2.0);
        Dtype w3 = pow((u-(Dtype)x1)*sqrt((Dtype)3.0)/((Dtype)2.0),(Dtype)2.0) +
                   pow((v+(Dtype)0.5*u)-((Dtype)y1+(Dtype)0.5*(Dtype)x1),(Dtype)2.0);
        int_tp yshift0 = 0;
        int_tp yshift1 = 0;
        int_tp yshift2 = 0;
        int_tp yshift3 = 0;
        if ((y0 < 0) && (2*y0+x0 >= 0)) {
          yshift0 = srcheight;
        }
        if ((y0 < 0) && (2*y0+x1 >= 0)) {
          yshift1 = srcheight;
        }
        if ((y1 < 0) && (2*y1+x0 >= 0)) {
          yshift2 = srcheight;
        }
        if ((y1 < 0) && (2*y1+x1 >= 0)) {
          yshift3 = srcheight;
        }
        Dtype rw = 1.0;
        if (w0 < rw && w0 < (Dtype)1.0) {
          if (x0 < 0 || (y0+yshift0) < 0 || x0 >= srcwidth || (y0+yshift0) >= srcheight || 2*y0+x0 >= 2*srcheight) {
            value = (Dtype)0.0;
          } else {
            value = bottom_data[x0+(yshift0+y0+c*srcheight)*srcwidth];
          }
          rw = w0;
        }
        if (w1 < rw && w1 < (Dtype)1.0) {
          if (x1 < 0 || (y0+yshift1) < 0 || x1 >= srcwidth || (y0+yshift1) >= srcheight || 2*y0+x1 >= 2*srcheight) {
            value = (Dtype)0.0;
          } else {
            value = bottom_data[x1+(yshift1+y0+c*srcheight)*srcwidth];
          }
          rw = w1;
        }
        if (w2 < rw && w2 < (Dtype)1.0) {
          if (x0 < 0 || (y1+yshift2) < 0 || x0 >= srcwidth || (y1+yshift2) >= srcheight || 2*y1+x0 >= 2*srcheight) {
            value = (Dtype)0.0;
          } else {
            value = bottom_data[x0+(yshift2+y1+c*srcheight)*srcwidth];
          }
          rw = w2;
        }
        if (w3 < rw && w3 < (Dtype)1.0) {
          if (x1 < 0 || (y1+yshift3) < 0 || x1 >= srcwidth || (y1+yshift3) >= srcheight || 2*y1+x1 >= 2*srcheight) {
            value = (Dtype)0.0;
          } else {
            value = bottom_data[x1+(yshift3+y1+c*srcheight)*srcwidth];
          }
          rw = w3;
        }
      }

      if (interpol == INTERPOL_BILINEAR) {
        if (fs > (Dtype)2.0) {
          int_tp count = 0;
          value = (Dtype)0.0;
          int_tp x = (int_tp)round(u);
          int_tp y = (int_tp)round(v);
          int_tp yshift = 0;
          if ((y < 0) && (2*y+x >= 0)) {
            yshift = srcheight;
          }
          if (x < 0 || (y+yshift) < 0 || x >= srcwidth || (y+yshift) >= srcheight || 2*y+x >= 2*srcheight) {
            value = (Dtype)0.0;
          } else {
            for (int_tp ix = (int_tp)floor(u-fs/(sqrt((Dtype)3.0)));
                        ix < (int_tp)ceil(u+fs/(sqrt((Dtype)3.0)));
                      ++ix) {
              for (int_tp iy = (int_tp)floor(v-fs*(sqrt((Dtype)3.0)-(Dtype)1.0)/((Dtype)2.0*sqrt((Dtype)3.0)));
                        iy < (int_tp)ceil(v+fs*(sqrt((Dtype)3.0)-(Dtype)1.0)/((Dtype)2.0*sqrt((Dtype)3.0)));
                      ++iy) {
                Dtype w = pow((u-(Dtype)ix)*sqrt((Dtype)3.0)/((Dtype)2.0),(Dtype)2.0) +
                           pow((v+(Dtype)0.5*u)-((Dtype)iy+(Dtype)0.5*(Dtype)ix),(Dtype)2.0);
                if ((iy < 0) && (2*iy+ix >= 0)) {
                   yshift = srcheight;
                 }
                 if (pow(fs/(Dtype)2.0,(Dtype)2.0) >= w && !(ix < 0 || (iy+yshift) < 0 || ix >= srcwidth || (iy+yshift) >= srcheight || 2*iy+ix >= 2*srcheight)) {
                  value += bottom_data[ix+(iy+yshift+c*srcheight)*srcwidth];
                  ++count;
                }
              }
            }
          }
          if (count > 0) {
            value /= (Dtype)count;
          }
        }
        if (fs <= (Dtype)2.0) {
          int_tp x0 = (int_tp)floor(u);
          int_tp y0 = (int_tp)floor(v);
          int_tp x1 = (int_tp)ceil(u);
          int_tp y1 = (int_tp)ceil(v);
          Dtype w0 = pow((u-(Dtype)x0)*sqrt((Dtype)3.0)/((Dtype)2.0),(Dtype)2.0) +
                     pow((v+(Dtype)0.5*u)-((Dtype)y0+(Dtype)0.5*(Dtype)x0),(Dtype)2.0);
          Dtype w1 = pow((u-(Dtype)x1)*sqrt((Dtype)3.0)/((Dtype)2.0),(Dtype)2.0) +
                     pow((v+(Dtype)0.5*u)-((Dtype)y0+(Dtype)0.5*(Dtype)x1),(Dtype)2.0);
          Dtype w2 = pow((u-(Dtype)x0)*sqrt((Dtype)3.0)/((Dtype)2.0),(Dtype)2.0) +
                     pow((v+(Dtype)0.5*u)-((Dtype)y1+(Dtype)0.5*(Dtype)x0),(Dtype)2.0);
          Dtype w3 = pow((u-(Dtype)x1)*sqrt((Dtype)3.0)/((Dtype)2.0),(Dtype)2.0) +
                     pow((v+(Dtype)0.5*u)-((Dtype)y1+(Dtype)0.5*(Dtype)x1),(Dtype)2.0);
          if (w0 > w1 && w0 > w2 && w0 > w3) {
            w0 = (Dtype)2.0;
          }
          if (w1 > w0 && w1 > w2 && w1 > w3) {
            w1 = (Dtype)2.0;
          }
          if (w2 > w1 && w2 > w0 && w2 > w3) {
            w2 = (Dtype)2.0;
          }
          if (w3 > w1 && w3 > w2 && w3 > w0) {
            w3 = (Dtype)2.0;
          }
          int_tp yshift0 = 0;
          int_tp yshift1 = 0;
          int_tp yshift2 = 0;
          int_tp yshift3 = 0;
          if ((y0 < 0) && (2*y0+x0 >= 0)) {
            yshift0 = srcheight;
          }
          if ((y0 < 0) && (2*y0+x1 >= 0)) {
            yshift1 = srcheight;
          }
          if ((y1 < 0) && (2*y1+x0 >= 0)) {
            yshift2 = srcheight;
          }
          if ((y1 < 0) && (2*y1+x1 >= 0)) {
            yshift3 = srcheight;
          }
          Dtype rw = 0.0;
          if (w0 < (Dtype)1.0) {
            if (!(x0 < 0 || (y0+yshift0) < 0 || x0 >= srcwidth || (y0+yshift0) >= srcheight || 2*y0+x0 >= 2*srcheight)) {
              value += bottom_data[x0+(yshift0+y0+c*srcheight)*srcwidth]*((Dtype)1.0-w0);
            }
            rw += ((Dtype)1.0-w0);
          }
          if (w1 < (Dtype)1.0) {
            if (!(x1 < 0 || (y0+yshift1) < 0 || x1 >= srcwidth || (y0+yshift1) >= srcheight || 2*y0+x1 >= 2*srcheight)) {
              value += bottom_data[x1+(yshift1+y0+c*srcheight)*srcwidth]*((Dtype)1.0-w1);
            }
            rw += ((Dtype)1.0-w1);
          }
          if (w2 < (Dtype)1.0) {
            if (!(x0 < 0 || (y1+yshift2) < 0 || x0 >= srcwidth || (y1+yshift2) >= srcheight || 2*y1+x0 >= 2*srcheight)) {
              value += bottom_data[x0+(yshift2+y1+c*srcheight)*srcwidth]*((Dtype)1.0-w2);
            }
            rw += ((Dtype)1.0-w2);
          }
          if (w3 < (Dtype)1.0) {
            if (!(x1 < 0 || (y1+yshift3) < 0 || x1 >= srcwidth || (y1+yshift3) >= srcheight || 2*y1+x1 >= 2*srcheight)) {
              value += bottom_data[x1+(yshift3+y1+c*srcheight)*srcwidth]*((Dtype)1.0-w3);
            }
            rw += ((Dtype)1.0-w3);
          }
          value /= rw;
        }
      }
    }

    if (srctype == TYPE_CARTESIAN && dsttype == TYPE_CARTESIAN) {
      Dtype fs = 0.0;
      if (interpol == INTERPOL_NEAREST) {
        // TODO
      }
      if (interpol == INTERPOL_BILINEAR) {
        if (fs > (Dtype)2.0) {
          // TODO
        }
        if (fs <= (Dtype)2.0) {
          // TODO
        }
      }
    }

    if (srctype == TYPE_HEXGRID && dsttype == TYPE_HEXGRID) {
      Dtype fs = 0.0;
      if (interpol == INTERPOL_NEAREST) {
        // TODO
      }
      if (interpol == INTERPOL_BILINEAR) {
        if (fs > (Dtype)2.0) {
          // TODO
        }
        if (fs <= (Dtype)2.0) {
          // TODO
        }
      }
    }

    top_data[index] = value;
  }
}
#endif  // USE_CUDA


template<typename Dtype>
void CoordInterpolationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    Interpolation<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(top[0]->count()),
                                  CAFFE_CUDA_NUM_THREADS)(
                                      top[0]->count(),
                                      bottom_data,
                                      bottom[0]->shape(2),
                                      bottom[0]->shape(3),
                                      src_grid_type_,
                                      top[0]->shape(2),
                                      top[0]->shape(3),
                                      dst_grid_type_,
                                      interp_mode_,
                                      scale_mode_,
                                      top_data);
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
#endif  // USE_GREENTEA
  }
}

template<typename Dtype>
void CoordInterpolationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                       const vector<bool>& propagate_down,
                                       const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

  if (this->device_->backend() == BACKEND_CUDA) {
#ifdef USE_CUDA
    Interpolation<Dtype> CUDA_KERNEL(CAFFE_GET_BLOCKS(top[0]->count()),
                                  CAFFE_CUDA_NUM_THREADS)(
                                      bottom[0]->count(),
                                      top_diff,
                                      top[0]->shape(2),
                                      top[0]->shape(3),
                                      dst_grid_type_,
                                      bottom[0]->shape(2),
                                      bottom[0]->shape(3),
                                      src_grid_type_,
                                      interp_mode_,
                                      scale_mode_,
                                      bottom_diff);
#endif  // USE_CUDA
  } else {
#ifdef USE_GREENTEA
#endif  // USE_GREENTEA
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CoordInterpolationLayer);

}  // namespace caffe
