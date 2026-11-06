#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>


namespace py = pybind11;
using namespace pybind11::literals;

namespace {

  // Constants that never change
  constexpr double LN10  = 2.302585092994046;
  constexpr double LN2   = 0.6931471805599453;
  constexpr double S     = 0.02;
  constexpr double INV_S = 1.0 / S;
  constexpr double SUB_DT = 2.0 / 86400.0;                        // 2 s in days
  constexpr int    N_SUB  = 100;                                  // = round(CADENCE / SUB_DT)
  constexpr int N_BLOCK = 10;
  constexpr int BLOCK_LEN = 10; 
  constexpr double CADENCE_DAYS = N_SUB*SUB_DT;

  struct fwd {
    double lnF0, lntb, a1, c, invS;
    fwd(double logF0, double logtb, double k, double p) {
      const double a2   = (12*p - 12 - 3*p*k + 5*k) / (4.0 * (4.0 - k));
      
      lnF0 = logF0 * LN10;
      lntb = logtb * LN10;
      a1   = -(2.0 - k) / (4.0 - k);
      invS = 1.0 / S;
      c    = (a2 - a1) * S;

    }
    double operator()(double t) const {
      if (t <= 0.0) return 0.0;
      double lx = std::log(t) - lntb;
      double u  = lx * INV_S;
      double sp = std::fmax(u, 0.0) + std::log1p(std::exp(-std::fabs(u)));  // ln(1+e^u)

      return std::exp(lnF0 - a1 * lx - c * (sp - LN2));
    }
  };
  struct rev {
    double lnF0, lntb, a1, c, invS;
    rev(double logF0, double logtb, double k, double p, double a1, double a2) {
      
      lnF0 = logF0 * LN10;
      lntb = logtb * LN10;
      invS = 1.0 / S;
      c    = (a2 - a1) * S;
      this->a1=a1;

    }
    double operator()(double t) const {
      if (t <= 0.0) return 0.0;
      double lx = std::log(t) - lntb;
      double u  = lx * INV_S;
      double sp = std::fmax(u, 0.0) + std::log1p(std::exp(-std::fabs(u)));  // ln(1+e^u)

      return std::exp(lnF0 - a1 * lx - c * (sp - LN2));
    }
  };
  struct inj {
  double kE, tinj, tend, eta;
    inj(double logtinj, double logtend, double k, double eta, double p) {
      kE = ((4 - k)*(p + 3) - 2*k)/ (4*(4-k)) ; 
      tinj = std::pow(10,logtinj) ; 
      tend = std::pow(10,logtend) ; 
      this->eta=eta;
      }
    double operator()(double t) const{
      if (t < tinj) return 1.0;
      else if (t < tend) return std::pow(t/tinj, kE*eta);
      else return std::pow(tend/tinj, kE*eta);
      }
  };
}


// forcecast + c_style: any float/int array or list gets converted to a contiguous double array
using darray = py::array_t<double, py::array::c_style | py::array::forcecast>;

py::array_t<double> model(darray t, double k, double logF0, double logtb, double p,
                          double logF0rev, double logtbrev, double a1, double a2, 
                          double logtinj, double logdeltatinj, double eta, double logEinj){
  auto tin = t.unchecked<1>();   //throws if t is not 1-D
  const py::ssize_t n = tin.shape(0);

  py::array_t <double> out(n);
  auto o = out.mutable_unchecked<1>();
  double tinj = std::pow(10, logtinj) ; 
  double logtend = std::log10(tinj + std::pow(10,logdeltatinj));
  fwd f1(logF0, logtb, k, p);
  rev f2(logF0rev, std::log10(std::pow(10, logtbrev) - tinj), k, p, a1, a2);
  inj f3(logtinj, logtend, k, eta, p);
  for (py::ssize_t i=0; i<n; ++i){
    const double t_start = tin(i) ; 
    double total = 0.0;
    // Replicates TESS's cosmic ray mitigation
    for (int b=0;b<N_BLOCK;++b){
      double sum = 0.0, lo = INFINITY, hi = -INFINITY;
      for (int s=0; s<BLOCK_LEN; ++s){
        const double ts = t_start + (b*BLOCK_LEN + s + 0.5)*SUB_DT;
        const double fl = f1(ts)*f3(ts)+f2(ts-tinj);
        sum += fl; 
        lo = std::fmin(lo,fl);
        hi = std::fmax(hi, fl);
        }
      total += sum - lo - hi;
      }
    o(i) = total;
    }
  return out;
  }

void register_injection(py::module_ &m){
  m.def("injection", &model, "t"_a, "k"_a, "logf0"_a, "logtb"_a, 
        "p"_a,"logF0rev"_a,"logtbrev"_a,"a1"_a, 
        "a2"_a, "logtinj"_a, "logdeltatinj"_a,
        "eta"_a,"logEinj"_a,
        "Compute model flux as times t");
  }

