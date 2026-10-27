#include <pybind11/pybind11.h>
#include <cmath> 
#include "forwardshock.h"
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>

namespace py = pybind11; 

// The model itself is just a smoothly broken powerlaw
double sbpl(double t, double f0, double tbreak, double k,
            double delta,double p){
    if (t <= 0.0)
        return 0.0;
    double alpha1=(2-k)/(4-k); 
    double alpha2=(12*p-12-3*p*k+5*k)/(4*(4-k));
    return f0 * std::pow((t/tbreak),-alpha1) * std::pow(( (1 + std::pow((t/tbreak),(1/delta)))/2),(alpha1-alpha2)*delta) ;
    }
// Need this to connect it to the integration library
double FSgsl(double t, void* params) {
    auto* p = static_cast<FSparams*>(params);
    return sbpl(t, p->f0, p->tbreak, p->k, p->delta, p->p);
    }

// The model fitting will, basically, take the smoothly broken powerlaw, then simulate the TESS algos
// Onboard, TESS reads out every 2.0 seconds. These 2.0 seconds are then coadded in sets of 10.
// (10 x 2 second images) The highest and lowest values are dropped. The remaining 8 are summed. 
// Then it is all added to give the full downlinked cadence. 

double FSmodel(double xdata, double ydata, double f0, 
               double k, double delta, double p){
  //
  // initialize GSL stuff so that it only happens once
  // TODO: DEFINE PARAMS!
  // FSparams = FSparams {f0, tb ,...... etc  ; 
  gsl_function f;
  f.function = &FSgsl;
  F.params = &FSparams;
