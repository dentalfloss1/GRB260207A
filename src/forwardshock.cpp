#include <cmath> 
#include "forwardshock.h"

double FSmodel(double t, double f0, double tbreak, double k,
            double delta,double p){
    if (t <= 0.0)
        return 0.0;
    double alpha1=(2-k)/(4-k); 
    double alpha2=(12*p-12-3*p*k+5*k)/(4*(4-k));
    return f0 * std::pow((t/tbreak),-alpha1) * std::pow(( (1 + std::pow((t/tbreak),(1/delta)))/2),(alpha1-alpha2)*delta) ;
    }

double FSgsl(double t, void* params) {
    auto* p = static_cast<FSparams*>(params);
    return FSmodel(t, p->f0, p->tbreak, p->k, p->delta, p->p);
    }
