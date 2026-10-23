#include <iostream>
#include <cmath>
#include <vector> 
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>

using namespace std; 

constexpr double t0 = 1.0f ; 

#define TRIGGERtjd 4078.7378894677386f
#define MASTERjd 4078.740327546373f
#define PREBURSTmin -1.0f
#define PREBURSTmax 0.0f

double pl(double tbins, double f0, double alpha) {
    return  f0*(std::pow(tbins/t0,alpha)) ;
    }

struct Params {
    double f0;
    double alpha;
    };
    
double pl_gsl(double t, void* params) {
    auto* p = static_cast<Params*>(params);
    return pl(t, p->f0, p->alpha);
    }


int main() {
    double tmin = 1.0 ; 
    double tmax = 10.0 ; 
    Params params{100, 0.2};
    
    gsl_function F;
    F.function = &pl_gsl;
    F.params = &params; 
    gsl_set_error_handler_off();

    auto* workspace = 
        gsl_integration_workspace_alloc(1000);

    double result = 0.0 ; 
    double error = 0.0 ;

    int status = gsl_integration_qags(
        &F,
        tmin,
        tmax,
        0.0, // absolute tolerance
        1e-8, // relative tolerance
        1000, // Maximum subintervals
        workspace,
        &result,
        &error
        );
    gsl_integration_workspace_free(workspace);

    if (status != GSL_SUCCESS) { 
        cerr << gsl_strerror(status) << endl;
        return 1;
        }
    cout << "Integral: " << result << endl; 
    cout << "Estimated error: " << error << endl;

    return 0;
    }




