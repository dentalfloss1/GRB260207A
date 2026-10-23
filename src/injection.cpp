#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <cmath>
#include <vector> 
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>

using namespace std; 

constexpr double t0 = 1.0 ; 
constexpr double cadence = 200.0/86400.0
constexpr double TRIGGERtjd = 4078.7378894677386 ; 
constexpr double MASTERjd = 4078.740327546373 ; 
constexpr double  PREBURSTmin = -1.0 ;
constexpr double  PREBURSTmax  = 0.0 ;

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

    std::vector<double> time ; 
    std::vector<double> flux ; 
    std::vector<double> err ; 

    std::ifstream file("lc_GRB260207A_cand41148_geo");

    if (!file) {
        std::cerr << "Cannot open file\n";
        return 1;
        }
    std::string line; 
    double row[13];

    while (std::getline(file,line)) {
        std::istringstream ss(line);

        int i = 0;

        while (i<13 && (ss >> row[i])) { 
            ++i;
            }
        if (i != 13)
            continue ; 

        // Process row
        time.push_back(row[0]);
        flux.push_back(row[1]);
        err.push_back(row[2]);

        }
    for (size_t i=0;i<time.size();i++){
        // These are the start times in days post burst. We'll feed this into the model and compute stop times.
        // No more doing midtimes because that becomes inaccurate for flux that changes inside of the observation.
        time[i] = time[i] - TRIGGERtjd ; 
        // cts to cts_per_sec accounting for cosmic ray and efficiency, then to Jansky
        flux[i] = 1.6110051544201104e-05*(flux[i]/(cadence*0.8*0.99*86400.0)) ;
        err[i] = 1.6110051544201104e-05*(err[i]/(cadence*0.8*0.99*86400.0)) ; 
        }

    // Now we have the measured flux values. Everything is in the correct units: time is in days post burst, flux is in Jansky, error is in Jansky.

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




