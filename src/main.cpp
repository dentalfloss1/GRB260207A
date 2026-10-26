#include <pybind11/pybind11.h>
#include <iostream>
#include "lightcurve.h"
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>

namespace py = pybind11; 


int main() { 
  // Read in the data
  LightCurve data = read_lightcurve("lc_GRB260207A_cand41148_geo") ; 
   //// Almost certainly wrong. Where do we allocate and free?   
  auto* workspace = gsl_integration_workspace_alloc(1000);



  gsl_integration_workspace_free(workspace);
  return 0;
 }

