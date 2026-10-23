#pragma once

struct FSparams {
  double f0; 
  double tbreak;
  double k;
  double delta;
  double p; 
  };

double FSmodel(double t, double f0,
               double tbreak, double k,
               double delta, double p);
double FSgsl(double t, void* params);
