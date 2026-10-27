#pragma once

#include <string>
#include <vector> 

struct LightCurve { 
  std::vector<double> time; 
  std::vector<double> flux; 
  std::vector<double err; 
  }

LightCurve read_lightcurve(const std::string& filename);


