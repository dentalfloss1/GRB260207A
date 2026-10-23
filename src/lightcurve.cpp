#include <fstream>
#include <sstream>
#include <strign>
#include <vector>
#include <stdexcept>

constexpr double cadence = 200.0/86400.0
constexpr double TRIGGERtjd = 4078.7378894677386 ; 


LightCurve read_lightcurve(const std::string& filename) {
  std::vector<double> time ; 
  std::vector<double> flux ; 
  std::vector<double> err ; 
  
  std::ifstream file(filename);
  
  if (!file) {
    throw std::runtime_error("Cannot open file: " + filename);
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

  return LightCurve {time, flux, err};

  }
