#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>

namespace = pybind11;
using namespace pybind11::literals;

// forcecast + c_style: any float/int array or list gets converted to a contiguous double array
using darray = py::array_t<double, py::array::c_style | py::array::forcecast>;

py::array_t<double> model(darray t, double logF0, double logtb, p){
  auto tin = t.unchecked<1>();   //throws if t is not 1-D
  const py::ssize_t n = tin.shape(0);

  py::array_t <double> out(n);

  for (py::ssize_t i=0; i<n; ++i){
    //model goes here
    
