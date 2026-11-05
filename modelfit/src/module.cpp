#include <pybind11/pybind11.h>
namespace py = pybind11;

void register_forwardshock(py::module_ &m);
void register_injection(py::module_ &m);
PYBIND11_MODULE(lcmodel, m){
  m.doc() = "Light curve model";
  register_forwardshock(m);
  register_injection(m);
  }
