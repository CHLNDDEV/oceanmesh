#include <algorithm>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <set>
#include <tuple>
#include <vector>

#include <ctime>
#include <iostream>
#include <chrono>

namespace py = pybind11;
using py::ssize_t;

#if 0 // DPZ non-portable
class Timer {
public:
  Timer() { clock_gettime(CLOCK_REALTIME, &beg_); }

  double elapsed() {
    clock_gettime(CLOCK_REALTIME, &end_);
    return end_.tv_sec - beg_.tv_sec +
           (end_.tv_nsec - beg_.tv_nsec) / 1000000000.;
  }

  void reset() { clock_gettime(CLOCK_REALTIME, &beg_); }

private:
  timespec beg_, end_;
};
#endif
class Timer {
public:
  Timer() { reset(); }

  double elapsed() {
    end_ = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dur = end_ - beg_;
    return dur.count();
  }

  void reset() { beg_ = std::chrono::high_resolution_clock::now();  }

private:
  std::chrono::time_point<std::chrono::high_resolution_clock> beg_, end_;
};


template <typename T>
std::vector<T> vectorSortIntArr(std::vector<std::array<T, 2>> v) {
  std::sort(v.begin(), v.end());
  // double t = tmr.elapsed();
  // tmr.reset();
  auto iter = std::unique(v.begin(), v.end());
  // t = tmr.elapsed();
  // std::cout << t << std::endl;

  size_t len = iter - v.begin();
  std::vector<T> outvec;
  outvec.reserve(len * 2);
  for (auto i = v.begin(); i != iter; ++i) {
    outvec.push_back(i->at(0));
    outvec.push_back(i->at(1));
  }
  return outvec;
}

py::array unique_edges(
    py::array_t<int, py::array::c_style | py::array::forcecast> edges) {

  std::vector<int> cedges(edges.size());
  std::memcpy(cedges.data(), edges.data(), edges.size() * sizeof(int));

  std::vector<std::array<int, 2>> tl;

  tl.reserve(cedges.size());
  for (size_t i = 0; i < cedges.size(); i += 2) {
    tl.push_back({std::min(cedges[i], cedges[i + 1]),
                  std::max(cedges[i], cedges[i + 1])});
  }

  auto u_edges = vectorSortIntArr<int>(std::move(tl));

  int num_edges = (int)u_edges.size();
  ssize_t sint = sizeof(int);
  std::vector<ssize_t> shape = {num_edges / 2, 2};
  std::vector<ssize_t> strides = {sint * 2, sint};
  return py::array(
      py::buffer_info(u_edges.data(), /* data as contiguous array  */
                      sizeof(int),    /* size of one scalar        */
                      py::format_descriptor<int>::format(), /* data type */
                      2,      /* number of dimensions      */
                      shape,  /* shape of the matrix       */
                      strides /* strides for each axis     */
                      ));
}

PYBIND11_MODULE(_fast_geometry, m) {
  m.def("unique_edges", &unique_edges);
}
