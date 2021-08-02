/* solve the Hamilton-Jacobi equation to smooth a raster field
   Persson, PO. Engineering with Computers (2006) 22: 95.
   https://doi.org/10.1007/s00366-006-0014-1 kjr, usp, 2019
*/
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <assert.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <chrono>

#define EPS 1e-9

namespace py = pybind11;

// for column major order with 1-based indexing
int sub2ind(const int row, const int col, const int zpos, const int nrows,
            const int ncols) {
  return (col - 1) * nrows + row +
         (zpos - 1) * (nrows * ncols); // trailing -1 is for zero-based indexing
}

//
void ind2sub(const int index, const int nrows, const int ncols, int *i, int *j,
             int *k) {
  int tmp2;
  double tmp = (double)index;
  double a = nrows * ncols;
  *k = std::ceil(tmp / a);
  tmp2 = (int)tmp - (*k - 1) * nrows * ncols;
  *j = 1 + std::floor((tmp2 - 1) / nrows);
  *i = tmp2 - (*j - 1) * nrows;
  assert(*i > 0);
  assert(*j > 0);
  assert(*k > 0);
}

//
bool IsNegative(int i) { return (i < 0); }

// find indices in linear time where A==value
std::vector<int> findIndices(const std::vector<int> &A, const int value) {
  std::vector<int> B;
  for (std::size_t i = 0; i < A.size(); i++) {
    if (A[i] == value) {
      B.push_back(i);
    }
  }
  return B;
}

// solve the Hamilton-Jacobi equation
std::vector<double> c_gradient_limit(const std::vector<int> &dims, 
                              const double &elen,
                              const double &dfdx, const int &imax,
                              const std::vector<double> &ffun) {

  assert(dims[0] > 0 && dims[1] > 0 && dims[2] > 0);

  std::vector<int> aset(dims[0] * dims[1] * dims[2], -1);

  double ftol = *(std::min_element(ffun.begin(), ffun.end())) * std::sqrt(EPS);

  std::array<int, 9> npos;
  npos.fill(0);

  double elend = elen * std::sqrt(2.0);

  // allocate output
  std::vector<double> ffun_s;
  ffun_s.resize(ffun.size());
  ffun_s = ffun;
  

  int maxSz = dims[0] * dims[1] * dims[2];

  for (int iter = 0; iter < imax; iter++) {

    //------------------------- find "active" nodes this pass
    auto aidx = findIndices(aset, iter - 1);

    //------------------------- convergence
    if (aidx.empty()) {
      // std::cout << "INFO: Converged in " << iter << " iterations." <<
      // std::endl;
      break;
    }

    for (std::size_t i = 0; i < aidx.size(); i++) {

      //----- map triply indexed to singly indexed
      int inod = aidx[i] + 1; // add one to match 1-based indexing

      //----- calculate the i,j,k position
      int ipos, jpos, kpos;
      ind2sub(inod, dims[0], dims[1], &ipos, &jpos, &kpos);

      // ---- gather indices centered on inod
      npos[0] = inod;
      // right
      npos[1] =
          sub2ind(ipos, std::min(jpos + 1, dims[1]), kpos, dims[0], dims[1]);
      // left
      npos[2] = sub2ind(ipos, std::max(jpos - 1, 1), kpos, dims[0], dims[1]);
      // top
      npos[3] = sub2ind(std::min(ipos + 1, dims[0]), jpos, kpos, dims[0], dims[1]);
      // bottom
      npos[4] = sub2ind(std::max(ipos - 1, 1), jpos, kpos, dims[0], dims[1]);

      // top right diagonal
      npos[5] = sub2ind(std::min(ipos +1, dims[0]), std::min(jpos +1, dims[1]), kpos, dims[0], dims[1]);
      // top left diagonal
      npos[6] = sub2ind(std::max(ipos -1 , 1), std::min(jpos +1, dims[1]), kpos, dims[0], dims[1]);
      // bottom left diagonal
      npos[7] = sub2ind(std::max(ipos -1 , 1), std::max(jpos -1, 1), kpos, dims[0], dims[1]);
      // bottom right diagonal
      npos[8] = sub2ind(std::min(ipos +1 , dims[0]), std::min(jpos+1, dims[1]), kpos, dims[0], dims[1]);

      for (std::size_t u = 0; u < 9; u++)
        npos[u]--;

      int nod1 = npos[0];
      assert(nod1 < ffun_s.size());
      assert(nod1 > -1);

      for (std::size_t p = 1; p < 9; p++) {

        int nod2 = npos[p];
        assert(nod2 < ffun_s.size());
        assert(nod2 > -1);

        if (p < 5) {
        
          double elenp = elen;

        } else {
        
          double elenp = elend;
        }


        //----------------- calc. limits about min.-value
        if (ffun_s[nod1] > ffun_s[nod2]) {

          double fun1 = ffun_s[nod2] + elenp * dfdx;
          if (ffun_s[nod1] > fun1 + ftol) {
            ffun_s[nod1] = fun1;
            aset[nod1] = iter;
          }

        } else {

          double fun2 = ffun_s[nod1] + elenp * dfdx;
          if (ffun_s[nod2] > fun2 + ftol) {
            ffun_s[nod2] = fun2;
            aset[nod2] = iter;
          }
        }
      }
    }
    // std::cout << "ITER: " << iter << std::endl;
  }
  return ffun_s;
}

// Python wrapper
py::array
gradient_limit(py::array_t<int, py::array::c_style | py::array::forcecast> dims,
        const double elen, const double dfdx, const int imax,
        py::array_t<double, py::array::c_style | py::array::forcecast> ffun) {
  int num_points = ffun.size();

  std::vector<double> cffun(num_points);
  std::vector<int> cdims(3);

  std::memcpy(cffun.data(), ffun.data(), num_points * sizeof(double));
  std::memcpy(cdims.data(), dims.data(), 3 * sizeof(int));

  std::vector<double> sffun = c_gradient_limit(cdims, elen, dfdx, imax, cffun);

  ssize_t sodble = sizeof(double);
  std::vector<ssize_t> shape = {num_points, 1};
  std::vector<ssize_t> strides = {sodble, sodble};

  // return 2-D NumPy array
  return py::array(
      py::buffer_info(sffun.data(),   /* data as contiguous array  */
                      sizeof(double), /* size of one scalar        */
                      py::format_descriptor<double>::format(), /* data type */
                      2,      /* number of dimensions      */
                      shape,  /* shape of the matrix       */
                      strides /* strides for each axis     */
                      ));
}

PYBIND11_MODULE(HamiltonJacobi, m) {

  m.doc() = "pybind11 module for gradient limiting a scalar field";

  m.def("gradient_limit", &gradient_limit,
        "The function which gradient limits a scalar field reshaped to a "
        "vector.");
}