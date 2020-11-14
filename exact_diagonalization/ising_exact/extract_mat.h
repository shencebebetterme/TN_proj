#pragma once

#include <cmath>
#include <string>
#include <cstring>
#include <vector>
//#include <array>
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <complex>
#include <cstdlib>
//#include <ctime>
#include <chrono>
#include <tuple>
#include <omp.h>
//#include <random>
#include <execution>
#include <numeric>


#include "itensor/all.h"
#include "itensor/util/print_macro.h"
#include <armadillo>

using namespace itensor;
using namespace std;
using namespace std::chrono;
using namespace arma;



arma::sp_mat extract_spmat(const ITensor& T){
    auto di = T.index(1).dim();
    auto dj = T.index(2).dim();

    auto extractReal = [](Dense<Real> const& d)
    {
        return d.store;
    };

    auto data_vec = applyFunc(extractReal,T.store());

    arma::mat denseT(&data_vec[0], di, dj, false);
    arma::sp_mat sparseT(denseT);
    return sparseT;
}



arma::mat extract_mat(const ITensor& T){
    auto di = T.index(1).dim();
    auto dj = T.index(2).dim();

    auto extractReal = [](Dense<Real> const& d)
    {
        return d.store;
    };

    auto data_vec = applyFunc(extractReal,T.store());

    arma::mat denseT(&data_vec[0], di, dj, true);
    return denseT;
}


// arma::mat extract_mat(const ITensor& T){
//     auto di = T.index(1).dim();
//     auto dj = T.index(2).dim();

//     auto extractReal = [](Dense<Real> const& d)
//     {
//         return d.store;
//     };

//     auto data_vec = applyFunc(extractReal,T.store());

//     arma::mat denseT(&data_vec[0], di, dj, false);
//     //arma::sp_mat sparseT(denseT);
//     return denseT;
// }