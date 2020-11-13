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
#include <thread>
#include <mutex>
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

#define PI 3.1415926535


arma::mat extract_mat(const ITensor& T){
    printf("\nStart converting.\n\n");
    if (T.order()!=2) {printf("\nThe input tensor is not a matrix!\n");}
    Index Ti = T.index(1);
    Index Tj = T.index(2);
    auto di = Ti.dim();
    auto dj = Tj.dim();
    if (di!=dj) {printf("\nThe input tensor is not a square matrix!\n");}
    // extract matrix elements
    auto extract_start = high_resolution_clock::now();
    arma::mat Tmat(di,dj,fill::zeros);
    //
    long c = 0;
    double progress=0;
    double progress_last=0;
    for (auto i : range1(di)){
        for (auto j=1; j<=dj; j++){
            //armadillo index is 0-based
            Tmat(i-1,j-1) = T.elt(Ti=i,Tj=j);
            // monitor conversion progress
            c++;
            if ((c%2000000) == 0){
                progress = c*100.0/(di*dj);
                if(progress>progress_last+1){
                    printf("Converting %.2f %%\n",progress);
                    progress_last = progress;
                }
            }
        }
    }
    auto extract_stop = high_resolution_clock::now();
    auto extract_duration = duration_cast<seconds>(extract_stop-extract_start).count();
    printf("\nConversion complete, %d seconds elapsed.\n",extract_duration);
    return Tmat;
}


arma::sp_mat extract_sparse_mat(const ITensor& T){
    // here T is sparse matrix ITensor
    Index Ti = T.index(1);
    Index Tj = T.index(2);
    auto di = Ti.dim();
    auto dj = Tj.dim();
    arma::sp_mat Tmat(di,dj);
    double val = 0;
    //   
    for (int i=1; i<=di; i++){
        for (int j=1; j<=dj; j++){
            val = eltC(T, Ti=i, Tj=j).real();
            if (val != 0){
                Tmat(i-1,j-1) = val;
            }
        }
    }
    return Tmat;
}


// this openmp implementation of parallel data transfer results in 
// unexpected behaviour

//  // M is extremely sparse, so choose a different strategy
// arma::sp_mat extract_sparse_mat_par(const ITensor& T){
//     // here T is sparse matrix ITensor
//     Index Ti = T.index(1);
//     Index Tj = T.index(2);
//     auto di = Ti.dim();
//     auto dj = Tj.dim();
//     arma::sp_mat Tmat(di,dj);
//     double val = 0;
//     //  
    
//     omp_set_num_threads(num_threads);
// #pragma omp parallel for  
//     for (int i=1; i<=di; i++){
//         for (int j=1; j<=dj; j++){
//             val = eltC(T, Ti=i, Tj=j).real();
//             if (val != 0){
//                 Tmat(i-1,j-1) = val;
//             }
//         }
//     }
//     return Tmat;
// }



// not using multi threading, data race in T?
void transferIthRow(const ITensor& T, arma::sp_mat& Tmat, int i){
    Index Ti = T.index(1);
    Index Tj = T.index(2);
    auto di = Ti.dim();
    auto dj = Tj.dim();
    double val = 0;
    
    for (int j=1; j<=dj; j++){
        val = eltC(T, Ti=i, Tj=j).real();
        if (val != 0){
            Tmat(i-1,j-1) = val;
        }
    }
}

arma::sp_mat extract_sparse_mat_par(const ITensor& T){
    // here T is sparse matrix ITensor
    Index Ti = T.index(1);
    Index Tj = T.index(2);
    auto di = Ti.dim();
    auto dj = Tj.dim();
    arma::sp_mat Tmat(di,dj);
    //   
    std::vector<int> mat_row(di);
    std::iota (mat_row.begin(), mat_row.end(), 1);

    auto lambda = [&](int n){
        transferIthRow(T, Tmat, n);
    };

    std::for_each(std::execution::par_unseq, mat_row.begin(), mat_row.end(), lambda);

    return Tmat;
}



// does not use multi threading either
struct pointVal{
    int i;
    int j;
    double val;
};

arma::sp_mat extract_sparse_mat_par2(const ITensor& T){
    // here T is sparse matrix ITensor
    Index Ti = T.index(1);
    Index Tj = T.index(2);
    auto di = Ti.dim();
    auto dj = Tj.dim();
    
    // obtain the info of non-zero elements, and store them in val_vec
    std::vector<int> mat_ele(di*dj);
    std::iota (mat_ele.begin(), mat_ele.end(), 1);
    std::vector<pointVal> val_vec = {};
    //std::mutex m;
    //
    auto getNon0 = [&](int n){
        //std::lock_guard<std::mutex> guard(m);
        int i = (n-1)/dj + 1;
        int j = (n-1)%dj + 1;
        double val = eltC(T, Ti=i, Tj=j).real();
        if (val != 0){
            struct pointVal pv;
            pv.i = i; pv.j = j; pv.val = val;
            val_vec.push_back(pv);
        }
    };
    std::for_each(std::execution::par, mat_ele.begin(), mat_ele.end(), getNon0);
    
    // construct a sparse matrix from the pointVal vector
    arma::sp_mat Tmat(di,dj);
    auto fillNon0 = [&](pointVal& pv){
        Tmat(pv.i-1, pv.j-1) = pv.val;
    };
    std::for_each(val_vec.begin(), val_vec.end(), fillNon0);

    //float density = Tmat.n_nonzero / Tmat.n_elem;
    //std::cout << "\ndensity is " << density << std::endl;
    return Tmat;
}




arma::sp_mat extract_sparse_mat_par3(const ITensor& T){
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