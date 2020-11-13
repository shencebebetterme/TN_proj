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
#if 0
 // M is extremely sparse, so choose a different strategy
arma::sp_mat extract_sparse_mat_par(const ITensor& T){
    // here T is sparse matrix ITensor
    Index Ti = T.index(1);
    Index Tj = T.index(2);
    auto di = Ti.dim();
    auto dj = Tj.dim();
    arma::sp_mat Tmat(di,dj);
    double val = 0;
    //  
    
    omp_set_num_threads(num_threads);
#pragma omp parallel for  
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
#endif


// not using multi threading, data race in T?
// then try to generate n armadillo sparse rows in parallel
// then construct an armadillo sparse matrix from these rows??
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