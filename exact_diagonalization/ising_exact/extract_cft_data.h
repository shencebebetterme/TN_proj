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
//#include <omp.h>
//#include <thread>
//#include <random>
//#include <execution>
#include <numeric>

#include "itensor/all.h"
#include "itensor/util/print_macro.h"
#include <armadillo>


using namespace itensor;
using namespace std;
using namespace std::chrono;
using namespace arma;

#define PI 3.1415926535

extern int len_chain;
extern int num_states;

void extract_cft_data(arma::sp_mat& TM_sparse){
    //arma::sp_mat TM_sparse(TM_dense);
    int nT = TM_sparse.n_rows;
    if(nT<num_states){
        printf("\nToo many states requested!\n");
    }
    // eigen
    //printf("\nStart eigenvalue decomposition\n\n");
    //auto eig_start = high_resolution_clock::now();
    cx_vec eigval = eigs_gen(TM_sparse,num_states,"lm",0.0001);
    //auto eig_stop = high_resolution_clock::now();
    //auto eig_duration = duration_cast<seconds>(eig_stop-eig_start).count();
    //printf("Eigenvalue decomposition finished, lasting %d seconds\n\n",eig_duration);
    //eigval.print("eigval=");
    //eigval.print();
    std::vector<double> log_lambdas = {};
    std::vector<double> scaling_dims = {};
    std::vector<double> spins = {};
    //obtain spins
    for (auto lambda_i_tilde : eigval){
        //eigen_vals.push_back(lambda_i_tilde);
        Cplx log_lambda_i_tilde = std::log(lambda_i_tilde);
        log_lambdas.push_back(log_lambda_i_tilde.real());
        spins.push_back(len_chain/(2*PI)*log_lambda_i_tilde.imag());
    }
    //obtain scaling dimensions
    double log_lambda_max = log_lambdas[0];
    for (const auto& log_lambda_i : log_lambdas){
        double i_th_scaling_dim = -len_chain/(2*PI)*(log_lambda_i-log_lambda_max);
        scaling_dims.push_back(i_th_scaling_dim);
    }
    //print
    for(int i = 0; i<num_states; i++){
        printf("%6.6f\t%6.6f\n",scaling_dims[i],spins[i]);
    }
}
