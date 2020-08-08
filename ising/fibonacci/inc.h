#pragma once
// make && ./ising_arma
#include "itensor/all.h"
#include "itensor/util/print_macro.h"
//#include "omp.h"
#include <armadillo>

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
//#include <random>


#define PI 3.1415926535
#define USE_MAINTEXT_CONVENTION 1

using namespace itensor;
using namespace std;
using namespace std::chrono;
using namespace arma;

extern int new_dim0;
extern int len_chain;
extern int num_states;



// in {a,b,c} one is 1 and the other two are 2
// suppose a,b,c \in {1,2}
bool oneIs1(int a, int b, int c){
    //if ((a==1)&&(b==2)&&(c==2)) {return true;}
    //else if ((a==2)&&(b==1)&&(c==2)) {return true;}
    //else if ((a==2)&&(b==2)&&(c==1)) {return true;}
    if (a+b+c == 5) return true;
    else return false;
}




void extract_cft_data(const arma::sp_mat& TM_sparse){
    //arma::sp_mat TM_sparse(TM);
    int nT = TM_sparse.n_rows;
    if(nT<num_states){
        printf("\nToo many states requested!\n");
    }
    // eigen
    printf("\nStarting eigenvalue decomposition\n\n");
    auto eig_start = high_resolution_clock::now();
    cx_vec eigval = eigs_gen(TM_sparse,num_states,"lm",0.0001);
    auto eig_stop = high_resolution_clock::now();
    auto eig_duration = duration_cast<seconds>(eig_stop-eig_start).count();
    printf("Eigenvalue decomposition finished, lasting %d seconds\n\n",eig_duration);
    //eigval.print("eigval");
    //eigval.print("eigval=");
    //eigval.print();
    std::vector<double> log_lambdas = {};
    std::vector<double> scaling_dims = {};
    std::vector<double> spins = {};
    //obtain spins
    for (auto lambda_i_tilde : eigval){
        //eigen_vals.push_back(lambda_i_tilde);
        Cplx log_lambda_i_tilde = std::log(lambda_i_tilde);
        //std::cout<< log_lambda_i_tilde << std::endl;
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
    printf("scal_dim \t spin\n");
    for(int i = 0; i<num_states; i++){
        printf("%6.6f\t%6.6f\n",scaling_dims[i],spins[i]);
    }
}




// the inverse process of combining indices
// e.g.,  len_val =10, num_inds=4, dim=2, then 10=8+2 -> {1,0,1,0}
std::vector<int> legToLegs(const int& leg_val, const int& num_inds, const int& dim){
    //int num_inds = inds.size();
    //leg_val is 0-based
    if (leg_val > std::pow(dim,num_inds)-1) print("\nleg_val is too large for the Legs set\n");
    //initial vector
    std::vector<int> result = {};
    int val = leg_val;
    while (val>=dim)
    {
        int quotient = val/dim;
        int remainder = val%dim;
        result.push_back(remainder);
        val = quotient;
    }
    result.push_back(val);
    // 0 padding
    int len = result.size();
    result.insert(result.end(), num_inds-len, 0);
    //std::reverse(result.begin(), result.end());
    return result;
}

int legsToLeg(const std::vector<int>& vec, const int& num_inds, const int& dim){
    int leg_val = 0;
    if (vec.size()!=num_inds) printf("\ndimension doesn't match\n");
    for (int i=0; i<num_inds; i++){
        leg_val += vec[i]*std::pow(dim,i);
    }
    return leg_val;
}

//translate to the right by one site
template<typename T>
void permute(std::vector<T>& vec){
    int len = vec.size();
    //inline int next(int i) {return (i+1) % len;};
    T last = vec[len-1];
    for (auto i=len-1; i>0; i--){
        vec[i] = vec[i-1];
    }
    vec[0] = last;
}

int ind_translate(const int& input, const int& num_inds, const int& dim){
    auto vec = legToLegs(input, num_inds, dim);
    permute(vec);
    auto output = legsToLeg(vec, num_inds, dim);
    //printf("%d translated to %d\n", input, output);
    return output;
}


 // M is extremely sparse, so choose a different strategy
arma::sp_mat extract_sparse_mat(const ITensor& T, bool twisted=false){
    // here T is sparse matrix ITensor
    Index Ti = T.index(1);
    Index Tj = T.index(2);
    auto di = Ti.dim();
    auto dj = Tj.dim();
    arma::sp_mat Tmat(di,dj);
    //
    for (auto it : iterInds(T)){
        double val = T.real(it);
        if (val != 0){
            IndexVal iv_i = it[0];//iv_i is 1-based
            IndexVal iv_j = it[1];
            int i = iv_i.val -1;// i is 0-based
            int j = iv_j.val -1;
            // the parameter leg_val is 0-based
            if (twisted) {
                int twisted_j = ind_translate(j,len_chain,new_dim0);
                printf("j is %d, twisted_j is %d\n", j, twisted_j);
                Tmat(i,twisted_j) = val;
            }
            else Tmat(i,j) = val;
        }
    }
    return Tmat;
}