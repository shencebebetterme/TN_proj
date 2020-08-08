// make && ./ising_arma
#include "itensor/all.h"
#include "itensor/util/print_macro.h"
#include "omp.h"
#include <armadillo>

#include <cmath>
#include <string>
#include <cstring>
#include <vector>
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <complex>
#include <cstdlib>
#include <chrono>


#define PI 3.1415926535

using namespace itensor;
using namespace std;
using namespace std::chrono;
using namespace arma;

double beta_c = 0.5*log(1+sqrt(2));//critical beta
int dim0 = 2;//initial A tensor leg dimension
int len_chain = 5; // the actual length is len_chain + 1
int period = len_chain + 1;
int num_states = 20;// final number of dots in the momentum diagram

// convert ITensor to arma matrix
arma:mat extract_mat(const ITensor& T){
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
    int c = 0;
    double progress=0;
    for (auto i : range1(di)){
        for (auto j=1; j<=dj; j++){
            //armadillo index is 0-based
            Tmat(i-1,j-1) = T.elt(Ti=i,Tj=j);
            c++;
            if ((c%200000) == 0){
                progress = c*100.0/(di*dj);
                printf("Converting %.2f %%\n",progress);
            }
        }
    }
    auto extract_stop = high_resolution_clock::now();
    auto extract_duration = duration_cast<seconds>(extract_stop-extract_start).count();
    printf("Conversion finished, lasting %d seconds\n\n",extract_duration);
    return Tmat;
}

//done: add an arg parser

int main(int argc, char* argv[]){
    if (argc==2) {
        len_chain = atoi(argv[1]);
        period = len_chain+1;
    }
    Index s(dim0);
    // initial tensor legs
    Index u = addTags(s,"up");
    Index r = addTags(s,"right");
    Index d = addTags(s,"down");
    Index l = addTags(s,"left");

    // the initial tensor for Ising
    ITensor A = ITensor(u,r,d,l);

    // Fill the A tensor with correct Boltzmann weights:
	auto Sig = [](int s) { return 1. - 2. * (s - 1); };
    // 1 -> 1
    // 2-> -1
	for (auto sl : range1(dim0))
		for (auto sd : range1(dim0))
			for (auto sr : range1(dim0))
				for (auto su : range1(dim0))
				{
					auto E = Sig(sl) * Sig(sd) + Sig(sd) * Sig(sr)
						+ Sig(sr) * Sig(su) + Sig(su) * Sig(sl);
					auto P = exp(E * beta_c);
					A.set(l(sl), r(sr), u(su), d(sd), P);
				}
    // normalize A to prevent it from being too large
    double TrA = elt(A * delta(l, r) * delta(u, d));
	A /= TrA;
    Print(TrA);
    //PrintData(A);

//////////////////////////////////////////////////////////////////////////////
    // transfer matrix
    Index u0 = addTags(u,"u_i="+str(0));
    Index d0 = addTags(d,"d_i="+str(0));
    Index l0 = addTags(l,"site="+str(0));
    Index r0 = addTags(r,"site="+str(0));
    //initial TM tensor
    ITensor TM = A * delta(u,u0) * delta(d,d0) * delta(l,l0) * delta(r,r0);
    //Print(M);
    //ITensor TM = M;
    // indexset for decomposition
    std::vector<Index> up_inds_vec = {u0};
    std::vector<Index> down_inds_vec = {d0};
    std::vector<Index> down_inds_shift_vec = {};
    //IndexSet up_inds = IndexSet(u0);
    //IndexSet down_inds = IndexSet(d0);

    for (int i : range1(len_chain)){
        // make a copy of the original A tensor
        // with different labels for the indices
        Index ui = addTags(u,"u_i="+str(i));
        Index di = addTags(d,"d_i="+str(i));
        Index li = addTags(l,"site="+str(i));
        Index ri = addTags(r,"site="+str(i));
        //construct the i-th copy of A tensor
        ITensor Ai = A * delta(u,ui) * delta(d,di) * delta(r,ri) * delta(l,li);
        Index previous_ri = addTags(r,"site="+str(i-1));
        //Print(M);
        //M *= (Ai * delta(previous_ri, li));
        //M /= TrA;
        //Print(M);
        TM *= (Ai * delta(previous_ri, li));
        //TM /= TrA;
        TM.swapTags("d_i="+str(i),"d_i="+str(i-1));
        //
        up_inds_vec.push_back(ui);
        down_inds_vec.push_back(di);
        down_inds_shift_vec.push_back(di);
    }
    down_inds_shift_vec.push_back(d0);
    // contract the leftmost leg and the rightmost leg to form a ring
    Index r_last = addTags(r,"site="+str(len_chain));
    //M *= delta(l0,r_last);//now M has 2*(len_chain+1) legs
    TM *= delta(l0,r_last);
    //Print(M);
    Print(TM);
    //PrintData(M-TM);

    IndexSet up_inds(up_inds_vec);
    IndexSet down_inds(down_inds_vec);
    IndexSet down_inds_shift(down_inds_shift_vec);
    //Print(down_inds==down_inds_shift);
    //ITensor TM = swapInds(M,IndexSet(down_inds_vec),IndexSet(down_inds_shift_vec));
    //ITensor TM = permute(M,IndexSet(down_inds_vec));
    //PrintData(M-TM);

    //regroup the top indices and bottom indices to make a TM matrix
    auto[uLC,uL] = combiner(up_inds);
    auto[dLC,dL] = combiner(down_inds);
    //auto[dsLC,dsL] = combiner(IndexSet(down_inds_shift_vec));
    //ITensor Mmat = M * uLC * dLC;
    //double TrMmat = elt(Mmat * delta(uL,dL));
    //Mmat /= TrMmat;
    TM *= uLC;
    TM *= dLC;
    //Print(TM);
    double TrTM = elt(TM * delta(uL,dL));
    Print(TrTM);
    //TM /= TrTM;
    //PrintData(Mmat-TMmat);
    //PrintData(TM);
    
    /// use armadillo to deal with TM
    auto TM_matrix_dense = extract_mat(TM);
    sp_mat TM_matrix(TM_matrix_dense);

    int nT = TM_matrix.n_rows;
    if(nT<num_states){
        printf("\nToo many states requested!\n");
        return 0;
    }

    // eigen
    printf("\nStarting eigenvalue decomposition\n\n");
    auto eig_start = high_resolution_clock::now();
    cx_vec eigval = eigs_gen(TM_matrix,num_states,"lm",0.0001);
    auto eig_stop = high_resolution_clock::now();
    auto eig_duration = duration_cast<seconds>(eig_stop-eig_start).count();
    printf("Eigenvalue decomposition finished, lasting %d seconds\n\n",eig_duration);
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
        spins.push_back(period/(2*PI)*log_lambda_i_tilde.imag());
    }
    //obtain scaling dimensions
    double log_lambda_max = log_lambdas[0];
    for (const auto& log_lambda_i : log_lambdas){
        double i_th_scaling_dim = -period/(2*PI)*(log_lambda_i-log_lambda_max);
        scaling_dims.push_back(i_th_scaling_dim);
    }
    //print
    for(int i = 0; i<num_states; i++){
        printf("%6.6f\t%6.6f\n",scaling_dims[i],spins[i]);
    }

    return 0;
}

//void getM(ITensor& )
//todo: write afunction template to extract diagonal matrix elements