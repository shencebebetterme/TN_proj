#include "itensor/all.h"
#include "itensor/util/print_macro.h"
#include <cmath>
#include <string>
#include <cstring>
#include <vector>
#include <iostream>
#include <complex>
#include <armadillo>

#define PI 3.1415926535

using namespace itensor;
using namespace std;
using namespace arma;

double beta_c = 0.5*log(1+sqrt(2));//critical beta
int dim0 = 2;//initial A tensor leg dimension
int len_chain = 10; // the actual length is len_chain + 1
int period = len_chain + 1;
int num_states = 20;// final number of dots in the momentum diagram

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
    //PrintData(A);

//////////////////////////////////////////////////////////////////////////////
    // transfer matrix
    Index u0 = addTags(u,"site="+str(0));
    Index d0 = addTags(d,"site="+str(0));
    Index l0 = addTags(l,"site="+str(0));
    Index r0 = addTags(r,"site="+str(0));
    //initial M tensor
    ITensor M = A * delta(u,u0) * delta(d,d0) * delta(l,l0) * delta(r,r0);
    // indexset for decomposition
    IndexSet up_inds = IndexSet(u0);
    IndexSet down_inds = IndexSet(d0);

    for (int i : range1(len_chain)){
        // make a copy of the original A tensor
        // with different labels for the indices
        Index ui = addTags(u,"site="+str(i));
        Index di = addTags(d,"site="+str(i));
        Index li = addTags(l,"site="+str(i));
        Index ri = addTags(r,"site="+str(i));
        //construct the i-th copy of A tensor
        ITensor Ai = A * delta(u,ui) * delta(d,di) * delta(r,ri) * delta(l,li);
        Index previous_ri = addTags(r,"site="+str(i-1));
        M *= (Ai * delta(previous_ri, li));
        // add the new up and down indices to the index set
        up_inds = IndexSet(up_inds, ui);
        down_inds = IndexSet(down_inds, di);
    }
    // contract the leftmost leg and the rightmost leg to form a ring
    Index r_last = addTags(r,"site="+str(len_chain));
    M *= delta(l0,r_last);//now M has 2*(len_chain+1) legs
    Print(M);
    //regroup the top indices and bottom indices
    //auto[uLC,uL] = combiner(up_inds);
    //auto[dLC,dL] = combiner(down_inds);
    //ITensor M_comb = M * uLC * dLC; // now M_comb has 2 legs, namely uL, dL
    //Print(M_comb);

//deprecated
    //construct the last A tensor
    //Index u_last = addTags(u,"site="+str(len_chain));
    //Index d_last = addTags(d,"site="+str(len_chain));
    //Index l_last = addTags(l,"site="+str(len_chain));
    //Index r_last = addTags(r,"site="+str(len_chain));
    //ITensor A_last = A * delta(u,u_last) * delta(d,d_last) * delta(l,l_last) * delta(r,r_last);
    //Print(A_last);
    //contract M and the last A to form a ring
    //ITensor M_final = M_comb * A_last * delta(r_near_last,l_last) * delta(l0,r_last);//now M_final is a 4-leg tensor
    //Print(M_final);
   

//////////////////////////////////////////////////////////////////////////////
// only extract singular values of transfer matrix
// because the transfer matrix is a symmetric matrix
// so its eigenvalues are equal to singular values??
#if 0
 //now regroup the 4 indices into two indices, up and down
//IndexSet M_up = IndexSet(uL,u_last);
//IndexSet M_down = IndexSet(dL,d_last);
auto [U,S,V] = svd(M,up_inds,{"MaxDim=",num_states});  
// extract singular values
Index Si = S.index(1);
Index Sj = S.index(2);
//auto [Si,Sj] = S.inds();
//double a = S.elt(Si=1,Sj=1);
std::vector<double> svals = {};//store singular values
std::vector<double> conf_dims = {};//store conformal dimensions
// extract diagonal elements from S
// and obtain the conformal dimension
for (int k : range1(num_states)){
    double k_th_sval = S.elt(Si=k,Sj=k);
    //printfln("%d",k_th_sval);
    svals.push_back(k_th_sval);
    double k_th_conf_dim = -len_chain*(log(svals[k-1])-log(svals[0]))/(2*PI);
    conf_dims.push_back(k_th_conf_dim);
    printfln("%d",k_th_conf_dim);
}
#endif



//////////////////////////////////////////////////////////////////////////////
// compose with translational operator
// i -> i+1 under periodic bc
//auto next = [](int i){
//    return (i+1)%period;
//};
//construct T.M
// the indices of T should be in down_inds and primed up_inds
#if 1
ITensor TM = M;
for (auto i : range(period)){
    //Index ui = addTags(u,"site="+str(i));
    //Index di = addTags(d,"site="+str(i));
    //Index dd_i_next = addTags(u,"site="+str(next(i)));
    //TM = M * delta(di, prime(dd_i_next));//now TM contains pairs of primed and unprimed legs
    Index& di = *(down_inds.begin()+i);
    auto dd_i_next_pt = (i==len_chain)? up_inds.begin() : (up_inds.begin()+i+1);
    TM *= delta(di,prime(*dd_i_next_pt));
}
Print(TM);
//PrintData(TM);
//PrintData(M);
//Print(*up_inds.begin());

//todo: eigenvalue decomposition of TM
printf("\nStarting eigenvalue decomposition of TM\n");
auto [U,D] = eigen(TM);
printf("\nEigenvalue decomposition finished.\n\n");
//PrintData(D);
std::vector<Cplx> eigen_vals = {};
std::vector<double> log_lambdas = {};
std::vector<double> scaling_dims = {};
std::vector<double> spins = {};
Index Di = D.index(1);
Index Dj = D.index(2);
if (Di.dim()<num_states) {println("Too few spins for num_states."); return 1;}

for (auto i : range1(num_states)){
    Cplx lambda_i_tilde = D.eltC(Di=i,Dj=i);
    eigen_vals.push_back(lambda_i_tilde);
    Cplx log_lambda_i_tilde = std::log(lambda_i_tilde);
    log_lambdas.push_back(log_lambda_i_tilde.real());
    spins.push_back(period/(2*PI)*log_lambda_i_tilde.imag());
}

//todo: sort the eigenvalues by modulus
// or export the matrix to armadillo sparse matrix
// and find the k largest eigenvalues

double log_lambda_max = log_lambdas[0];
for (const auto& log_lambda_i : log_lambdas){
    double i_th_scaling_dim = -period/(2*PI)*(log_lambda_i-log_lambda_max);
    scaling_dims.push_back(i_th_scaling_dim);
}

for(int i = 0; i<num_states; i++){
    printfln("%6.6f\t%6.6f",scaling_dims[i],spins[i]);
}
#endif

//use armadillo
arma::mat ar = arma::randu<arma::mat>(3, 3);

//construct M + sa * T  where sa is a small parameter
//todo: write down the explicit tensor form of T, and add it to M
    return 0;
}

//void getM(ITensor& )
//todo: write afunction template to extract diagonal matrix elements