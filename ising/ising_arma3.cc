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

using namespace itensor;
using namespace std;
using namespace std::chrono;
using namespace arma;

double beta_c = 0.5*log(1+sqrt(2));//critical beta
int dim0 = 2;//initial A tensor leg dimension
int len_chain = 6; // the actual length is len_chain + 1
int len_chain_L = 3;
int len_chain_R = 3;
//int period = len_chain + 1;
int num_states = 20;// final number of dots in the momentum diagram



std::vector<int> legToLegs(const int& leg_val, const int& num_inds, const int& dim);
int legsToLeg(const std::vector<int>& vec, const int& num_inds, const int& dim);
template<typename T>
void permute(std::vector<T>& vec);
int ind_translate(const int& input, const int& num_inds, const int& dim);


// convert ITensor matrix to arma matrix
arma::mat extract_mat(const ITensor& T);
arma::mat extract_mat_twisted(const ITensor& T);

void extract_cft_data(const arma::mat& TM);


int main(int argc, char* argv[]){
    if (argc==2) {
        len_chain = atoi(argv[1]);
    }
    // decompose the chain into  two parts
    // stupid way to reduce memory usage and avoid
    // lapack 32bit 64vit dgemm parameter 3 issue
    if (len_chain%2==0) {
        len_chain_L = len_chain_R = len_chain/2;
    }
    else
    {
        len_chain_L = (len_chain-1)/2;
        len_chain_R = len_chain - len_chain_L;
    }
    
    // initial tensor legs
    Index s(dim0);
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
                    //A.set(l(sl), r(sr), u(su), d(sd), 0);
				}
    // normalize A to prevent it from being too large
    double TrA = elt(A * delta(l, r) * delta(u, d));
	A /= (TrA/2);
    Print(TrA);
    //PrintData(A);

//////////////////////////////////////////////////////////////////////////////
    // build left and right transfer matrix independently
    // and contract them in the end
    Index u0L = addTags(u,"L,u_i="+str(0));
    Index d0L = addTags(d,"L,d_i="+str(0));
    Index l0L = addTags(l,"L,site="+str(0));
    Index r0L = addTags(r,"L,site="+str(0));
    Index u0R = addTags(u,"R,u_i=" +str(len_chain_L));
    Index d0R = addTags(d,"R,d_i=" +str(len_chain_L));
    Index l0R = addTags(l,"R,site="+str(len_chain_L));
    Index r0R = addTags(r,"R,site="+str(len_chain_L));
    //initial TM tensor
    ITensor ML = A * delta(u,u0L) * delta(d,d0L) * delta(l,l0L) * delta(r,r0L);
    ITensor MR = A * delta(u,u0R) * delta(d,d0R) * delta(l,l0R) * delta(r,r0R);
    // indexset for decomposition
    std::vector<Index> upL_inds_vec = {u0L};
    std::vector<Index> downL_inds_vec = {d0L};
    std::vector<Index> upR_inds_vec = {u0R};
    std::vector<Index> downR_inds_vec = {d0R};

    for (int i : range1(len_chain_L-1)){
        Index uiL = addTags(u,"L,u_i="+str(i));
        Index diL = addTags(d,"L,d_i="+str(i));
        Index liL = addTags(l,"L,site="+str(i));
        Index riL = addTags(r,"L,site="+str(i));
        //construct the i-th copy of A tensor
        ITensor AiL = A * delta(u,uiL) * delta(d,diL) * delta(r,riL) * delta(l,liL);
        Index previous_riL = addTags(r,"L,site="+str(i-1));
        printf("eating %dth A tensor, left\n",i);
        ML *= (AiL * delta(previous_riL, liL));
        //TM.swapTags("d_i="+str(i),"d_i="+str(i-1));
        upL_inds_vec.push_back(uiL);
        downL_inds_vec.push_back(diL);
    }
    for (int i : range1(len_chain_R-1)){
        Index uiR = addTags(u,"R,u_i="+ str(i+len_chain_L));
        Index diR = addTags(d,"R,d_i="+ str(i+len_chain_L));
        Index liR = addTags(l,"R,site="+str(i+len_chain_L));
        Index riR = addTags(r,"R,site="+str(i+len_chain_L));
        //construct the i-th copy of A tensor
        ITensor AiR = A * delta(u,uiR) * delta(d,diR) * delta(r,riR) * delta(l,liR);
        Index previous_riR = addTags(r,"R,site="+str(i-1+len_chain_L));
        printf("eating %dth A tensor, right\n",i);
        MR *= (AiR * delta(previous_riR, liR));
        //TM.swapTags("d_i="+str(i),"d_i="+str(i-1));
        upR_inds_vec.push_back(uiR);
        downR_inds_vec.push_back(diR);
    }

    Index L_rightmost = addTags(r,"L,site="+str(len_chain_L-1));
    Index R_rightmost = addTags(r,"R,site="+str(len_chain_L+len_chain_R-1));
    printf("\ncombining left and right\n\n");
    // use reference and *= to reduce memory usage
    ITensor& M = ML;
    M *= MR * delta(L_rightmost,l0R) * delta(l0L,R_rightmost);
    Print(M);

    //group up and down indices
    std::vector<Index>& up_inds = upL_inds_vec;
    up_inds.insert(up_inds.end(), upR_inds_vec.begin(), upR_inds_vec.end());
    std::vector<Index>& down_inds = downL_inds_vec;
    down_inds.insert(down_inds.end(), downR_inds_vec.begin(), downR_inds_vec.end());
    // combine the up and down indices into uL and dL
    auto[uLC,uL] = combiner(up_inds);
    auto[dLC,dL] = combiner(down_inds);
    M *= uLC;
    M *= dLC;
    Print(M);
    // now M is a matrix
    //Index Mi = M.index(1);
    //Index Mj = M.index(2);
    auto TM_dense = extract_mat_twisted(M);
    
    extract_cft_data(TM_dense);

    return 0;
}



void extract_cft_data(const arma::mat& TM_dense){
    arma::sp_mat TM_sparse(TM_dense);
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


// convert ITensor to arma matrix
arma::mat extract_mat_twisted(const ITensor& T){
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
    for (auto j : range1(dj)){
        //armadillo index is 0-based
        int twisted_j = ind_translate(j-1,len_chain,dim0)+1;
        //printf("%d\t%d\n",j,twisted_j);
        for (auto i : range1(di)){
            Tmat(i-1,j-1) = T.elt(Ti = i, Tj = twisted_j);
            //////////////////////////
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

//permute to the right
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

